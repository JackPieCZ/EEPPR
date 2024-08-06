import argparse
import logging
import numpy as np
import sparse
import torch
from torch.nn import functional as F
from torch.nn import Unfold
import einops
from tqdm import tqdm
from scipy.optimize import fminbound
from scipy.signal import find_peaks
from utils import load_events, quantize_events, find_template_depth, viz_corr_resps

logger = logging.getLogger(__name__)


class EE3P3D:
    def __init__(self, args: argparse.Namespace):
        """
        Initialize the EE3P3D method.

        Args:
            args (argparse.Namespace): Command-line arguments.
        """
        self.input_file_path = args.file
        self.output_folder_path = args.output_dir
        self.aggreg_t = args.aggreg_t
        self.read_t = args.read_t
        self.win_size = args.win_size
        self.template_event_count = args.event_count
        self.roi_coords = args.roi_coords
        self.aggreg_fn = args.aggreg_fn
        self.precision = args.decimals
        self.viz_corr_resp = args.viz_corr_resp
        self.device = torch.device(args.device)

    def run(self):
        """
        Run the EE3P3D method.

        Returns:
            float: Estimated frequency in Hz.
        """
        sparse_ev_arr = load_events(
            self.input_file_path, self.read_t, self.roi_coords)
        quantized_events = torch.from_numpy(quantize_events(self.input_file_path, self.read_t, self.roi_coords, self.aggreg_t)[None])\
            .to(self.device)

        roi_width = self.roi_coords['x1'] - self.roi_coords['x0']
        roi_height = self.roi_coords['y1'] - self.roi_coords['y0']

        x_windows_num = (roi_width + 1) // self.win_size
        y_windows_num = (roi_height + 1) // self.win_size

        unfold_fn = Unfold(kernel_size=(self.win_size,
                           self.win_size), stride=self.win_size)
        partitioned_events = einops.rearrange(unfold_fn(quantized_events), '1 (b w h) (y x) -> b w h y x',
                                              x=x_windows_num, y=y_windows_num, w=self.win_size, h=self.win_size)

        estim_rps_per_win = self.correlate_3d(
            sparse_ev_arr, x_windows_num, y_windows_num, partitioned_events)
        if estim_rps_per_win is None:
            self.aggreg_t *= 2
            return self.run()

        aggreg_fn = getattr(np, self.aggreg_fn)
        result = np.round(
            aggreg_fn(estim_rps_per_win[estim_rps_per_win > 0]), self.precision)

        return result

    def correlate_3d(self, sparse_ev_arr: sparse.COO, x_windows_num: int, y_windows_num: int, partitioned_events: torch.Tensor):
        """
        Perform 3D correlation on the event data.

        Args:
            sparse_ev_arr (sparse.COO): Unfiltered sparse array of events.
            x_windows_num (int): Number of windows in x direction.
            y_windows_num (int): Number of windows in y direction.
            partitioned_events (torch.Tensor): Partitioned events tensor.

        Returns:
            np.ndarray: Array of estimated rotations per second for each window.
        """
        win_rps_arr = np.full((x_windows_num, y_windows_num), -1.0)

        for x_win, y_win in tqdm(np.ndindex(x_windows_num, y_windows_num), desc='Analysing events within windows', total=x_windows_num * y_windows_num):
            x_start = x_win * self.win_size + self.roi_coords['x0']
            x_end = x_start + self.win_size
            y_start = y_win * self.win_size + self.roi_coords['y0']
            y_end = y_start + self.win_size

            sparse_win_arr = sparse_ev_arr[x_start:x_end,
                                           y_start:y_end]
            if sparse_win_arr.nnz < self.template_event_count:
                continue

            template_depth = find_template_depth(
                sparse_win_arr, self.template_event_count, self.read_t, self.aggreg_t)
            logger.debug(
                f'Template depth: {template_depth}, max depth: {self.read_t // self.aggreg_t}')

            if template_depth >= (self.read_t // self.aggreg_t) / 8:
                logger.debug(
                    'Template events are too sparse. Skipping window.')
                continue

            template = partitioned_events[:template_depth, :, :, y_win, x_win]
            window = partitioned_events[:, :, :, y_win, x_win]

            corr_out = F.conv3d(window[None][None],
                                template[None][None]).squeeze()

            if torch.topk(corr_out, 2)[0][-1] <= 0:
                continue

            corr_out = corr_out.detach().cpu().numpy()

            derivative = np.diff(corr_out)
            max_derivative = np.max(np.abs(derivative))
            logger.debug(f'Max derivative: {max_derivative}')
            if max_derivative > 3900:
                return

            peaks = self.find_periodic_peaks(corr_out)
            period_timemarks = np.multiply(peaks, self.aggreg_t).flatten()
            periods = np.diff(period_timemarks)
            if periods.size == 0:
                continue
            win_rps_arr[y_win, x_win] = np.median(1e6 / periods)
            logger.debug(f'Estimated frequency: {np.median(1e6 / periods)} Hz')

        return win_rps_arr

    def find_periodic_peaks(self, corr_out: np.ndarray) -> np.ndarray:
        """
        Find periodic peaks in the correlation reponses.

        Args:
            corr_out (np.ndarray): Correlation reponse array.

        Returns:
            np.ndarray: Array of peak indices.
        """
        min_peak_height = (np.partition(corr_out, -2)
                           [-2] + max(np.min(corr_out), 0)) // 2
        peaks, properties = find_peaks(corr_out, height=min_peak_height)

        if np.std(np.diff(peaks)) < 5:
            if self.viz_corr_resp:
                viz_corr_resps(peaks, properties, corr_out,
                               self.aggreg_t, min_peak_height)
            return peaks
        else:
            opt_min_peak_dist = fminbound(
                func=lambda dist: np.std(
                    np.diff(find_peaks(corr_out, height=min_peak_height, distance=dist)[0])),
                x1=2,
                x2=160,
                xtol=.1,
                disp=1)

            opt_min_peak_height = fminbound(
                func=lambda bound: np.std(
                    np.diff(find_peaks(corr_out, height=bound, distance=opt_min_peak_dist)[0])),
                x1=min_peak_height / 2,
                x2=min_peak_height * 1.2,
                xtol=.1,
                disp=1)

            peaks, properties = find_peaks(
                corr_out, height=opt_min_peak_height, distance=opt_min_peak_dist)
            if self.viz_corr_resp:
                viz_corr_resps(peaks, properties, corr_out,
                               self.aggreg_t, opt_min_peak_height)
            return peaks
