import argparse
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
from logger import logger


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
        # Load events from input file
        sparse_ev_arr = load_events(
            self.input_file_path, self.read_t, self.roi_coords)
        
        # Quantize events and convert to torch tensor
        quantized_events = torch.from_numpy(quantize_events(self.input_file_path, self.read_t, self.roi_coords, self.aggreg_t)[None])\
            .to(self.device)

        # Calculate the number of windows in x and y directions
        roi_width = self.roi_coords['x1'] - self.roi_coords['x0']
        roi_height = self.roi_coords['y1'] - self.roi_coords['y0']
        x_windows_num = (roi_width + 1) // self.win_size
        y_windows_num = (roi_height + 1) // self.win_size

        # Create an unfold function to partition the events into windows
        unfold_fn = Unfold(kernel_size=(self.win_size,
                           self.win_size), stride=self.win_size)
        partitioned_events = einops.rearrange(unfold_fn(quantized_events), '1 (b w h) (y x) -> b w h y x',
                                              x=x_windows_num, y=y_windows_num, w=self.win_size, h=self.win_size)

        # Perform 3D correlation on the event data
        estim_rps_per_win = self.correlate_3d(
            sparse_ev_arr, x_windows_num, y_windows_num, partitioned_events)
        
        # If no estimations are found, double the aggreg_t value and run again
        if estim_rps_per_win is None:
            self.aggreg_t *= 2
            return self.run()

        # Apply the aggregation function to aggregate the estimations from all windows
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
        # Initialize the result array with -1.0
        win_rps_arr = np.full((x_windows_num, y_windows_num), -1.0)

        # Iterate over each window in the grid
        for x_win, y_win in tqdm(np.ndindex(x_windows_num, y_windows_num), desc='Analysing events within windows', total=x_windows_num * y_windows_num):
            logger.debug(
                f'Processing window ({x_win}, {y_win}), {self.win_size}x{self.win_size} px, total windows: {x_windows_num * y_windows_num}')
            
            # Calculate the start and end coordinates for the current window
            x_start = x_win * self.win_size + self.roi_coords['x0']
            x_end = x_start + self.win_size
            y_start = y_win * self.win_size + self.roi_coords['y0']
            y_end = y_start + self.win_size

            # Extract the sparse array for the current window
            sparse_win_arr = sparse_ev_arr[x_start:x_end, y_start:y_end]
            
            # Skip if the number of non-zero elements is less than the template event count
            if sparse_win_arr.nnz < self.template_event_count:
                continue

            # Find the depth of the template
            template_depth = find_template_depth(
                sparse_win_arr, self.template_event_count, self.read_t, self.aggreg_t)
            logger.debug(
                f'Template depth: {template_depth}, max depth: {self.read_t // self.aggreg_t}')

            # Skip if the template depth is too sparse
            if template_depth >= (self.read_t // self.aggreg_t) / 8:
                logger.debug(
                    'Template events are too sparse. Skipping window.')
                continue

            # Extract the template and window tensors
            template = partitioned_events[:template_depth, :, :, y_win, x_win]
            window = partitioned_events[:, :, :, y_win, x_win]

            # Perform 3D convolution
            corr_out = F.conv3d(window[None][None].float(),
                                template[None][None].float()).squeeze()

            # Skip if the second highest value in the correlation output is non-positive
            if torch.topk(corr_out, 2)[0][-1] <= 0:
                continue

            # Convert the correlation output to a numpy array
            corr_out = corr_out.detach().cpu().numpy()

            # Calculate the derivative of the correlation output
            derivative = np.diff(corr_out)
            max_derivative = np.max(np.abs(derivative))
            
            # Skip if the maximum derivative is too high
            if max_derivative > 3900:
                logger.debug(f'Max derivative: {max_derivative}')
                return

            # Find periodic peaks in the correlation output
            peaks = self.find_periodic_peaks(corr_out)
            period_timemarks = np.multiply(peaks, self.aggreg_t).flatten()
            periods = np.diff(period_timemarks)
            
            # Skip if no periods are found
            if periods.size == 0:
                continue
            
            # Calculate the median rotations per second for the current window
            win_rps_arr[y_win, x_win] = np.median(1e6 / periods)
            logger.debug(f'Estimated frequency: {np.median(1e6 / periods)} Hz')

        # Return the array of estimated rotations per second for each window
        return win_rps_arr

    def find_periodic_peaks(self, corr_out: np.ndarray) -> np.ndarray:
            """
            Find periodic peaks in the correlation responses.

            Args:
                corr_out (np.ndarray): Correlation response array.

            Returns:
                np.ndarray: Array of peak indices.
            """
            # Calculate the minimum peak height as the average of the second highest value and the maximum of the minimum value and 0
            min_peak_height = (np.partition(corr_out, -2)
                               [-2] + max(np.min(corr_out), 0)) // 2
            # Find peaks in the correlation output with the calculated minimum peak height
            peaks, properties = find_peaks(corr_out, height=min_peak_height)

            # Check if the standard deviation of the differences between timemarks of peaks is less than 5
            if np.std(np.diff(peaks)) < 5:
                # If visualization is enabled, visualize the correlation responses
                if self.viz_corr_resp:
                    viz_corr_resps(peaks, properties, corr_out,
                                   self.aggreg_t, min_peak_height)
                # Return the found peaks
                return peaks
            else:
                # Optimize the minimum peak distance to minimize the standard deviation of the differences between peaks
                opt_min_peak_dist = fminbound(
                    func=lambda dist: np.std(
                        np.diff(find_peaks(corr_out, height=min_peak_height, distance=dist)[0])),
                    x1=2,
                    x2=160,
                    xtol=.1,
                    disp=1)

                # Optimize the minimum peak height to minimize the standard deviation of the differences between peaks
                opt_min_peak_height = fminbound(
                    func=lambda bound: np.std(
                        np.diff(find_peaks(corr_out, height=bound, distance=opt_min_peak_dist)[0])),
                    x1=min_peak_height / 2,
                    x2=min_peak_height * 1.2,
                    xtol=.1,
                    disp=1)

                # Find peaks with the optimized minimum peak height and distance
                peaks, properties = find_peaks(
                    corr_out, height=opt_min_peak_height, distance=opt_min_peak_dist)
                # If visualization is enabled, visualize the correlation responses
                if self.viz_corr_resp:
                    viz_corr_resps(peaks, properties, corr_out,
                                   self.aggreg_t, opt_min_peak_height)
                # Return the found peaks
                return peaks
