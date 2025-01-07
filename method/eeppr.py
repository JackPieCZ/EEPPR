import argparse
import os

import einops
import matplotlib.pyplot as plt
import numpy as np
import sparse
import torch
from matplotlib import ticker
from scipy.optimize import fminbound
from scipy.signal import find_peaks
from torch.nn import functional as F
from tqdm.auto import tqdm

from logger import logger
from utils import load_events, quantize_events, find_template_depth


class EEPPR:
    def __init__(self, args: argparse.Namespace, run_dir: str, raw_reader):
        """
        Initialize the EEPPR method.

        Args:
            args (argparse.Namespace): Command-line arguments.
        """
        self.reader = raw_reader
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
        self.run_dir = run_dir
        self.log_level = args.log
        self.win_coords = -1, -1
        self.verbose = args.verbose

    def run(self):
        """
        Runs the EEPPR method.

        Returns:
            float: Estimated frequency in Hz.
        """
        if self.log_level in ['DEBUG', 'INFO']:
            main_pbar = tqdm(total=5, desc='Running EEPPR method', leave=False,
                             bar_format="{l_bar}|{bar}| {n_fmt}/{total_fmt}{postfix}")
            main_pbar.set_description('Loading events...')
        logger.debug(f'Loading events from input file: {self.input_file_path=}, {self.read_t=}, {self.roi_coords=}')
        # Load events from input file
        sparse_ev_arr = load_events(
            self.input_file_path, self.reader, self.read_t, self.roi_coords)
        if sparse_ev_arr is None:
            # The sparse array is empty, end of sequence reached
            if self.log_level in ['DEBUG', 'INFO']:
                main_pbar.close()
            return None, None, None

        if self.log_level in ['DEBUG', 'INFO']:
            main_pbar.update(1)
            main_pbar.set_description('Quantizing events...')
        # Quantize events and convert to torch tensor
        logger.debug(f'Quantizing events: {self.aggreg_t=}')
        quantized_events = torch.from_numpy(quantize_events(
            self.input_file_path, self.read_t, self.roi_coords, self.aggreg_t)[None])
        if self.log_level in ['DEBUG', 'INFO']:
            main_pbar.update(1)

        # Calculate the number of windows in x and y directions
        roi_width = self.roi_coords['x1'] - self.roi_coords['x0']
        roi_height = self.roi_coords['y1'] - self.roi_coords['y0']
        x_windows_num = (roi_width + 1) // self.win_size
        y_windows_num = (roi_height + 1) // self.win_size

        # Create an unfold function to partition the events into windows
        if self.log_level in ['DEBUG', 'INFO']:
            main_pbar.set_description('Partitioning events...')
        logger.debug(f'Partitioning events into windows: {x_windows_num=}, {y_windows_num=}, {self.win_size=}')

        unfolded_events = self.unfold(quantized_events, kernel_size=(
            self.win_size, self.win_size), stride=self.win_size)
        del quantized_events
        partitioned_events = einops.rearrange(unfolded_events, '1 (b w h) (y x) -> b w h y x',
                                              x=x_windows_num, y=y_windows_num, w=self.win_size, h=self.win_size) \
            .to(self.device)

        # Delete the original events tensor to free up memory
        del unfolded_events
        if self.log_level in ['DEBUG', 'INFO']:
            main_pbar.update(1)
            main_pbar.set_description('Computing 3D correlation...')
        # Perform 3D correlation on the event data
        logger.debug('Performing 3D correlation on the event data')
        estim_freq_arr, corr_out = self.correlate_3d(sparse_ev_arr, x_windows_num, y_windows_num, partitioned_events)
        if self.log_level in ['DEBUG', 'INFO']:
            main_pbar.update(1)

        # Delete the sparse array and partitioned events tensor to free up memory
        del sparse_ev_arr, partitioned_events

        # Apply the aggregation function to aggregate the estimations from all windows
        if self.log_level in ['DEBUG', 'INFO']:
            main_pbar.set_description('Aggregating results...')
        aggreg_fn = getattr(np, self.aggreg_fn.lower())
        result = np.round(aggreg_fn(estim_freq_arr[estim_freq_arr > 0]), self.precision)
        if self.log_level in ['DEBUG', 'INFO']:
            main_pbar.update(1)
            main_pbar.close()

        return result, estim_freq_arr, corr_out

    def unfold(self, input_tensor, kernel_size, stride=1, padding=0, dilation=1):
        """
        Re-implementation of the torch.nn.Unfold function as the original only supports torch.float16/32 input tensors.

        Args:
            input_tensor (torch.Tensor): Input tensor of shape (N, C, H, W).
            kernel_size (int or tuple): Size of the sliding blocks.
            stride (int or tuple, optional): Stride of the sliding blocks. Default: 1.
            padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0.
            dilation (int or tuple, optional): Spacing between kernel elements. Default: 1.

        Returns:
            torch.Tensor: Unfolded tensor of shape (N, C * prod(kernel_size), L).
        """
        # Ensure kernel_size, stride, padding, and dilation are tuples
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        # Pad the input tensor
        input_tensor = F.pad(input_tensor, (padding[1], padding[1], padding[0], padding[0]))

        # Extract dimensions
        N, C, H, W = input_tensor.shape
        KH, KW = kernel_size
        SH, SW = stride
        DH, DW = dilation

        # Calculate output dimensions
        OH = (H - (DH * (KH - 1) + 1)) // SH + 1
        OW = (W - (DW * (KW - 1) + 1)) // SW + 1

        # Use unfold to extract sliding blocks
        unfolded = input_tensor.unfold(2, KH, SH).unfold(3, KW, SW)

        # Rearrange the dimensions to match the expected output
        unfolded = unfolded.permute(0, 2, 3, 1, 4, 5).contiguous()
        unfolded = unfolded.view(N, OH * OW, C * KH * KW).transpose(1, 2)

        return unfolded

    def correlate_3d(self, sparse_ev_arr: sparse.COO, x_windows_num: int, y_windows_num: int,
                     partitioned_events: torch.Tensor):
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
        corr_arr = np.full((x_windows_num, y_windows_num, self.read_t // self.aggreg_t), np.nan)
        freq_arr = np.full((x_windows_num, y_windows_num), -1.0)
        logger.debug(f'Number of windows: {x_windows_num=}, {y_windows_num=}')

        # Iterate over each window in the grid
        if self.log_level in ['DEBUG', 'INFO']:
            pbar = tqdm(np.ndindex(x_windows_num, y_windows_num), total=x_windows_num * y_windows_num, position=1,
                        leave=False)
        else:
            pbar = np.ndindex(x_windows_num, y_windows_num)
        for x_win, y_win in pbar:
            self.win_coords = (x_win, y_win)
            logger.debug(f'Analysing events within window x={x_win}, y={y_win}')
            if self.log_level in ['DEBUG', 'INFO']:
                pbar.set_description(f'Window x={x_win}, y={y_win}')

            # Calculate the start and end coordinates for the current window
            x_start = x_win * self.win_size + self.roi_coords['x0']
            x_end = x_start + self.win_size
            y_start = y_win * self.win_size + self.roi_coords['y0']
            y_end = y_start + self.win_size

            # Extract the sparse array for the current window
            sparse_win_arr = sparse_ev_arr[x_start:x_end, y_start:y_end]

            # Skip if the number of non-zero elements is less than the template event count
            if sparse_win_arr.nnz < self.template_event_count:
                logger.debug(
                    f'Number of events ({sparse_win_arr.nnz}) in the window is less than the '
                    f'template event count ({self.template_event_count}). Skipping window {self.win_coords}.')
                if self.verbose:
                    tqdm.write(
                        f'Number of events ({sparse_win_arr.nnz}) in the window is less than the '
                        f'template event count ({self.template_event_count}). Skipping window {self.win_coords}.')
                continue

            # Find the depth of the template
            template_depth = find_template_depth(sparse_win_arr, self.template_event_count, self.read_t, self.aggreg_t)
            logger.debug(f'Template depth: {template_depth}, max depth: {self.read_t // self.aggreg_t}')

            # Skip if the template depth is too sparse
            if template_depth >= (self.read_t // self.aggreg_t) / 8:
                logger.debug(f'Template events are too sparse. Skipping window {self.win_coords}.')
                if self.verbose:
                    tqdm.write(f'Template events are too sparse. Skipping window {self.win_coords}.')
                continue

            # Extract the template and window tensors
            template = partitioned_events[:template_depth, :, :, y_win, x_win]
            window = partitioned_events[:, :, :, y_win, x_win]

            # Perform 3D convolution
            corr_out = F.conv3d(window[None][None].float(), template[None][None].float()).squeeze()

            # Skip if the second highest value in the correlation output is non-positive
            if torch.topk(corr_out, 2)[0][-1] <= 0:
                logger.debug(f'No positive correlation responses found. Skipping window  {self.win_coords}.')
                if self.verbose:
                    tqdm.write(f'No positive correlation responses found. Skipping window  {self.win_coords}.')
                continue

            # Convert the correlation output to a numpy array
            corr_out = corr_out.detach().cpu().numpy()
            corr_arr[x_win, y_win, :corr_out.size] = corr_out

            # Find periodic peaks in the correlation output
            peaks = self.find_periodic_peaks(corr_out)
            period_timemarks = np.multiply(peaks, self.aggreg_t).flatten()
            if period_timemarks.size < 2:
                logger.debug(f'Less than 2 peaks found. Skipping window  {self.win_coords}.')
                if self.verbose:
                    tqdm.write(f'Less than 2 peaks found. Skipping window  {self.win_coords}.')
            periods = np.diff(period_timemarks)

            # Skip if no periods are found
            if periods.size == 0:
                logger.debug(f'No periods found. Skipping window  {self.win_coords}.')
                if self.verbose:
                    tqdm.write(f'No periods found. Skipping window  {self.win_coords}.')
                continue

            # Calculate the median rotations per second for the current window
            freq_arr[x_win, y_win] = np.median(1e6 / periods)
            logger.debug(f'Estimated frequency: {freq_arr[x_win, y_win]} Hz')

        # Return the array of estimated rotations per second for each window
        return freq_arr, corr_arr

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
                self.viz_corr_resps(peaks, properties, corr_out, min_peak_height)
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
            peaks, properties = find_peaks(corr_out, height=opt_min_peak_height, distance=opt_min_peak_dist)
            # If visualization is enabled, visualize the correlation responses
            if self.viz_corr_resp:
                self.viz_corr_resps(peaks, properties, corr_out, opt_min_peak_height)
            # Return the found peaks
            return peaks

    def viz_corr_resps(self, peaks: np.ndarray, properties: dict, conv_out: np.ndarray, lower_bound: float) -> None:
        """
        Visualize the correlation responses.

        Args:
            peaks (np.ndarray): Array of peak indices.
            properties (dict): Dictionary of peak properties.
            conv_out (np.ndarray): Array of correlation responses.
            lower_bound (float): Lower bound of peaks.

        Returns:
            None
        """
        fig, ax = plt.subplots(dpi=300, figsize=(8, 4.5))
        fontsize = 14

        # Normalize the correlation response
        lower_bound /= np.max(conv_out)
        properties['peak_heights'] /= np.max(conv_out)
        conv_out /= np.max(conv_out)

        plt.title('Correlation responses for window x={}, y={}'.format(
            self.win_coords[0], self.win_coords[1]), fontsize=fontsize)
        plt.ylabel("Normalized correlation response", fontsize=fontsize)
        plt.xlabel("Time in milliseconds", fontsize=fontsize)
        plt.xlim(0, np.max(peaks))
        plt.ylim(max(np.min(conv_out) - 0.025, 0),
                 np.partition(conv_out, -2)[-2] + 0.025)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)

        # Format x-axis labels to display time in milliseconds
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: '{:.0f}'.format(x / (1000 / self.aggreg_t))))
        ax.plot(conv_out, label='Correlation responses')
        ax.plot(peaks, properties['peak_heights'], "x", label='Peaks', markersize=fontsize)
        plt.plot(np.full_like(conv_out, lower_bound), "--", color="gray", label='Lower bound of peaks')
        plt.legend(loc='lower right', fontsize=fontsize)
        plt.tight_layout()
        plt.grid()

        if self.viz_corr_resp:
            plt.savefig(os.path.join(self.run_dir, f'corr_resps_win-{self.win_coords[0]}-{self.win_coords[1]}.jpg'))
        plt.close('all')
