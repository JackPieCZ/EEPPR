import matplotlib.pyplot as plt
import numpy as np
import sparse
from matplotlib.widgets import Button, Slider
from metavision_core.event_io import EventsIterator
from metavision_sdk_core import BaseFrameGenerationAlgorithm, RoiFilterAlgorithm

from logger import logger


def generate_event_frame(events: np.ndarray, height: int, width: int) -> np.ndarray:
    """
    Generate a frame for the given events.

    Args:
        events (np.ndarray): The events to be displayed.
        height (int): The height of the frame.
        width (int): The width of the frame.

    Returns:
        np.ndarray: An RGB numpy array of the frame.
    """
    image = np.zeros((height, width, 3), dtype=np.uint8)
    BaseFrameGenerationAlgorithm.generate_frame(events, image)
    return image[:, :, ::-1]  # Convert BGR to RGB


def setup_roi(raw_reader, roi: dict, win_size: int) -> dict:
    """
    Set up the Region of Interest (ROI) for event processing.

    Args:
        raw_reader (RawReader): RawReader object for the video file.
        roi (dict): Initial ROI coordinates.
        win_size (int): Size of a window See paper for details.

    Returns:
        dict: Selected ROI coordinates.
    """
    selected_roi = []
    windows_patches = []
    height, width = raw_reader.get_size()
    acc_events = raw_reader.load_n_events(5e5)
    image = generate_event_frame(acc_events, height, width)

    fig, ax = plt.subplots(6, 1, figsize=(12, 9.5), gridspec_kw={'height_ratios': [5, .4, .1, .1, .1, .2]})
    plt.get_current_fig_manager().window.wm_geometry("+0+0")

    def remove_windows(windows_patches):
        """Remove all window patches from the plot."""
        for patch in windows_patches:
            patch.remove()
        windows_patches.clear()

    def draw_windows(x, y, size, win_size):
        """Draw windows on the plot according to RoI."""
        if size < win_size:
            return
        remove_windows(windows_patches)
        if size < 2 * win_size:
            window_patch = plt.Rectangle(
                (x, y), win_size, win_size, facecolor='none', edgecolor='blue', linewidth=1)
            ax_img.add_patch(window_patch)
            windows_patches.append(window_patch)
        else:
            for i in range(size // win_size):
                for j in range(size // win_size):
                    window_patch = plt.Rectangle((x + i * win_size, y + j * win_size),
                                                 win_size, win_size, facecolor='none', edgecolor='blue', linewidth=1)
                    ax_img.add_patch(window_patch)
                    windows_patches.append(window_patch)

    def update_roi_info(prefix):
        """Update the ROI info text."""
        windows_count = (size.val // win_size) ** 2
        text.set_text(
            f"{prefix} Currently {windows_count} windows for analysis. "
            f"Estimated compute time: {0.3 * windows_count:.1f} - {1.1 * windows_count:.1f} sec.")

    def update(_):
        """Update the position and size of the RoI."""
        pending_roi.set_xy((x.val, y.val))
        pending_roi.set_width(size.val)
        pending_roi.set_height(size.val)
        draw_windows(x.val, y.val, size.val, win_size)
        update_roi_info('Confirm or change the RoI.')
        fig.canvas.draw_idle()

    def add_roi(_):
        """Add or replace the current ROI."""
        nonlocal current_roi
        selected_roi.clear()
        selected_roi.append({
            'x0': int(x.val),
            'y0': int(y.val),
            'x1': int(x.val) + int(size.val),
            'y1': int(y.val) + int(size.val)
        })
        current_roi.remove()
        current_roi = plt.Rectangle((x.val, y.val), size.val, size.val, facecolor='none', edgecolor='orange')
        ax_img.add_patch(current_roi)
        logger.info(f"Set new RoI {selected_roi[-1]}")
        update_roi_info('New RoI set.')
        fig.canvas.draw_idle()

    def fullscreen_roi(_):
        selected_roi.clear()
        selected_roi.append({
            'x0': 0,
            'y0': 0,
            'x1': width,
            'y1': height
        })
        logger.info(f"Set new RoI {selected_roi[-1]}")
        update_roi_info('New RoI set.')
        plt.close(fig)

    def save(_):
        """Close the figure and save the ROI."""
        plt.close(fig)

    def reset(_):
        """Reset the sliders and restore the previous ROI if available."""
        nonlocal current_roi
        x.reset()
        y.reset()
        size.reset()
        if roi:
            selected_roi.clear()
            selected_roi.append(roi)
            current_roi.remove()
            current_roi = plt.Rectangle((roi['x0'], roi['y0']), roi['x1'] - roi['x0'],
                                        roi['y1'] - roi['y0'], facecolor='none', edgecolor='orange', linewidth=1)
            ax_img.add_patch(current_roi)
        else:
            add_roi(None)
        fig.canvas.draw_idle()

    # Set up axes
    ax_img, ax_text, ax_x, ax_y, ax_size, ax_buttons = ax
    ax_text.axis('off')
    ax_buttons.axis('off')

    ax_img.imshow(image)
    text = ax_text.text(0.5, 0.5, '', horizontalalignment='center', verticalalignment='center', fontsize=16)

    # Create sliders
    size = Slider(ax_size, 'Size', 45, 720, valinit=roi['x1'] - roi['x0'] if roi else 200, valstep=1)
    x = Slider(ax_x, 'X0', 0, width, valinit=roi['x0'] if roi else width // 2 - size.val // 2, valstep=1,
               color='green')
    y = Slider(ax_y, 'Y0', 0, height, valinit=roi['y0'] if roi else height // 2 - size.val // 2, valstep=1,
               color='green')

    # Create buttons
    ax_reset = fig.add_axes([0.29, 0.05, 0.1, 0.075])
    ax_addroi = fig.add_axes([0.4, 0.05, 0.1, 0.075])
    ax_save = fig.add_axes([0.51, 0.05, 0.1, 0.075])
    ax_fullscreen = fig.add_axes([0.62, 0.05, 0.1, 0.075])

    save_button = Button(ax_save, 'Confirm & Exit')
    reset_button = Button(ax_reset, 'Reset to default')
    addroi_button = Button(ax_addroi, 'Set RoI')
    fullscreen_button = Button(
        ax_fullscreen, "Analyse full\nresolution instead")

    # Setup initial square
    pending_roi = plt.Rectangle((x.val, y.val), size.val, size.val,
                                facecolor='none', edgecolor='red', linewidth=1)
    ax_img.add_patch(pending_roi)
    current_roi = pending_roi

    if roi:
        current_roi = plt.Rectangle((roi['x0'], roi['y0']), roi['x1'] - roi['x0'],
                                    roi['y1'] - roi['y0'], facecolor='none', edgecolor='orange', linewidth=1)
        ax[0].add_patch(current_roi)
        selected_roi.append(roi)
        draw_windows(roi['x0'], roi['y0'], roi['x1'] - roi['x0'], win_size)
        update_roi_info(
            'Confirm the RoI or use sliders to modify it, then press Set RoI.\n')
    else:
        draw_windows(x.val, y.val, size.val, win_size)
        update_roi_info('Use sliders to select a RoI, then press Set RoI.')
        add_roi(None)

    fig.canvas.draw_idle()

    # Connect event handlers
    x.on_changed(update)
    y.on_changed(update)
    size.on_changed(update)
    reset_button.on_clicked(reset)
    save_button.on_clicked(save)
    addroi_button.on_clicked(add_roi)
    fullscreen_button.on_clicked(fullscreen_roi)

    plt.show()
    raw_reader.reset()
    return selected_roi[0]


def list_to_dict(coords: list) -> dict:
    """
    Convert a list of coordinates to a dictionary.

    Args:
        coords (list): List of coordinates [x0, y0, x1, y1].

    Returns:
        dict: Dictionary with keys 'x0', 'y0', 'x1', 'y1'.
    """
    x0, y0, x1, y1 = coords
    if x0 >= x1 or y0 >= y1:
        raise ValueError(
            "Invalid coordinates: x0 must be less than x1 and y0 must be less than y1")
    return {'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1}


def load_events(video_path: str, raw_reader, microseconds_to_read: int, roi_dict: dict,
                reset_reader=False) -> sparse.COO:
    """
    Load events from the video file within the specified ROI.

    Args:
        video_path (str): Path to the video file.
        microseconds_to_read (int): Number of microseconds to read.
        roi_dict (dict): ROI coordinates.

    Returns:
        sparse.COO: Sparse array of events.
    """
    if reset_reader:
        # Reset the reader to the beginning of the video
        raw_reader.reset()
    # Load events for the specified duration
    events = raw_reader.load_delta_t(microseconds_to_read)
    if events.size == 0:
        logger.info(f"EOF reached or no events found in the video file {video_path}")
        return
    # Normalize event timestamps to start from zero
    events['t'] -= events['t'].min()

    # Initialize the ROI filter with the provided coordinates
    template_roi_filter = RoiFilterAlgorithm(
        **roi_dict, output_relative_coordinates=False)
    filtered_events_buffer = template_roi_filter.get_empty_output_buffer()
    # Filter the events based on the ROI
    template_roi_filter.process_events(events, filtered_events_buffer)
    # Convert the filtered events buffer to a numpy array
    events = filtered_events_buffer.numpy()

    # Create a sparse array from the filtered events
    sparse_arr = sparse.COO(
        np.array((events['x'], events['y'], events['p'], events['t'])),
        data=True,
        shape=(raw_reader.width, raw_reader.height, 2, microseconds_to_read))

    return sparse_arr


def quantize_events(video_path: str, microseconds_to_analyze: int, roi_dict: dict, aggreg_t: int) -> np.ndarray:
    """
    Quantize events into a 3D array.

    Args:
        video_path (str): Path to the video file.
        microseconds_to_analyze (int): Number of microseconds to analyze.
        roi_dict (dict): ROI coordinates.
        aggreg_t (int): Aggregation time interval.

    Returns:
        np.ndarray: Quantized events array.
    """
    start_x, start_y, end_x, end_y = roi_dict.values()

    # Calculate the number of intervals based on the total time to analyze and the aggregation time
    num_intervals = int(microseconds_to_analyze / aggreg_t)

    quantized_events = np.zeros(shape=(num_intervals, end_y - start_y + 1, end_x - start_x + 1), dtype=np.int8)

    ev_it = EventsIterator(video_path, delta_t=aggreg_t, max_duration=microseconds_to_analyze)

    it = 0
    for ev in ev_it:
        if ev.size == 0 and it == 0:
            # If the interval at the start of the sequence contains no events, skip to the next interval
            # This is due to the fact that RAW files always start with a few ms of no events
            continue

        # Create a mask to filter events within the ROI bounds
        in_bounds_mask = (ev['y'] >= start_y) & \
                         (ev['y'] < end_y) & \
                         (ev['x'] >= start_x) & \
                         (ev['x'] < end_x)

        # Store events with relative coordinates and normalized polarity (0, 1) -> (-1, 1)
        quantized_events[it, ev['y'][in_bounds_mask] - start_y,
                         ev['x'][in_bounds_mask] - start_x] = ev['p'][in_bounds_mask] * 2 - 1
        it += 1

    return quantized_events


def find_template_depth(sparse_win_arr: sparse.COO, template_event_count: int, read_t: int, aggreg_t: int) -> int:
    """
    Find the appropriate depth for the template using binary search.

    Args:
        sparse_win_arr (sparse.COO): Sparse array of events in the window.
        template_event_count (int): Desired number of events in the template.
        read_t (int): Total time to read in microseconds.
        aggreg_t (int): Aggregation time interval.

    Returns:
        int: Appropriate depth for the template.
    """
    # initilize lower and upper bounds
    left = 0
    right = read_t // aggreg_t

    while left < right:
        mid = (left + right) // 2
        # Count the number of non-zero elements in the sparse array
        event_count = sparse_win_arr[:, :, :, :mid * aggreg_t].nnz
        if event_count < template_event_count:
            # If the event count is less than the desired count, move the left pointer to mid + 1
            left = mid + 1
        else:
            # If the event count is greater than or equal to the desired count, move the right pointer to mid - 1
            right = mid - 1
    return left


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
