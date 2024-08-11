import os
import json
import argparse
import numpy as np
from ee3p3d import EE3P3D
from datetime import datetime
from matplotlib import pyplot as plt, use
from utils import setup_roi, list_to_dict
from loader import get_sequence_path_roi
from logger import setup_logger, logger, assert_and_log
from metavision_core.event_io import RawReader


def main(args):
    """
    Main function to run the EE3P3D method.

    Args:
        args (argparse.Namespace): Command-line arguments.
    """
    setup_logger(run_dir, args.log)
    assert_and_log(0 < args.aggreg_t < args.read_t,
                   "Aggregation interval must be greater than 0 and less than read_t, reccomended 100")
    assert_and_log(args.aggreg_t < args.read_t,
                   "Read time must be greater than aggreg_t, reccomended 1000000")
    assert_and_log(args.win_size > 0,
                   "Window size must be greater than 0, reccomended 45")
    assert_and_log(args.event_count > 0,
                   "Event count threshold must be greater than 0, reccomended 1800")

    logger.info(
        f"EE3P3D method started with sequence '{args.file}', Window size: {args.win_size}px, Event count threshold: {args.event_count}, Aggregation interval: {args.aggreg_t} us, Read time: {args.read_t} us, Aggregation function: {args.aggreg_fn}")

    # Check if file is from EE3P3D dataset
    if args.file in seq_names:
        args.file, args.roi_coords = get_sequence_path_roi(args.file)
    else:
        assert_and_log(os.path.exists(args.file),
                       f"File {args.file} not found")
        assert_and_log(args.file.lower().endswith(
            '.raw'), "File must be in .raw format")

    raw_reader = RawReader(args.file, max_events=int(3e7))

    if isinstance(args.roi_coords, list):
        args.roi_coords = list_to_dict(args.roi_coords)
    # Verify, modify or set new ROI coordinates
    if args.roi_coords is None or not args.skip_roi_gui:
        args.roi_coords = setup_roi(
            raw_reader, args.roi_coords, args.win_size)
    logger.debug(f"Selected RoI: {args.roi_coords}")

    # Initialize and run EE3P3D
    ee3p3d = EE3P3D(args, run_dir, raw_reader)
    result, freq_arr = ee3p3d.run()

    if args.all_results:
        # Log the frequency estimation for each window
        logger.info("Estimated frequency per window:")
        for row in freq_arr:
            formatted_row = [
                f'{np.round(freq, args.decimals):.{args.decimals}f}' if freq > 0 else 'X' for freq in row]
            logger.info('[' + ' | '.join(f'{{:>{5 + args.decimals}}}'.format(item)
                                         for item in formatted_row) + ']')

        # Log the results of the other aggregation functions
        results = []
        for func_name in ['min', 'median', 'mean', 'max']:
            value = getattr(np, func_name)(freq_arr[freq_arr > 0])
            results.append(
                f"{func_name.capitalize()}: {value:.{args.decimals}f} Hz")
        logger.info(" | ".join(results))
        return freq_arr
    else:
        logger.info(f"Estimated {args.aggreg_fn} frequency: {result} Hz")
        return result


if __name__ == "__main__":
    # Configure matplotlib and numpy settings
    use('TkAgg')
    plt.rcParams["font.family"] = "Times New Roman"
    np.set_printoptions(suppress=True, formatter={'float_kind': '{:f}'.format})

    # Load the dataset configuration data
    dataset_dir = os.path.join(os.path.dirname(
        os.path.dirname(__file__)), 'dataset')
    try:
        config_filepath = os.path.join(dataset_dir, 'config.json')
        with open(config_filepath, 'r') as f:
            config_data = json.load(f)
            f.close()
    except FileNotFoundError:
        logger.error(
            f"Dataset configuration file {config_filepath} not found. Please verify you have downloaded the whole EE3P3D repository.")
        raise FileNotFoundError(
            f"Dataset configuration file {config_filepath} not found. Please verify you have downloaded the whole EE3P3D repository.")
    seq_names = config_data['sequence_names']

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Measure the frequency of periodic phenomena (rotation, vibration, flicker, etc.) in event-based sequence using the EE3P3D method.')
    parser.add_argument('--file', '-f', required=True, type=str,
                        help=f'Filepath to the file to read events from (.raw) or name of a sequence from EE3P3D dataset: {seq_names}')
    parser.add_argument('--roi_coords', '-rc', type=int, nargs=4, metavar=('X0', 'Y0', 'X1', 'Y1'),
                        help='RoI coordinates of the object to track (X0 Y0 X1 Y1)')
    parser.add_argument('--aggreg_t', '-t', type=int, default=100,
                        help='Events aggregation interval in microseconds (default: 100)')
    parser.add_argument('--read_t', '-r', type=int, default=1000000,
                        help='Number of microseconds to read events from the file (default: 1000000)')
    parser.add_argument('--aggreg_fn', '-afn', type=str, default='median', choices=['mean', 'median', 'max', 'min'],
                        help='Name of a NumPy function used to aggregate measurements from all windows (default: median)')
    parser.add_argument('--decimals', '-dp', type=int, default=1,
                        help='Number of decimal places to round the result to (default: 1)')
    parser.add_argument('--skip_roi_gui', '-srg', action='store_true',
                        help='Flag to skip the RoI setup GUI if --roi_coords are provided')
    parser.add_argument('--win_size', '-w', type=int, default=45,
                        help='Window size in pixels (default: 45, recommended not to change, see our paper)')
    parser.add_argument('--event_count', '-N', type=int, default=1800,
                        help='Threshold for template event count (default: 1800, recommended not to change, see our paper)')
    parser.add_argument('--viz_corr_resp', '-vcr', action='store_true',
                        help='Visualize correlation responses for each window')
    parser.add_argument('--all_results', '-ar', action='store_true',
                        help='Show results from all windows')
    parser.add_argument('--device', '-d', type=str, default='cuda:0',
                        help='Device to run 3D correlation computations on (default: cuda:0)')
    parser.add_argument('--log', '-l', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level (default: INFO)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose mode')
    parser.add_argument('--output_dir', '-o', type=str,
                        help='Name of output directory (default: ./ee3p3d_out)', default='./ee3p3d_out')

    args = parser.parse_args()
    run_dir = os.path.join(
        args.output_dir, f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')
    os.makedirs(run_dir, exist_ok=True)
    print(main(args))
