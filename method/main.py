import os
import json
import argparse
import numpy as np
from eeppr import EEPPR
from datetime import datetime
from matplotlib import pyplot as plt, use
from utils import setup_roi, list_to_dict
from loader import get_sequence_path_roi
from logger import setup_logger, logger, assert_and_log, raise_and_log
from metavision_core.event_io import RawReader


def parse_arguments():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Measure the frequency of periodic phenomena '
        '(rotation, vibration, flicker, etc.) in an event-based sequence using the EEPPR method.')
    parser.add_argument('--file', '-f', required=True, type=str,
                        help=f'Filepath to the file to read events from (.raw) or name of a sequence from EEPPR dataset: {seq_names}')
    parser.add_argument('--roi_coords', '-rc', type=int, nargs=4, metavar=('X0', 'Y0', 'X1', 'Y1'),
                        help='RoI coordinates of the object to track (X0 Y0 X1 Y1)')
    parser.add_argument('--full_resolution', '-fr', action='store_true',
                        help='Flag to set ROI to full resolution of the input file')
    parser.add_argument('--aggreg_t', '-t', type=int, default=100,
                        help='Events aggregation interval in microseconds (default: 100)')
    parser.add_argument('--read_t', '-r', type=int,
                        help='Number of microseconds to read events from the file')
    parser.add_argument('--full_seq_analysis', '-fsa', action='store_true',
                        help='Analyze the whole sequence at the step length of --read_t microseconds, updating the template every step (default: False)')
    parser.add_argument('--aggreg_fn', '-afn', type=str, default='median', choices=['mean', 'median', 'max', 'min'],
                        help='Name of a NumPy function used to aggregate measurements from all windows (default: median)')
    parser.add_argument('--decimals', '-dp', type=int, default=1,
                        help='Number of decimal places to round the result to (default: 1)')
    parser.add_argument('--skip_roi_gui', '-srg', action='store_true',
                        help='Flag to skip the RoI setup GUI if --roi_coords are provided')
    parser.add_argument('--win_size', '-w', type=int, default=45,
                        help='Window size in pixels (default: 45, recommended not to change, see our paper)')
    parser.add_argument('--event_count', '-N', type=int, default=1800,
                        help='The threshold for template event count (default: 1800, recommended not to change, see our paper)')
    parser.add_argument('--viz_corr_resp', '-vcr', action='store_true',
                        help='Visualize correlation responses for each window')
    parser.add_argument('--all_results', '-ar', action='store_true',
                        help='Output estimates from all windows (NumPy 2D array X x Y) and all correlation responses (NumPy 3D array X x Y x read_t/aggreg_t)')
    parser.add_argument('--device', '-d', type=str, default='cuda:0',
                        help='Device to run 3D correlation computations on (default: cuda:0)')
    parser.add_argument('--log', '-l', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level (default: INFO)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose mode')
    parser.add_argument('--output_dir', '-o', type=str,
                        help='Name of output directory (default: ./eeppr_out)', default='./eeppr_out')

    args = parser.parse_args()
    if args.read_t is None:
        if args.file in seq_names and not args.roi_coords:
            args.read_t = 1000000
        if args.full_resolution:
            args.read_t = 250000
        else:
            args.read_t = 500000

    return args


def log_results(result, freq_arr, full=True, prefix=''):
    if full:
        # Log the frequency estimation for each window
        logger.info("Estimated frequency per window:")
        for row in freq_arr:
            formatted_row = [
                f'{np.round(freq, args.decimals):.{args.decimals}f}'
                if freq > 0 else 'X' for freq in row]
            logger.info('[' + ' | '.join(f'{{:>{5 + args.decimals}}}'.format(item)
                                         for item in formatted_row) + ']')

    logger.info(f"{prefix}Estimated {args.aggreg_fn} frequency: {result} Hz")


def main(args):
    """
    Main function to run the EEPPR method.

    Args:
        args (argparse.Namespace): Command-line arguments.
    """
    setup_logger(run_dir, args.log)
    assert_and_log(0 < args.aggreg_t < args.read_t,
                   "Aggregation interval must be greater than 0 and less than read_t, "
                   "reccomended 100")
    assert_and_log(args.aggreg_t < args.read_t,
                   "Read time must be greater than aggreg_t, reccomended 1000000")
    assert_and_log(args.win_size > 0,
                   "Window size must be greater than 0, reccomended 45")
    assert_and_log(args.event_count > 0,
                   "Event count threshold must be greater than 0, reccomended 1800")

    logger.info(
        f"EEPPR method started with sequence '{args.file}', "
        f"{f'Full sequence analysis with step length {args.read_t} us' if args.full_seq_analysis else f'Partial sequence analysis of length {args.read_t} us'}, "
        f"Window size: {args.win_size}px, "
        f"Event count template threshold: {args.event_count}, "
        f"Aggregation interval: {args.aggreg_t} us, "
        f"Aggregation function: {args.aggreg_fn}")

    # Check if file is from EEPPR dataset
    if args.file in seq_names:
        args.file, args.roi_coords = get_sequence_path_roi(args.file)
    else:
        assert_and_log(os.path.exists(args.file),
                       f"File {args.file} not found")
        assert_and_log(args.file.lower().endswith('.raw'),
                       "File must be in .raw format")

    raw_reader = RawReader(args.file, max_events=int(4e7))

    if isinstance(args.roi_coords, list):
        args.roi_coords = list_to_dict(args.roi_coords)

    # Verify, modify or set new ROI coordinates
    if args.full_resolution:
        args.roi_coords = {'x0': 0, 'y0': 0,
                           'x1': raw_reader.get_size()[0],
                           'y1': raw_reader.get_size()[1]}
    if args.roi_coords is None and not args.skip_roi_gui:
        args.roi_coords = setup_roi(raw_reader, args.roi_coords, args.win_size)
    logger.debug(f"Selected RoI: {args.roi_coords}")

    # Initialize and run EEPPR
    eeppr = EEPPR(args, run_dir, raw_reader)

    if args.full_seq_analysis:
        results = []
        freq_arrs = []
        corr_arrs = []
        i_step = 0
        while True:
            # Run EEPPR analysis for each time segment
            result, freq_arr, corr_arr = eeppr.run()

            # Break the loop if no more data is available
            if result is None or freq_arr is None or corr_arr is None:
                break

            # Append results to respective lists
            results.append(result)
            freq_arrs.append(freq_arr)
            corr_arrs.append(corr_arr)

            # Log the results for the current segment
            log_results(result, freq_arr,
                        prefix=f'Sequence step {i_step * args.read_t}--{(i_step + 1) * args.read_t}us: ')
            i_step += 1

        # Convert lists to numpy arrays
        results = np.array(results)
        freq_arrs = np.stack(freq_arrs)
        corr_arrs = np.concatenate(corr_arrs, axis=2)
        if args.all_results:
            return results, freq_arrs, corr_arrs
        else:
            return results
    else:
        # Run EEPPR analysis for a single time segment
        result, freq_arr, corr_arr = eeppr.run()
        if result is None or freq_arr is None or corr_arr is None:
            return None
        # Log the results
        log_results(result, freq_arr)

        if args.all_results:
            return result, freq_arr, corr_arr
        else:
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
        raise_and_log(
            FileNotFoundError,
            f"Dataset configuration file {config_filepath} not found."
            "Please verify you have downloaded the whole EEPPR repository.")
    seq_names = config_data['sequence_names']

    args = parse_arguments()
    run_dir = os.path.join(
        args.output_dir, f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')
    os.makedirs(run_dir, exist_ok=True)
    print(main(args))
