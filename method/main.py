import argparse
import json
import logging
import os
import numpy as np
from matplotlib import pyplot as plt, use
from ee3p3d import EE3P3D
from utils import setup_roi, list_to_dict
from loader import get_sequence_path_roi


def main(args):
    """
    Main function to run the EE3P3D method.

    Args:
        args (argparse.Namespace): Command-line arguments.
    """
    logger.debug('Test1')
    logger.info(f"EE3P3D method started with sequence '{args.file}'")

    # Check if file is from EE3P3D dataset
    if args.file in seq_names:
        args.file, args.roi_coords = get_sequence_path_roi(args.file)
    else:
        assert os.path.exists(args.file), f"File {args.file} not found"

    if isinstance(args.roi_coords, list):
        args.roi_coords = list_to_dict(args.roi_coords)
    # Verify, modify or set new ROI coordinates
    if args.roi_coords is None or not args.skip_roi_gui:
        args.roi_coords = setup_roi(args.file, args.roi_coords, args.win_size)
    logger.info(f"Selected RoI: {args.roi_coords}")

    # Initialize and run EE3P3D
    ee3p3d = EE3P3D(args)
    result = ee3p3d.run()

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
    with open(os.path.join(dataset_dir, 'config.json'), 'r') as f:
        config_data = json.load(f)
    seq_names = config_data['sequence_names']

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Analyse event-based recording using EE3P3D method')
    parser.add_argument('--file', '-f', required=True, type=str,
                        help=f'Filepath to the file to read events from (.raw) or name of a sequence from EE3P3D dataset: {seq_names}')
    parser.add_argument('--aggreg_t', '-t', type=int, default=100,
                        help='Events aggregation interval in microseconds (default: 100)')
    parser.add_argument('--read_t', '-r', type=int, default=1000000,
                        help='Number of microseconds to read events from the file (default: 1000000)')
    parser.add_argument('--win_size', '-w', type=int, default=45,
                        help='Window size in pixels (default: 45, recommended not to change, see our paper)')
    parser.add_argument('--event_count', '-N', type=int, default=1800,
                        help='Threshold for template event count (default: 1800, recommended not to change, see our paper)')
    parser.add_argument('--roi_coords', '-c', type=int, nargs=4, metavar=('X0', 'Y0', 'X1', 'Y1'),
                        help='RoI coordinates of the object to track (X0 Y0 X1 Y1)')
    parser.add_argument('--skip_roi_gui', action='store_true',
                        help='Flag to skip the RoI setup GUI if --roi_coords are provided')
    parser.add_argument('--device', '-d', type=str, default='cuda:0',
                        help='Device to run 3D correlation computations on (default: cuda:0)')
    parser.add_argument('--aggreg_fn', '-afn', type=str, default='median', choices=['mean', 'median', 'max', 'min'],
                        help='Function used to aggregate measurements from all windows (default: median)')
    parser.add_argument('--log', '-l', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level (default: INFO)')
    parser.add_argument('--decimals', '-dp', type=int, default=1,
                        help='Number of decimal places to round the result to (default: 1)')
    parser.add_argument('--viz_corr_resp', '-vcr', action='store_true',
                        help='Visualize correlation responses for each window')

    args = parser.parse_args()
    print(main(args))