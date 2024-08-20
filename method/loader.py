import os
import gdown
import json
from logger import assert_and_log, logger


def download_sequence(sequence_name: str, file_path: str, url_id: str) -> None:
    """
    Download the sequence file from its URL and save it to the specified file path.
    """
    # Get the download URL from the dictionary
    if url_id is None:
        raise ValueError(
            f"No download URL found for sequence '{sequence_name}'.")

    gdown.download(
        f'https://drive.google.com/uc?id={url_id}', file_path, quiet=False)


def get_sequence_path_roi(sequence_name: str) -> str:
    """
    Get the file path for a given sequence name from EEPPR dataset.
    """
    # Get the path of the EEPPR dataset directory
    dataset_dir = os.path.join(os.path.dirname(
        os.path.dirname(__file__)), 'dataset')

    # Load the dataset configuration data
    with open(os.path.join(dataset_dir, 'config.json'), 'r') as f:
        config_data = json.load(f)
        f.close()
    seq_names = config_data['sequence_names']

    # Ensure the provided sequence name is in the list of valid names
    assert_and_log(sequence_name in seq_names,
                   f"Sequence {sequence_name} not found. The sequence name must be one of {seq_names}")

    seq_info = config_data['sequence_info'][sequence_name]
    file_path = os.path.join(dataset_dir, seq_info['raw_filepath'])

    # Remove temporary index files if they exist
    for file in os.listdir(os.path.dirname(file_path)):
        if file.endswith(".tmp_index"):
            try:
                os.remove(os.path.join(os.path.dirname(file_path), file))
            except PermissionError:
                logger.warning(
                    f"Unable to remove temporary index file {file}. The file will be removed on the next run.")

    # Check if the file exists, if not, attempt to download it from Metavision dataset
    if not os.path.exists(file_path):
        if seq_info['id']:
            download_sequence(sequence_name, file_path, seq_info['id'])
        else:
            raise FileNotFoundError(
                f"File {file_path} not found.")
    assert_and_log(os.path.exists(file_path), f"File {file_path} not found")

    # Load the proposed ROI coordinates for the sequence
    roi_coords = seq_info['roi']
    assert_and_log(roi_coords, f"No ROI coordinates found for sequence {sequence_name}. "
                   "Please open an issue on GitHub.")
    return file_path, roi_coords


if __name__ == '__main__':
    # Example usage -> prints the file path and roi of the given sequence
    print(get_sequence_path_roi('led'))
    print(get_sequence_path_roi('screen'))
