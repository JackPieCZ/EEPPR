import os
import gdown

# List of valid sequence names
SEQ_NAMES = [
    'highcontrastline',
    'velcro_front',
    'velcro_side',
    'highcontrastdot',
    'handspinner',
    'spider',
    'led',
    'screen',
    'speaker',
    'motor',
    'chain_side',
    'chain_top'
]

METAVISION_SEQ_LINKS = {
    'highcontrastdot': r'1PwRXsNgZjrjYcKG6KqXSy07W70TrVRRp',
    'handspinner': r'1I_hL-MP8J6QGMZvXNUaReoihxfysxx5N',
    'motor': r'1p0b8hM5HzxgvptC4z6Itw27c9NizOZs7'
}


def download_sequence(sequence_name: str, file_path: str) -> None:
    """
    Download the sequence file from its URL and save it to the specified file path.
    """
    # Get the download URL from the dictionary
    url = METAVISION_SEQ_LINKS.get(sequence_name)
    if url is None:
        raise ValueError(
            f"No download URL found for sequence '{sequence_name}'.")

    gdown.download(
        f'https://drive.google.com/uc?id={url}', file_path, quiet=False)


def get_sequence_file_path(sequence_name: str) -> str:
    """
    Get the file path for a given sequence name from EE3P3D dataset.
    """
    # Ensure the provided sequence name is in the list of valid names
    assert sequence_name in SEQ_NAMES, f"Sequence {sequence_name} not found."\
        "The sequence name must be one of {SEQ_NAMES}"
        
    # Get the path of the EE3P3D dataset directory
    base_dir = os.path.join(os.path.dirname(
        os.path.dirname(__file__)), 'dataset')

    # List all subfolders in the base directory
    subfolders = [f.name for f in os.scandir(base_dir) if f.is_dir()]
    # Find the first subfolder that contains the sequence name
    sequence_dir = next((f for f in subfolders if sequence_name in f), None)
    # If no such subfolder exists, raise an error
    if sequence_dir is None:
        raise ValueError(
            f"No subfolder containing '{sequence_name}' found in {base_dir}.")

    # Construct the file path by joining the base directory, sequence directory, and file name
    file_path = os.path.join(base_dir, sequence_dir, f'{sequence_dir}.raw')

    # Check if the file exists, if not, attempt to download it from Metavision dataset
    if not os.path.exists(file_path):
        if sequence_name in METAVISION_SEQ_LINKS:
            download_sequence(sequence_name, file_path)
        else:
            raise FileNotFoundError(
                f"File {file_path} not found.")
    assert os.path.exists(file_path), f"File {file_path} not found"
    return file_path


if __name__ == '__main__':
    # Example usage -> prints the file path of the 'highcontrastline' sequence
    print(get_sequence_file_path('highcontrastline'))
