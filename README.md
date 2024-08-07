[![arXiv](https://img.shields.io/badge/arXiv-0000.00000-b31b1b.svg)](https://arxiv.org/abs/0000.00000) [![DOI:10.48550/ARXIV.0000.00000](https://zenodo.org/badge/doi/10.48550/ARXIV.0000.00000.svg)](https://doi.org/10.48550/arXiv.0000.00000)

# EE3P3D: Event-based Estimation of Periodic Phenomena Frequency using 3D Correlation
<!-- Kolář, J., Špetlík, R., Matas, J. (2024) EE3P3D: Event-based Estimation of Periodic Phenomena Frequency using 3D Correlation. In Proceedings of , 2024
 -->

Paper Link: (soon)

## Dataset
The dataset features 12 sequences of periodic phenomena (rotation - `01-06`, flicker - `07-08`, vibration - `09-10` and movement - `11-12`) with GT frequencies ranging from 3.2Hz up to 2000Hz in `.raw` and `.hdf5` file formats.

Data capture demonstration: [Youtube video](https://youtu.be/QlfQtvbaYy8)

Each event-based sequence was rendered as a video in 3 playback speeds: [Youtube playlist](https://www.youtube.com/playlist?list=PLK466i9CoYqQ2780OXJg7WgtUtWMEqbkS)

![Ground-truth frequencies of experiments](./dataset/xx_images/experiments_freqs.png)

Sequences `04_highcontrastdot`, `05_handspinner` and `10_motor` originate from the Metavision Dataset. To download these sequences, use the `method/loader.py` script. The `method/main.py` script handles the downloading of these sequences automatically.

## Method

![Method's diagram](https://github.com/user-attachments/assets/90d8ddb3-23a7-4fa9-a24c-ee5b2560b80d)

1. Data captured from an event camera is aggregated into a 3D array,    
2. the array is split into same-sized areas, and in each area, a template depth is automatically selected,
3. a 3D correlation of the template with the event stream is computed,
4. a frequency is calculated from the median of time deltas measured between correlation peaks for each window,
5. the output frequency is computed as a median of measurements from all windows.

### Prerequisites
1. **Anaconda**: For managing the Python environment.
   - Download and install Miniconda from the [official Anaconda website](https://docs.anaconda.com/miniconda/miniconda-install/).

3. **CUDA Toolkit**: This method supports GPU acceleration for 3D correlation computing using CUDA PyTorch. The method was tested and the repo contains environment setups for CUDA versions `11.7`, `12.1` and `12.4`. If none of these versions are compatible with your system, a `cpu only` environment setup is available as well.
   - Download and install the appropriate version from the [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive).

4. ** Metavision SDK **: The Metavision SDK is required for I/O operations with event-based sequences (.raw or .hdf5). File formats from other event camera manufacturers are currently not supported. However, if you implemented support for one, feel free to open a pull request.
   - Follow installation instructions for [Windows](https://docs.prophesee.ai/4.5.2/installation/windows.html#chapter-installation-windows) or [Linux](https://docs.prophesee.ai/4.5.2/installation/linux.html#chapter-installation-linux). Check out their [Operating System Support](https://docs.prophesee.ai/4.5.2/installation/index.html#operating-system-support)

### Installation

1. Clone the repository:
```console
git clone https://github.com/JackPieCZ/EE3P3D.git
cd EE3P3D
```

2. Create a new Anaconda environment:
- If you have CUDA XX.Y version installed:
```console
conda env create -f ./setup/environment_cudaXX_Y.yml
```
- If you prefer to only use CPU:
```console
conda env create -f ./setup/environment_cpu.yml
```

3. Activate the environment:
```console
conda activate ee3p3d
```

4. Verifying the Installation:

To verify that the environment with CUDA is set up correctly, you can run a quick CUDA check:

```python
python
import torch
print(f"CUDA is available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"PyTorch version: {torch.__version__}")
exit()
```

### Usage

After setting up your environment, you can run the EE3P3D method using the main.py script. Here are some example commands:

1. Basic usage with a file (`--file`/`-f`):
```console
python ./method/main.py --file path/to/your/event_file.raw
```

2. For analysing any sequence from the EE3P3D dataset simply enter the sequence name. For example, `led`. For all sequence names, check the `dataset` folder or `dataset/config.json`.
```console
python ./method/main.py -f led
```

3. Specifying RoI coordinates (optional, `--roi_coords`/`-rc`) in `X0 Y0 X1 Y1` format. RoI for sequences from the dataset are provided automatically:
```console
python ./method/main.py -f path/to/your/event_file.raw -rc 100 100 300 300
```

4. A simple GUI is presented to the user for verifying, selecting, modifying and replacing the RoI. If `--roi_coords` are provided (or the sequence if from the EE3P3D dataset), the GUI can be skipped by using `--skip_roi_gui`/`-srg` flag:
```console
python ./method/main.py -f path/to/your/event_file.raw -rc 100 100 300 300 --skip_roi_gui
```

5. Using a different aggregation function (`mean`, `median`, `max`, `min` or other NumPy functions) for aggregating measurements from all windows (optional, `--aggreg_fn`/`-afn`). By default, `median` is used.
```console
python ./method/main.py -f path/to/your/event_file.raw --aggreg_fn mean
```

6. Visualize correlation responses and their peaks for each window (optional, `--viz_corr_resp`/`-vcr`):
```console 
python ./method/main.py -f path/to/your/event_file.raw --aggreg_fn mean
```

7. Running on a specific device (optional, `--device`/`-d`, default is `cuda:0`):
```console
python ./method/main.py -f path/to/your/event_file.raw --device cuda:1
python ./method/main.py -f path/to/your/event_file.raw --device cpu
```

8. For a full list of available options, run:
```console
python ./method/main.py -h 

usage: main.py [-h] --file FILE [--roi_coords X0 Y0 X1 Y1] [--aggreg_t AGGREG_T] [--read_t READ_T] [--aggreg_fn {mean,median,max,min}] [--decimals DECIMALS] [--skip_roi_gui] [--win_size WIN_SIZE] [--event_count EVENT_COUNT] [--viz_corr_resp] [--device DEVICE] [--log {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
               [--output_dir OUTPUT_DIR]

optional arguments:
  -h, --help            show this help message and exit
  --file FILE, -f FILE  Filepath to the file to read events from (.raw) or name of a sequence from EE3P3D dataset: ['highcontrastline', 'velcro_front', 'velcro_side', 'highcontrastdot', 'handspinner', 'spider', 'led', 'screen', 'speaker', 'motor', 'chain_side', 'chain_top']
  --roi_coords X0 Y0 X1 Y1, -rc X0 Y0 X1 Y1
                        RoI coordinates of the object to track (X0 Y0 X1 Y1)
  --skip_roi_gui        Flag to skip the RoI setup GUI if --roi_coords are provided
  --viz_corr_resp, -vcr
                        Visualize correlation responses for each window
  --device DEVICE, -d DEVICE
                        Device to run 3D correlation computations on (default: cuda:0)
  --aggreg_t AGGREG_T, -t AGGREG_T
                        Events aggregation interval in microseconds (default: 100)
  --read_t READ_T, -r READ_T
                        Number of microseconds to read events from the file (default: 1000000)
  --aggreg_fn {mean,median,max,min}, -afn {mean,median,max,min}
                        The function used to aggregate measurements from all windows (default: median)
  --decimals DECIMALS, -dp DECIMALS
                        Number of decimal places to round the result to (default: 1)
  --win_size WIN_SIZE, -w WIN_SIZE
                        Window size in pixels (default: 45, recommended not to change, see our paper)
  --event_count EVENT_COUNT, -N EVENT_COUNT
                        Threshold for template event count (default: 1800, recommended not to change, see our paper)
  --log {DEBUG,INFO,WARNING,ERROR,CRITICAL}, -l {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        Logging level (default: INFO)
```

## Troubleshooting

If you encounter any issues during installation or running the method, please check the following:

1. Ensure your CUDA installation matches the version specified in the Anaconda environment.
2. Make sure all prerequisites are correctly installed.
3. Verify that you're using the correct Python version (3.9) within the Anaconda environment.

If problems persist, please open an issue with details about your setup and the error you're encountering.

## License:

The code and dataset is provided under the GPL-3.0 license. Please refer to the LICENSE file for details.
We encourage you to use them responsibly and cite the paper if you use it in your work:
```
(soon)
``` 
