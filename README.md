[![arXiv](https://img.shields.io/badge/arXiv-0000.00000-b31b1b.svg)](https://arxiv.org/abs/0000.00000) [![DOI:10.48550/ARXIV.0000.00000](https://zenodo.org/badge/doi/10.48550/ARXIV.0000.00000.svg)](https://doi.org/10.48550/arXiv.0000.00000)

# EE3P3D: Event-based Estimation of Periodic Phenomena Frequency using 3D Correlation
<!-- Kolář, J., Špetlík, R., Matas, J. (2024) EE3P3D: Event-based Estimation of Periodic Phenomena Frequency using 3D Correlation. In Proceedings of , 2024
 -->

Paper Link: (soon)

## Dataset
The dataset features 12 sequences of periodic phenomena (rotation - `01-06`, flicker - `07-08`, vibration - `09-10` and movement - `11-12`) with GT frequencies ranging from 3.2 up to 2000 Hz in `.raw` and `.hdf5` file formats.

Data capture demonstration: [Youtube video](https://youtu.be/QlfQtvbaYy8)

Each event-based sequence was rendered as a video in 3 playback speeds: [Youtube playlist](https://www.youtube.com/playlist?list=PLK466i9CoYqQ2780OXJg7WgtUtWMEqbkS)

![Ground-truth frequencies of experiments](https://github.com/user-attachments/assets/5175b810-81f7-4508-83f4-7903f2c6c27d)

Sequences `highcontrastdot`, `handspinner` and `motor` originate from the Metavision Dataset. To download these sequences, use the `method/loader.py` script. The `method/main.py` script handles the downloading of these sequences automatically.

## Method

![Method's diagram](https://github.com/user-attachments/assets/90d8ddb3-23a7-4fa9-a24c-ee5b2560b80d)

1. Data captured from an event camera is aggregated into a 3D array,    
2. the array is split into same-sized areas, and in each area, a template depth is automatically selected,
3. a 3D correlation of the template with the event stream is computed,
4. a frequency is calculated from the median of time deltas measured between correlation peaks for each window,
5. the output frequency is computed as a median of measurements from all windows.

## Installation

Compatibility: 
- Windows 10/11, Linux Ubuntu 20.04 and Ubuntu 22.04
- Python 3.9
- CUDA 11.7, 12.1, 12.4 or CPU only

Tested on, but same versions (probably) not required:
- Miniconda3-py39-24.5.0
- Metavision SDK 4.4.0

### Prerequisites
0. [**Git**](https://git-scm.com/downloads) or [**GitHub Desktop**](https://github.com/apps/desktop) app

1. **Anaconda**: For managing the Python environment.
   - Download and install Miniconda for **Python 3.9** from the [official Anaconda website](https://docs.anaconda.com/miniconda/miniconda-other-installer-links/).
   - When installing, consider adding Miniconda3 to PATH. Otherwise, you will need to run a specific Anaconda terminal (instructions on activating it below).

2. **CUDA Toolkit**: This method supports GPU acceleration for 3D correlation computing using CUDA PyTorch. The method was tested and the repo contains environment setups for CUDA versions `11.7`, `12.1` and `12.4`. If none of these versions are compatible with your system, a `cpu only` environment setup is available as well and you can skip this step.
   - Download and install the appropriate version from the [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive).
   - Feel free to use the Expert installation mode and uncheck installing any non CUDA related packages (Nsight VS, GeForce drivers,...)

3. **Metavision SDK**: The Metavision SDK is required for I/O operations with event-based sequences (.raw or .hdf5). File formats from other event camera manufacturers are currently not supported. However, if you implemented support for one, feel free to open a pull request.
   - For Windows
      - Open installation instructions for [Windows](https://docs.prophesee.ai/4.5.2/installation/windows.html#chapter-installation-windows)
      - Verify your system is compatible by checking the [Required Configuration section](https://docs.prophesee.ai/4.5.2/installation/windows.html#required-configuration)
      - Whether you are installing the SDK for the first time or are currently using a previous version, you must download our Windows installer, which is hosted in an installer repository. The URL of this repository can be retrieved this way:
         - If you are a Prophesee customer, retrieve the link in the [Knowledge Center Download section](https://support.prophesee.ai/portal/en/kb/prophesee-1/metavision-sdk/download-center). ([request an account](https://www.prophesee.ai/kc-access-request/) if you don’t have one yet).
         - Otherwise, you must [sign-up for the SDK](https://www.prophesee.ai/metavision-intelligence-sdk-download) to get the link.

      - Once you have access to our installer repository, among the list of SDK installers, download the `Metavision_SDK_440_Setup.exe` and install it using default settings.
   - For Linux
      - Open installation instructions for [Linux](https://docs.prophesee.ai/4.5.2/installation/linux.html#chapter-installation-linux)
      - Verify your system is compatible by checking the [Required Configuration section](https://docs.prophesee.ai/4.5.2/installation/linux.html#required-configuration)
      - Whether you are installing the SDK for the first time or are currently using a previous version, to install Metavision SDK on Linux, you need our APT repository configuration file `metavision.list`.
         - If you are a Prophesee customer, retrieve the link in the [Knowledge Center Download section](https://support.prophesee.ai/portal/en/kb/prophesee-1/metavision-sdk/download-center). ([request an account](https://www.prophesee.ai/kc-access-request/) if you don’t have one yet).
         - Otherwise, you must [sign-up for the SDK](https://www.prophesee.ai/metavision-intelligence-sdk-download) to get the link.

      - Continue with [Installing Dependencies](https://docs.prophesee.ai/4.5.2/installation/linux.html#installing-dependencies) and [Installation](https://docs.prophesee.ai/4.5.2/installation/linux.html#installation) steps
   
### Setup
1. Clone the repository at a path that does not contain any special characters (e.g. ě, š, č, ř, ž, ý, á, í, é). Otherwise, the Metavision SDK Driver raises an exception (Error 103001):
```console
git clone https://github.com/JackPieCZ/EE3P3D.git
cd EE3P3D
```

2. Create a new Anaconda environment:
   - Open the Anaconda terminal 
      - If you used **default path** when installing in for **all users** in Windows you can use the following command
      ```bash
      %windir%\System32\cmd.exe "/K" C:\ProgramData\miniconda3\Scripts\activate.bat C:\ProgramData\miniconda3
      ```
      - If you used **default path** when installing in for **current user** with `<user_name>` in Windows you can use the following command
      ```bash
      %windir%\System32\cmd.exe "/K" C:\Users\<user_name>\Scripts\activate.bat C:\Users\<user_name>\miniconda3
      ```
   - Verify conda is installed. A list of commands should appear.
   ```console
   conda
   ```
   - Move to the EE3P3D directory using `cd` if you are not already there
   - If you have the CUDA XX.Y version installed (11.7, 12.1, 12.4), run the following command:
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

4. Verify the Installation:

You can run a quick check to verify that the environment with CUDA is set up correctly. No errors should appear.

```python
python
import torch
print(f"CUDA version: {torch.version.cuda}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA is available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
   print("CUDA is not available.")
   print("When running the EE3P3D method, please use '--device cpu' flag.")

exit()

python ./method/test_imports.py

```

### Usage

After setting up your environment, you can run the EE3P3D method using the main.py script. Here are some example commands:

1. Basic usage with a file (`--file`/`-f`):
```console
python ./method/main.py --file path/to/your/event_file.raw
```

2. For analysing any sequence from the EE3P3D dataset simply enter the sequence name. For example, `led`, `highcontrastdot`, `screen`, `motor`, ect. For all sequence names, check the `dataset` [folder](https://github.com/JackPieCZ/EE3P3D/tree/main/dataset) or `dataset/config.json` [file](https://github.com/JackPieCZ/EE3P3D/blob/175736d322d484b46277459ba09a71a9fc23d58a/dataset/config.json#L2).
```console
python ./method/main.py -f led
```

3. Specifying RoI coordinates (optional, `--roi_coords`/`-rc`) in `X0 Y0 X1 Y1` format. RoI for sequences from the dataset are provided automatically. If the coordinates are not provided, a GUI will be shown where the user can setup RoI easily.
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
python ./method/main.py -f handspinner -vcr
```

7. Running on a specific device (optional, `--device`/`-d`, default is `cuda:0`):
```console
python ./method/main.py -f path/to/your/event_file.raw --device cuda:1
python ./method/main.py -f path/to/your/event_file.raw --device cpu
```

8. For a full list of available options, run:
```console
python ./method/main.py -h 
```
```
usage: main.py [-h] --file FILE [--roi_coords X0 Y0 X1 Y1] [--aggreg_t AGGREG_T] [--read_t READ_T] [--aggreg_fn {mean,median,max,min}] [--decimals DECIMALS] [--skip_roi_gui] [--win_size WIN_SIZE] [--event_count EVENT_COUNT] [--viz_corr_resp] [--device DEVICE]
               [--log {DEBUG,INFO,WARNING,ERROR,CRITICAL}] [--verbose] [--output_dir OUTPUT_DIR]

Measure the frequency of periodic phenomena (rotation, vibration, flicker, etc.) in event-based sequence using the EE3P3D method.

optional arguments:
  -h, --help            show this help message and exit
  --file FILE, -f FILE  Filepath to the file to read events from (.raw) or name of a sequence from EE3P3D dataset: ['highcontrastline', 'velcro_front', 'velcro_side', 'highcontrastdot', 'handspinner', 'spider', 'led', 'screen', 'speaker', 'motor', 'chain_side', 'chain_top']
  --roi_coords X0 Y0 X1 Y1, -rc X0 Y0 X1 Y1
                        RoI coordinates of the object to track (X0 Y0 X1 Y1)
  --aggreg_t AGGREG_T, -t AGGREG_T
                        Events aggregation interval in microseconds (default: 100)
  --read_t READ_T, -r READ_T
                        Number of microseconds to read events from the file (default: 1000000)
  --aggreg_fn {mean,median,max,min}, -afn {mean,median,max,min}
                        Name of a NumPy function used to aggregate measurements from all windows (default: median)
  --decimals DECIMALS, -dp DECIMALS
                        Number of decimal places to round the result to (default: 1)
  --skip_roi_gui, -srg  Flag to skip the RoI setup GUI if --roi_coords are provided
  --win_size WIN_SIZE, -w WIN_SIZE
                        Window size in pixels (default: 45, recommended not to change, see our paper)
  --event_count EVENT_COUNT, -N EVENT_COUNT
                        Threshold for template event count (default: 1800, recommended not to change, see our paper)
  --viz_corr_resp, -vcr
                        Visualize correlation responses for each window
  --device DEVICE, -d DEVICE
                        Device to run 3D correlation computations on (default: cuda:0)
  --log {DEBUG,INFO,WARNING,ERROR,CRITICAL}, -l {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        Logging level (default: INFO)
  --verbose, -v         Verbose mode
  --output_dir OUTPUT_DIR, -o OUTPUT_DIR
                        Name of output directory (default: ./ee3p3d_out)
```

## Troubleshooting

If you encounter any issues during installation or running the method, please check the following:

- `Metavision SDK Driver error 103001`: You are trying to open a file using a filepath that contains some special characters ([Metavision FAQ](https://docs.prophesee.ai/stable/faq.html#why-do-i-get-errors-when-trying-to-read-recorded-files-raw-or-hdf5-with-studio-or-the-api-on-windows))
- For other Metavision related issues, see their [Troubleshooting guide](https://docs.prophesee.ai/stable/faq.html#troubleshooting)
- `RuntimeError: Found no NVIDIA driver on your system`: If you have NVIDIA GPU, check that you have updated its driver, otherwise use the `--device cpu` flag when running the method
- Make sure all prerequisites are correctly installed by running the `method/test_imports.py` script
- Ensure your CUDA installation matches the version specified by the Anaconda environment version

If problems persist, please open an issue with details about your setup and the error you're encountering.

## License:

The code and dataset is provided under the GPL-3.0 license. Please refer to the LICENSE file for details.
We encourage you to use them responsibly and cite the paper if you use it in your work:
```
(soon)
``` 
