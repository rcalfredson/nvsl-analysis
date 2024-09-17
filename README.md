# nvsl-analysis
Analysis code for paper "Parallel use of self-generated olfactory and motion cues for learning a spatial goal location in Drosophila melanogaster"

## Setup

This document outlines the steps required to set up the data analysis environment for our lab's code base. These steps are designed to be OS-independent, with notes provided for specific instructions on Linux, Windows, and macOS.

### Prerequisites

- Python 3.10

### Installation Steps

#### 1. Install Python 3.10

Ensure that Python 3.10 is installed on your system. This can typically be done through your operating system's package manager, the Windows Store, or by downloading from the [Python website](https://www.python.org/downloads/).

**Note:** Installing Python 3.10 should also install `pip` (Python's package installer). However, it does **not** automatically install `virtualenv`.

#### 2. Install `virtualenv` (if necessary)

If `virtualenv` is not installed (it's not included with Python 3.10 by default), install it globally using pip:

```bash
pip install virtualenv
```

#### 3. Create a Virtual Environment

Navigate to your project directory and create a virtual environment. Replace myenv with your desired environment name.

- Linux/macOS

  ```bash
  python3 -m venv myenv
  ```
- Windows

  ```cmd

  python -m venv myenv
  ```

#### 4. Activate the Virtual Environment

- Linux/macOS

  ```bash

  source myenv/bin/activate
  ```

- Windows

  ```cmd

  .\myenv\Scripts\activate
  ```

#### 5. Install Dependencies

With the virtual environment activated, install the required dependencies using:

```bash

pip install -r requirements.txt
```

#### 6. Build Cython Extensions

Finally, build the Cython extensions. This step assumes Cython is listed as a dependency in your requirements.txt and thus has been installed in the previous step.

```bash

python setup.py build_ext --inplace
```

#### 7. Create drive mapping file (optional)

By default, paths will be handled using their respective conventions by OS - `C:\Users\you\your_video.avi` for Windows and `/home/you/your_video.avi` for Unix. However, `analyze.py` can also be configured to use Unix paths universally- just save a file to this directory named `drive_mapping.json` that maps Unix-style paths to Windows-style ones. For example:

```json
{
    "/media/Synology4/": "Y:\\",
}
```

If the program detects `drive_mapping.json`, then it attempts to map all inputted paths to their Windows equivalents, so that `/media/Synology4/you/your_video.avi`, for example, would be converted to `Y:\you\your_video.avi`.

### Deactivation

To exit the virtual environment, simply run:

```bash

deactivate
```

### Troubleshooting

- If you encounter any issues during the setup, ensure that you have the correct version of Python installed and that your virtual environment is activated before installing dependencies.
- For problems related to Cython compilation, verify that you have a C compiler installed and configured correctly for your operating system.

## Running the first analysis
Once your environment is set up, you can run a standard analysis using the following command:
```bash
python analyze.py -v "$path_to_video" -f "0-9"
```
- `-v` specifies the path to the video file.
- `-f` defines the range of fly numbers (depending on how many flies your chamber supports).

This command will perform a standard analysis with no extra features. Be sure to replace "$path_to_video" with the path to your video file and adjust the fly number range as needed.

## Testing
The repository includes a regression testing script to verify that the current version of the code produces the same results as checked-in reference data. The script reads commands from earlier results and compares the outputs.

To run the test, execute the following:

```bash

python regression_test.py
```

The script will:
- Parse commands from log files and CSVs.
- Compare the output of the current script with reference logs, CSV files, and images.
- Print discrepancies if there are any mismatches between the generated files and the reference data.

The test checks both textual output (logs, CSVs) and images (using binary comparison).

For example, the script checks logs like:
- `__analyze_13F02.log`
- `__analyze_blind.log`
- `__analyze_wall_full_bucket.log`

And CSV files such as:
- `learning_stats_alt.csv`
- `learning_stats_blind.csv`
- `learning_stats_heatmap.csv`

It also compares images specified in `test-imgs/key.json` with output images.

If there are any discrepancies or errors, they will be reported during the test run.
