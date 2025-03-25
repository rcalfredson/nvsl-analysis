# nvsl-analysis
Analysis code for "Spatial learning in feature-impoverished environments in Drosophila" (Yang Chen, Robert Alfredson, Dorsa Motevalli, Ulrich Stern, Chung-Hui Yang, bioRxiv 2024.09.28.615625; doi: https://doi.org/10.1101/2024.09.28.615625)

## Setup

This document provides instructions for setting up the data analysis environment for this project. The software has been tested on **Ubuntu 22.04** and **Windows 11**, but the instructions should be broadly applicable to other OS versions. No special hardware is required beyond a standard desktop or laptop computer.

### Prerequisites

- **Python 3.10** (either system-installed or provided via Conda)
- A C compiler (e.g., `gcc` on Linux, Xcode command line tools on macOS, or Visual Studio Build Tools on Windows)

---
### Installation Methods
You can set up this project using either of two recommended methods:
- **Method 1**: Virtualenv (requires system-wide Python installation)
- **Method 2**: Conda (provides isolated Python environments and package management)
Choose the method most suitable for your needs.
---

### Method 1: Virtualenv
Follow this method if you already have a system-wide Python installation and prefer using `virtualenv`.

#### 1. Install Python 3.10

Use your OS package manager (apt, brew, etc.) or download from the [official Python website](https://www.python.org/downloads/).

#### 2. Install `virtualenv` (if necessary)

```bash
pip install virtualenv
```

#### 3. Create a Virtual Environment

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

```bash

pip install -r requirements.txt
```

#### 6. Build Cython Extensions

```bash

python scripts/setup.py build_ext --inplace
```
---
### Method 2: Conda

Follow this method for a more isolated environment, particularly if you do not have Python 3.10 installed system-wide or prefer easy package management.

#### 1. Install Miniconda
Download and install from the [official Miniconda website](https://docs.conda.io/en/latest/miniconda.html), selecting the appropriate installer for your OS.

#### 2. Create and Activate a Conda Environment
```bash
conda create -n nvsl-analysis python=3.10
conda activate nvsl-analysis
```

#### 3. Install Dependencies
With the environment activated, install packages listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

#### 4. Build Cython Extensions
```bash
python scripts/setup.py build_ext --inplace
```
---
### (Optional) Create Drive Mapping File

By default, the script will use OS-specific path conventions (e.g., `/home/you/your_video.avi` vs. `C:\Users\you\your_video.avi`). To universally use Unix-style paths on Windows, create a JSON file `drive_mapping.json`:

```json
{
    "/media/Synology4/": "Y:\\\\"
}
```

---

### Typical Installation Time
If Python 3.10 is **already installed**, setting up the virtual environment and installing dependencies (including building Cython extensions) typically takes about **2 minutes** on a normal desktop computer. If you need to install Python or Miniconda from scratch, plan for an additional **3–5 minutes**, depending on your internet speed and operating system.

---

### Deactivation

Exit your environment:
- **Virtualenv:**
  ```bash
  deactivate
  ```
- **Conda:**
  ```bash
  conda deactivate
  ```

### Troubleshooting

- **Dependency errors**: Double-check you have Python 3.10 and your virtual environment is activated before installing.
- **Cython compilation issues**: Confirm that you have a working C compiler (e.g., gcc, Xcode CLT, or VS Build Tools).

---

## Basic Usage
Once your environment is ready, you can run a standard analysis with:
```bash
python analyze.py -v "/actual/path/to/your_video.avi" -f "0-9"
```
Where:
- `-v` is **required** and specifies the path to your video file.
- `-f` is **required** if analyzing a chamber that supports multiple flies; use a range like `0-9`.

### Additional Flags: Turn Detection

You can detect turn events in different areas with the --turn flag:
- `circle`
- `agarose`
- `boundary`

For example:
```bash
python analyze.py -v "/actual/path/to/your_video.avi" -f "0-9" --turn circle
```
For more details on optional flags, see the top of `analyze.py` or run `python analyze.py --help`.

---

## Demo
Below is a short demo to verify the pipeline with *real* data.
1. **Download the sample data**
   Grab the files from [this Figshare dataset](https://figshare.com/articles/dataset/NVSL_Analysis_Demo_Sample_Experiments/28228976).
2. **Run an example command**
   ```bash
   python analyze.py \
       -v /path/to/your/demo/files/c*.avi \
       -f 0-9 \
       --turn circle \
       --turn-prob-by-dist 2,3,4,5,6
   ```
   - The analysis takes about **4 minutes** to complete on a typical desktop.
   - This produces:
       - `__analyze.log`: A log summarizing results per fly.
       - `learning_stats.csv`: Metrics such as walking speed, number of rewards, etc.
       - `imgs/`: A folder with around 48 plots covering various metrics.
3. **Compare outputs to expected**
    - Download the reference outputs from [this Figshare dataset](https://figshare.com/articles/dataset/NVSL_Analysis_Demo_Expected_Outputs/28229126).
    - Compare each file (log, CSV, images) to confirm the results match.

---

## Testing
A regression test checks whether the latest code matches prior reference outputs:
```bash
python scripts/regression_test.py
```
It:
  - Parses log/CSV commands from past runs
  - Executes them on your current code
  - Compares generated outputs (logs, CSVs, images) with the reference data
  - Prints any discrepancies

Logs like `__analyze_13F02.log` or `__analyze_blind.log` and CSVs like `learning_stats_alt.csv` or `learning_stats_blind.csv` are checked, and images in `test-imgs/key.json` are compared via binary checks.

*Thank you for using `nvsl-analysis`! If you have any feedback or questions, please open an issue on GitHub or email the authors.*