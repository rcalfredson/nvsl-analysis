"""
This script verifies the output of analyses by comparing logs, CSVs, and images
to reference files. It parses and runs commands from the input files, then checks 
whether the outputs match the expected results, reporting any discrepancies.

The script processes three types of files:
- Log files: commands are extracted, executed, and their logs are compared.
- CSV files: their contents are compared with reference CSVs.
- Image files: byte-by-byte comparison is performed between reference and output images.

The script organizes this process using functions to parse commands, run them, and compare outputs.
The `main` function coordinates the file associations, command execution, and discrepancy reporting.
"""

import os
import subprocess
import json
from collections import defaultdict

# Directory containing the project files
PROJECT_DIR = "."

# Directories for reference files
REF_DIR = os.path.join(PROJECT_DIR, "test/reference")
CSV_REF_DIR = os.path.join(REF_DIR, "csv")
LOG_REF_DIR = os.path.join(REF_DIR, "log")
IMG_REF_DIR = os.path.join(REF_DIR, "img")

IMAGE_KEY_FILE = os.path.join(IMG_REF_DIR, "key.json")

# List of log and CSV files
log_files = [
    "__analyze_13F02.log",
    "__analyze_alt.log",
    "__analyze_blind.log",
    "__analyze_htl.log",
    "__analyze_htl2.log",
    "__analyze_on_agarose.log",
    "__analyze_wall_full_bucket.log",
]

csv_files = [
    "learning_stats_alt.csv",
    "learning_stats_blind.csv",
    "learning_stats_boundary_contact.csv",
    "learning_stats_heatmap.csv",
    "learning_stats_on_agarose.csv",
    "learning_stats_on-off.csv",
    "learning_stats_pltThm.csv",
    "learning_stats_wall_noyc.csv",
    "learning_stats_wall_yc.csv",
    "learning_stats_wt.csv",
]


def parse_command(file_path):
    """Parse the command from the first line of the file."""
    with open(file_path, "r") as file:
        first_line = file.readline().strip()
        if first_line.startswith("# command:"):
            command = first_line[len("# command: ") :].strip()
            if "[" in command and command.endswith("]"):
                command = command[: command.rindex("[")].strip()
            return f"python {command}"
    return None


def run_command(command):
    """Run the given command."""
    print(f"Running command: {command}")
    result = subprocess.run(command, shell=True, capture_output=True)
    if result.returncode != 0:
        print(f"Command failed: {command}\n{result.stderr.decode()}")
        return False
    return True


def compare_files(file1, file2):
    """Compare two files ignoring the first line."""
    with open(file1, "r") as f1, open(file2, "r") as f2:
        f1_lines = f1.readlines()[1:]
        f2_lines = f2.readlines()[1:]
        return f1_lines == f2_lines


def compare_images(img1, img2):
    """Compare two images byte by byte."""
    with open(img1, "rb") as f1, open(img2, "rb") as f2:
        return f1.read() == f2.read()


def main():
    """
    Main function that associates commands with their respective log, CSV, or image files. 
    It parses commands from these files, runs the commands, and compares the outputs to 
    reference files (logs, CSVs, and images). Discrepancies between outputs and reference 
    files are reported.
    """
    command_to_files = defaultdict(list)

    # Collect commands and associated files from logs
    for log_file in log_files:
        log_path = os.path.join(LOG_REF_DIR, log_file)
        command = parse_command(log_path)
        if command:
            command_to_files[command].append(log_file)

    # Collect commands and associated files from CSVs
    for csv_file in csv_files:
        csv_path = os.path.join(CSV_REF_DIR, csv_file)
        command = parse_command(csv_path)
        if command:
            command_to_files[command].append(csv_file)

    # Collect commands and associated files from images key.json
    if os.path.exists(IMAGE_KEY_FILE):
        with open(IMAGE_KEY_FILE, "r") as file:
            image_commands = json.load(file)
            for command, file_pairs in image_commands.items():
                command_to_files[command].extend(file_pairs)

    # Run each command and compare files
    for command, files in command_to_files.items():
        if run_command(command):
            for file_info in files:
                if isinstance(file_info, dict):  # Handling image file pairs
                    reference_image = os.path.join(IMG_REF_DIR, file_info["reference"])
                    output_image = os.path.join(
                        PROJECT_DIR, "imgs", file_info["output"]
                    )
                    if not compare_images(reference_image, output_image):
                        print(f"Discrepancy found in {file_info['output']}")
                    else:
                        print(f"No discrepancy in {file_info['output']}")
                else:  # Handling log and CSV files
                    if file_info.startswith("__analyze"):
                        new_file = "__analyze.log"
                    else:
                        new_file = "learning_stats.csv"

                    if not compare_files(
                        os.path.join(
                            (
                                LOG_REF_DIR
                                if file_info.startswith("__analyze")
                                else CSV_REF_DIR
                            ),
                            file_info,
                        ),
                        os.path.join(PROJECT_DIR, new_file),
                    ):
                        print(f"Discrepancy found in {file_info}")
                    else:
                        print(f"No discrepancy in {file_info}")
        else:
            print(f"Failed to run command: {command}")


if __name__ == "__main__":
    main()
