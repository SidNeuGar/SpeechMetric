# Speech Metrics Evaluation

This script evaluates different speech quality metrics, including SECS, nMOS, and CER, based on the provided audio input. It allows users to run specific metrics by passing appropriate arguments.

## Features
- **SECS (Speech Enhancement Comparison Score):** Requires a reference file for comparison.
- **nMOS (Neural Mean Opinion Score):** Estimates the quality of the input speech.
- **CER (Character Error Rate):** Measures the accuracy of speech transcription against a ground truth text.

## Installation
Ensure you have Python installed. Clone this repository and navigate to the directory and install the necessary dependencies:

```sh
pip install -r requirements.txt
```

## Usage
**Note:** The script only handles one execution at a time, multiple files will be suppported later on. You must run it separately for different metric configurations.
Run the script with the required arguments:

```sh
python metric.py [--secs] [--nmos] [--cer] input_file [--ref_file REF_FILE] [--ground_truth GROUND_TRUTH]
```

### Arguments
- `input_file` (required): Path to the input audio file.
- `--secs` (optional): Runs SECS metric (requires `--ref_file`).
- `--nmos` (optional): Runs nMOS metric.
- `--cer` (optional): Runs CER metric (requires `--ground_truth`).
- `--ref_file` (optional): Reference file path for SECS.
- `--ground_truth` (optional): Ground truth text for CER.

### Example Commands
#### Run SECS metric:
```sh
python metric.py --secs input.wav --ref_file reference.wav
```

#### Run nMOS metric:
```sh
python metric.py --nmos input.wav
```

#### Run CER metric:
```sh
python metric.py --cer input.wav --ground_truth "expected transcript"
```

#### Run all metrics:
```sh
python metric.py --secs --nmos --cer input.wav --ref_file reference.wav --ground_truth "expected transcript"
```

## Error Handling
- If `--secs` is used without `--ref_file`, the script will return:
  ```
  Error: Reference file is required for SECS metric.
  ```
- If `--cer` is used without `--ground_truth`, the script will return:
  ```
  Error: Ground truth text is required for CER metric.
  ```


