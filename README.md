# AutoAnot - Automatic Audio Annotation System

AutoAnot is a Streamlit-based application for automatic audio recording, annotation, and dataset generation. The application records audio, automatically labels segments based on RMS (Root Mean Square) changes, and allows saving the annotated data as a dataset.

## Features

- Audio recording with customizable sampling rate and duration
- Automatic annotation of 1-second audio segments based on RMS change threshold
- Visual representation of audio waveforms with color-coded annotations (green for "OK", red for "NG")
- RMS value display for each audio segment
- Dataset generation and saving as .npz file for machine learning applications

## Requirements

- Python 3.x
- streamlit
- numpy
- sounddevice
- matplotlib

## Installation

Clone the repository and install the required dependencies:

```bash
# Clone the repository
git clone <repository-url>
cd AutoAnot

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install streamlit numpy sounddevice matplotlib
```

## Usage

1. Run the application:

```bash
# Option 1: Using the provided script
chmod +x run_app.sh
./run_app.sh

# Option 2: Direct streamlit command
streamlit run app.py
```

2. In the web interface:
   - Set the sampling frequency (Hz)
   - Set the recording duration (seconds)
   - Set the RMS change threshold (%)
   - Click "Record" to start recording
   - After recording, review the automatically annotated segments
   - Click "Save Dataset" to save the annotated data as a dataset.npz file

## Dataset Format

The saved dataset is in .npz format with the following components:
- `waveforms`: Array of audio segments (shape: number_of_segments × samples_per_segment)
- `labels`: Array of labels ("OK" or "NG") for each segment
- `fs`: Sampling frequency used for recording

## Customization

- Modify the RMS threshold to adjust the sensitivity of automatic annotation
- Edit `app.py` to change the annotation logic or visualization parameters
- Modify `run_app.sh` to change server address or add additional streamlit parameters

## License

[Specify your license here]