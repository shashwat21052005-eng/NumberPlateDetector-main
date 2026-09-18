# Lightweight Number Plate Detection Project

This project detects vehicle number plates from the video in `src/car.mp4` using a lightweight YOLOv8 setup and OpenCV preprocessing.

## Features
- GPU support when CUDA is available
- Lightweight YOLOv8n model for faster inference
- Plate candidate extraction using contour-based filtering
- Saving annotated output video and snapshots
- Easy to run in a local Python environment

## Folder structure
- `src/car.mp4` – input video
- `src/number_plate_detection.py` – detection pipeline
- `output/` – generated video and plate crops

## Setup
1. Create and activate a Python virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the project:
   ```bash
   python src/number_plate_detection.py
   ```

## GPU note
The script checks for CUDA automatically and uses `cuda:0` when available. If CUDA is not available, it falls back to CPU.

## Output
- Annotated video is saved in `output/annotated_video.mp4`
- Plate crops are saved in `output/plates/`
- A summary JSON file is saved in `output/summary.json`
