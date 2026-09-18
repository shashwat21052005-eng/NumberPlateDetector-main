import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent.parent
VIDEO_PATH = ROOT / 'src' / 'car.mp4'
MODEL_PATH = ROOT / 'src' / 'yolov8n.pt'
OUTPUT_DIR = ROOT / 'output'
PLATE_DIR = OUTPUT_DIR / 'plates'


def get_device():
    if torch.cuda.is_available():
        return 'cuda:0'
    return 'cpu'


def save_plate_crop(frame, bbox, index):
    x1, y1, x2, y2 = map(int, bbox)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)

    if x2 <= x1 or y2 <= y1:
        return

    crop = frame[y1:y2, x1:x2]
    plate_file = PLATE_DIR / f'plate_{index:04d}.jpg'
    cv2.imwrite(str(plate_file), crop)


def find_plate_candidates(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(thresh, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    image_area = frame.shape[0] * frame.shape[1]
    frame_h, frame_w = frame.shape[:2]

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w <= 0 or h <= 0:
            continue

        aspect_ratio = w / float(h)
        area = w * h
        if not (2.0 <= aspect_ratio <= 6.0):
            continue
        if not (0.004 * image_area <= area <= 0.25 * image_area):
            continue
        if not (0.12 * frame_w <= w <= 0.8 * frame_w):
            continue
        if not (0.04 * frame_h <= h <= 0.25 * frame_h):
            continue

        center_x = x + w / 2.0
        center_y = y + h / 2.0
        if center_y < 0.35 * frame_h or center_y > 0.9 * frame_h:
            continue
        if abs(center_x - frame_w / 2.0) > 0.7 * frame_w:
            continue

        # reject very large, non-plate regions that span most of a vehicle
        if w > 0.8 * frame_w or h > 0.25 * frame_h:
            continue

        candidates.append((x, y, x + w, y + h))

    if not candidates:
        return []

    candidates.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    return candidates[:3]


def detect_number_plate_video(video_path: str | Path, output_dir: Path = OUTPUT_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)
    PLATE_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device()
    model = YOLO(str(MODEL_PATH))
    model.to(device)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f'Could not open video file: {video_path}')

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    output_video = output_dir / 'annotated_video.mp4'
    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height),
    )

    records = []
    frame_index = 0
    plate_index = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_index += 1
        results = model(frame, imgsz=640, conf=0.25, device=device, verbose=False)[0]

        detected_plates = []
        detected_vehicle = False

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0])
            label = model.names[cls_id].lower()

            if label in {'car', 'truck', 'bus', 'motorcycle'}:
                detected_vehicle = True
                vehicle_roi = frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
                candidates = find_plate_candidates(vehicle_roi)

                for candidate in candidates:
                    cx1, cy1, cx2, cy2 = candidate
                    plate_box = (
                        x1 + cx1,
                        y1 + cy1,
                        x1 + cx2,
                        y1 + cy2,
                    )
                    detected_plates.append(plate_box)

        for plate_box in detected_plates:
            x1, y1, x2, y2 = plate_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, 'NUMBER PLATE', (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            save_plate_crop(frame, plate_box, plate_index)
            plate_index += 1

        if detected_vehicle:
            records.append({
                'frame': frame_index,
                'plates_detected': len(detected_plates),
                'status': 'vehicle_detected',
            })

        writer.write(frame)

    cap.release()
    writer.release()

    summary_path = output_dir / 'summary.json'
    summary_path.write_text(json.dumps(records, indent=2), encoding='utf-8')
    print(f'Video processed successfully. Output saved to: {output_video}')
    print(f'Plate crops saved to: {PLATE_DIR}')
    print(f'Summary saved to: {summary_path}')
    print(f'Using device: {device}')


if __name__ == '__main__':
    detect_number_plate_video(VIDEO_PATH)
