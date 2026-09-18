# Project Statement: Lightweight Automatic Number Plate Detection System

**Project Name:** Automatic Number Plate Recognition (ANPR) & Detection Pipeline  
**Status:** Approved  

---

## 1. Problem Statement

### 1.1 Context & Background
Urban traffic management, automated toll collection, parking facility administration, and law enforcement rely heavily on accurate vehicle tracking and identification. Manual monitoring of video feeds or physical inspection at checkpoints is inefficient, prone to human error, and financially unscalable as vehicle density grows.

### 1.2 Core Challenges & Pain Points
* **High Computational Cost:** Traditional computer vision models for vehicle and license plate identification often require expensive dedicated hardware, making real-time deployment at edge checkpoints costly.
* **Environmental & Visual Variations:** Lighting fluctuations, varying camera angles, motion blur, and background clutter complicate accurate plate candidate localization.
* **Processing Speed vs. Accuracy Trade-off:** Deploying heavy deep-learning networks on frame-by-frame video leads to high latency, dropping operational frame rates below acceptable real-time thresholds.
* **Unstructured Output Data:** Simple visual detection without organized metadata logging limits integration into downstream systems such as security databases, access control barriers, and automated billing software.

### 1.3 Solution Objective
This project implements a lightweight, automated pipeline combining YOLOv8 vehicle detection with OpenCV morphological contour filtering. It optimizes detection speed and accuracy, generating annotated video outputs, isolated license plate crop snapshots, and structured JSON logs without requiring enterprise-grade server infrastructure.

---

## 2. Scope of the Project

### 2.1 In-Scope Capabilities
* **Automated Vehicle Detection:** Integration of YOLOv8n (nano architecture) for detecting vehicle classes (`car`, `truck`, `bus`, `motorcycle`) in full-HD video streams (`src/car.mp4`).
* **Region of Interest (ROI) & Plate Candidate Extraction:** Image preprocessing using OpenCV (grayscale conversion, Gaussian blur, Otsu thresholding, Canny edge detection) and geometric contour filtering (aspect ratio, area bounding, frame relative positioning) to locate plate boundaries within detected vehicle boundaries.
* **Hardware Acceleration Management:** Dynamic execution fallback between CUDA GPU (`cuda:0`) and CPU environments based on system availability.
* **Artifact Generation & Output Management:**
  * Bounding box and label annotation on processed video saved to `output/annotated_video.mp4`.
  * High-resolution cropped image files for each detected plate candidate saved in `output/plates/`.
  * Structured frame-by-frame detection metadata exported to `output/summary.json`.

### 2.2 Out-of-Scope (Future Extensions)
* **Optical Character Recognition (OCR):** Reading alphanumeric text off cropped plates (e.g., via Tesseract or EasyOCR) is deferred to downstream processing modules.
* **Direct Database Integration:** Live streaming into SQL/NoSQL databases or cloud access control APIs.
* **Multi-Camera Synchronized Tracking:** Multi-camera re-identification across different physical entry points.

---

## 3. Target Users

| User Category | Description & Persona | Primary Needs & Use Cases | Key Benefits |
| :--- | :--- | :--- | :--- |
| **Smart City & Traffic Engineers** | Operators monitoring traffic flow and municipal roadway security. | Automated monitoring of vehicle movement and density analysis. | Low hardware cost, real-time video stream processing capabilities. |
| **Parking Facility Administrators** | Managers of commercial parking garages and gated residential access. | Automated vehicle entry/exit verification and plate logging. | Eliminates manual ticket validation, saves cropped plate snapshots for audit logs. |
| **Law Enforcement & Toll Operators** | Security personnel and automated toll collection agencies. | Rapid identification of target vehicle license plates from recorded or live video. | Structured JSON data logs for easy integration with security systems. |
| **Edge Device & Embedded Developers** | Developers building lightweight vision solutions on edge devices (Raspberry Pi, Jetson Nano). | Modular, efficient codebase with low memory and compute requirements. | Optimized Python script supporting CUDA acceleration and CPU fallback. |

---

## 4. High-Level Features

### 4.1 Dual-Stage Detection Pipeline
* **Object-Level Vehicle Recognition:** Leverages lightweight YOLOv8n to identify vehicle regions while discarding non-relevant background clutter.
* **Sub-ROI Contour Analysis:** Isolates the vehicle bounding box and applies edge detection, aspect-ratio checking (2.0 ≤ AR ≤ 6.0), and relative area constraints to extract candidate plate regions.

### 4.2 Hardware Aware Execution Engine
* **Automatic Device Detection:** Queries PyTorch runtime to select GPU hardware acceleration (`cuda:0`) or CPU fallback automatically.
* **Frame-by-Frame Processing:** Optimizes image resolution and inference configuration for fast throughput.

### 4.3 Output & Artifact Generation
* **Annotated Video Renderer:** Writes bounding boxes and `NUMBER PLATE` overlay labels directly onto the processed video at native frame rates.
* **Plate Snapshot Extractor:** Automatically cuts, crops, and formats individual license plate regions into JPEG snapshots (`plate_XXXX.jpg`).
* **Structured Analytics Exporter:** Generates a clean `summary.json` containing frame indices, detection counts, and vehicle presence status for downstream logging.
