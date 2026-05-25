# 🚗 Indian & GTSRB Traffic Sign AI Detector

An end-to-end, high-performance, and real-time traffic sign detection system designed to identify Vienna Convention-compliant traffic signs (common across India and the GTSRB dataset). 

This project addresses severe class imbalance in traffic sign datasets through **procedural generative synthetic data**, implements aggressive **monsoon rain and lighting augmentations**, and runs YOLOv8 with custom **Early Stopping callbacks**, feeding bounding box detections into an **OpenAI LLM reasoning layer** to produce natural language safety advice for the driver.

---

## 📂 Repository Structure

The project workspace contains the following core files:

- **[traffic_sign_detector.py](file:///C:/Users/Admin/.gemini/antigravity/scratch/traffic-sign-detector/traffic_sign_detector.py)**: The single workable Python file comprising the generative synthetic dataset builder, dataset validation script, environmental augmentation pipeline, YOLOv8 trainer with custom Early Stopping hooks, OpenAI alert logic, and the Streamlit dashboard.
- **[requirements.txt](file:///C:/Users/Admin/.gemini/antigravity/scratch/traffic-sign-detector/requirements.txt)**: Python package list mapping core dependencies (`streamlit`, `ultralytics`, `opencv-python`, `pandas`, `numpy`, `pillow`, `openai`, `pyyaml`).
- **[README.md](file:///C:/Users/Admin/.gemini/antigravity/scratch/traffic-sign-detector/README.md)**: Project documentation (this file).

---

## 🛠️ Instructions for Running & Testing

### Step 1: Install Python
Ensure Python 3.10+ is installed on your Windows machine. Remember to check **"Add python.exe to PATH"** during setup.

### Step 2: Install Package Dependencies
Open a terminal in this directory and execute:
```cmd
python -m pip install -r requirements.txt
