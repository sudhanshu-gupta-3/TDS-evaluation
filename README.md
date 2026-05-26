# 4. README.md

## 🚗 Project Overview
An end-to-end, high-performance, and real-time traffic sign detection system designed to identify Vienna Convention-compliant traffic signs (common across India and the GTSRB dataset).

This project addresses severe class imbalance in traffic sign datasets through procedural generative synthetic data, implements aggressive environmental augmentations (such as monsoon rain, glare, and dust), and runs YOLOv8 with custom Early Stopping callbacks. It also features an OpenAI LLM reasoning layer that converts bounding box detections into natural language safety advice for the driver.

## 📂 Repository Structure

```text
TDS-evaluation/
├── Golden Response/       # Complete traffic sign detection system code
│   ├── app.py             # Streamlit real-time detection dashboard
│   ├── data/              # Dataset configuration files (.yaml)
│   ├── models/            # Directory for trained models (YOLOv8 weights)
│   ├── scripts/           # Scripts for training, evaluation, and data preparation
│   ├── src/               # Core source code (inference logic, utilities, UI styles)
│   ├── test_images/       # Sample images for testing the dashboard
│   ├── Dockerfile         # Docker configuration for containerized deployment
│   └── requirements.txt   # Python package dependencies
├── Justification.md       # Detailed justification document explaining architectural decisions
├── Prompt.md              # Original prompt/requirements specification
└── README.md              # Project documentation (this file)
```

## 🛠️ Instructions for Running/Testing the Code

**Step 1: Install Python**  
Ensure Python 3.10+ is installed on your machine.

**Step 2: Clone the Repository and Navigate to the Code Directory**  
Open a terminal and run:
```bash
git clone https://github.com/sudhanshu-gupta-3/TDS-evaluation.git
cd TDS-evaluation/"Golden Response"
```

**Step 3: Install Package Dependencies**  
Execute the following to install all required libraries:
```bash
pip install -r requirements.txt
```
*(Dependencies include `streamlit`, `ultralytics`, `opencv-python`, `pandas`, `numpy`, `pillow`, `openai`, and `pyyaml`.)*

**Step 4: Start the Streamlit Dashboard**  
Launch the web interface locally:
```bash
streamlit run app.py
```

**Step 5: Testing the Pipeline**  
- **API Key Setup:** Enter your `OPENAI_API_KEY` in the sidebar to enable GPT-generated safety warnings.
- **Dataset Synthesis & Training:** Use the sidebar to trigger data generation, dataset balancing, and YOLOv8 training loops via the provided scripts.
- **Detection Dashboard:** Upload static images/videos from `test_images/` or use the webcam feed to test real-time detections, distance estimation, and AI safety alerts.

## 📊 Brief Explanation of the Evaluation Methodology

Our model evaluation balances standard machine learning metrics, safety-critical class assessments, and real-time execution constraints:

1. **Hard Target Accuracy (mAP@0.5):**  
   The primary objective is a validation testing accuracy exceeding 90% mAP@0.5, calculated at the end of each training epoch using standard IoU intersection metrics.

2. **Evaluating the "Tail" (Rare Classes):**  
   To combat the long-tail distribution typical of traffic datasets (where critical signs like "Stop" appear rarely), we use procedural generation to scale minority classes to 1,000 images each. We specifically monitor the validation score (mAP) for these synthetically augmented minority classes.

3. **Custom Early Stopping:**  
   To optimize training compute, the training loop monitors validation fitness. If the validation metric stagnates for 5 consecutive epochs, training terminates automatically, and the best-performing serialized weights are saved.

4. **Latency Optimization (Sub-10ms):**  
   Designed for highway driving assistance, the target inference speed is sub-10ms per frame. Bounding box extraction, distance estimation, and local overlay calculations are vectorized in NumPy to avoid frame skipping on edge hardware.
