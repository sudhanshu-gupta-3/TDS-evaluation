🚗 Indian & GTSRB Traffic Sign AI Detector
An end-to-end, high-performance, and real-time traffic sign detection system designed to identify Vienna Convention-compliant traffic signs (common across India and the GTSRB dataset).

This project addresses severe class imbalance in traffic sign datasets through procedural generative synthetic data, implements aggressive monsoon rain and lighting augmentations, and runs YOLOv8 with custom Early Stopping callbacks, feeding bounding box detections into an OpenAI LLM reasoning layer to produce natural language safety advice for the driver.

📂 Repository Structure

TDS-evaluation/
├── Golden Response.py     # Complete single-file Python implementation: synthetic data
│                          # generator, dataset validator, augmentation pipeline, YOLOv8
│                          # trainer with Early Stopping, OpenAI alert engine, and
│                          # Streamlit real-time detection dashboard.
│
├── Justification.md       # Detailed justification document explaining architectural
│                          # decisions, model selection rationale, and design trade-offs.
│
├── Prompt.md              # Original prompt/requirements specification used for
│                          # building this project.
│
└── Readme.md              # Project documentation — overview, setup instructions,
                           # repository structure, and evaluation methodology (this file).
🛠️ Instructions for Running & Testing
Step 1: Install Python
Ensure Python 3.10+ is installed on your Windows machine. Remember to check "Add python.exe to PATH" during setup.

Step 2: Install Package Dependencies
Open a terminal in this directory and execute:

cmd

python -m pip install -r requirements.txt
Note: If requirements.txt is not present in the repo, install the following packages manually:

cmd

pip install streamlit ultralytics opencv-python pandas numpy pillow openai pyyaml
Step 3: Start the Streamlit Dashboard
Launch the dashboard local webserver:

cmd

streamlit run "Golden Response.py"
Step 4: Run the Pipeline Steps in the UI
API Key Setup: Add your OPENAI_API_KEY in the sidebar to test GPT-generated safety warnings. If omitted, the system falls back to a rules-based offline generator.
Synthesize & Balance Dataset:
Go to the ⚙️ Model Training Console tab.
Select either Fast Demo Mode (120 images per class) or Full Production Mode (1,200 images per class).
Click Generate & Balance Dataset to start the synthetic templating engine.
Inspect the Pandas validation log summarizing image channels, file existence, bounding boxes, and corrupt files.
Train YOLOv8:
Click Start YOLOv8 Training to kick off the 50-epoch training loop.
Note the custom Early Stopping logs showing plateaus and halts.
Detect Signs:
Navigate to the 🚗 Real-Time Detection Dashboard.
Upload static images, upload recorded MP4 videos, or toggle the webcam feed to test detections, distance estimation, and AI safety alerts in real-time.
Augmentation Preview:
Explore the 📊 Augmentation & Dataset Preview tab to visually compare clean signs against monsoon rain, glare, dust, and CutMix transformations.
📊 Evaluation Methodology
Our model evaluation balances standard machine learning metrics, safety critical class assessments, and real-time execution constraints:

1. Hard Target Accuracy (mAP@.5)
The core target is a validation testing accuracy exceeding 90% mAP@0.5.
Validation is performed at the end of each epoch using standard IoU intersection metrics.
2. Evaluating the "Tail" (Rare Classes)
Classic traffic datasets suffer from a long-tail distribution where critical signs (like Stop or Yield) have few training images compared to common speed limits.
By applying procedural generative templates, we scale up all rare classes to exactly 1,000 images per class, ensuring class balance and eliminating the risk of rare-class neglect during training.
We specifically monitor the validation score (mAP) for these minority classes during evaluation.
3. Custom Early Stopping
To optimize training compute, we monitor validation fitness.
If the validation metric fails to improve for 5 consecutive epochs (constant accuracy), the training terminates automatically, and the best-performing serialized weights are saved.
4. Latency Optimization (Sub-10ms)
Designed for highway driving assistance, the target inference speed is sub-10ms per frame (running on edge hardware/GPU).
Bounding box extraction, distance estimation, and local overlay calculations are vectorized in NumPy to run inside the sub-millisecond range, avoiding frame skipping.
