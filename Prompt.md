# Prompt

## Context and Role

You are a Senior ML Engineer specializing in Computer Vision and Real-Time AI Systems, you are responsible for designing and implementing a production-grade Traffic Sign Detection System for Indian roads.
The system must accurately detect and classify all major Indian traffic signs in real time and maintain extremely low latency and high reliability under diverse environmental conditions.

The project should support static image inference, recorded video analytics, and live-stream webcam detection. Additionally, the solution must integrate an LLM-powered reasoning layer capable of generating contextual driving alerts based on detected traffic signs.

# Objective

Develop a complete end-to-end Traffic Sign Detection System that:

* Detects and classifies Indian traffic signs with high accuracy.
* Operates in real time with ultra-low inference latency.
* Supports partially visible and occluded traffic signs.
* Works across diverse weather and lighting conditions.
* Provides contextual navigation alerts using an LLM layer.
* Includes a live visualization dashboard for monitoring detections and alerts.
* Produces a single fully working Python implementation.

# Dataset Requirements

## Dataset Selection

Use the German Traffic Sign Recognition Benchmark dataset as the primary training dataset.

Additionally:

* Curate supplementary Indian traffic sign datasets if required.
* Ensure all traffic sign categories relevant to Indian roads are covered.
* Maintain sufficient image distribution across all classes.

# Data Processing Requirements

## Dataset Balancing

For classes having fewer than 1,000 images:

* Use generative synthetic augmentation techniques.
* Generate realistic synthetic samples to reduce class imbalance.
* Ensure balanced representation across all sign categories.

## Preprocessing Pipeline

Using Pandas and NumPy, implement a robust preprocessing and validation pipeline that:

* Detects and removes corrupt image files.
* Validates annotation consistency.
* Handles missing labels and malformed metadata.
* Normalizes image dimensions and pixel distributions.

## Environmental Augmentation

To improve robustness against real-world conditions, apply aggressive augmentation techniques including:

* Random cropping
* CutMix augmentation
* Simulated occlusions
* Shadow overlays
* Tree branch interference
* Vehicle obstruction simulation
* Motion blur
* Rain effects
* Dust storm simulation
* Night-time glare
* Low-light enhancement
* Brightness and contrast randomization

The model must learn to detect:

* Partially visible traffic signs
* Damaged signs
* Distant signs
* Motion-blurred signs

# Input Requirements

The system must support:

## Static Images

Used for:

* Batch processing
* Road audits
* Offline traffic analysis

## Recorded Video

Used for:

* Fleet monitoring
* Post-drive analytics
* Surveillance applications

## Live Stream

Provide real-time webcam/video-stream integration for:

* In-vehicle driver assistance
* Live traffic monitoring
* Smart transportation systems

# Model and Inference Requirements

## Inference Engine

Use **YOLOv8** as the primary object detection architecture.

Alternative architectures may be considered if they provide:

* Better latency
* Higher mAP
* Improved edge-device performance

## Performance Optimization

Optimize the model for:

* GPU acceleration
* TensorRT/ONNX export
* Batch inference
* Quantization
* Mixed precision inference

Target:

* Sub-10ms inference latency per frame
* Stable FPS for highway-speed detection

# LLM Integration Requirements

## Contextual Alert Generation

Integrate **OpenAI** as a reasoning layer that converts detections into actionable driver alerts.

Instead of basic labels like:

> “Stop Sign Detected”

Generate contextual alerts such as:

> “Detected ‘No Entry’ sign 30 meters ahead; rerouting to the nearest legal turn.”

The LLM should:

* Interpret detection confidence
* Generate safe navigation guidance
* Produce human-readable contextual notifications

# Dashboard and Visualization Requirements

## Live Monitoring Interface

Build a high-performance dashboard using **Streamlit** that provides:

* Real-time bounding box visualization
* Detection confidence display
* Live alert logs
* Video stream rendering
* Detection statistics
* FPS monitoring
* System latency metrics

# UI and Visualization Requirements

The dashboard must include:

* Animated real-time detection overlays
* Detection history panel
* Alert notification feed
* Confidence score visualization
* Performance analytics

The interface must be:

* Responsive
* Lightweight
* Accessible
* Optimized for low-latency rendering

# Training and Evaluation Requirements

## Training Strategy

Train the model for:

* 50 epochs minimum

Implement:

* Early stopping callback
* Best-weight checkpoint serialization
* Validation monitoring

Training must terminate automatically if:

* Validation accuracy stagnates for 5 consecutive epochs

## Evaluation Metrics

The system must achieve:

* mAP@0.5 greater than 90%
* High precision on rare traffic signs
* Low false-positive rate
* Stable inference under real-world conditions

Evaluate specifically on:

* Rare sign categories
* Occluded signs
* Low-light environments
* Adverse weather conditions

# Output Requirements

Upon detecting a traffic sign, the system must:

* Identify and classify the sign
* Draw bounding boxes in real time
* Estimate confidence scores
* Estimate approximate sign distance
* Trigger contextual LLM-generated alerts
* Display results in the Streamlit dashboard

The output pipeline must include:

* Real-time annotated frames
* Alert logs
* Detection summaries
* JSON-formatted inference outputs

# Error Handling and Documentation Requirements

## Error Handling

Gracefully handle:

* Corrupt image files
* Missing annotations
* Webcam failures
* Stream interruptions
* Backend inference failures
* LLM API failures

Provide:

* Structured error responses
* Robust logging mechanisms
* Debugging-friendly error traces

## Documentation

Provide detailed documentation covering:

* Folder structure
* Dataset preparation
* Training instructions
* Environment setup
* Model export process
* Streamlit dashboard setup
* API key configuration
* Deployment instructions

# Performance and Scalability Requirements

Optimize the system for:

* Low memory consumption
* Efficient GPU utilization
* Scalable inference pipelines
* Multi-stream processing
* High-throughput deployments

Ensure:

* Minimal frame drops
* Smooth real-time detection
* Efficient batching
* Non-blocking UI rendering

# Technology Stack

## Core AI Stack

* Python - For the backend
* YOLOv8 - Building the model
* OpenCV - For Image and Video Analysis
* PyTorch - For implementing Machine Learning

## Data Processing

* NumPy 
* Pandas
* Albumentations

## LLM Integration

* OpenAI API

## Dashboard

* Streamlit

## Optimization Tools

* ONNX
* TensorRT
* CUDA

## Optional

* Docker
* FastAPI
* Redis Queue
* PostgreSQL or MongoDB for detection logs

# Final Requirements

The final system must:

* Detect all major Indian traffic signs
* Achieve testing accuracy above 90%
* Maintain ultra-low latency
* Detect partially visible signs reliably
* Operate in real time
* Generate contextual AI-powered driving alerts
* Provide a fully working single Python implementation suitable for production deployment
