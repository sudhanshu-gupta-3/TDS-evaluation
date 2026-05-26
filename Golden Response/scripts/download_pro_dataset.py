from roboflow import Roboflow
import os

def main():
    api_key = "rf_i6ix1UQJDarbHRQkxXqM"
    rf = Roboflow(api_key=api_key)
    # Downloading the most popular YOLOv8 Traffic Sign dataset (approx 50k images)
    # Using a reliable public GTSRB dataset
    project = rf.workspace("roboflow-jvuw7").project("traffic-sign-detection-gtsrb")
    dataset = project.version(1).download("yolov8")
    
    print(f"Dataset downloaded to: {dataset.location}")
    print("You can now train with: python scripts/train.py --data dataset/data.yaml --model yolov8m.pt --epochs 100")

if __name__ == "__main__":
    main()
