from ultralytics import YOLO

def main():
    # Load a medium YOLOv8 model for better accuracy/speed balance
    model = YOLO("yolov8m.pt")
    
    # Train on the built-in TT100K dataset (100,000 images)
    # WARNING: This will download approx 10GB of data.
    # It is recommended to run this on a machine with a GPU (NVIDIA).
    results = model.train(
        data="tt100k.yaml", 
        epochs=100, 
        imgsz=640,
        batch=16,
        project="runs/detect",
        name="tt100k_pro"
    )
    
    print("Professional training complete!")
    print(f"Weights saved to: {results.save_dir}/weights/best.pt")

if __name__ == "__main__":
    main()
