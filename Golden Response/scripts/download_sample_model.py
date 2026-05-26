from pathlib import Path
from ultralytics import YOLO

def main() -> None:
    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)
    target = model_dir / "traffic_sign_best.pt"

    if target.exists():
        print(f"Model already exists at {target}")
        return

    # Downloads YOLOv8n weights and stores them as the local starter checkpoint.
    # Replace this with your fine-tuned checkpoint for best traffic-sign accuracy.
    model = YOLO("yolov8n.pt")
    src = Path(model.ckpt_path)
    target.write_bytes(src.read_bytes())
    print(f"Starter model saved to {target}")

if __name__ == "__main__":
    main()
