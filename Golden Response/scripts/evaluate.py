import argparse
from ultralytics import YOLO

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate YOLOv8 traffic sign detector")
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--imgsz", type=int, default=960)
    return p.parse_args()

def main():
    args = parse_args()
    model = YOLO(args.weights)
    metrics = model.val(data=args.data, imgsz=args.imgsz)
    print("mAP50:", metrics.box.map50)
    print("mAP50-95:", metrics.box.map)

if __name__ == "__main__":
    main()
