import argparse
from ultralytics import YOLO

def parse_args():
    p = argparse.ArgumentParser(description="Train YOLOv8 for traffic sign detection")
    p.add_argument("--data", type=str, required=True, help="Path to YOLO data YAML")
    p.add_argument("--model", type=str, default="yolov8m.pt", help="Base model checkpoint")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--project", type=str, default="runs/detect")
    p.add_argument("--name", type=str, default="traffic_sign_train")
    p.add_argument("--hsv_h", type=float, default=0.015)
    p.add_argument("--hsv_s", type=float, default=0.7)
    p.add_argument("--hsv_v", type=float, default=0.4)
    p.add_argument("--degrees", type=float, default=8.0)
    p.add_argument("--translate", type=float, default=0.1)
    p.add_argument("--scale", type=float, default=0.5)
    p.add_argument("--shear", type=float, default=2.0)
    p.add_argument("--perspective", type=float, default=0.0005)
    p.add_argument("--flipud", type=float, default=0.0)
    p.add_argument("--fliplr", type=float, default=0.5)
    p.add_argument("--mosaic", type=float, default=1.0)
    p.add_argument("--mixup", type=float, default=0.1)
    return p.parse_args()

def main():
    args = parse_args()
    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        pretrained=True,
        cache=True,
        cos_lr=True,
        optimizer="AdamW",
        patience=25,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        shear=args.shear,
        perspective=args.perspective,
        flipud=args.flipud,
        fliplr=args.fliplr,
        mosaic=args.mosaic,
        mixup=args.mixup,
    )

if __name__ == "__main__":
    main()
