from collections import Counter
from typing import Any

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO


def load_model(weights_path: str) -> YOLO:
    return YOLO(weights_path)


def _serialize_detections(result: Any) -> list[dict]:
    detections = []
    names = result.names
    for box in result.boxes:
        cls_id = int(box.cls.item())
        detections.append(
            {
                "class": names.get(cls_id, str(cls_id)),
                "confidence": round(float(box.conf.item()), 4),
                "xyxy": [round(float(v), 1) for v in box.xyxy[0].tolist()],
            }
        )
    return detections


def predict_image(model: YOLO, image: Image.Image, conf: float = 0.35, iou: float = 0.45):
    rgb = np.array(image)
    result = model.predict(source=rgb, conf=conf, iou=iou, verbose=False)[0]
    plotted = result.plot()
    plotted = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)
    return plotted, _serialize_detections(result)


def predict_video(model: YOLO, input_path: str, output_path: str, conf: float = 0.35, iou: float = 0.45):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    class_counter = Counter()
    frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        result = model.predict(source=frame, conf=conf, iou=iou, verbose=False)[0]
        writer.write(result.plot())

        for d in _serialize_detections(result):
            class_counter[d["class"]] += 1
        frames += 1

    cap.release()
    writer.release()

    summary = [{"class": k, "count": v} for k, v in class_counter.items()]
    summary.append({"class": "_frames_processed", "count": frames})
    return summary
