import os
import random
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageEnhance

def augment_image(img):
    # Random brightness
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(random.uniform(0.7, 1.3))
    
    # Random contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(random.uniform(0.7, 1.3))
    
    # Random rotation
    img = img.rotate(random.uniform(-10, 10), resample=Image.BICUBIC, expand=False)
    
    return img

def main():
    source_dir = Path("C:/Users/user/.gemini/antigravity/brain/5b752ed2-29a2-4ec8-bd92-ff5fba97b4aa")
    dest_images = Path("datasets/pro_synthetic/images")
    dest_labels = Path("datasets/pro_synthetic/labels")
    dest_images.mkdir(parents=True, exist_ok=True)
    dest_labels.mkdir(parents=True, exist_ok=True)

    # Class Mapping (Simplified for this booster run)
    # 0: speed_limit_50, 1: stop, 2: no_parking, 3: yield
    mapping = {
        "speed_50": 0,
        "stop": 1,
        "no_parking": 2,
        "yield": 3
    }

    # Find all generated images
    files = list(source_dir.glob("*.png"))
    
    count = 0
    for f in files:
        cls_key = None
        for k in mapping:
            if k in f.name.lower():
                cls_key = k
                break
        
        if cls_key is None:
            continue
            
        cls_id = mapping[cls_key]
        img_base = Image.open(f).convert("RGB")
        
        # Generate 50 augmented versions for each base image
        for i in range(50):
            aug_img = augment_image(img_base)
            name = f"{cls_key}_{count}_{i}.jpg"
            aug_img.save(dest_images / name)
            
            # YOLO Label: Class X_center Y_center Width Height (Normalized)
            # Assuming the sign is roughly centered and 50% of the image
            with open(dest_labels / f"{cls_key}_{count}_{i}.txt", "w") as lf:
                lf.write(f"{cls_id} 0.5 0.5 0.5 0.5\n")
        
        count += 1
    
    print(f"Generated {count * 50} augmented training images.")

    # Create YAML
    with open("data/pro_synthetic.yaml", "w") as y:
        y.write(f"path: {os.path.abspath('datasets/pro_synthetic')}\n")
        y.write("train: images\nval: images\n\n")
        y.write("names:\n  0: speed_limit_50\n  1: stop\n  2: no_parking\n  3: yield\n")

if __name__ == "__main__":
    main()
