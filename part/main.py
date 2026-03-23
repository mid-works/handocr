from paddleocr import PaddleOCR
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import cv2
import numpy as np
import os
import json
import time

IMAGE_PATH = "swe.png"
CROP_DIR = "crops"
FINAL_OUTPUT = "final_output.json"
sttime= time.time()
os.makedirs(CROP_DIR, exist_ok=True)

print("Loading PaddleOCR...")
ocr = PaddleOCR(
    use_angle_cls=True,
    lang='en',
    use_gpu=False,
    det_limit_side_len=2000,
    det_db_thresh=0.12,
    det_db_box_thresh=0.28,
    det_db_unclip_ratio=3.0
)

print("Loading TrOCR...")
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

original = cv2.imread(IMAGE_PATH)
img = original.copy()


results_normal = ocr.ocr(img, cls=True)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

thresh = cv2.adaptiveThreshold(
    gray, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    25, 5
)

kernel = np.ones((1,2), np.uint8)
thresh = cv2.dilate(thresh, kernel, iterations=1)

results_pre = ocr.ocr(thresh, cls=True)


all_boxes = []

for line in results_normal[0]:
    if line[1][1] > 0.4:
        all_boxes.append(line[0])

for line in results_pre[0]:
    if line[1][1] > 0.4:
        all_boxes.append(line[0])

def box_center(box):
    box = np.array(box)
    return np.mean(box, axis=0)

unique_boxes = []

for box in all_boxes:
    center = box_center(box)
    duplicate = False

    for ubox in unique_boxes:
        if np.linalg.norm(center - box_center(ubox)) < 15:
            duplicate = True
            break

    if not duplicate:
        unique_boxes.append(box)

unique_boxes = sorted(unique_boxes, key=lambda b: np.mean(b, axis=0)[1])


final_results = []

def crop_rotated(img, box):
    pts = np.array(box, dtype="float32")

    width = int(max(
        np.linalg.norm(pts[0] - pts[1]),
        np.linalg.norm(pts[2] - pts[3])
    ))

    height = int(max(
        np.linalg.norm(pts[0] - pts[3]),
        np.linalg.norm(pts[1] - pts[2])
    ))

    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(img, M, (width, height))

for idx, box in enumerate(unique_boxes):

    crop = crop_rotated(original, box)
    crop_path = os.path.join(CROP_DIR, f"line_{idx}.png")
    cv2.imwrite(crop_path, crop)

    # Run TrOCR
    image = Image.open(crop_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    with torch.no_grad():
        generated_ids = model.generate(pixel_values)

    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    final_results.append({
        "line": idx,
        "text": text
    })

    print(f"Line {idx}: {text}")


with open(FINAL_OUTPUT, "w", encoding="utf-8") as f:
    json.dump(final_results, f, indent=4)
edtime = time.time()
print("Full OCR Pipeline Completed ✅")
print('ttime = ',edtime -sttime)