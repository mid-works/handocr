from paddleocr import PaddleOCR
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
from spellchecker import SpellChecker
import torch
import cv2
import numpy as np
import os
import json
import re
import time

# ================= SETTINGS =================
IMAGE_PATH = "swe.png"
CROP_DIR = "crops"
FINAL_OUTPUT = "final_output.json"
Y_THRESHOLD = 25
# ============================================

start_time = time.time()
os.makedirs(CROP_DIR, exist_ok=True)

# Clear old crops
for file in os.listdir(CROP_DIR):
    if file.endswith(".png"):
        os.remove(os.path.join(CROP_DIR, file))

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

spell = SpellChecker()

original = cv2.imread(IMAGE_PATH)
if original is None:
    raise ValueError("Image not found!")

img = original.copy()

# ================= DETECTION =================
results_normal = ocr.ocr(img, cls=True)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

thresh = cv2.adaptiveThreshold(
    gray, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    25, 5
)

kernel = np.ones((1, 2), np.uint8)
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
    return np.mean(np.array(box), axis=0)

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

unique_boxes = sorted(unique_boxes, key=lambda b: (b[0][1], b[0][0]))

lines = []

for box in unique_boxes:
    y_center = (box[0][1] + box[2][1]) / 2
    placed = False

    for line in lines:
        if abs(y_center - line["y_center"]) < Y_THRESHOLD:
            line["boxes"].append(box)
            line["y_center"] = np.mean([
                (b[0][1] + b[2][1]) / 2 for b in line["boxes"]
            ])
            placed = True
            break

    if not placed:
        lines.append({
            "boxes": [box],
            "y_center": y_center
        })

merged_lines = []

for line in lines:
    line_boxes = line["boxes"]

    x_min = min(b[0][0] for b in line_boxes)
    y_min = min(b[0][1] for b in line_boxes)
    x_max = max(b[2][0] for b in line_boxes)
    y_max = max(b[2][1] for b in line_boxes)

    merged_box = [
        [x_min, y_min],
        [x_max, y_min],
        [x_max, y_max],
        [x_min, y_max]
    ]

    merged_lines.append(merged_box)

print(f"Total lines detected: {len(merged_lines)}")


def crop_rotated(img, box, padding=10):
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
    warped = cv2.warpPerspective(img, M, (width, height))

    warped = cv2.copyMakeBorder(
        warped,
        padding, padding, padding, padding,
        cv2.BORDER_CONSTANT,
        value=[255, 255, 255]
    )

    return warped

# ================= SPELL CORRECTION =================
# def correct_text(text):
#     words = text.split()
#     corrected_words = []

#     for word in words:
#         prefix = re.match(r'^\W*', word).group()
#         suffix = re.search(r'\W*$', word).group()
#         core = re.sub(r'^\W+|\W+$', '', word)

#         if len(core) <= 2 or core.lower() in spell:
#             corrected_words.append(word)
#             continue

#         correction = spell.correction(core)

#         if correction:
#             corrected_words.append(prefix + correction + suffix)
#         else:
#             corrected_words.append(word)

#     return " ".join(corrected_words)

# ================= RECOGNITION =================
final_results = []

for idx, box in enumerate(merged_lines):

    crop = crop_rotated(original, box)
    crop_path = os.path.join(CROP_DIR, f"line_{idx}.png")
    cv2.imwrite(crop_path, crop)

    image = Image.open(crop_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            pixel_values,
            max_new_tokens=120,
            num_beams=5,
            early_stopping=True
        )

    raw_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    # corrected_text = correct_text(raw_text)

    final_results.append({
        "line": idx,
        "trocr_raw": raw_text,
        # "trocr_corrected": corrected_text
    })

    print(f"\nLine {idx}")
    print("RAW      →", raw_text)
    # print("CORRECT  →", corrected_text)

# ================= SAVE =================
with open(FINAL_OUTPUT, "w", encoding="utf-8") as f:
    json.dump(final_results, f, indent=4)

end_time = time.time()

print("\nFull OCR Pipeline Completed ✅")
print("Total Time:", end_time - start_time)