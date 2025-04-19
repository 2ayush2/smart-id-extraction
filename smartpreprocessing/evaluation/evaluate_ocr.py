import json
import os
from difflib import SequenceMatcher

# ---------------------- CONFIG ----------------------
# Path to predicted output (OCR result)
predicted_path = os.path.join("..", "ocr_result", "0af50be5-1d41-4214-b37c-2252937713eb", "final_combined.json")

# Path to correct values
ground_truth_path = os.path.join("..", "test_data", "ground_truth.json")

# Similarity threshold
threshold = 0.85

# ---------------------- UTILS ----------------------
def normalize(text):
    return text.strip().lower().replace("  ", " ")

def text_similarity(a, b):
    return SequenceMatcher(None, normalize(a), normalize(b)).ratio()

def is_invalid(value):
    return value.strip().lower() in {"unknown", "invalid", "invalid date", "n/a", ""}

# ---------------------- EVALUATION ----------------------
with open(predicted_path, "r", encoding="utf-8") as f:
    predicted = json.load(f)

with open(ground_truth_path, "r", encoding="utf-8") as f:
    ground_truth = json.load(f)

correct = 0
total = 0

print("\n📊 Field-wise Evaluation:\n")

for field, true_value in ground_truth.items():
    pred_value = predicted.get(field, "")
    
    if is_invalid(pred_value):
        similarity = 0.0
    else:
        similarity = text_similarity(true_value, pred_value)

    print(f"{field}:\n  ✅ GT:   {true_value}\n  🧠 Pred: {pred_value}\n  🔍 Similarity: {similarity:.2f}\n")

    if similarity >= threshold:
        correct += 1
    total += 1

accuracy = (correct / total) * 100
print(f"✅ Final OCR Accuracy: {accuracy:.2f}%\n")
