import os
from ultralytics import YOLO

model = YOLO("model.pt")

TEST_IMAGES_DIR = "dataset/test/images"
TEST_LABELS_DIR = "dataset/test/labels"
labels = ['Defect', 'Product']
print("0: 'Defect', 1: 'Product'")
total_correct = 0
total_wrong = 0

for img_name in os.listdir(TEST_IMAGES_DIR):
    img_path = os.path.join(TEST_IMAGES_DIR, img_name)
    label_path = os.path.join(TEST_LABELS_DIR, img_name.replace(".jpg", ".txt"))

    try:
        with open(label_path, "r") as f:
            original_classes = []
            for line in f:
                class_id = int(line.strip().split()[0])
                original_classes.append(class_id)
    except:
        continue

    results = model.predict(img_path, save=True, verbose=False, show_boxes=False)
    for r in results:

        try:
            predicted_classes = []
            for box in r.boxes.data:
                predicted_class = int(box[5].item())
                predicted_classes.append(predicted_class)
        except:
            predicted_classes = []

        correct_predictions = 0
        wrong_predictions = 0
        
        for pred_class in predicted_classes:
            if pred_class in original_classes:
                correct_predictions += 1
                total_correct += 1
            else:
                wrong_predictions += 1
                total_wrong += 1
                print(f"Wrong prediction: {labels[pred_class]}, Original classes: {[labels[c] for c in original_classes]}, Image: {img_name}")

        missed_classes = len(original_classes) - correct_predictions
        if missed_classes > 0:
            total_wrong += missed_classes
            print(f"Missed classes in {img_name}: {[labels[c] for c in original_classes if c not in predicted_classes]}, Predicted classes: {[labels[p] for p in predicted_classes]}")

print("-------------------------------------------------")
print("Total Predictions: ", total_correct + total_wrong)
print("Total Correct: ", total_correct)
print("Total Wrong: ", total_wrong)
print("Success Rate: ", (total_correct / (total_correct + total_wrong))*100)
print("-------------------------------------------------")