import os
from ultralytics import YOLO
from ultralytics.data.annotator import auto_annotate
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the model
#auto_annotate(data="bottle_dataset/valid/images", det_model="detection.pt", sam_model="sam2.1_l.pt", device="cuda")


def show_segmentation(image_path, label_path, output_path):
    # Read image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create overlay for transparency
    overlay = image.copy()
    
    # Read segmentation coordinates
    with open(label_path, 'r') as f:
        for line in f:
            values = line.strip().split()
            class_id = int(values[0])
            coords = np.array([float(x) for x in values[1:]]).reshape(-1, 2)
            
            # Denormalize coordinates
            coords[:, 0] *= image.shape[1]
            coords[:, 1] *= image.shape[0]
            coords = coords.astype(np.int32)
            
            # Fill polygon with semi-transparent color
            cv2.fillPoly(overlay, [coords], (0, 255, 0))  # Green fill
            cv2.polylines(image, [coords], True, (0, 255, 0), 2)  # Green border
    
    # Apply transparency
    alpha = 0.3  # Transparency factor
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    # Instead of display, save the figure
    plt.figure(figsize=(10, 10), frameon=False)
    plt.imshow(image)
    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0,0)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()  # Close the figure to free memory

# Directory paths
images_dir = "segmentation_bottle_dataset/train/images"
labels_dir = "segmentation_bottle_dataset/train/labels"
output_dir = "segmented"  # New output directory

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Example usage for one image
for label_file in os.listdir(labels_dir):
    if label_file.endswith('.txt'):
        image_file = label_file.replace('.txt', '.jpg')
        image_path = os.path.join(images_dir, image_file)
        label_path = os.path.join(labels_dir, label_file)
        output_path = os.path.join(output_dir, f"segmented_{image_file}")
        
        if os.path.exists(image_path):
            print(f"Processing: {image_file}")
            show_segmentation(image_path, label_path, output_path)
