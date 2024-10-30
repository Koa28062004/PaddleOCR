import cv2
import json
import numpy as np
import os

# Path to your Label.txt file
label_file = "200_ngoai canh/Label.txt"

# Base directory for images
image_base_dir = ""

# Directory to save annotated images
output_dir = "det_ground_truth/"
os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist

# Function to draw rectangles based on points
def draw_rectangle(img, points):
    # Convert points to tuples
    pts = [tuple(point) for point in points]
    # Draw the polygon based on the four points
    cv2.polylines(img, [np.array(pts)], isClosed=True, color=(0, 255, 0), thickness=2)

# Read Label.txt
with open(label_file, 'r') as f:
    lines = f.readlines()

# Process each line in Label.txt
for line in lines:
    # Split each line into image path and JSON annotation
    image_path, annotation = line.strip().split("\t")
    annotation_data = json.loads(annotation)
    
    # Load the image
    full_image_path = os.path.join(image_base_dir, image_path)
    img = cv2.imread(full_image_path)
    if img is None:
        print(f"Failed to load image: {full_image_path}")
        continue

    # Draw each bounding box
    for item in annotation_data:
        points = item['points']
        draw_rectangle(img, points)
    
    # Construct output path and save the image
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, img)
    print(f"Annotated image saved to {output_path}")
