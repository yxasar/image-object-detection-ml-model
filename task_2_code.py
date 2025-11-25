import os
import cv2
import numpy as np

# ---------------- SETTINGS ---------------- #
input_folder = "task_2_input"            # Folder containing X.jpg and X~2.jpg
output_folder = "task_2_output"          # Output folder to store processed files

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# ---------------- PROCESSING ---------------- #
files = [f for f in os.listdir(input_folder) if f.endswith(".jpg") and "~2" not in f]

for before_name in files:

    before_path = os.path.join(input_folder, before_name)

    # Construct after image name: X~2.jpg
    base_name = before_name.replace(".jpg", "")
    after_name = f"{base_name}~2.jpg"
    after_path = os.path.join(input_folder, after_name)

    if not os.path.exists(after_path):
        print(f"Missing after image for {before_name}, skipping...")
        continue

    # Load images
    before_img = cv2.imread(before_path)
    after_img = cv2.imread(after_path)

    # Convert to grayscale
    before_gray = cv2.cvtColor(before_img, cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(after_img, cv2.COLOR_BGR2GRAY)

    # Compute absolute difference
    diff = cv2.absdiff(before_gray, after_gray)

    # Threshold to get binary mask of changes
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    # Morphological operations to group areas
    kernel = np.ones((5,5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Find contours of differences
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    annotated = after_img.copy()

    # Draw bounding boxes
    for contour in contours:
        if cv2.contourArea(contour) > 300:  # Ignore small noise
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Save output as X~3.jpg inside task_2_output
    output_name = f"{base_name}~3.jpg"
    cv2.imwrite(os.path.join(output_folder, output_name), annotated)
    print(f"[DONE] {before_name} -> {output_name}")

print("\nProcessing complete!")
