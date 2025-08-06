from ultralytics import YOLO
import cv2
import os

# --- Step 1: Define the paths to your files ---
trained_model_path = 'best.pt'
test_image_filename = 'image.png' # <--- IMPORTANT: REPLACE with your test image name!

# --- Step 2: Load your trained model ---
if os.path.exists(trained_model_path):
    trained_model = YOLO(trained_model_path)
    print(f"Successfully loaded trained model from: {trained_model_path}")
else:
    print(f"Error: Trained model not found at {trained_model_path}.")
    print("Please ensure 'best.pt' is in the same directory as detect.py.")
    exit()

if not os.path.exists(test_image_filename):
    print(f"Error: Test image '{test_image_filename}' not found.")
    print("Please ensure the image file is in the same directory as detect.py.")
    exit()
print(f"Running inference on: {test_image_filename}")
results = trained_model.predict(test_image_filename, save=True, conf=0.25)

print("\nInference complete. The image with detections has been saved.")
print("You can find the output in the 'runs/detect/predict/' folder.")