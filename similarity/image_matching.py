from ultralytics import YOLO
import cv2
import numpy as np
import os
import math
import concurrent.futures

# Load the trained YOLO model
model = YOLO("/home/mundax/Projects/Location_tracking/model/yolov11m_saved.pt")

# Load the input image for ORB matching
input_image_path = "/home/mundax/Projects/Location_tracking/model/data/images/hiLife___18.jpg"
img = cv2.imread(input_image_path)
input_image = cv2.resize(img, (640, 480))

# Convert input image to grayscale
gray_input = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

# Create ORB detector
orb = cv2.ORB_create(nfeatures=3000)

# Detect keypoints and compute descriptors for the input image
keypoints_input, descriptors_input = orb.detectAndCompute(gray_input, None)

# Initialize variables to track the most similar image
most_similar_score = 0  # Start with zero for maximum matches
best_match_image_path = ""

# Directory containing images to compare with
image_folder = "/home/mundax/Projects/Location_tracking/model/data/images/train/"

def process_image(image_path):
    # Load and process each image
    image = cv2.imread(image_path)

    # Resize for faster processing
    image_resized = cv2.resize(image, (640, 480))
    
    # Convert to grayscale
    gray_image = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)

    # Detect keypoints and compute descriptors for the current image
    keypoints_image, descriptors_image = orb.detectAndCompute(gray_image, None)

    # Use KNN matcher to find matches
    knn_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = knn_matcher.knnMatch(descriptors_input, descriptors_image, k=2)

    # Apply Lowe's ratio test
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

    return len(good_matches), image_path

# Use ThreadPoolExecutor for multithreading
with concurrent.futures.ThreadPoolExecutor() as executor:
    # Prepare the list of image paths to process
    image_paths = [os.path.join(image_folder, filename) for filename in os.listdir(image_folder)
                   if filename.endswith(('.jpg', '.jpeg', '.png', '.JPG'))]

    # Process images in parallel
    results = list(executor.map(process_image, image_paths))

# Find the most similar image based on the results
for score, path in results:
    if score > most_similar_score:
        most_similar_score = score
        best_match_image_path = path

# Function to draw bounding boxes on an image based on YOLO results
def draw_detections(image, results):
    for result in results:
        boxes = result.boxes  # Get detected boxes
        for box in boxes:
            # Get bounding box coordinates and class ID
            x1, y1, x2, y2 = box.xyxy[0]  # Top-left and bottom-right coordinates
            class_id = int(box.cls[0])  # Class ID
            
            # Draw bounding box on the image
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # # Put label text on the image
            # cv2.putText(image, f'Class: {class_id}', (int(x1), int(y1) + 20), 
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

# Display results for the most similar image found and run YOLO on it
if best_match_image_path:
    best_match_image = cv2.imread(best_match_image_path)
    
    # Resize best match image to match the input image height
    best_match_image = cv2.resize(best_match_image, (640, 480))  # Resize to the same dimensions

    # Perform inference on the best match image using YOLO model
    best_match_results = model(best_match_image)

    # Perform inference on the input image using YOLO model
    input_results = model(input_image)

    # Draw detections on both images
    draw_detections(input_image, input_results)
    draw_detections(best_match_image, best_match_results)

    print(f'Most Similar Image: {best_match_image_path}')
    print(f'Number of Good Matches: {most_similar_score}')
    
    # Concatenate images side by side
    combined_image = np.hstack((input_image, best_match_image))
    
    # Show combined image
    cv2.imwrite("/home/mundax/Projects/Location_tracking/similarity/input_with_detections.jpg", input_image)
    cv2.imwrite("/home/mundax/Projects/Location_tracking/similarity/best_match_with_detections.jpg", best_match_image)

    cv2.imshow('Input Image (Left) and Best Match with Detections (Right)', combined_image)
    cv2.waitKey(0)
else:
    print("No similar images found.")

cv2.destroyAllWindows()
