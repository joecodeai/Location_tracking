# Get class ID

import cv2
import numpy as np
from ultralytics import YOLO

# Function to calculate the center of a bounding box
def calculate_center(x1, y1, x2, y2):
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    return center_x, center_y

# Function to calculate the distance between two points
def calculate_distance(pt1, pt2):
    return np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)

# Function to draw matching coordinates and IDs on both images
def draw_matching_coordinates(input_image, similar_image, similar_results, input_results, distance_threshold=100):
    similar_boxes = similar_results[0].boxes  # Get detected boxes from similar image
    input_boxes = input_results[0].boxes      # Get detected boxes from input image
    
    id_counter = 1  # Start ID counter
    matched_ids = {}  # Dictionary to store matched IDs for objects

    for sim_box in similar_boxes:
        sim_x1, sim_y1, sim_x2, sim_y2 = map(int, sim_box.xyxy[0])  # Bounding box coordinates
        sim_class_id = int(sim_box.cls[0])  # Class ID

        # Calculate center of the similar image box
        sim_center = calculate_center(sim_x1, sim_y1, sim_x2, sim_y2)

        for inp_box in input_boxes:
            inp_x1, inp_y1, inp_x2, inp_y2 = map(int, inp_box.xyxy[0])  # Bounding box coordinates
            inp_class_id = int(inp_box.cls[0])  # Class ID

            # Calculate center of the input image box
            inp_center = calculate_center(inp_x1, inp_y1, inp_x2, inp_y2)

            # Check if class IDs match and centers are within the distance threshold
            if sim_class_id == inp_class_id and calculate_distance(sim_center, inp_center) < distance_threshold:
                # Assign ID if not already assigned
                if (sim_class_id, sim_center) not in matched_ids:
                    matched_ids[(sim_class_id, sim_center)] = id_counter
                    id_counter += 1

                # Get the assigned ID
                obj_id = matched_ids[(sim_class_id, sim_center)]

                # Draw on input image with ID
                cv2.putText(input_image, f'ID: {obj_id} ({inp_center[0]}, {inp_center[1]})', 
                            (inp_center[0], inp_center[1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.circle(input_image, inp_center, 5, (0, 255, 0), -1)

                # Draw on similar image with ID
                cv2.putText(similar_image, f'ID: {obj_id} ({sim_center[0]}, {sim_center[1]})', 
                            (sim_center[0], sim_center[1] + 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.circle(similar_image, sim_center, 5, (0, 255, 0), -1)

# Load the images
# input_image_path = "/home/mundax/Projects/Location_tracking/similarity/input_with_detections.jpg"
# similar_image_path = "/home/mundax/Projects/Location_tracking/similarity/best_match_with_detections.jpg"

input_image_path = "/home/mundax/Projects/Location_tracking/model/data/images/train/post___90.jpg"
similar_image_path = "/home/mundax/Projects/Location_tracking/model/data/images/train/post___91.jpg"

input_image = cv2.imread(input_image_path)
similar_image = cv2.imread(similar_image_path)

# Load the YOLO model
model = YOLO("/home/mundax/Projects/Location_tracking/model/yolov11m_saved.pt")

# Perform inference on both images
similar_results = model(similar_image)
input_results = model(input_image)

# Draw matching coordinates and IDs from both images
draw_matching_coordinates(input_image, similar_image, similar_results, input_results)

# Resize images to the same height
height = 480  # Desired height
input_image_resized = cv2.resize(input_image, (int(input_image.shape[1] * height / input_image.shape[0]), height))
similar_image_resized = cv2.resize(similar_image, (int(similar_image.shape[1] * height / similar_image.shape[0]), height))

# Stack images horizontally
combined_image = np.hstack((input_image_resized, similar_image_resized))

# Show the resulting image
cv2.imshow('Input Image and Best Match with Matching Coordinates', combined_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
