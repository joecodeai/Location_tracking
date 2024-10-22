from ultralytics import YOLO
import cv2
import numpy as np

# Load the trained YOLO model
model = YOLO("/home/mundax/Projects/Location_tracking/model/yolov11m_new_saved.pt")

# Train the model on the Open Images V7 dataset
# results = model.train(data="/home/mundax/Projects/Location_tracking/model/data.yaml", epochs=10, imgsz=416)

# Save the model and weights
# model.save("/home/mundax/Projects/Location_tracking/model/yolo11m_saved.pt")

# Read an image using OpenCV
image_path = "/home/mundax/Projects/Location_tracking/model/data/images/train/hiLife___17.jpg"
image = cv2.imread(image_path)

# Get image dimensions
height, width = image.shape[:2]

# Calculate the center of the image
image_center_x = width / 2

# Define maximum angle for maximum distance (adjust as needed)
max_angle = 45  # Maximum angle in degrees
max_distance = width / 2  # Maximum distance from center in pixels

# Perform inference on the image
results = model(image)

# Process results to get bounding boxes, labels, and angles
for result in results:
    boxes = result.boxes  # Get the detected boxes
    for box in boxes:
        # Get bounding box coordinates and class ID
        x1, y1, x2, y2 = box.xyxy[0]  # Top-left and bottom-right coordinates
        class_id = int(box.cls[0])  # Class ID
        
        # Calculate width and height of the bounding box
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        
        # Calculate the center coordinates of the bounding box
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        # Calculate distance from image center
        distance_from_center = center_x - image_center_x

        # Map pixel distance to angle
        angle = (distance_from_center / max_distance) * max_angle

        # Draw bounding box on the image
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        
        # Draw the center point on the image
        cv2.circle(image, (center_x, center_y), 5, (255, 0, 0), -1)

        # Define text properties
        font_scale = 0.4
        font_color = (255, 255, 255)
        thickness = 1
        background_color = (0, 0, 0)  # Background color for text

        # Prepare text strings
        class_text = f'Class: {class_id}'
        center_text = f'Center: ({center_x}, {center_y})'
        angle_text = f'Angle: {angle:.2f}'

        # Calculate text sizes for background rectangles
        (class_text_width, class_text_height), _ = cv2.getTextSize(class_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        (center_text_width, center_text_height), _ = cv2.getTextSize(center_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        (angle_text_width, angle_text_height), _ = cv2.getTextSize(angle_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

        # Draw background rectangles for each text
        cv2.rectangle(image, (int(x1), int(y1)), (int(x1) + class_text_width + 10, int(y1) + class_text_height + 10), background_color, -1)
        cv2.rectangle(image, (center_x - 40, center_y + 5), (center_x + center_text_width - 40, center_y + center_text_height + 15), background_color, -1)
        cv2.rectangle(image, (int(x1) + 70, int(y1)), (int(x1) + 60 + angle_text_width + 10, int(y1) + angle_text_height + 10), background_color, -1)

        # Put label text on the image with background
        cv2.putText(image, class_text, (int(x1)+10, int(y1) + class_text_height + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, thickness)
        
        # Display central coordinates on the image with background
        cv2.putText(image, center_text, (center_x - 40, center_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, thickness)  
        
        # Display angle on the image with background
        cv2.putText(image, angle_text, (int(x1)+70, int(y1) + angle_text_height + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, thickness)

# Show the image with detections
cv2.imshow("Detections", image)
cv2.waitKey(0)
cv2.destroyAllWindows()