import sqlite3
import cv2
import os
import torch
from ultralytics import YOLO

# Connect to your SQLite database
db_path = '/home/mundax/SQLite/My_Database.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Function to retrieve object coordinates from the database for a specific id
def get_object_coordinates(primary_key_id):
    cursor.execute('''
        SELECT obj1_latitude, obj1_longitude, 
               obj2_latitude, obj2_longitude, 
               obj3_latitude, obj3_longitude
        FROM location
        WHERE id = ?
    ''', (primary_key_id,))
    return cursor.fetchone()

# Function to find the most similar image using ORB feature matching
def find_most_similar_image(new_image_path, folder_path):
    new_img = cv2.imread(new_image_path)
    if new_img is None:
        print(f"Error: Unable to load image at {new_image_path}")
        return None
    
    orb = cv2.ORB_create(nfeatures=3000)
    keypoints1, descriptors1 = orb.detectAndCompute(new_img, None)
    
    best_match_file = None
    max_matches = 0

    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is None:
                continue

            keypoints2, descriptors2 = orb.detectAndCompute(img, None)
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(descriptors1, descriptors2)
            num_matches = len(matches)

            if num_matches > max_matches:
                max_matches = num_matches
                best_match_file = filename

    print(f"Best match: {best_match_file} with {max_matches} matches.")
    return best_match_file

# Function to display the new image with object coordinates
def display_objects_on_image(image_path, primary_key_id, detections):
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Error: Unable to load image at {image_path}")
        return

    coordinates = get_object_coordinates(primary_key_id)
    
    if coordinates is None:
        print(f"No coordinates found for primary key id {primary_key_id}.")
        return

    height, width, _ = img.shape
    image_center = (width // 2, height // 2)

    for i, (lat, lon) in enumerate(zip(coordinates[::2], coordinates[1::2]), start=1):
        if lat is not None and lon is not None:
            print(f"Object {i}: Latitude: {lat}, Longitude: {lon}")
            
            if detections:  # Only if detections are present
                for (x1, y1, x2, y2) in detections:
                    object_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    distance_from_center = ((object_center[0] - image_center[0]) ** 2 + 
                                            (object_center[1] - image_center[1]) ** 2) ** 0.5
                    max_angle = 45
                    max_distance = width / 2
                    angle = (distance_from_center / max_distance) * max_angle

                    cv2.putText(img, f"Obj{i}: ({lat}, {lon}), Angle: {angle:.2f}", 
                                (object_center[0], object_center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    cv2.putText(img, f"Angle: {angle:.2f}", 
                                (object_center[0], object_center[1] + 20),  # Offset the y-coordinate
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    cv2.circle(img, object_center, 5, (0, 255, 0), -1)  # Draw center point

    cv2.imshow('Image with Object Coordinates', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Function to run YOLO model and get detections
def run_yolo_model(image_path, model_path):
    model = YOLO(model_path)  # Load custom YOLO model
    results = model(image_path)
    
    detections = []
    
    # Check if results are available
    if results:
        for result in results:
            boxes = result.boxes  # Get the boxes from the result
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
                detections.append((x1, y1, x2, y2))
                
    return detections

# Example usage:
new_image_path = '/home/mundax/Projects/Location_tracking/model/data/images/train/post___91.jpg'
folder_path = '/home/mundax/Projects/Location_tracking/model/data/images/Matching/'
best_match_filename = find_most_similar_image(new_image_path, folder_path)

if best_match_filename:
    primary_key_id = os.path.splitext(best_match_filename)[0]
    detections = run_yolo_model(new_image_path, '/home/mundax/Projects/Location_tracking/model/yolov11m_new_saved.pt')
    display_objects_on_image(new_image_path, primary_key_id, detections)

# Close the connection
conn.close()
