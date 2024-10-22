import sqlite3
import math
import cv2
import numpy as np
from ultralytics import YOLO
import time

# Connect to your SQLite database
db_path = '/home/mundax/SQLite/My_Database.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Load YOLO model
model = YOLO("/home/mundax/Projects/Location_tracking/model/yolov11m_new_saved.pt")

# Function to calculate new GPS coordinates after moving a certain distance
def calculate_new_coordinates(lat, lon, distance, angle):
    R = 6378137  # Earth radius in meters
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)

    new_lat = lat + (distance / R) * (180 / math.pi)
    new_lon = lon + (distance / R) * (180 / math.pi) / math.cos(lat_rad)

    return new_lat, new_lon

# Define initial GPS coordinates for the first picture
initial_latitude = 77.78 
initial_longitude = -89.41  

# Calculate new GPS coordinates after moving 100 meters at an angle of 0 degrees (north)
new_latitude, new_longitude = calculate_new_coordinates(initial_latitude, initial_longitude, 100, 0)

# Function to detect objects in an image using YOLO and calculate angles
def detect_objects_and_angles(image_path, camera_angle):
    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    image_center_x = width / 2

    results = model(img)  # Run inference
    obj_data = []

    for result in results:
        boxes = result.boxes  # Get detected boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].numpy()  # Top-left and bottom-right coordinates
            class_id = int(box.cls[0].item())  # Class ID
            
            # Calculate the center X coordinate of the bounding box
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            # Calculate distance from image center
            distance_from_center = center_x - image_center_x

            # Map pixel distance to angle
            max_angle = 45  # Maximum angle in degrees
            max_distance = width / 2  # Maximum distance from center in pixels
            angle = (distance_from_center / max_distance) * max_angle
            
            # Store object data: (class_id, center_x, center_y, angle)
            obj_data.append((class_id, center_x, center_y, angle + camera_angle))

    return obj_data[:3]  # Return only up to 3 objects

# Detect objects and angles in both images
detections_pic1 = detect_objects_and_angles('/home/mundax/Projects/Location_tracking/model/data/images/Matching/7.jpg', 45)
detections_pic2 = detect_objects_and_angles('/home/mundax/Projects/Location_tracking/model/data/images/Matching/8.jpg', 90)

# Function to insert row into the database for one picture
def insert_row(camera_id, latitude, longitude, detections):
    try:
        obj1_class, obj1_x, obj1_y, angle1 = detections[0] if len(detections) > 0 else (None, None, None, None)
        obj2_class, obj2_x, obj2_y, angle2 = detections[1] if len(detections) > 1 else (None, None, None, None)
        obj3_class, obj3_x, obj3_y, angle3 = detections[2] if len(detections) > 2 else (None, None, None, None)

        cursor.execute('''
            INSERT INTO location (
                camera_id, gps_latitude, gps_longitude,
                angle_1, angle_2, angle_3,
                obj1_x, obj1_y, obj2_x, obj2_y, obj3_x, obj3_y,
                class_obj1, class_obj2, class_obj3,
                obj1_latitude, obj1_longitude,
                obj2_latitude, obj2_longitude,
                obj3_latitude, obj3_longitude
            ) VALUES (?, ?, ?,
                      ?, ?, ?,
                      ?, ?, ?, ?, ?, ?,
                      ?, ?, ?,
                      ?, ?,
                      ?, ?,
                      ?, ?)
        ''', (camera_id, latitude, longitude,
              angle1, angle2, angle3,
              obj1_x, obj1_y, obj2_x, obj2_y, obj3_x, obj3_y,
              obj1_class, obj2_class, obj3_class,
              None, None,
              None, None,
              None, None))
        conn.commit()  # Commit the changes
        return cursor.lastrowid
    except Exception as e:
        print(f"Error inserting row for camera_id {camera_id}: {e}")
        return None

# Insert rows for both pictures and get unique IDs
id1 = insert_row(1, initial_latitude, initial_longitude, detections_pic1)
id2 = insert_row(2, new_latitude, new_longitude, detections_pic2)

def calculate_distance(pt1, pt2):
    return np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)

# Match objects and triangulate their positions
def update_matched_objects(detections1, detections2, latitude, longitude, new_latitude, new_longitude, id1, id2, distance_threshold=100):
    # Store detected objects in camera 1
    matched_objects = {}

    # For each object detected in the first image
    for detection1 in detections1:
        class_id1, obj1_x, obj1_y, angle1 = detection1
        
        # Store the class ID and angle for camera 1
        matched_objects[class_id1] = (angle1, latitude, longitude, obj1_x, obj1_y)

    # Search for matching objects in the second image
    for detection2 in detections2:
        class_id2, obj2_x, obj2_y, angle2 = detection2
        
        if class_id2 in matched_objects:
            angle1, lat1, lon1, obj1_x, obj1_y = matched_objects[class_id2]
            lat2, lon2 = new_latitude, new_longitude
            # Calculate the distance between the two objects
            dist = calculate_distance((obj1_x, obj1_y), (obj2_x, obj2_y))
            
            # Only proceed if the distance is within the threshold
            if dist < distance_threshold:
                # Calculate object coordinates using the intersection of two lines
                angle1_rad = math.radians(angle1)
                angle2_rad = math.radians(angle2)

                # Calculate slopes based on the angles
                slope1 = math.tan(angle1_rad)
                slope2 = math.tan(angle2_rad)

                # Calculate intercepts
                b1 = lat1 - slope1 * lon1
                b2 = lat2 - slope2 * lon2

                # Ensure the lines are not parallel
                if slope1 != slope2:
                    obj_longitude = (b2 - b1) / (slope1 - slope2)
                    obj_latitude = slope1 * obj_longitude + b1
                    
                    # Update the specific object coordinates in the database for camera 1
                    cursor.execute(f'''
                        UPDATE location SET
                            obj1_latitude = ?, obj1_longitude = ?
                        WHERE id = ?
                    ''', (obj_latitude, obj_longitude, id1))

                    # Update the specific object coordinates in the database for camera 2
                    cursor.execute(f'''
                        UPDATE location SET
                            obj1_latitude = ?, obj1_longitude = ?
                        WHERE id = ?
                    ''', (obj_latitude, obj_longitude, id2))
                    
                    print(f"Updated coordinates for object class {class_id2}: ({obj_latitude}, {obj_longitude})")
                else:
                    print(f"Lines are parallel for object class {class_id2}, cannot determine coordinates.")
            else:
                print(f"Objects of class {class_id2} are too far apart to match. Distance: {dist:.2f} meters.")

    # Commit changes for each update to ensure individual updates
    conn.commit()
        
# Update matched objects in the database
update_matched_objects(detections_pic1, detections_pic2, initial_latitude, initial_longitude, new_latitude, new_longitude, id1, id2)

# Close the connection
conn.close()

print("Inserted and updated object coordinates into the location table.")
