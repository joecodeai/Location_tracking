import sqlite3
import cv2
import os
from ultralytics import YOLO
from math import radians, cos

# Connect to your SQLite database
db_path = '/home/mundax/SQLite/My_Database.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Retrieve object coordinates from the database for a specific id
def get_object_coordinates(primary_key_id):
    cursor.execute('''
        SELECT obj1_latitude, obj1_longitude 
        FROM location 
        WHERE id = ?
    ''', (primary_key_id,))
    return cursor.fetchone()

# Find the most similar image using ORB feature matching
def find_most_similar_image(new_image_path, folder_path):
    new_img = cv2.imread(new_image_path)
    if new_img is None:
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

    return best_match_file

# Calculate bearing using camera orientation and object's position
def calculate_bearing(camera_orientation, image_width, object_position):
    camera_fov = 90
    center_x = image_width / 2
    distance_from_center = object_position - center_x
    max_angle = camera_fov / 2
    max_distance = image_width / 2
    angle_offset = (distance_from_center / max_distance) * max_angle
    return camera_orientation + angle_offset

# Run YOLO model and get detections
def run_yolo_model(image_path, model_path):
    model = YOLO(model_path)
    results = model(image_path)
    
    detections = []
    if results:
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append((x1, y1, x2, y2))
    return detections

# Find intersection of two bearings
def find_intersection(lat1, lon1, bearing1, lat2, lon2, bearing2):
    lat1_rad = radians(lat1)
    lon1_rad = radians(lon1)
    lat2_rad = radians(lat2)
    lon2_rad = radians(lon2)

    R = 6371  # Radius of the Earth in kilometers
    distance = 100  # Distance in kilometers

    end_lat1 = lat1_rad + (distance / R) * cos(radians(bearing1))
    end_lon1 = lon1_rad + (distance / R) * cos(radians(bearing1)) / cos(lat1_rad)

    end_lat2 = lat2_rad + (distance / R) * cos(radians(bearing2))
    end_lon2 = lon2_rad + (distance / R) * cos(radians(bearing2)) / cos(lat2_rad)

    intersection_lat = (lat1 + lat2) / 2
    intersection_lon = (lon1 + lon2) / 2

    return intersection_lat, intersection_lon

# Main function to handle two images and estimate location
def main(image_paths, folder_path, model_path):
    image_data = []
    camera_orientations = [45, 90]  # Camera orientations for the two images

    for idx, image_path in enumerate(image_paths):
        best_match_filename = find_most_similar_image(image_path, folder_path)
        
        if not best_match_filename:
            print(f"Skipping {image_path}: No matching image found.")
            continue
        
        primary_key_id = os.path.splitext(best_match_filename)[0]
        detections = run_yolo_model(image_path, model_path)
        
        if not detections:
            print(f"Skipping {image_path}: No objects detected.")
            continue
        
        coordinates = get_object_coordinates(primary_key_id)
        
        if not coordinates:
            print(f"No coordinates found for primary key id {primary_key_id}.")
            continue
        
        image_data.append((coordinates, detections))

    if len(image_data) != 2:
        print("Error: Need data from 2 images for calculations.")
        return
    
    lat1, lon1 = image_data[0][0]
    lat2, lon2 = image_data[1][0]

    image_width = 640  
    object_position1 = image_data[0][1][0][0]
    object_position2 = image_data[1][1][0][0]

    bearing1 = calculate_bearing(camera_orientations[0], image_width, object_position1)
    bearing2 = calculate_bearing(camera_orientations[1], image_width, object_position2)

    intersection_lat, intersection_lon = find_intersection(lat1, lon1, bearing1, lat2, lon2, bearing2)

    print(f"Calculated Intersection Location: Latitude = {intersection_lat}, Longitude = {intersection_lon}")

# Example usage
image_paths = [
    '/home/mundax/Projects/Location_tracking/model/data/images/Matching/1.jpg',
    '/home/mundax/Projects/Location_tracking/model/data/images/Matching/4.jpg'
]

folder_path = '/home/mundax/Projects/Location_tracking/model/data/images/Matching/'
model_path = '/home/mundax/Projects/Location_tracking/model/yolov11m_new_saved.pt'

# Run the main function
main(image_paths, folder_path, model_path)

# Close the database connection
conn.close()
