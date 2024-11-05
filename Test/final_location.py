import sqlite3
import cv2
import os
import numpy as np
from ultralytics import YOLO
from math import radians, degrees

class ImageMatcher:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.orb = cv2.ORB_create(nfeatures=3000)

    def find_most_similar_image(self, image_path):
        """Find the most similar image from the folder using ORB feature matching."""
        new_img = cv2.imread(image_path)
        if new_img is None:
            return None
        
        keypoints1, descriptors1 = self.orb.detectAndCompute(new_img, None)
        best_match_file = None
        max_matches = 1000

        for filename in os.listdir(self.folder_path):
            if filename.endswith(('.jpg', '.png', '.jpeg')):
                img_path = os.path.join(self.folder_path, filename)
                img = cv2.imread(img_path)
                if img is None:
                    continue

                keypoints2, descriptors2 = self.orb.detectAndCompute(img, None)
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = bf.match(descriptors1, descriptors2)
                num_matches = len(matches)

                if num_matches > max_matches:
                    max_matches = num_matches
                    best_match_file = filename

        return best_match_file

class DatabaseHandler:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()

    def get_object_coordinates(self, primary_key_id):
        """Retrieve object coordinates from the database for a specific id."""
        self.cursor.execute('''
            SELECT obj1_latitude, obj1_longitude 
            FROM location 
            WHERE id = ?
        ''', (primary_key_id,))
        return self.cursor.fetchone()

    def close(self):
        """Close the database connection."""
        self.conn.close()

class ObjectDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def run_yolo_model(self, image_path):
        """Run YOLO object detection on the given image."""
        results = self.model(image_path)
        detections = []
        if results:
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections.append((x1, y1, x2, y2))
        return detections

class LocationCalculator:
    @staticmethod
    def calculate_bearing(camera_orientation, image_width, object_position):
        """Calculate the bearing using the camera orientation and object position in the image."""
        camera_fov = 55  # Camera field of view (degrees)
        center_x = image_width / 2 
        distance_from_center = object_position - center_x  # Object's horizontal offset from image center
        max_angle = camera_fov / 2  # Maximum angle the camera can capture
        max_distance = image_width / 2  # Maximum pixel distance from center to edge

        # Calculate the angle offset of the object relative to the camera's center
        angle_offset = (distance_from_center / max_distance) * max_angle
        # The total bearing is the camera's orientation plus the offset
        total_bearing = camera_orientation + angle_offset

        return total_bearing

    @staticmethod
    def find_intersection(lat1, lon1, bearing1, lat2, lon2, bearing2):
        """
        Triangulate to find the intersection of two bearings (from two known points) on a plane.
        
        Parameters:
            lat1, lon1 (float): Latitude and longitude of the first camera.
            bearing1 (float): Bearing (angle) from the first camera.
            lat2, lon2 (float): Latitude and longitude of the second camera.
            bearing2 (float): Bearing (angle) from the second camera.

        Returns:
            tuple: Estimated location (latitude, longitude) of the camera.
        """
        # Convert degrees to radians
        lat1_rad = radians(lat1)
        lon1_rad = radians(lon1)
        lat2_rad = radians(lat2)
        lon2_rad = radians(lon2)
        bearing1_rad = radians(bearing1)
        bearing2_rad = radians(bearing2)

        # Use the slope of the bearings to set up a system of equations
        slope1 = np.tan(bearing1_rad)
        slope2 = np.tan(bearing2_rad)

        # The equations of the lines in slope-intercept form
        intercept1 = lat1_rad - slope1 * lon1_rad

        # For object 2
        intercept2 = lat2_rad - slope2 * lon2_rad

        # Set the two equations equal to find x (longitude)
        x_long = (intercept2 - intercept1) / (slope1 - slope2)

        # Now substitute x back into one of the original equations to find y (latitude)
        y_lat = slope1 * x_long + intercept1

        # Convert back to degrees
        lat_deg = degrees(y_lat)
        lon_deg = degrees(x_long)

        return lat_deg, lon_deg

class LocationEstimator:
    def __init__(self, db_path, model_path, folder_path):
        self.db_handler = DatabaseHandler(db_path)
        self.detector = ObjectDetector(model_path)
        self.matcher = ImageMatcher(folder_path)
        self.calculator = LocationCalculator()

    def estimate_location(self, image_paths, camera_orientations):
        """Estimate the exact location using two images and camera orientations."""
        image_data = []

        for idx, image_path in enumerate(image_paths):
            best_match_filename = self.matcher.find_most_similar_image(image_path)
            
            if not best_match_filename:
                print(f"Skipping {image_path}: No matching image found.")
                continue
            
            primary_key_id = os.path.splitext(best_match_filename)[0]
            detections = self.detector.run_yolo_model(image_path)
            
            if not detections:
                print(f"Skipping {image_path}: No objects detected.")
                continue
            
            coordinates = self.db_handler.get_object_coordinates(primary_key_id)
            
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

        bearing1 = self.calculator.calculate_bearing(camera_orientations[0], image_width, object_position1)
        bearing2 = self.calculator.calculate_bearing(camera_orientations[1], image_width, object_position2)

        intersection_lat, intersection_lon = self.calculator.find_intersection(lat1, lon1, bearing1, lat2, lon2, bearing2)

        print(f"Calculated Intersection Location: Latitude = {intersection_lat}, Longitude = {intersection_lon}")

    def close(self):
        self.db_handler.close()

if __name__ == "__main__":
    db_path = r"D:\Jupyter\Road_Object_Detection\SQLite\My_Database.db"
    model_path = r"D:\Jupyter\Road_Object_Detection\yolov11m_new_saved.pt"
    folder_path = r"D:\Jupyter\Road_Object_Detection\Captured"
    
    image_paths = [
        r"D:\Jupyter\Road_Object_Detection\NEW_Captured\1.jpg",
        r"D:\Jupyter\Road_Object_Detection\NEW_Captured\2.jpg"
    ]
    
    camera_orientations = [45, 90]  # Example camera orientations for the two images
    
    estimator = LocationEstimator(db_path, model_path, folder_path)
    estimator.estimate_location(image_paths, camera_orientations)
    estimator.close()