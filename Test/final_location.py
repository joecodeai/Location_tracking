import sqlite3
import cv2
import os
import numpy as np
from ultralytics import YOLO
from math import radians, degrees
import logging

# Configure logging for better traceability and debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ImageMatcher:
    """Finds the most similar image in a folder using ORB feature matching."""

    def __init__(self, folder_path):
        """
        Initializes the ImageMatcher.

        Parameters:
            folder_path (str): Path to the folder containing images for comparison.
        """
        self.folder_path = folder_path
        self.orb = cv2.ORB_create(nfeatures=3000)

    def find_most_similar_image(self, image_path):
        """
        Finds the most similar image from the folder using ORB feature matching.

        Parameters:
            image_path (str): Path to the image to compare.

        Returns:
            str: Filename of the most similar image, or None if no match is found.
        """
        new_img = cv2.imread(image_path)
        if new_img is None:
            logging.error(f"Image not found: {image_path}")
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
    """Handles SQLite database interactions."""

    def __init__(self, db_path):
        """
        Initializes the DatabaseHandler by connecting to the SQLite database.

        Parameters:
            db_path (str): Path to the SQLite database file.
        """
        try:
            self.conn = sqlite3.connect(db_path)
            self.cursor = self.conn.cursor()
            logging.info(f"Connected to database: {db_path}")
        except sqlite3.Error as e:
            logging.error(f"Database connection error: {e}")
            raise

    def get_object_coordinates(self, primary_key_id):
        """
        Retrieves object coordinates (latitude and longitude) from the database for a given ID.

        Parameters:
            primary_key_id (str): The primary key (image ID) to query.

        Returns:
            tuple: Latitude and longitude of the object, or None if not found.
        """
        self.cursor.execute('''
            SELECT obj1_latitude, obj1_longitude 
            FROM location 
            WHERE id = ?
        ''', (primary_key_id,))
        return self.cursor.fetchone()

    def close(self):
        """Closes the database connection."""
        try:
            self.conn.close()
            logging.info("Database connection closed.")
        except sqlite3.Error as e:
            logging.error(f"Error closing database: {e}")


class ObjectDetector:
    """Runs YOLO object detection on images."""

    def __init__(self, model_path):
        """
        Initializes the ObjectDetector with a YOLO model.

        Parameters:
            model_path (str): Path to the YOLO model.
        """
        self.model = YOLO(model_path)

    def run_yolo_model(self, image_path):
        """
        Runs the YOLO model on the image to detect objects.

        Parameters:
            image_path (str): Path to the image to process.

        Returns:
            list: List of detected bounding boxes (x1, y1, x2, y2) or an empty list if no objects detected.
        """
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
    """Calculates object location based on camera orientations and detected positions."""

    CAMERA_FOV = 55  # Camera field of view (degrees)

    @staticmethod
    def calculate_bearing(camera_orientation, image_width, object_position):
        """
        Calculates the bearing (angle) based on the object position in the image.

        Parameters:
            camera_orientation (float): Camera orientation angle (in degrees).
            image_width (int): Width of the image in pixels.
            object_position (int): Horizontal position of the detected object in the image.

        Returns:
            float: Calculated bearing angle (in degrees).
        """
        center_x = image_width / 2 
        distance_from_center = object_position - center_x  # Horizontal offset
        max_angle = LocationCalculator.CAMERA_FOV / 2  # Maximum angle captured by the camera
        max_distance = image_width / 2  # Max pixel distance from center to edge

        # Calculate the angle offset from the center of the image
        angle_offset = (distance_from_center / max_distance) * max_angle
        total_bearing = camera_orientation + angle_offset

        return total_bearing

    @staticmethod
    def find_intersection(lat1, lon1, bearing1, lat2, lon2, bearing2):
        """
        Triangulates the intersection of two bearings to estimate an object's location.

        Parameters:
            lat1, lon1 (float): Coordinates of the first camera.
            bearing1 (float): Bearing from the first camera.
            lat2, lon2 (float): Coordinates of the second camera.
            bearing2 (float): Bearing from the second camera.

        Returns:
            tuple: Estimated object location (latitude, longitude).
        """
        try:
            # Convert degrees to radians
            lat1_rad = radians(lat1)
            lon1_rad = radians(lon1)
            lat2_rad = radians(lat2)
            lon2_rad = radians(lon2)
            bearing1_rad = radians(bearing1)
            bearing2_rad = radians(bearing2)

            # Calculate slopes of the bearings
            slope1 = np.tan(bearing1_rad)
            slope2 = np.tan(bearing2_rad)

            intercept1 = lat1_rad - slope1 * lon1_rad
            intercept2 = lat2_rad - slope2 * lon2_rad

            # Solve the system of equations to find the intersection
            x_long = (intercept2 - intercept1) / (slope1 - slope2)
            y_lat = slope1 * x_long + intercept1

            # Convert the result back to degrees
            lat_deg = degrees(y_lat)
            lon_deg = degrees(x_long)

            return lat_deg, lon_deg
        except Exception as e:
            logging.error(f"Error calculating intersection: {e}")
            raise


class LocationEstimator:
    """Estimates the location of an object based on two images and their camera orientations."""

    def __init__(self, db_path, model_path, folder_path):
        """
        Initializes the LocationEstimator.

        Parameters:
            db_path (str): Path to the SQLite database.
            model_path (str): Path to the YOLO model.
            folder_path (str): Path to the folder containing images for matching.
        """
        self.db_handler = DatabaseHandler(db_path)
        self.detector = ObjectDetector(model_path)
        self.matcher = ImageMatcher(folder_path)
        self.calculator = LocationCalculator()

    def estimate_location(self, image_paths, camera_orientations):
        """
        Estimates the object's location based on two images and camera orientations.

        Parameters:
            image_paths (list): List of two image paths to compare.
            camera_orientations (list): List of two camera orientations (in degrees).
        """
        image_data = []

        for idx, image_path in enumerate(image_paths):
            best_match_filename = self.matcher.find_most_similar_image(image_path)
            
            if not best_match_filename:
                logging.warning(f"Skipping {image_path}: No matching image found.")
                continue
            
            primary_key_id = os.path.splitext(best_match_filename)[0]
            detections = self.detector.run_yolo_model(image_path)
            
            if not detections:
                logging.warning(f"Skipping {image_path}: No objects detected.")
                continue
            
            coordinates = self.db_handler.get_object_coordinates(primary_key_id)
            
            if not coordinates:
                logging.warning(f"No coordinates found for primary key id {primary_key_id}.")
                continue
            
            image_data.append((coordinates, detections))

        if len(image_data) != 2:
            logging.error("Error: Need data from exactly 2 images for calculations.")
            return
        
        lat1, lon1 = image_data[0][0]
        lat2, lon2 = image_data[1][0]

        image_width = 640  # Example image width
        object_position1 = image_data[0][1][0][0]  # Horizontal position of object in first image
        object_position2 = image_data[1][1][0][0]  # Horizontal position of object in second image

        bearing1 = self.calculator.calculate_bearing(camera_orientations[0], image_width, object_position1)
        bearing2 = self.calculator.calculate_bearing(camera_orientations[1], image_width, object_position2)

        intersection_lat, intersection_lon = self.calculator.find_intersection(lat1, lon1, bearing1, lat2, lon2, bearing2)

        logging.info(f"Calculated Intersection Location: Latitude = {intersection_lat}, Longitude = {intersection_lon}")

    def close(self):
        """Closes the database connection."""
        self.db_handler.close()


if __name__ == "__main__":
    directory = os.getcwd()
    db_path = os.path.join(directory,"My_Database.db")
    model_path = os.path.join(directory,"model/yolov11m_new_saved.pt")
    folder_path = os.path.join(directory,"Captured")

    image_paths = [
        os.path.join(directory,"New_Captured/1.jpg"),
        os.path.join(directory,"New_Captured/2.jpg")
    ]
    
    camera_orientations = [45, 90]  # Example camera orientations for the two images
    
    estimator = LocationEstimator(db_path, model_path, folder_path)
    estimator.estimate_location(image_paths, camera_orientations)
    estimator.close()
