import cv2
import numpy as np
import sqlite3
from ultralytics import YOLO
from gps import get_gps_data, capture_images, save_images

class ObjectDetector:
    """
    Object detection class using the YOLO model.
    It detects objects, calculates angles, and processes the image.
    """
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_objects(self, image_path, camera_angle):
        """
        Detects objects in an image and calculates angles relative to the camera's angle.

        Parameters:
            image_path (str): Path to the image file.
            camera_angle (float): Camera's orientation angle in degrees.

        Returns:
            list: A list of detected objects with their angles and coordinates (up to 3 objects).
        """
        # Load the image
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
        
        height, width = img.shape[:2]
        image_center_x = width / 2

        # Run object detection inference
        results = self.model(img)
        obj_data = []

        # Process each detection result
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].numpy()  # Bounding box coordinates
                class_id = int(box.cls[0].item())  # Object class ID

                # Calculate bounding box center
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                # Calculate angle from the center of the image
                distance_from_center = center_x - image_center_x
                max_angle = 45  # Max horizontal field of view angle
                max_distance = width / 2  # Max horizontal pixel distance from center
                angle = (distance_from_center / max_distance) * max_angle

                # Adjust the angle with the camera angle and store object data
                obj_data.append((class_id, center_x, center_y, angle + camera_angle))

        return obj_data[:3]  # Return up to 3 detected objects

class DatabaseHandler:
    """
    Handles interactions with the SQLite database for storing object detection data.
    """
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()

    def insert_row(self, camera_id, latitude, longitude, detections):
        """
        Inserts detected objects into the SQLite database.

        Parameters:
            camera_id (int): Unique identifier of the camera.
            latitude (float): GPS latitude.
            longitude (float): GPS longitude.
            detections (list): List of detected objects with angles and coordinates.

        Returns:
            int: The row ID of the inserted entry, or None if insertion fails.
        """
        try:
            # Unpack up to 3 objects from the detection results
            obj1_class, obj1_x, obj1_y, angle1 = detections[0] if len(detections) > 0 else (None, None, None, None)
            obj2_class, obj2_x, obj2_y, angle2 = detections[1] if len(detections) > 1 else (None, None, None, None)
            obj3_class, obj3_x, obj3_y, angle3 = detections[2] if len(detections) > 2 else (None, None, None, None)

            # Insert the data into the database
            self.cursor.execute('''
                INSERT INTO location (
                    camera_id, gps_latitude, gps_longitude,
                    angle_1, angle_2, angle_3,
                    obj1_x, obj1_y, obj2_x, obj2_y, obj3_x, obj3_y,
                    class_obj1, class_obj2, class_obj3
                ) VALUES (?, ?, ?,
                          ?, ?, ?,
                          ?, ?, ?, ?, ?, ?,
                          ?, ?, ?)
            ''', (camera_id, latitude, longitude,
                  angle1, angle2, angle3,
                  obj1_x, obj1_y, obj2_x, obj2_y, obj3_x, obj3_y,
                  obj1_class, obj2_class, obj3_class))
            
            # Commit changes and return the last inserted row ID
            self.conn.commit()
            return self.cursor.lastrowid
        except sqlite3.Error as e:
            print(f"Database error: {e}")
            return None

    def close(self):
        """Closes the SQLite database connection."""
        self.conn.close()

class MainApp:
    """
    The main application class that integrates object detection and database interaction.
    """
    def __init__(self, model_path, db_path):
        self.detector = ObjectDetector(model_path)
        self.db_handler = DatabaseHandler(db_path)

    def process_image(self, image_path, camera_id, latitude, longitude, camera_angle):
        """
        Processes a single image for object detection and stores the result in the database.

        Parameters:
            image_path (str): Path to the image file.
            camera_id (int): Camera ID.
            latitude (float): GPS latitude.
            longitude (float): GPS longitude.
            camera_angle (float): Camera's orientation angle in degrees.
        """
        try:
            # Perform object detection
            detections = self.detector.detect_objects(image_path, camera_angle)

            # Insert the detection results into the database
            row_id = self.db_handler.insert_row(camera_id, latitude, longitude, detections)
            if row_id:
                print(f"Successfully inserted row with ID: {row_id}")
            else:
                print("Failed to insert row into the database.")
        except FileNotFoundError as fnf_error:
            print(f"Error: {fnf_error}")
        except Exception as e:
            print(f"An error occurred: {e}")

    def close(self):
        """Closes the database connection."""
        self.db_handler.close()

if __name__ == "__main__":
    # Initialize the MainApp with the YOLO model path and database path
    app = MainApp(
        model_path="/home/mundax/Projects/Location_tracking/model/yolov11m_new_saved.pt",
        db_path="/home/mundax/SQLite/My_Database.db"
    )

    # Get GPS data
    latitude, longitude = get_gps_data()
    if latitude is None or longitude is None:
        print("Failed to retrieve GPS data.")
    else:
        # Capture images from both cameras
        frame1, frame2 = capture_images(camera1_id=0, camera2_id=1)

        if frame1 is not None and frame2 is not None:
            # Save captured images and get their paths
            img1_path, img2_path = save_images(frame1, frame2)

            # Process images for both cameras
            camera_data = [
                {
                    "image_path": img1_path,
                    "camera_id": 1,
                    "latitude": latitude,
                    "longitude": longitude,
                    "camera_angle": 45
                },
                {
                    "image_path": img2_path,
                    "camera_id": 2,
                    "latitude": latitude,
                    "longitude": longitude,
                    "camera_angle": 90
                }
            ]

            for data in camera_data:
                app.process_image(
                    image_path=data["image_path"],
                    camera_id=data["camera_id"],
                    latitude=data["latitude"],
                    longitude=data["longitude"],
                    camera_angle=data["camera_angle"]
                )
        else:
            print("Could not capture images from cameras.")

    # Close the database connection
    app.close()