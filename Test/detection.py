import cv2
import numpy as np
import sqlite3
from ultralytics import YOLO
from gps import get_gps_data, capture_images
#, save_images
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
        
        fixed_width = 1000
        height, width = img.shape[:2]
        aspect_ratio = height / width
        fixed_height = int(fixed_width * aspect_ratio)

        # If the fixed dimensions are smaller than the original, crop around the center
        if fixed_width < width or fixed_height < height:
            # Calculate the center of the original image
            center_x, center_y = width // 2, height // 2
            
            # Calculate the top-left corner of the crop (centered)
            crop_x1 = max(center_x - fixed_width // 2, 0)
            crop_y1 = max(center_y - fixed_height // 2, 0)

            # Calculate the bottom-right corner of the crop (centered)
            crop_x2 = min(center_x + fixed_width // 2, width)
            crop_y2 = min(center_y + fixed_height // 2, height)

            # Crop the image centered around the middle
            cropped_img = img[crop_y1:crop_y2, crop_x1:crop_x2]
        else:
            # If fixed dimensions are larger, we need to pad the image
            pad_height = fixed_height - height
            pad_width = fixed_width - width
            # Apply mirror padding to the image using cv2.copyMakeBorder
            cropped_img = cv2.copyMakeBorder(
                img, 
                pad_height // 2, pad_height // 2,  # Padding on top and bottom (split evenly)
                pad_width // 2, pad_width // 2,    # Padding on left and right (split evenly)
                cv2.BORDER_REPLICATE  # Mirror padding
            )

        image_center_x = fixed_width / 2

        # Run object detection inference
        results = self.model(cropped_img)
        obj_data = []

        # Process each detection result
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].numpy()  # Bounding box coordinates
                class_id = int(box.cls[0].item())  # Object class ID
                
                scale_x = fixed_width / width
                scale_y = fixed_height / height
                x1_resized = x1 * scale_x
                y1_resized = y1 * scale_y
                x2_resized = x2 * scale_x
                y2_resized = y2 * scale_y

                # Calculate bounding box center
                center_x = (x1_resized + x2_resized) / 2
                center_y = (y1_resized + y2_resized) / 2

                # Calculate angle from the center of the image
                distance_from_center = center_x - image_center_x
                cropped_ratio = min(fixed_width / width, 1)
                max_angle = (55/2) * cropped_ratio  # Max horizontal field of view angle
                max_distance = fixed_width / 2  # Max horizontal pixel distance from center
                angle = (distance_from_center / max_distance) * max_angle

                # Adjust the angle with the camera angle and store object data
                obj_data.append((class_id, center_x, center_y, angle + camera_angle))

        return obj_data[:3]  # Return up to 3 detected objects

class DatabaseHandler:
    """
    Handles interactions with the SQLite database for storing object detection data.
    """
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path, timeout=2.0)
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
            latitude = float(latitude)
            longitude = float(longitude)

            # Ensure obj_class remains int and other fields are cast to float
            obj1_class, obj1_x, obj1_y, angle1 = self.format_detection(detections, 0)
            obj2_class, obj2_x, obj2_y, angle2 = self.format_detection(detections, 1)
            obj3_class, obj3_x, obj3_y, angle3 = self.format_detection(detections, 2)

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

    def format_detection(self, detections, index):
        """
        Formats the detection tuple with appropriate types:
        obj_class as int and other values as float.
        """
        try:
            detection = detections[index]
            obj_class = int(detection[0]) if detection[0] is not None else None
            obj_x = float(detection[1]) if detection[1] is not None else None
            obj_y = float(detection[2]) if detection[2] is not None else None
            angle = float(detection[3]) if detection[3] is not None else None
            return obj_class, obj_x, obj_y, angle
        except IndexError:
            # Return None values if detection at the given index does not exist
            return None, None, None, None

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
        model_path='/home/mundax/Projects/Location_tracking/model/yolov11m_new_saved.pt',
        db_path='/home/mundax/SQLite/My_Database.db'
    )

    # Get GPS data
    latitude, longitude = get_gps_data()
    if latitude is None or longitude is None:
        print("Failed to retrieve GPS data.")
    else:
        # Capture images from both cameras
        img1_path, img2_path = capture_images(camera_index1=1, camera_index2=2)

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

    # Close the database connection
    app.close()