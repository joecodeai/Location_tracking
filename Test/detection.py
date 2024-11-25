import cv2
import sqlite3
import logging
import numpy as np
import os
from ultralytics import YOLO
from gps import get_gps_data, capture_images


# Configure logging for better traceability and debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ObjectDetector:
    """
    Object detection class using the YOLO model.
    It detects objects, calculates angles, and processes the image.
    """

    def __init__(self, model_path):
        """
        Initializes the object detector with the given YOLO model.

        Parameters:
            model_path (str): Path to the YOLO model weights file.
        """
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
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
        
        fixed_width = 1000
        height, width = img.shape[:2]
        aspect_ratio = height / width
        fixed_height = int(fixed_width * aspect_ratio)

        # Crop or pad the image to fixed dimensions
        cropped_img = self._process_image_dimensions(img, fixed_width, fixed_height)

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

                # Rescale bounding box coordinates to the fixed image size
                scale_x = fixed_width / width
                scale_y = fixed_height / height
                x1_resized, y1_resized, x2_resized, y2_resized = self._rescale_coordinates(
                    x1, y1, x2, y2, scale_x, scale_y)

                # Calculate bounding box center
                center_x = (x1_resized + x2_resized) / 2
                center_y = (y1_resized + y2_resized) / 2

                # Calculate angle from the center of the image
                distance_from_center = center_x - image_center_x
                cropped_ratio = min(fixed_width / width, 1)
                max_angle = (55 / 2) * cropped_ratio  # Max horizontal field of view angle
                max_distance = fixed_width / 2  # Max horizontal pixel distance from center
                angle = (distance_from_center / max_distance) * max_angle

                # Adjust the angle with the camera angle and store object data
                obj_data.append((class_id, center_x, center_y, angle + camera_angle))

        return obj_data[:3]  # Return up to 3 detected objects

    def _process_image_dimensions(self, img, fixed_width, fixed_height):
        """
        Processes the image to fit fixed dimensions by either cropping or padding.

        Parameters:
            img (ndarray): Input image.
            fixed_width (int): Target width.
            fixed_height (int): Target height.

        Returns:
            ndarray: Processed image with fixed dimensions.
        """
        height, width = img.shape[:2]
        
        if fixed_width < width or fixed_height < height:
            return self._crop_image(img, width, height, fixed_width, fixed_height)
        else:
            return self._pad_image(img, width, height, fixed_width, fixed_height)

    def _crop_image(self, img, width, height, fixed_width, fixed_height):
        """
        Crops the image to fit fixed dimensions while maintaining the center.

        Parameters:
            img (ndarray): Input image.
            width (int): Original width.
            height (int): Original height.
            fixed_width (int): Target width.
            fixed_height (int): Target height.

        Returns:
            ndarray: Cropped image.
        """
        center_x, center_y = width // 2, height // 2
        crop_x1 = max(center_x - fixed_width // 2, 0)
        crop_y1 = max(center_y - fixed_height // 2, 0)
        crop_x2 = min(center_x + fixed_width // 2, width)
        crop_y2 = min(center_y + fixed_height // 2, height)

        return img[crop_y1:crop_y2, crop_x1:crop_x2]

    def _pad_image(self, img, width, height, fixed_width, fixed_height):
        """
        Pads the image to fit fixed dimensions using mirror padding.

        Parameters:
            img (ndarray): Input image.
            width (int): Original width.
            height (int): Original height.
            fixed_width (int): Target width.
            fixed_height (int): Target height.

        Returns:
            ndarray: Padded image.
        """
        pad_height = fixed_height - height
        pad_width = fixed_width - width
        return cv2.copyMakeBorder(
            img,
            pad_height // 2, pad_height // 2,
            pad_width // 2, pad_width // 2,
            cv2.BORDER_REPLICATE
        )

    def _rescale_coordinates(self, x1, y1, x2, y2, scale_x, scale_y):
        """
        Rescales bounding box coordinates to match the fixed image dimensions.

        Parameters:
            x1, y1, x2, y2 (float): Original bounding box coordinates.
            scale_x, scale_y (float): Scaling factors for x and y coordinates.

        Returns:
            tuple: Rescaled coordinates (x1, y1, x2, y2).
        """
        return x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y


class DatabaseHandler:
    """
    Handles interactions with the SQLite database for storing object detection data.
    """
    
    def __init__(self, db_path):
        """
        Initializes the database handler.

        Parameters:
            db_path (str): Path to the SQLite database.
        """
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
            latitude, longitude = float(latitude), float(longitude)

            # Format detections for insertion
            formatted_detections = [self.format_detection(detections, i) for i in range(3)]

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
                  *formatted_detections[0], *formatted_detections[1], *formatted_detections[2]))

            # Commit changes and return the last inserted row ID
            self.conn.commit()
            return self.cursor.lastrowid
        except sqlite3.Error as e:
            logging.error(f"Database error: {e}")
            return None

    def format_detection(self, detections, index):
        """
        Formats the detection tuple with appropriate types:
        obj_class as int and other values as float.

        Parameters:
            detections (list): List of detected objects.
            index (int): Index of the detection to format.

        Returns:
            tuple: Formatted detection (class_id, x, y, angle).
        """
        try:
            detection = detections[index]
            return (
                int(detection[0]) if detection[0] is not None else None,
                float(detection[1]) if detection[1] is not None else None,
                float(detection[2]) if detection[2] is not None else None,
                float(detection[3]) if detection[3] is not None else None
            )
        except IndexError:
            return None, None, None, None

    def close(self):
        """Closes the SQLite database connection."""
        self.conn.close()


class MainApp:
    """
    The main application class that integrates object detection and database interaction.
    """
    
    def __init__(self, model_path, db_path):
        """
        Initializes the main application with the object detector and database handler.

        Parameters:
            model_path (str): Path to the YOLO model weights file.
            db_path (str): Path to the SQLite database.
        """
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
                logging.info(f"Successfully inserted row with ID: {row_id}")
            else:
                logging.warning("Failed to insert row into the database.")
        except FileNotFoundError as fnf_error:
            logging.error(f"Error: {fnf_error}")
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")

    def close(self):
        """Closes the database connection."""
        self.db_handler.close()


if __name__ == "__main__":
    directory = os.getcwd()
    # Initialize the MainApp with the YOLO model path and database path
    app = MainApp(
        model_path = os.path.join(directory,"model/yolov11m_new_saved.pt"),
        db_path = os.path.join(directory,"My_Database.db")
    )

    # Get GPS data
    latitude, longitude = get_gps_data()
    if latitude is None or longitude is None:
        logging.error("Failed to retrieve GPS data.")
    else:
        # Capture images from both cameras
        img1_path, img2_path = capture_images(camera_index1=1, camera_index2=2)

        # Process images for both cameras
        camera_data = [
            {"image_path": img1_path, "camera_id": 1, "latitude": latitude, "longitude": longitude, "camera_angle": 45},
            {"image_path": img2_path, "camera_id": 2, "latitude": latitude, "longitude": longitude, "camera_angle": 90}
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
