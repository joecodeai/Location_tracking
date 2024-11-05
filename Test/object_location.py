import cv2
import numpy as np
import sqlite3
import os

class ImageComparator:
    """
    Compares two images using SIFT feature matching to determine similarity.
    """
    def __init__(self):
        self.sift = cv2.SIFT_create(nfeatures=1000)
        self.bf = cv2.BFMatcher(cv2.NORM_L2)  # Use L2 norm for SIFT

    def compare_images(self, image_path1, image_path2):
        """
        Compares two images and returns True if they are similar enough.
        
        Parameters:
            image_path1 (str): Path to the first image.
            image_path2 (str): Path to the second image.

        Returns:
            bool: True if the images are similar, False otherwise.
        """
        # Read the two images in grayscale
        img1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)

        # Check if images are loaded
        if img1 is None or img2 is None:
            raise FileNotFoundError("One or both image files not found.")

        # Detect SIFT keypoints and descriptors
        kp1, des1 = self.sift.detectAndCompute(img1, None)
        kp2, des2 = self.sift.detectAndCompute(img2, None)

        # If descriptors cannot be computed, return False
        if des1 is None or des2 is None:
            return False

        # Use BFMatcher with correct data type check
        matches = self.bf.knnMatch(des1, des2, k=2)

        # Apply the ratio test to filter out poor matches
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

        # Set a minimum number of good matches to consider images as similar
        similarity_threshold = 20  # Adjust based on experimentation
        are_similar = len(good_matches) > similarity_threshold

        return are_similar


class DatabaseHandler:
    """
    Handles interactions with the SQLite database for checking and updating object coordinates.
    """
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()

    def get_unprocessed_entries(self):
        """
        Fetch rows where object coordinates are not yet processed.

        Returns:
            list: List of tuples (id, camera_id, gps_latitude, gps_longitude).
        """
        self.cursor.execute('''
            SELECT id, camera_id, gps_latitude, gps_longitude 
            FROM location 
            WHERE obj1_latitude IS NULL OR obj2_latitude IS NULL OR obj3_latitude IS NULL
        ''')
        return self.cursor.fetchall()
    
    def get_camera_angle(self, row_id):
        """
        Retrieves the camera angle for a specific image entry.

        Parameters:
            row_id (int): The primary key of the row to retrieve the angle from.

        Returns:
            float: The camera angle in degrees.
        """
        self.cursor.execute('''
            SELECT angle_1 FROM location WHERE id = ?
        ''', (row_id,))
        result = self.cursor.fetchone()
        return result[0] if result else None

    def update_object_coordinates(self, row_id, obj_latitude, obj_longitude, obj_number):
        """
        Updates the object latitude and longitude in the database for a specific object number.
        
        Parameters:
            row_id (int): The primary key of the row to update.
            obj_latitude (float): The updated latitude of the object.
            obj_longitude (float): The updated longitude of the object.
            obj_number (int): The object number (1, 2, or 3).
        """
        self.cursor.execute(f'''
            UPDATE location
            SET obj{obj_number}_latitude = ?, obj{obj_number}_longitude = ?
            WHERE id = ?
        ''', (obj_latitude, obj_longitude, row_id))
        self.conn.commit()

    def close(self):
        """Closes the SQLite database connection."""
        self.conn.close()


def calculate_coordinates(lat1, lon1, lat2, lon2, angle1, angle2):
    """
    Calculates the object's latitude and longitude based on GPS coordinates and angles.

    Parameters:
        lat1, lon1 (float): GPS coordinates of camera 1.
        lat2, lon2 (float): GPS coordinates of camera 2.
        angle1, angle2 (float): Object angles from camera 1 and camera 2.

    Returns:
        tuple: Object's latitude and longitude.
    """
    angle1_rad = np.radians(angle1)
    angle2_rad = np.radians(angle2)

    slope1 = np.tan(angle1_rad)
    slope2 = np.tan(angle2_rad)

    # Calculate intercepts
    b1 = lat1 - slope1 * lon1
    b2 = lat2 - slope2 * lon2

    # Ensure lines are not parallel
    if slope1 != slope2:
        obj_longitude = (b2 - b1) / (slope1 - slope2)
        obj_latitude = slope1 * obj_longitude + b1
        return np.degrees(obj_latitude), np.degrees(obj_longitude)
    else:
        raise ValueError("The angles result in parallel lines. Cannot calculate coordinates.")


def process_images_and_update_db(db_handler, image_comparator):
    """
    Main logic to compare images and update object coordinates in the database.

    Parameters:
        db_handler (DatabaseHandler): The database handler object.
        image_comparator (ImageComparator): The image comparator object.
    """
    entries = db_handler.get_unprocessed_entries()
    processed_ids = set()  # Track processed image IDs
    checked_pairs = []  # List to store checked image pairs

    # Create a set of unprocessed IDs for quick lookup
    unprocessed_ids = {entry[0] for entry in entries}

    # Iterate through each entry
    for i in range(len(entries)):
        id1, camera_id1, lat1, lon1 = entries[i]

        # Skip if this image has already been processed
        if id1 in processed_ids:
            continue

        # Retrieve the angle for the first camera
        angle1 = db_handler.get_camera_angle(id1)
        if angle1 is None:
            print(f"Error: Camera angle not found for ID {id1}")
            continue

        # Compare with every other entry
        for j in range(i + 1, len(entries)):
            id2, camera_id2, lat2, lon2 = entries[j]

            # Skip if the two entries are from the same image (same primary key)
            if id1 == id2 or (id1, id2) in checked_pairs or (id2, id1) in checked_pairs:
                continue

            # Only process if the camera IDs are different and both images are unprocessed
            if camera_id1 != camera_id2 and id2 in unprocessed_ids:
                # Retrieve the angle for the second camera
                angle2 = db_handler.get_camera_angle(id2)
                if angle2 is None:
                    print(f"Error: Camera angle not found for ID {id2}")
                    continue

                # Compare images
                image_path1 = os.path.join(r"D:\Jupyter\Road_Object_Detection\Captured", f"{id1}.jpg")
                image_path2 = os.path.join(r"D:\Jupyter\Road_Object_Detection\Captured", f"{id2}.jpg")

                if image_comparator.compare_images(image_path1, image_path2):
                    print(f"Images {id1} and {id2} are similar. Processing...")

                    try:
                        # Calculate the object coordinates using the retrieved angles
                        obj_latitude, obj_longitude = calculate_coordinates(lat1, lon1, lat2, lon2, angle1, angle2)

                        # Update object coordinates for both entries
                        db_handler.update_object_coordinates(id1, obj_latitude, obj_longitude, 1)
                        db_handler.update_object_coordinates(id2, obj_latitude, obj_longitude, 1)

                        print(f"Updated object coordinates in database for {id1} and {id2}.")
                        
                        # Mark these IDs as processed
                        processed_ids.add(id1)
                        processed_ids.add(id2)

                        # Add the pair to checked pairs list
                        checked_pairs.append((id1, id2))

                    except ValueError as e:
                        print(f"Error calculating coordinates: {e}")
    
    print("Processing completed.")

if __name__ == "__main__":
    db_path = r"D:\Jupyter\Road_Object_Detection\SQLite\My_Database.db"
    
    db_handler = DatabaseHandler(db_path)
    
    image_comparator = ImageComparator()

    process_images_and_update_db(db_handler, image_comparator)

    db_handler.close()
