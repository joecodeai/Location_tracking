import os
import time
import cv2
import serial
import sqlite3
import threading
import logging
import pynmea2

# Configure logging for better traceability and debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# GPS Configuration
SERIAL_PORT = '/dev/ttyUSB0'
BAUD_RATE = 9600  # Typical baud rate for GPS modules
GPS_TIMEOUT = 1  # Timeout for GPS data in seconds

# Database Configuration
directory = os.getcwd()
DB_PATH = os.path.join(directory,"My_Database.db")
IMAGE_SAVE_DIR = os.path.join(directory,"Captured")

# Create the image save directory if it doesn't exist
os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)

def get_gps_data():
    """
    Reads GPS data from the serial port and returns latitude and longitude.

    Returns:
        tuple: (latitude, longitude) if successful, else (None, None)
    """
    try:
        with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=GPS_TIMEOUT) as ser:
            while True:
                line = ser.readline().decode('ascii', errors='replace').strip()
                if line.startswith('$GNGGA') or line.startswith('$GNRMC'):
                    try:
                        msg = pynmea2.parse(line)
                        if isinstance(msg, (pynmea2.GGA, pynmea2.RMC)):
                            return msg.latitude, msg.longitude
                    except pynmea2.ParseError:
                        logging.warning("Failed to parse GPS message: %s", line)
                        continue
    except serial.SerialException as e:
        logging.error("GPS Serial error: %s", e)
    return None, None


def get_latest_row_id(db_path):
    """
    Fetches the latest row ID from the 'location' table in the SQLite database.

    Parameters:
        db_path (str): Path to the SQLite database file.

    Returns:
        int: Latest row ID from the 'location' table, or 0 if not found.
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM location ORDER BY id DESC LIMIT 1")
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else 0
    except sqlite3.Error as e:
        logging.error("Database error: %s", e)
        return 0


def capture_frames(cap, frames):
    """
    Captures frames from a camera and appends them to the provided frames list.

    Parameters:
        cap (cv2.VideoCapture): Opened video capture object.
        frames (list): List to store captured frames.
    """
    while True:
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            logging.warning("Failed to capture frame from camera.")
            break


def capture_images(camera_indices=(1, 2)):
    """
    Captures images from multiple cameras and saves them to disk.

    Parameters:
        camera_indices (tuple): Indices of cameras to capture from (default is (1, 2)).

    Returns:
        tuple: File paths of the saved images from the cameras, or None for failed captures.
    """
    caps = []
    threads = []
    frames = [[] for _ in camera_indices]  # List of lists to store frames for each camera

    # Fetch the latest row ID from the database
    latest_row_id = get_latest_row_id(DB_PATH) + 1  # Increment by 1 for the new images

    # Open the specified cameras
    for index in camera_indices:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            logging.info("Camera found at index %d", index)
            caps.append(cap)
            # Start a new thread to capture frames from the camera
            thread = threading.Thread(target=capture_frames, args=(cap, frames[len(caps) - 1]))
            thread.start()
            threads.append(thread)
        else:
            logging.error("No camera found at index %d", index)

    # Check if at least one camera is available
    if not caps:
        logging.error("No cameras available for capture.")
        return None, None

    # Display live video feed from the cameras for 2 seconds (can be adjusted)
    display_duration = 2  # Duration in seconds
    start_time = time.time()
    while time.time() - start_time < display_duration:
        if frames:
            # Only show frames from cameras that have captured data
            combined_frames = [f for f in frames if len(f) > 0]
            if combined_frames:
                for i, frame in enumerate(combined_frames):
                    if len(frame) > 0:
                        cv2.imshow(f"Camera Feed {camera_indices[i]}", frame[-1])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Capture and save one image from each camera
    image_paths = []
    for i, frame_list in enumerate(frames):
        if frame_list:
            try:
                # Save the last captured frame
                image_filename = os.path.join(IMAGE_SAVE_DIR, f"{latest_row_id + i}.jpg")
                cv2.imwrite(image_filename, frame_list[-1])
                logging.info("Captured and saved image from camera %d as %s", camera_indices[i], image_filename)
                image_paths.append(image_filename)
            except Exception as e:
                logging.error("Failed to save image from camera %d: %s", camera_indices[i], e)

    # Release the cameras and close OpenCV windows
    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()

    # Return paths for both images or None if a capture failed
    return (
        image_paths[0] if len(image_paths) > 0 else None,
        image_paths[1] if len(image_paths) > 1 else None
    )


if __name__ == "__main__":
    # Example of capturing images
    camera_images = capture_images(camera_indices=(1, 2))
    if camera_images[0] and camera_images[1]:
        logging.info("Images captured successfully: %s, %s", camera_images[0], camera_images[1])
    else:
        logging.error("Image capture failed.")
