import serial
import pynmea2
import cv2
from datetime import datetime
import os
import time
import threading
import sqlite3

# GPS configuration
# serial_port = '/dev/ttyUSB0'  # Adjust based on your system
serial_port = 'COM3'
baud_rate = 9600  # Typical baud rate for GPS modules

def get_gps_data():
    """
    Reads GPS data from the serial port and returns latitude and longitude.

    Returns:
        tuple: (latitude, longitude) if successful, else (None, None)
    """
    try:
        with serial.Serial(serial_port, baud_rate, timeout=1) as ser:
            while True:
                line = ser.readline().decode('ascii', errors='replace').strip()
                if line.startswith('$GNGGA') or line.startswith('$GNRMC'):
                    try:
                        msg = pynmea2.parse(line)
                        if isinstance(msg, pynmea2.GGA) or isinstance(msg, pynmea2.RMC):
                            return msg.latitude, msg.longitude
                    except pynmea2.ParseError:
                        continue
    except serial.SerialException as e:
        print(f"Serial error: {e}")
    return None, None

def get_latest_row_id(db_path):
    """
    Fetches the latest row ID from the 'location' table in the SQLite database.

    Parameters:
        db_path (str): Path to the SQLite database file.

    Returns:
        int: Latest row ID from the 'location' table.
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM location ORDER BY id DESC LIMIT 1")
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else 0
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return 0

def capture_frames(cap, frames):
    while True:
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break

def capture_images(camera_index1=1, camera_index2=2):
    """
    Captures images from two cameras and saves them.

    Parameters:
        camera_index1 (int): ID of the first camera.
        camera_index2 (int): ID of the second camera.

    Returns:
        tuple: File paths of the saved images from camera 1 and camera 2.
    """
    camera_indices = (camera_index1, camera_index2)
    caps = []
    threads = []
    frames = [[] for _ in camera_indices]  # List of lists to store frames for each camera
    
    # Specify the directory to save images
    save_dir = r"D:\Jupyter\Road_Object_Detection\Captured"
    db_path = r"D:\Jupyter\Road_Object_Detection\SQLite\My_Database.db"
    os.makedirs(save_dir, exist_ok=True)

    # Open the specified cameras
    for index in camera_indices:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            print(f"Camera found at index {index}")
            caps.append(cap)
            # Start a new thread for capturing frames
            thread = threading.Thread(target=capture_frames, args=(cap, frames[len(caps) - 1]))
            thread.start()
            threads.append(thread)
        else:
            print(f"No camera at index {index}")

    # Check if at least one camera is available
    if not caps:
        print("No cameras to display.")
        return None, None

    # Display video feed from the cameras briefly
    display_duration = 2  # Display duration in seconds
    start_time = time.time()
    while time.time() - start_time < display_duration:
        if len(frames) > 0:
            # Check if we have frames from all cameras
            combined_frames = [f for f in frames if len(f) > 0 and f[-1] is not None]  # Only take non-empty frames
            if len(combined_frames) > 0:
                # Show each frame individually
                for i, frame in enumerate(combined_frames):
                    if len(frame) > 0:  # Ensure there is at least one frame
                        cv2.imshow(f"Camera Feed {camera_indices[i]}", frame[-1])  # Show the latest frame

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Capture and save one image from each camera
    image_paths = []  # To store the paths of saved images
    latest_row_id = get_latest_row_id(db_path) + 1  # Fetch the latest row ID and increment it

    for i, frame in enumerate(frames):
        if len(frame) > 0:  # Ensure there is at least one frame
            # Save the frame as an image with an incremented ID for each camera
            image_filename = os.path.join(save_dir, f"{latest_row_id + i}.jpg")
            try:
                cv2.imwrite(image_filename, frame[-1])  # Save the last captured frame
                print(f"Captured and saved image from camera {camera_indices[i]} as {image_filename}.")
                image_paths.append(image_filename)  # Store the image path
            except Exception as e:
                print(f"Failed to save image from camera {camera_indices[i]}: {e}")

    # Release the cameras and close windows
    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()
    
    # Ensure we return paths for both images
    return (image_paths[0] if len(image_paths) > 0 else None, 
            image_paths[1] if len(image_paths) > 1 else None)
