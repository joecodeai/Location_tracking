import serial
import pynmea2
import cv2
from datetime import datetime

# GPS configuration
serial_port = '/dev/ttyUSB0'  # Adjust based on your system
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

def capture_images(camera1_id=0, camera2_id=1):
    """
    Captures images from two cameras.

    Parameters:
        camera1_id (int): ID of the first camera.
        camera2_id (int): ID of the second camera.

    Returns:
        tuple: (frame1, frame2) captured from camera 1 and camera 2, respectively.
    """
    cam1 = cv2.VideoCapture(camera1_id)
    cam2 = cv2.VideoCapture(camera2_id)
    
    if not cam1.isOpened() or not cam2.isOpened():
        print("Error: One or both cameras could not be opened.")
        return None, None

    ret1, frame1 = cam1.read()
    ret2, frame2 = cam2.read()
    
    cam1.release()
    cam2.release()

    if ret1 and ret2:
        return frame1, frame2
    else:
        print("Error: Could not capture frames from one or both cameras.")
        return None, None

def save_images(frame1, frame2):
    """
    Saves images from two frames with a timestamp.

    Parameters:
        frame1 (numpy.ndarray): Image from camera 1.
        frame2 (numpy.ndarray): Image from camera 2.

    Returns:
        tuple: File paths of saved images.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    img1_path = f"camera1_{timestamp}.jpg"
    img2_path = f"camera2_{timestamp}.jpg"
    cv2.imwrite(img1_path, frame1)
    cv2.imwrite(img2_path, frame2)
    return img1_path, img2_path
