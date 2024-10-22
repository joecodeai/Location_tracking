from ultralytics import YOLO
import cv2

# Load the trained YOLO model
model = YOLO("/home/mundax/Projects/Location_tracking/model/yolo11n_saved.pt")

# Path to the video file
video_path = "/home/mundax/Downloads/Shop Sign Board Design Ideas - Glow Sign Board Making - 3D Signage Types..mp4"

# Open the video file
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    results = model(frame)

    # Process results
    for result in results:
        boxes = result.boxes  # Get the bounding boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]  # Get coordinates of the bounding box
            conf = box.conf[0]  # Get confidence score
            cls = int(box.cls[0])  # Get class ID
            
            # Draw bounding box and label on the frame
            label = f'Class: {cls}, Conf: {conf:.2f}'
            frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            frame = cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Display the frame with detections
    cv2.imshow("Object Detection", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
