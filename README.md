# Location_tracking
# Image Matching and Location Estimation with YOLO and CLIP

This project provides a robust system for **image matching** and **location estimation** using object detection and camera orientation. The system utilizes **YOLO-based object detection**, **OpenAI's CLIP model** for image feature extraction, and **SQLite** for storing object coordinates and metadata. The overall goal is to estimate the location of an object based on images captured from two cameras and their respective orientations.

## Features
- **Image Matching**: Find the most similar image from a folder using CLIP-based feature extraction.
- **Location Estimation**: Estimate the location of an object based on its coordinates detected by two cameras with known orientations.
- **YOLO Object Detection**: Detect objects in images and match them with database entries for location information.
- **Scalable Design**: Built with production in mind, using powerful models like CLIP for image matching and location calculation.

## Key Technologies
- **YOLO**: A state-of-the-art object detection model for detecting objects in images.
- **CLIP**: OpenAI’s Contrastive Language-Image Pretraining model, used to extract high-quality image embeddings for similarity matching.
- **SQLite**: A lightweight SQL database for storing object coordinates and metadata.
- **Python**: The primary programming language used, along with popular libraries like `OpenCV`, `PyTorch`, and `NumPy`.

## Project Structure

requirements.txt # Python dependencies
/model # YOLO model for object detection
/Test/detection # Runs the model on images taken
/Test/object_location # Finds the location of the objects in the images taken
/Test/final_location # Finds our location based on new images taken and comparing with our database
/Captured # Folder containing images taken
/New_Captured # New images to process


## Requirements

Before running the project, make sure you have the following installed:

- Python 3.7 or higher
- CUDA (if using GPU acceleration)

Install dependencies using:


pip install -r requirements.txt

Setup and Usage

    Clone the repository.

Download the YOLO model:

    Ensure you create your own fine-tuned YOLO model (yolov11m.pt). You can download yolo model from YOLOv11 GitHub releases.

Setup your SQLite database:

    Make sure your SQLite database (e.g., My_Database.db) contains object coordinates and camera metadata.
    The database schema should include a table for object locations with columns like id, obj1_latitude, obj1_longitude.

Run the Location Estimation:

    You can now run the location estimation script. Replace the image_paths with your own images, and adjust the camera_orientations according to the orientations of your cameras.

from location_estimator import LocationEstimator

# Define paths to images, database, and model
db_path = '/path/to/your/database.db'
model_path = '/path/to/your/yolov11m.pt'
folder_path = '/path/to/your/images'

image_paths = [
    '/path/to/image1.jpg',
    '/path/to/image2.jpg'
]

camera_orientations = [45, 90]  # Example orientations for the cameras

# Initialize the estimator and estimate location
estimator = LocationEstimator(db_path, model_path, folder_path)
estimator.estimate_location(image_paths, camera_orientations)
estimator.close()

Image Matching:

    Use the ImageMatcher class to find the most similar image from a folder:

from image_matcher import ImageMatcher

folder_path = '/path/to/your/image/folder'
image_path = '/path/to/query/image.jpg'

matcher = ImageMatcher(folder_path)
best_match, similarity = matcher.find_best_match_or_alternate(image_path)

if best_match != 'No Match Found':
    print(f"Most similar image: {best_match} with similarity score: {similarity}")
else:
    print(best_match)

Test the System:

    To ensure everything works, run some sample images through the system and monitor the results.
    The system will automatically detect objects in the images, match them with similar images, and estimate the location using the two cameras.
