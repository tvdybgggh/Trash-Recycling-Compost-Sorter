Automated Waste Sorting System
Project Overview
The Automated Waste Sorting System aims to revolutionize waste management by leveraging advanced computer vision technologies. Utilizing the YOLO (You Only Look Once) object detection model, this system can accurately classify waste items into categories such as recycling, compost, and trash, facilitating more efficient waste processing and contributing to environmental sustainability.

Getting Started
Prerequisites
Python 3.x
Flask
OpenCV
NumPy
Installation
Clone the repository to your local machine.
Install the required Python libraries using pip install -r requirements.txt.
Download the YOLOv4 and YOLOv3-tiny model weights and place them in the project directory.
Running the Application
To start the Flask server for the waste sorting API, run python model0000.py.
For a standalone object detection test using YOLOv3-tiny, execute python V3.py.
System Architecture
This project comprises two main components:

Flask API (model0000.py): A REST API serving the object detection model that classifies images of waste into predefined categories.
YOLO Object Detection (V3.py): A script that utilizes YOLOv3-tiny for quick object detection within waste images, identifying recyclable materials.
Model Information
The system uses the YOLO object detection models trained on the COCO dataset for recognizing a wide range of items. Specific categories relevant to waste sorting (e.g., bottles for recycling) are identified for targeted action.

Future Work
Enhance the model's accuracy by training on a dataset specifically designed for waste materials.
Expand the list of identifiable waste categories.
Integrate with physical sorting mechanisms for a fully automated waste management solution.
