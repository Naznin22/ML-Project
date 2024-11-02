# YOLO Object Detection with Custom Bangla Labels
This project demonstrates object detection using the YOLO (You Only Look Once) model implemented with Darkflow, a TensorFlow adaptation of Darknet. Using YOLO for object detection is advantageous due to its speed and efficiency in real-time applications. YOLO processes the entire image at once, which allows it to detect multiple objects in a single evaluation. This makes it particularly suitable for applications requiring quick responses, such as autonomous driving or surveillance. Additionally, YOLO's architecture is designed to provide a balance between accuracy and performance, making it a popular choice in the field of computer vision.

## Overview
The goal of this project is to perform object detection on an image and display the labels in Bangla. The model is set up with YOLO's pre-trained weights, and it uses TensorFlow 1.15 and OpenCV for image processing. Bangla labels are applied by mapping the COCO dataset's default English labels to their corresponding Bangla translations.

## Requirements
Install the following dependencies:
* Cython
* TensorFlow 1.15
* OpenCV
* Numpy
* Matplotlib
* Pillow
Clone the darkflow repository, build it, and then set up the environment.

# Project Structure
### Installation and Setup:
The project installs necessary packages and clones the darkflow GitHub repository.
darkflow is set up for building and deploying YOLO within the Colab environment.
### Model Configuration:
The YOLO model configuration and weights are loaded with specified options (e.g., confidence threshold and GPU usage).
The cfg/yolo.cfg file is used to load YOLO with pre-trained weights (bin/yolo.weights).
### Bangla Label Conversion:
Labels in the default coco.names file are translated to Bangla.
A custom label file in Bangla replaces English labels during detection to display Bangla labels on bounding boxes.
### Bounding Box Drawing and Labeling:
The boxing function overlays bounding boxes and labels on detected objects in the image.
Custom Bangla font (Bangla-Kolom.ttf) is used for rendering text on images.
Objects with a confidence score above 0.3 are labeled and displayed.
### Running the Model
Load an image and run the YOLO model with tfnet.return_predict() to get predictions.
Apply boxing() to add bounding boxes and labels in Bangla.
### Visualization
The final output is displayed inline using Matplotlib to show bounding boxes and labels in Bangla on detected objects.

## Acknowledgments
This project utilizes:

* Darkflow for YOLO implementation on TensorFlow.
* COCO dataset label names (coco.names file) as the basis for object labels, which are translated to Bangla.
