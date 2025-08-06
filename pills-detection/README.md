# YOLO Object Detection for Tablets and Capsules

This project implements a YOLOv8-based object detection model for identifying and localizing tablets and capsules in images. The model was trained using a dataset of annotated images generated using Roboflow, a platform for managing and preparing computer vision datasets.

## Dataset Preparation
To prepare the dataset, images of tablets and capsules were collected and annotated using Roboflow's annotation tools. The resulting dataset was then exported in YOLOv8 format and used to train the object detection model.

## Model Training
The YOLOv8 object detection model was trained on a local machine. The model was trained for 100 epochs using a batch size of 8 and a learning rate of 0.01, achieving an mAP of 93.1% on the validation set.

## Model Evaluation
The trained model was evaluated on a test set of images, achieving an mAP of 91,7% and demonstrating its ability to accurately identify and localize tablets and capsules in real-world images.

## Usage
The trained YOLOv8 model can be tested on [Roboflow](https://universe.roboflow.com/seblful/pills-detection-s9ywn/model/19).
