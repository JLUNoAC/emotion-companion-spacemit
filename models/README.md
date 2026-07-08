# AI Models

This directory stores the neural network models required by the emotion recognition pipeline.

The models are not included in this repository because they are large binary files and may have independent licenses.

Please download the required models and place them in this directory before running the project.

## Required Models

### 1. YuNet Face Detection Model

File name:


face_detection_yunet_2023mar.onnx


Function:

- Face detection
- Provides face bounding boxes for further emotion recognition

Framework:

- ONNX
- OpenCV DNN


### 2. HSEmotion Emotion Recognition Model

File name:


enet_b2_7.onnx


Function:

- Facial expression classification
- Seven emotion categories:

  - Anger
  - Disgust
  - Fear
  - Happiness
  - Sadness
  - Surprise
  - Neutral


Framework:

- ONNX Runtime
- EfficientNet-B2 backbone


## Directory Example

After downloading the models, this directory should look like:


models/
├── README.md
├── face_detection_yunet_2023mar.onnx
└── enet_b2_7.onnx


The program will automatically load these files when running:


python3 main.py