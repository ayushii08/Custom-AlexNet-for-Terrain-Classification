Terrain Classification using Deep Learning

This project implements terrain classification using three different deep learning approaches: ELM (Extreme Learning Machine), SegNet, and AlexNet. The models are trained on two distinct datasets for terrain classification tasks.

Project Structure
Project/
├── Codes/
│   ├── ELM.ipynb        
│   └── SegNet.ipynb     
├── DL Model/
│   ├── Model.ipynb
│   ├── best_model_weights2.h5
│   └── best_model_weights2.h5
├── requirement.txt
└── README.md


Models
ELM (Extreme Learning Machine)

Fast single-hidden layer feedforward network
Optimized for quick training speeds
Input weights randomly assigned and fixed
Only output weights trained using least squares

SegNet

Deep encoder-decoder architecture for semantic segmentation
Each encoder layer has corresponding decoder layer
Preserves high-frequency details in segmentation
Maintains spatial information through max-pooling indices

AlexNet

Pre-trained CNN architecture, modified for terrain classification
5 convolutional layers, 3 fully connected layers
Modified final layers for terrain classes
Saved weights provided in weights/ directory

Datasets

Dataset A: Different Terrain Types Classification

Number of classes: 4
Image size: 4kB-350kB
Total samples:3196
link: https://www.kaggle.com/datasets/durgeshrao9993/different-terrain-types-classification

Dataset B: Terrain Recognition

Number of classes: 4
Image size: 25KB-170KB
Total samples: 38340
link: https://www.kaggle.com/datasets/atharv1610/terrain-recognition

Setup
Requirements
python==3.10.15
opencv-python==4.10.0.84
numpy==1.21.3
hpelm==1.0.10  
matplotlib==3.5.1
pillow==11.0.0
scikit-learn==1.5.2 
tensorflow==2.10.0   


Installation
bashCopygit clone https://github.com/Ricky0403/Custom-AlexNet-for-Terrain-Classification-DL-Project-
pip install -r requirements.txt
install the dataset from the given link in dataset.

Usage
Training
For ELM/SegNet change the path to the dataset in the code and run

# For AlexNet 
Training
For AlexNet change the dataset path

Using existing weights:
For Dataset A use weight_best_model_weights3.h5 (Accuracy 91%)
For Dataset B use weight_best_model_weights7.h5 (Accuracy 96.5%)

Change the weights path to use these weights.
