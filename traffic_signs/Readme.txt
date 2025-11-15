Traffic Sign Recognition
This is my small project for traffic sign recognition.  
I am using the German Traffic Sign Recognition Benchmark (GTSRB) dataset and a simple convolutional neural network (CNN).

The goal of this project is:
to learn how to work with image datasets in Python.
to train a basic CNN model that can classify traffic signs into 43 classes.
to understand how to evaluate a classification model (accuracy, F1 score, confusion matrix).

I tried to do my best with my knowledge of python language, I used some AI to clarify some places for me and I tried to see another project about this project.
Link for the video: https://www.youtube.com/watch?v=Vtc64rPHZ6I&t=2s
Right now it can do basic needs, check images, differenciate 43 types of road signs, I will try to make more functionality

Dataset
From the dataset I only need the Train folder:
Train/0, Train/1, ..., Train/42 subfolders with images for each class.

In my project, the folder structure is:
Artificial-Intelligence/
  traffic_signs/
    data/
      Meta.csv
      Test.csv
      Train.csv  
      Meta/
      Test/
      Train/
        0/
        1/
        ...
        42/
    traffic_sign.py
    requirements.txt
    Readme.txt