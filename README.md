# Brain Tumor Detection

Welcome to the Brain Tumor Detection project repository. 
This project aims to detect the presence of brain tumors in MRI images using machine learning techniques. 
It provides a user-friendly interface for users to upload an MRI image and receive a prediction about the presence of a brain tumor.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Dataset](#Dataset)
- [Setup](#setup)
- [Usage](#usage)

## Introduction

The Brain Tumor Detection application utilizes machine learning techniques to classify MRI images as either having a brain tumor or being tumor-free. It's built using Python, OpenCV, NumPy, scikit-learn, and Tkinter for the graphical user interface.

## Features

- Loading and preprocessing MRI images for training and testing.
- Perform dimensionality reduction using  Principal Component Analysis (PCA).
- Utilize machine learning models: Logistic Regression and Support Vector Machine.
- Preprocesses images using OpenCV and NumPy.
- Provides a graphical user interface (GUI) for easy interaction.
- Upload an MRI image for prediction and receive the tumor detection result instantly.

## Dataset

Kaggle - Brain Tumor Classification (MRI)
[link](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)


## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/juinaik-1/Brain-Tumor-Detection.git
   cd Brain-Tumor-Detection
   
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   
3. Run the application:
```bash
  python user_input.py
