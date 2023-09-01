import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Define functions and classes for tasks

def load_data(classes, data_path):
    X = []
    Y = []
    for cls, label in classes.items():
        pth = os.path.join(data_path, cls)
        for j in os.listdir(pth):
            img = cv2.imread(os.path.join(pth, j), 0)
            img = cv2.resize(img, (200, 200))
            X.append(img)
            Y.append(label)
    return np.array(X), np.array(Y)

def preprocess_data(X):
    X_flatten = X.reshape(len(X), -1) / 255.0
    pca = PCA(n_components=0.98)
    pca.fit(X_flatten)
    return X_flatten, pca

def train_models(X_train, y_train, pca):
    pca_train = pca.transform(X_train)
    
    lg = LogisticRegression(C=0.1)
    lg.fit(pca_train, y_train)
    
    sv = SVC()
    sv.fit(pca_train, y_train)
    
    return lg, sv

def predict_tumor(image, pca, sv):
    user_pca = pca.transform(image)
    user_prediction = sv.predict(user_pca)
    return user_prediction

print("successfully loaded the model")