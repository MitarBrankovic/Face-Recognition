# import libraries here
import os
import numpy as np
import cv2 # OpenCV
from sklearn.svm import SVC # SVM klasifikator
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier # KNN
from joblib import dump, load
import matplotlib
import matplotlib.pyplot as plt

def train_or_load_facial_expression_recognition_model(train_image_paths, train_image_labels):
    """
    Procedura prima listu putanja do fotografija za obucavanje i listu labela za svaku fotografiju iz prethodne liste

    Kada se procedura pozove, ona treba da trenira model ako on nije istraniran, ili da ga samo ucita ako je prethodno istreniran. 
    Ako serijalizujete model, serijalizujte ga odmah pored main.py, bez kreiranja dodatnih foldera.
    Napomena: Platforma ne vrsi download serijalizovanih modela i bilo kakvih foldera i sve ce se na njoj ponovo trenirati (vodite racuna o vremenu). 
    Serijalizaciju mozete raditi samo da ubrzate razvoj lokalno, da se model ne trenira svaki put.

    Vreme izvrsavanja celog resenja je ograniceno na maksimalno 1h.

    :param train_image_paths: putanje do fotografija za obucavanje
    :param train_image_labels: labele za sve fotografije iz liste putanja za obucavanje
    :return: Objekat modela
    """
    # TODO - Istrenirati model ako vec nije istreniran

    #anger, contempt, disgust, happiness, neutral, sadness, surprise



    model = None
    return model


def extract_facial_expression_from_image(trained_model, image_path):
    """
    Procedura prima objekat istreniranog modela za prepoznavanje ekspresije lica i putanju do fotografije na kojoj
    se nalazi novo lice sa koga treba prepoznati ekspresiju.

    Ova procedura se poziva automatski iz main procedure pa nema potrebe dodavati njen poziv u main.py

    :param trained_model: <Model> Istrenirani model za prepoznavanje karaktera
    :param image_path: <String> Putanja do fotografije sa koje treba prepoznati ekspresiju lica
    :return: <String>  Naziv prediktovane klase (moguce vrednosti su: 'anger', 'contempt', 'disgust', 'happiness', 'neutral', 'sadness', 'surprise'
    """
    facial_expression = ""
    # TODO - Prepoznati ekspresiju lica i vratiti njen naziv (kao string, iz skupa mogucih vrednosti)

    return facial_expression
