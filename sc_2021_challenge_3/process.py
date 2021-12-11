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

import functions as fu

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

    anger_imgs, contempt_imgs, disgust_imgs, happiness_imgs, neutral_imgs, sadness_imgs, suprise_imgs = fu.get_emotions(train_image_paths, train_image_labels)

    anger_features = []
    contempt_features = []
    disgust_features = []
    happiness_features = []
    neutral_features = []
    sadness_features = []
    surprise_features = []
    labels = []


    for img_path in anger_imgs:
        img = cv2.imread(img_path)
        img = cv2.resize(img, (78, 159))
        hog = fu.make_hog(img)
        img_gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        anger_features.append(hog.compute(img_gs))
        labels.append(0)

    for img_path in contempt_imgs:
        img = cv2.imread(img_path)
        img = cv2.resize(img, (78, 159))
        hog = fu.make_hog(img)
        img_gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        contempt_features.append(hog.compute(img_gs))
        labels.append(1)

    for img_path in disgust_imgs:
        img = cv2.imread(img_path)
        img = cv2.resize(img, (78, 159))
        hog = fu.make_hog(img)
        img_gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        disgust_features.append(hog.compute(img_gs))
        labels.append(2)

    for img_path in happiness_imgs:
        img = cv2.imread(img_path)
        img = cv2.resize(img, (78, 159))
        hog = fu.make_hog(img)
        img_gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        happiness_features.append(hog.compute(img_gs))
        labels.append(3)

    for img_path in neutral_imgs:
        img = cv2.imread(img_path)
        img = cv2.resize(img, (78, 159))
        hog = fu.make_hog(img)
        img_gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        neutral_features.append(hog.compute(img_gs))
        labels.append(4)

    for img_path in sadness_imgs:
        img = cv2.imread(img_path)
        img = cv2.resize(img, (78, 159))
        hog = fu.make_hog(img)
        img_gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sadness_features.append(hog.compute(img_gs))
        labels.append(5)

    for img_path in suprise_imgs:
        img = cv2.imread(img_path)
        img = cv2.resize(img, (78, 159))
        hog = fu.make_hog(img)
        img_gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        surprise_features.append(hog.compute(img_gs))
        labels.append(6)


    anger_features = np.array(anger_features)
    contempt_features = np.array(contempt_features)
    disgust_features = np.array(disgust_features)
    happiness_features = np.array(happiness_features)
    neutral_features = np.array(neutral_features)
    sadness_features = np.array(sadness_features)
    surprise_features = np.array(surprise_features)

    x_train = np.vstack((anger_features, contempt_features, disgust_features, happiness_features, neutral_features, sadness_features, surprise_features))
    y_train = np.array(labels)
    x_train = fu.reshape_data(x_train)
    clf_svm = SVC(kernel='linear', probability=True)
    clf_svm.fit(x_train, y_train)

    #dump(clf_svm, "faces.joblib")
    #clf_svm = load("faces.joblib")
    #print("Created " + "faces.joblib")
    model = clf_svm

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

    img = cv2.imread(image_path)
    img = cv2.resize(img, (78, 159))
    img_gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hog = fu.make_hog(img)

    features = []
    features.append(hog.compute(img))
    aa = np.array(features)
    reshaped = fu.reshape_data(aa)
    facial_expression = trained_model.predict(reshaped)

    if facial_expression[0] == 0:
        facial_expression = "anger"
    elif facial_expression[0] == 1:
        facial_expression = "contempt"
    elif facial_expression[0] == 2:
        facial_expression = "disgust"
    elif facial_expression[0] == 3:
        facial_expression = "happiness"
    elif facial_expression[0] == 4:
        facial_expression = "neutral"
    elif facial_expression[0] == 5:
        facial_expression = "sadness"
    else:
        facial_expression = "surprise"
    
    print(facial_expression)


    return facial_expression
