import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

import cv2

from sklearn.decomposition import PCA, KernelPCA
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

import os
import warnings

def show_image_classes(image, label, n=10):
    fig, axarr=plt.subplots(nrows=n, ncols=n, figsize=(18, 18))
    axarr=axarr.flatten()
    plt_id=0
    start_index=0
    for sign in range(10):
        sign_indexes=np.where(label==sign)[0]
        for i in range(n):

            image_index=sign_indexes[i]
            axarr[plt_id].imshow(image[image_index], cmap='gray')
            axarr[plt_id].set_xticks([])
            axarr[plt_id].set_yticks([])
            axarr[plt_id].set_title("Sign :{}".format(sign))
            plt_id=plt_id+1
    plt.suptitle("{} Sample for Each Classes".format(n))
    plt.show()


def fix_digit_labels(y):
    y_new=list()
    for target in y:
        y_new.append(np.argmax(target))
    y=np.array(y_new)

    label_map={0:9,1:0, 2:7, 3:6, 4:1, 5:8, 6:4, 7:3, 8:2, 9:5}
    y_new=list()
    for s in y:
        y_new.append(label_map[s])
    y=np.array(y_new)

    return y


class Clf_Helper:

    def __init__(self, X, y, n_components):
        self.train_test_split(X,y)
        self.n_components=n_components
        self.transform()


    def train_test_split(self, X, y):
        if np.max(X[0])>1:
            X=X/255.0
            print("Data scaled...")

        self.X_train, self.X_test, self.y_train, self.y_test=train_test_split(X, y,
                                                 stratify=y,
                                                 test_size=0.3,
                                                 random_state=42)
    def transform(self):
        self.pca_transform()

    def pca_transform(self):
        pca=PCA(n_components=self.n_components)
        pca.fit(self.X_train)
        self.X_train=pca.transform(self.X_train)
        self.X_test=pca.transform(self.X_test)
        print("PCA transform performed...")

    def best_model(self, models, show_metrics=False):
        print("INFO: Finding Accuracy Best Classifier...", end="\n\n")
        best_clf=None
        best_acc=0
        for clf in models:
            clf.fit(self.X_train, self.y_train)
            y_pred=clf.predict(self.X_test)
            acc=metrics.accuracy_score(self.y_test, y_pred)
            print(clf.__class__.__name__, end=" ")
            print("Accuracy:{:.3f}".format(acc))

            if best_acc<acc:
                best_acc=acc
                best_clf=clf
                best_y_pred=y_pred

        print("Best Classifier:{}".format(best_clf.__class__.__name__))
        if show_metrics:
            self.metrics(y_true=self.y_test, y_pred=best_y_pred)

    def cv_best_model(self, models, show_metrics=False):
        print("INFO: Finding Cross Validated Accuracy Best Classifier...", end="\n\n")
        kfold=KFold(n_splits=5,  shuffle=True, random_state=0)
        best_clf=None
        best_acc=0
        for clf in models:
            cv_scores=cross_val_score(clf, self.X_train, self.y_train, cv=kfold)
            print(clf.__class__.__name__, end=" ")
            cv_mean_acc=cv_scores.mean()
            print("CV Mean Accuracy:{:.3f}".format(cv_mean_acc))
            if best_acc<cv_mean_acc:
                  best_acc=cv_mean_acc
                  best_clf=clf

        print("CV Best Classifier:{}".format(best_clf.__class__.__name__))
        if show_metrics:
            y_pred = best_clf.predict(self.X_test)
            self.metrics(y_true=self.y_test, y_pred=y_pred)

        return best_clf

    def grid_searc_cv_for_best_model(self, model, params, show_metrics=False):
        kfold=KFold(n_splits=3,  shuffle=True, random_state=0)
        grid_search_cv=GridSearchCV(SVC(), params, cv=kfold, scoring="accuracy")
        grid_search_cv.fit(self.X_train, self.y_train)
        y_pred=grid_search_cv.predict(self.X_test)
        print("Best pamameters for {}:{}".format(model.__class__.__name__,
                                              grid_search_cv.best_params_))
        print("Accuracy:{:.3f}".format(metrics.accuracy_score(self.y_test, y_pred)))
        if show_metrics:
            self.metrics(y_true=self.y_test, y_pred=y_pred)

    def metrics(self, y_true, y_pred):
        print("Accuracy:{:.3f}".format(metrics.accuracy_score(y_true, y_pred)))
        print("Confusion Matrix:\n{}".format(metrics.confusion_matrix(y_true, y_pred)))
        print("Classification Report:\n{}".format(metrics.classification_report(y_true, y_pred)))
