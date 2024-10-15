import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import random
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def read_img_from_path(df,id_,type_ = 0,shape=[128,128]):
    # img = cv2.imread(df.loc[id_,'paths'],type_)
    img = cv2.imread(df.loc[id_,'paths'], type_)
    img = cv2.resize(img,[shape[0],shape[1]])

    # matplotlib
    # img = plt.imread(df.loc[id_,'paths'],type_) # type_=0 for grayscale
    # img = plt.imread(df.loc[id_,'paths'],type_) # type_=0 for grayscale

    return img

# Stand-in for sklearn's PCA implementation on Kaggle.
class PCA1:
    # TODO: Pull __init__ and get_data methods into functions.
    def __init__(self,df,img_type,img_size,train_size):
        self.df = df # Dataframe with image paths
        self.img_type = img_type # Img_type =0 for grayscale
        self.img_size = img_size # Final Image size
        self.train_size = train_size # Percentage of training images
        self.get_data()
        self.svd()

    def get_data(self):
        self.pca_data = []
        # random.seed(config.seed)
        train_ids = random.sample(range(len(self.df)),int(self.train_size*len(self.df)))
        train = self.df[self.df.index.isin(train_ids)]
        self.val = self.df[~self.df.index.isin(train_ids)]
        self.val.reset_index(inplace=True,drop=True)
        train.reset_index(inplace=True,drop=True)
        self.train = train

        for i in range(len(train)): # Read, flatten, normalize the images
            img = read_img_from_path(train,i,self.img_type,self.img_size)
            # img = np.reshape(img, (np.product(img.shape),))
            img = np.reshape(img, (np.prod(img.shape),))
            img = img/255
            self.pca_data.append(img)

        self.pca_data = np.asarray(self.pca_data)
        self.data_mean = self.pca_data.mean(axis=0) # Finding mean pokemon image
        self.pca_data = self.pca_data-self.data_mean # Subtract mean to get centered data
        print(f'train data shape is{self.train.shape}')
        print(f'val data shape is{self.val.shape}')

    def svd(self):
        u,s,v = np.linalg.svd(self.pca_data) #SVD on centred data
        self.eigenfaces = v # Principal Components
        self.eigenproj = u
        self.eigenvalues = s
        self.eigenweights = np.dot(self.pca_data, v.T) # Computing weights for all training images
        self.weight_means = self.eigenweights.mean(axis=0)
        self.weight_std = self.eigenweights.std(axis=0)

    def proj(self,n_comps,train,id_):
        if train==True:
            db = self.train
        else:
            db = self.val
        img = read_img_from_path(db,id_,self.img_type,self.img_size)
        # img = np.reshape(img, (np.product(img.shape),))
        img = np.reshape(img, (np.prod(img.shape),))
        img = img/255
        img = (img - self.data_mean)
        components = self.eigenfaces[:n_comps]
        proj = np.dot(np.dot(img,components.T),components)
        proj = (self.data_mean + proj).reshape(self.img_size)
        return img,proj


