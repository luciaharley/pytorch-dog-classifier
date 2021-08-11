import numpy as np 
import pandas as pd
import cv2
import os
import time
import random
from albumentations import RandomCrop, HorizontalFlip, CenterCrop, Compose, Normalize
from albumentations.pytorch.transforms import ToTensorV2
import torchvision.models as models
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader  
from torch.optim import SGD
import torchvision.transforms as transforms

def aug1():
        return Compose([RandomCrop(height = 299, width = 299, p = 1.0), 
                        HorizontalFlip(p = 0.5),  
                        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ToTensorV2()], p = 1)
def aug2():
    return Compose([Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
                    ToTensorV2()], p = 1)

def PCAColorAug(image, category = 'Tensor'):
    if type(image) == torch.Tensor:
        image = image.numpy()
        image = np.moveaxis(image, 0, 2)
    
    
    img_reshaped = image.reshape(-1, 3).astype('float32')
    mean, std = np.mean(img_reshaped, 0), np.std(img_reshaped, 0)
    img_rescaled = (img_reshaped - mean)/std
    cov_matrix = np.cov(img_rescaled, rowvar = False) # Covariant matrix of reshaped image.  Output is 3*3 matrix
    eigen_val, eigen_vec = np.linalg.eig(cov_matrix) # Compute Eigen Values and Eigen Vectors of the covariant matrix. eigen_vec is 3*3 matrix with eigen vectors as column. 
    alphas = np.random.normal(loc = 0, scale = 0.1, size = 3)
    vec1 = alphas*eigen_val
    valid = np.dot(eigen_vec, vec1) # Matrix multiplication
    pca_aug_norm_image = img_rescaled + valid
    pca_aug_image = pca_aug_norm_image*std + mean
    aug_image = np.maximum(np.minimum(pca_aug_image, 255), 0).astype('uint8')
    if category == 'Tensor':
        return torch.from_numpy(aug_image.reshape(3,299,299))
    else:
        return aug_image.reshape(299,299,3)

class StanfordDogs(Dataset):
    def __init__(self, transform1, transform2, X, Y, objective = 'train'):
        self.X = X
        self.Y = Y
        self.train_transform = transform1
        self.valid_transform = transform2
        self.objective = objective
        
    def __getitem__(self, idx):
        image_path = ''
        path = self.X['Path'][idx]
        label = self.Y.iloc[idx, :].values
        img = cv2.imread(os.path.join(image_path, path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Shortest side of image is scaled to 299 pixels and the other side is scaled so as to maintain aspect ratio
        
        h, w, _ = img.shape
        
        if h <= w:
            aspect_ratio = w/h
            dim = (299, int(299*aspect_ratio))
            img = cv2.resize(img, dim)
        else:
            aspect_ratio = h/w
            dim = (int(299*aspect_ratio), 299)
            img = cv2.resize(img, dim)

           
        img = CenterCrop(height = 299, width = 299, p = 1)(image = img)['image']
        
        if self.objective == 'train':
            random = np.random.uniform(size = 1)
            if random < 0.5:                            # PCA Augmentation carried out only 50 percent of time
                img = PCAColorAug(img, category = 'numpy')
                
            augmented = self.train_transform(image = img)
            img = augmented['image']
            
            return img, label
        
        elif ((self.objective == 'validation') |  (self.objective == 'test')):
            img = cv2.resize(img, (299, 299))
            augmented = self.valid_transform(image = img)
            img = augmented['image']  
            
            return img, label
        
    
    def __len__(self):
        return len(self.X)
    
def loader(data_X, data_Y, batch_size = 1, obj = 'train'):
    train_aug = aug1()
    valid_aug = aug2()
    data = StanfordDogs(train_aug, valid_aug, X = data_X, Y = data_Y, objective = obj)
    loader = DataLoader(data, batch_size = batch_size, shuffle = True)
    
    return loader