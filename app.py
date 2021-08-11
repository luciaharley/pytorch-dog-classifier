import streamlit as st
import pandas as pd
import numpy as np
import cv2
import pickle
import torch
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader  
from utils import *

st.title("Mixed-Breed Dog Classifier")

# load breed encodings
with open('idx2breed.pickle', 'rb') as handle:
    idx2breed = pickle.load(handle)

# load model
PATH = 'inception_finetuning.pth'
inception = models.inception_v3(pretrained = True, aux_logits=False)
# Freeze model parameters
for param in inception.parameters():
    param.requires_grad = False
# Change the final layer of Inception Model for Transfer Learning
fc_inputs = inception.fc.in_features
inception.fc = nn.Sequential(
    nn.Linear(fc_inputs, 2048),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(2048, 120),
    nn.LogSoftmax(dim=1) # For using NLLLoss()
)
inception.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
inception.eval()

# upload image
uploaded_file = st.file_uploader("Choose an image...", type=['png','jpeg','jpg'])
if uploaded_file is not None:
    
    # preprocess image
    image_path = ''
    path = [uploaded_file.name]
    image1 = cv2.imread(uploaded_file.name)
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image1 = cv2.resize(image1, (299,299))
    label = ['']
    single_test_df = pd.DataFrame()
    single_test_df['Path'] = path
    single_test_df['Label'] = label
    null_label = pd.get_dummies(single_test_df['Label'])
    single_test_loader = loader(single_test_df, null_label, batch_size = 1, obj = 'test')
    
    # display uploaded image
    st.image(image1, caption='Your Dog', use_column_width=False)
    st.write('')

    # run image through model
    inputs, _ = next(iter(single_test_loader))
    outputs = inception(inputs)
    parents_pred = np.argpartition(outputs.detach().numpy(), -2)[:,-2:]

    # get predicted parent breeds
    parent_breeds = []
    for breed_id in parents_pred[0]:
        parent_breeds.append(idx2breed[breed_id])
    demo_images_df = pd.read_csv('demo_photos.csv')
    parent1_row = demo_images_df.loc[demo_images_df['Label'].str.lower() == parent_breeds[0]]
    parent2_row = demo_images_df.loc[demo_images_df['Label'].str.lower() == parent_breeds[1]]

    # display results
    stanford_image_path = 'stanford-dogs-dataset/images/Images'
    image2 = cv2.imread(stanford_image_path + parent1_row.Path.values[0])
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    image2 = cv2.resize(image2, (299,299))

    image3 = cv2.imread(stanford_image_path + parent2_row.Path.values[0])
    image3 = cv2.cvtColor(image3, cv2.COLOR_BGR2RGB)
    image3 = cv2.resize(image3, (299,299))

    st.image([image2,image3], caption=['Predicted Parent 1','Predicted Parent 2'], use_column_width=False)

