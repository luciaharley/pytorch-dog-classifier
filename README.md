# Pytorch Mixed-Breed Dog Classifier
Mixed-breed dog classifier using Pytorch. 

Meet Lenny. 
<img src="https://user-images.githubusercontent.com/19161994/128657128-2708d61e-086d-417d-94e1-59886e1b5718.jpg" width="200" height="200">
Lenny is Lucia's one-year-old rescue puppy, found as a stray in the central valley of CA. Since he was found without a mother or father present, his breed isn't known. His owners think he looks like a cross between a golden retriever and a dachshund but have no evidence besides his outward appearance. 

This project serves as an exploration in finetuning pre-trained CNN models as well as a light-hearted voyage into detecting Lenny's heritage. 

Datasets:
1. ![Stanford Dogs Dataset](https://www.kaggle.com/jessicali9530/stanford-dogs-dataset)
Images of 120 purebred dog breeds from around the world. This dataset was used for training our CNN.
2. A hand-curated dataset of mixed breed dogs from Google Images (images: mixed_breeds.zip, annotations: mixed_breeds.csv). This dataset contains 400 images of mixed-breed dogs (20 mixed-breeds @ 20 photos each). Each mixed-breed is a mix of exactly 2 parent breeds found in the Stanford Dogs Dataset. This dataset was used for model testing.

Process: 
In this project, we attempted to recreate the model described in the methods section of this ![paper](https://www.academia.edu/33721767/Mixed_Breed_Dogs_Classification). First, we trained a fine-tuned Inception model on the Stanford Dogs Dataset. We then tested the model on our mixed-breed dataset, taking the top 2 highest predicted breeds as the putative parent breeds for each image. 

Like the paper, we calculated accuracy in detecting both parents and at least one parent, as well as topK accuracy for one and both parents. Also like the paper, we found that detecting parent breeds is very difficult, even with a well-trained CNN. As summarized in the paper, this is likely a combination of error from a) introducing new variability to a model tightly trained to one dataset, as well as b) introducing variants that blur inter-class differences and enlarge intra-class variations.  

This project was done for the course MSDS 631 - Deep Learning, in partial completion of the Masters in Data Science degree program at the University of San Francisco.

Contributors: 
Lucia Page-Harley 
Joshua Majano 
