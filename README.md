# Pytorch Mixed-Breed Dog Classifier
Meet Lenny. 

<img src="https://user-images.githubusercontent.com/19161994/128657128-2708d61e-086d-417d-94e1-59886e1b5718.jpg" width="200" height="200">

Lenny is Lucia's one-year-old rescue puppy, found as a stray in the central valley of CA. Since he was found without a mother or father present, his breed isn't known. His owners think he looks like a cross between a golden retriever and a dachshund but have no evidence besides his outward appearance. 

This project serves as an exploration in finetuning pre-trained CNN models as well as a light-hearted voyage into detecting Lenny's heritage. 

<img width="900" src="https://user-images.githubusercontent.com/19161994/128658405-02f839b7-edab-4376-8aa6-5c2c13a8b1c9.png">

Browse the model training and analysis process in [final_inception.ipynb](https://github.com/luciaharley/pytorch-dog-classifier/blob/main/final_inception.ipynb).
Or, test the model yourself using the streamlit app [app.py](https://github.com/luciaharley/pytorch-dog-classifier/blob/main/app.py).

To run app.py:
1. Clone the repository and unzip all zip files 
2. Create a virtual environment and download the necessary packackes using [requirements.txt](https://github.com/luciaharley/pytorch-dog-classifier/blob/main/requirements.txt):
```
 $ python -m venv torch-env
 $ source torch-env/bin/activate
 $ python -m pip install -r requirements.txt
```
4. Run app`streamlit run app.py`
3. Upload any image **from the same directory** and see what breed the doggy parents are!

## Datasets
1. [Stanford Dogs Dataset](https://www.kaggle.com/jessicali9530/stanford-dogs-dataset):
Images of 120 purebred dog breeds from around the world. This dataset was used for training our CNN.
2. A hand-curated dataset of mixed breed dogs from Google Images (images: [mixed_breeds.zip](https://github.com/luciaharley/pytorch-dog-classifier/blob/main/mixed_breeds.zip), annotations: [mixed_breeds.csv](https://github.com/luciaharley/pytorch-dog-classifier/blob/main/mixed_breeds.csv)). This dataset contains 400 images of mixed-breed dogs (20 mixed-breeds @ 20 photos each). Each mixed-breed is a mix of exactly 2 parent breeds found in the Stanford Dogs Dataset. This dataset was used for model testing.

## Process
In this project, we attempted to recreate the model described in the methods section of this [paper](https://www.academia.edu/33721767/Mixed_Breed_Dogs_Classification). First, we trained a fine-tuned Inception model on the Stanford Dogs Dataset. Specifically, we added an extra fully-connected layer and corresponding dropout layer to the existing model to assign scores to the 120 dog breeds. We then tested the model on our mixed-breed dataset, taking the top 2 highest predicted breeds as the putative parent breeds for each image. 

<img width="800" src="https://user-images.githubusercontent.com/19161994/128912897-2d094778-6bc6-413c-b969-bb26aa81fe75.png">

## Model Performance
Like the paper, we calculated accuracy in detecting both parents and at least one parent, as well as topK accuracy for one and both parents. Also like the paper, we found that detecting parent breeds is very difficult, even with a well-trained CNN. As summarized in the paper, this is likely a combination of error from a) introducing new variability to a model tightly trained to one dataset, as well as b) introducing variants that blur inter-class differences and enlarge intra-class variations.  

On the (purebred) training data, the model yielded 77.8% validation accuracy after 10 epochs. For the mixed-breed test data, TopK accuracy for one and both parents are as follows:

<img width="900" src="https://user-images.githubusercontent.com/19161994/128966747-6aac2da0-e98e-481e-8a04-3a7e3cb84c76.png">

Though accuracy scores are significantly lower than validation accuracy with the purebred model, these scores nearly perfectly align with the scores in the paper that we are replicating -- a clear next step would be to aggregate a labeled mixed breed dataset on the scale of the Stanford Dogs Dataset and train a multi-label CNN classifier directly on those images.

## Lenny's Results
And finally...the moment we have all been waiting for! Who is Lenny? 

<img width="900" src="https://user-images.githubusercontent.com/19161994/128931317-e00b05cb-eb83-41d6-8c09-9c23b2c5c5ee.png">

Though not what we expected at all, these breeds do seem to match the input photo in interesting ways. The ears on the Irish Terrier look very similar to Lenny's ears, and one can't deny that the smile Schipperke looks familiar as well.

## App demo
To upload a photo of your dog (or your friend) and see who their doggy parents are, follow the installation instructions above and run our locally-hosted Streamlit app: 

<img width="900" src="https://user-images.githubusercontent.com/19161994/129119476-96715a7f-a3b7-4a8a-b4b8-a76fda44ee4d.png">

This project was done for the course MSDS 631 - Deep Learning, in partial completion of the Masters in Data Science degree program at the University of San Francisco.

Contributors: 
Lucia Page-Harley 
Joshua Majano 
