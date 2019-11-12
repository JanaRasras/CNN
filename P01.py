'''
@ Jana Rasras
Train CNN to classify images with Pytorch

'''

# Import libraries
import numpy as np
from matplotlib import pyplot as plt 
from torch import nn # torch for nn and torchvision for datasets
from torchvision import datasets, transoforms #datasets for mnist # transforms for preprocessing


# define classes

# Define functions
def load_data():
    ''' load datasets from torchvision and preprocess it '''
    pass

def build_nn_model():
    ''' build NN'''
    pass

def train_nn(model, data):
    ''' train model on data. return trained model and accuracy . '''
    pass

def test_nn():
    ''' test trained model on new data'''
    pass




# main programm
def main():
    '''main function '''

    # load the data
    train_data, test_Pdata = load_data(root ='/data')

    # build NN
    ''' build CNN'''
    model = build_nn_model()

    ## Train NN
    trainedModel, accuracy = train_nn(model, train_data)

    # test NN model on the new data
    test_accuracy = test_nn(trainedModel, test_data)



if __name__ == "__main__":
    main()
