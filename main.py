'''
@ Jana Rasras
Train CNN to classify images with Pytorch

'''

# Import libraries
import numpy as np
from matplotlib import pyplot
from torch import nn # torch for nn and torchvision for datasets
from torchvision import datasets, transforms, utils #datasets for mnist # transforms for preprocessing # utils for making grid images
from torch.utils.data import DataLoader


# define classes

# Define functions
def load_data(path, batch_size = 32):
    ''' load datasets from torchvision and preprocess it 
        Inputs: 
        Path : where to save dataset
        batch_size : ideally all imgs at once
    '''
    # Apply these transforms to dataset
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean = [0.5], std=[0.5])  ]) # preprocessing
    trainDS = datasets.FashionMNIST(path, download=True, train=True, transform = tf)
    testDS =  datasets.FashionMNIST(path, download=True, train= False, transform = tf) 


    trainLoader = DataLoader(trainDS, batch_size = batch_size, shuffle=True)
    testLoader =  DataLoader(testDS,batch_size = batch_size, shuffle=True)

    # target labels = 
    return trainLoader, testLoader

def build_nn_model():
    ''' build NN'''
    pass

def train_nn(model, loader):
    ''' train model on data. return trained model and accuracy . '''
    
    for i in range(10): # train 10 times on all dataset 
        for batch in loader: # this is the actual images(batch)
            pass # do training here on the batch

        # once loop finishes, we say we trained on 1 ephoc 

def test_nn(model, loader):
    ''' test trained model on new data'''
    for batch in loader:
        pass

def plot_image_batch(batch):                                # batch: a small sample of dataset to train the model faster on smaller dataset. 
    ''' plot sample of images as 8*4 grid '''     #ephocs: train the model using all batches.

    grid = utils.make_grid(batch, nrow = 8, normalize = True)
    pyplot.imshow (np.transpose (grid, (1,2,0))) # re_arrange to(width,hight,channle)
    pyplot.axis('off')
    pyplot.show()


# main programm
def main():
    '''main function '''

    # load the data (as iterator not images)
    train_loader, test_loader = load_data('./data', batch_size =32) # loader is an iterator (function) that return a batch of images 

    # build NN
    ''' build CNN'''
    model = build_nn_model()

    ## Train NN
    trainedModel, accuracy = train_nn(model, train_loader)

    # test NN model on the new data
    test_accuracy = test_nn(trainedModel, test_loader)



if __name__ == "__main__":
    #main()
    train_loader, test_loader = load_data('./data', batch_size =32) # loader is an iterator (function) that return a batch of images 
    batch , label = next(iter(train_loader))
    plot_image_batch(batch)