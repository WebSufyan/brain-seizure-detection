import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from GRU_model import GRU

from tqdm import tqdm



x_train = pd.read_csv('x_train.csv')
x_test = pd.read_csv('x_test.csv')

y_train = pd.read_csv('y_train.csv')
y_test = pd.read_csv('y_test.csv')


# transform dataframe to numpy array then to torch tensors then feed it to tensordataset
# to combine samples with labels and then feed all of it to dataloader
train = TensorDataset(torch.from_numpy(x_train.values), torch.from_numpy(y_train.values))
test = TensorDataset(torch.from_numpy(x_test.values), torch.from_numpy(y_test.values))

# dataloader will split the data into batches fro every batch will be 16 samples
train_loader = DataLoader(train, batch_size=16)
test_loader = DataLoader(test, batch_size=16)



# hyper parameters for the model
input_size = 1
output_size = 1
hidden_size = 8
num_layers = 1


model = GRU(input_size, hidden_size, output_size, num_layers)

torch.manual_seed(101)

optimizer = torch.optim.Adam(model.parameters(), lr = 0.002)
criterion = nn.BCELoss()

epochs = 4
losses = []
test_losses = []


train_corrects = []
test_corrects = []

result_thresh = 0.50

for epoch in range(epochs):
    train_correct = 0 # reset train and test correct values
    test_correct = 0
    model.train # model training mode
    for idx, (i, j) in enumerate(train_loader):
        # i is of shape [batch_size, sequence]
        # .unsqueeze will add one dimention at the end to be input size because the model expects 3 dimensions
        # after adding the one dimention data will be of shape [batch size, sequence length, number of features]
        data = (i.unsqueeze(2)).float() # since our model expects tensors to be float type
        y_true = j.float() # cost function expects float too

        y_pred = model(data)
        cost = criterion(y_pred, y_true) # calculate the loss 

        optimizer.zero_grad()
        cost.backward() # calculate the derivatives startig from the loss up to the first layer of the model
        optimizer.step()
        
        # count how many correct samples for each batch and then accumulate the result until
        # the length of dataset has been reached
        y_cap = torch.where(y_pred > result_thresh, 1, 0)
        correct = (y_cap == y_true).sum()
        
        train_correct += correct.item()
    
    # append correct samples each batch
    train_corrects.append(train_correct)
    losses.append(cost.item())

# evaluation after training on entire dataset
    model.eval # model evaluation mode
    for idx2, (k, l) in enumerate(test_loader):
        with torch.no_grad():
            test_data = (k.unsqueeze(2)).float()
            test_label = (l).float()
            
            y_test = model(test_data)
        test_cost = criterion(y_test, test_label)
            
        test_y_cap = torch.where(y_test > result_thresh, 1, 0)
        test_corr = (test_y_cap == test_label).sum()
        
        test_correct += test_corr.item()

    test_corrects.append(test_correct)
    test_losses.append(test_cost.item())

    if epoch % 2 == 0: # print every two epochs
        print(f'train cost is: {cost.item()}.\n \
              test cost is: {test_cost.item()}.\n \
              train accuracy is: {(train_correct / 1870)*100:.2f}.\n \
              test accuracy is: {(test_correct / 330)*100:.2f}.')


# plot the training and evaluation losses
plt.plot(losses)
plt.plot(test_losses)

''' 
model train cost is: 0.2225491851568222.
model test cost is: 0.27880793809890747.
train accuracy is: 91.07 %.
test accuracy is: 90.00 %.
'''
# torch.save(model.state_dict(), 'brain_seizure_detection.pt')








