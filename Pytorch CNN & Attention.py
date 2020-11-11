#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing the libraries
import pandas as pd
import numpy as np
import os


# for creating validation set
from sklearn.model_selection import train_test_split

# for evaluating the model
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# PyTorch libraries and modules
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.utils.data import Dataset, DataLoader

#Transformers
from module import *


# In[2]:


# Load training data set (small amount to test if it works first)

xtrain = r"data/training_sample_NoSparse.csv.gz"
ytrain = r"data/training_label_NoSparse.csv.gz"
xtest = r"data/testing_sample_NoSparse.csv.gz"
ytest = r"data/testing_label_NoSparse.csv.gz"


# In[3]:


#check balance
samplesdf = pd.DataFrame()
for df in  pd.read_csv(ytrain,compression ="gzip",delimiter=',', chunksize = 10000, header=0):
    samplesdf = samplesdf.append(df)
y_train = samplesdf.to_numpy()

num0 = 0
num1 = 0
for x in y_train:
    if x == 0:
        num0 = num0 + 1
    else:
        num1 = num1 + 1
num0,num1 #checking if it is balanced


# In[4]:


df = pd.read_csv(xtrain,compression ="gzip",delimiter=',', nrows=123, header=0)
x_train = df.to_numpy()
print(x_train)


# In[5]:


class Dataset(Dataset):

    def __init__(self,samples,labels,numrows):

        self.data = pd.read_csv(samples,compression ="gzip",delimiter=',', nrows = numrows, header=0)
        self.label = pd.read_csv(labels,compression ="gzip",delimiter=',', nrows = numrows, header=0)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        rna = self.data.iloc[idx]
        seplb = self.label.iloc[idx]
        rna = np.array([rna])
        seplb = np.array([seplb])
        rna = rna.astype('double').reshape(-1,3273)
        i = 0
        d = np.zeros((len(seplb),2))
        for x in seplb:
            if x == 0:
                d[i] = [1,0]
            else:
                d[i] = [0,1]
        sample = {'rna': rna, 'label': d}

        return sample


# In[6]:


train_dataset = Dataset(samples=xtrain,labels=ytrain,numrows = 29255)
test_dataset = Dataset(samples = xtest,labels = ytest,numrows = 3241)


# In[7]:


# Hyper Parameters
EPOCH = 10             # train the training data n times, to save time, we just train 1 epoch
LR = 0.0001              # learning rate
batch_size = 32
wd = LR / EPOCH


# In[8]:


train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# In[9]:


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (x,1, 8949)
            nn.Conv1d(
                in_channels=1,              # input height
                out_channels=4,            # n_filters
                kernel_size=3,              # filter size
                stride=1,                   # filter movement/step
                padding=1,                  # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (x,64,8949)
            nn.BatchNorm1d(4),
            nn.ReLU(),                      # activation
        )
        self.conv2 = nn.Sequential(         # input shape (x,64, 8949)
            nn.Conv1d(4,4,3,1,1),            
            nn.ReLU(),  
            nn.Dropout(p=0.1),
            nn.MaxPool1d(kernel_size =2, stride=2,ceil_mode = True),                # output shape (x,64,4478)
        )
        self.conv3 = nn.Sequential(         # input shape (x,64,4478)
            nn.Conv1d(4, 4, 3, 1, 1),     # output shape (x,128,4478)
            nn.ReLU(),                      # activation
            nn.MaxPool1d(kernel_size =2, stride=2,ceil_mode = True),                # output shape (x,128,2238)
        )
        self.out = nn.Linear(3276 , 2)   # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output  


# cnn = CNN()
# optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
# loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted
# if torch.cuda.is_available():
#     cnn = cnn.cuda()
#     loss_func = loss_func.cuda()
# cnn = cnn.double()    
# print(cnn)

# In[10]:


class SetTransformer(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output,
            num_inds=32, dim_hidden=128, num_heads=2, ln=False):
        super(SetTransformer, self).__init__()
        self.enc = nn.Sequential(
                ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
                ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln))
        self.dec = nn.Sequential(
                PMA(dim_hidden, num_heads, num_outputs, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                nn.Linear(dim_hidden, dim_output))

    def forward(self, X):
        return self.dec(self.enc(X))


# In[19]:


trans = SetTransformer(3273,2,1)
optimizer = torch.optim.Adam(trans.parameters(), lr=LR)   
loss_func = nn.L1Loss()                      
if torch.cuda.is_available():
    loss_func = loss_func.cuda()
    trans = trans.cuda()
trans = trans.double()
print(trans)


# In[20]:


class CNN_Trans(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output,
            num_inds=32, dim_hidden=128, num_heads=4, ln=False):
        super(CNN_Trans, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (x,1, 8949)
            nn.Conv1d(
                in_channels=1,              # input height
                out_channels=4,            # n_filters
                kernel_size=3,              # filter size
                stride=1,                   # filter movement/step
                padding=1,                  # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (x,64,8949)
            nn.BatchNorm1d(4),
            nn.ReLU(),                      # activation
        )
        self.conv2 = nn.Sequential(         # input shape (x,64, 8949)
            nn.Conv1d(4,4,3,1,1),            
            nn.ReLU(),  
            nn.Dropout(p=0.2),
            nn.MaxPool1d(kernel_size =2, stride=2,ceil_mode = True),                # output shape (x,64,4478)
        )
        self.conv3 = nn.Sequential(         # input shape (x,64,4478)
            nn.Conv1d(4, 4, 3, 1, 1),     # output shape (x,128,4478)
            nn.ReLU(),                      # activation
            nn.MaxPool1d(kernel_size =2, stride=2,ceil_mode = True),                # output shape (x,128,2238)
        )
        self.enc = nn.Sequential(
            SAB(dim_in=1, dim_out=64, num_heads=4),
            SAB(dim_in=64, dim_out=64, num_heads=4),
        )
        self.dec = nn.Sequential(
            PMA(dim=64, num_heads=4, num_seeds=1),
            nn.Linear(in_features=64, out_features=2),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = x.reshape(-1,3276,1)
        return self.dec(self.enc(x))


# cnntrans = CNN_Trans(3276,1,2)
# optimizer = torch.optim.Adam(cnntrans.parameters(), lr=LR)   
# loss_func = nn.L1Loss()                      
# if torch.cuda.is_available():
#     loss_func = loss_func.cuda()
#     cnntrans = cnntrans.cuda()
# cnntrans = cnntrans.double()
# print(cnntrans)

# In[21]:


class SmallSetTransformer(nn.Module):
    def __init__(self,):
        super().__init__()
        self.enc = nn.Sequential(
            SAB(dim_in=1, dim_out=64, num_heads=4),
            SAB(dim_in=64, dim_out=64, num_heads=4),
        )
        self.dec = nn.Sequential(
            PMA(dim=64, num_heads=4, num_seeds=1),
            nn.Linear(in_features=64, out_features=2),
        )

    def forward(self, x):
        x = self.enc(x)
        x = self.dec(x)
        return x.squeeze(-1)


# smalltrans = SmallSetTransformer()
# optimizer = torch.optim.Adam(smalltrans.parameters(), lr=LR)   
# loss_func = nn.L1Loss()                      
# if torch.cuda.is_available():
#     loss_func = loss_func.cuda()
#     smalltrans = smalltrans.cuda()
# smalltrans = smalltrans.double()
# print(smalltrans)

# In[22]:


def test(model):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for batch in train_loader:
            rna,labels = batch["rna"], batch["label"] 
            if torch.cuda.is_available():
                rna = rna.cuda()
                labels = labels.cuda()
            labels = labels.reshape(-1,2)
            outputs = model(rna)
            outputs = outputs.reshape(-1,2)
            _, predicted = torch.max(outputs.data, 1)
            _, labels = torch.max(labels.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
           # print(total,correct)

        print('Train Accuracy of the model on the train rna: {} %'.format((correct / total) * 100))
    with torch.no_grad():
        correct = 0
        total = 0
        for batch in test_loader:
            rna,labels = batch["rna"], batch["label"] 
            if torch.cuda.is_available():
                rna = rna.cuda()
                labels = labels.cuda()
            labels = labels.reshape(-1,2)
            outputs = model(rna)
            outputs = outputs.reshape(-1,2)
            _, predicted = torch.max(outputs.data, 1)
            _, labels = torch.max(labels.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    #print(total,correct)

        print('Test Accuracy of the model on the test rna: {} %'.format((correct / total) * 100))


# In[23]:


def train(model): 
    total_step = len(train_loader)
    num_epoch = EPOCH
    for epoch in range(num_epoch):
        model.train()
        for i, batch in enumerate(train_loader):
            # Run the forward pass
            rna,labels = batch["rna"], batch["label"] 
            if torch.cuda.is_available():
                rna = rna.cuda()
                labels = labels.cuda()
            outputs = model(rna)
            labels = labels.long()
            #print(outputs)
            outputs = outputs.reshape(-1,2)
            labels = labels.reshape(-1,2)
            loss = loss_func(outputs, labels)
            train_losses.append(loss.item())

            # Backprop and perform Adam optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track the accuracy
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            _, labels = torch.max(labels.data, 1)
            correct = (predicted == labels).sum().item()
            train_acc.append(correct / total)
            if i % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                        .format(epoch + 1, num_epoch , i + 1, total_step, loss.item(),
                                (correct / total) * 100))
        test(model)


# In[ ]:


train_losses = []
train_acc = []
test_acc = []
train(trans)


# In[ ]:


plt.plot(train_losses, label='Training loss')
plt.plot(train_acc, label='Training accuracy')
plt.legend()
plt.show()


# In[ ]:



#plt.plot(train_acc[200:], label='Training accuracy')
plt.legend()
plt.show()


# In[ ]:


# Save the model and plot
torch.save(model.state_dict(), MODEL_STORE_PATH + 'conv_net_model.ckpt')


# In[ ]:




