
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import librosa
import torch
import Dataloader
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from Dataloader import AudioDataset
import torch.nn as nn


# In[155]:


class AudioDataset(Dataset):
    def __init__(self, filename, directory, mfcc_features):
        self.filename = filename
        self.directory = directory
        self.features = mfcc_features
        self.file = pd.read_pickle(self.directory + self.filename)

    def __len__(self):
        return len(self.file)
    
    def __getitem__(self, idx):
        audio_file = librosa.feature.mfcc(self.file.iloc[idx, 0], n_mfcc=self.features)
        if self.file.iloc[idx, 2] == 'female':
            gender = np.array((1,0))
        else:
            gender = np.array((0,1))
        sample = {'audio' : audio_file, 'target' : gender}
        return sample 


# In[84]:


mfcc_features = 50
data = AudioDataset('segment_valid_train_2.p', '', mfcc_features=mfcc_features)


# In[130]:


batch_size = 128
dataloader = DataLoader(data, batch_size = batch_size, shuffle = True, drop_last=True, num_workers=2)


# In[131]:



class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv1d(44, 25, 3, stride=2),
                        nn.GLU(),
        )
        
        self.layer2 = nn.Sequential(
                        nn.Linear(300,50),
                        nn.ELU(),
                        nn.Linear(50,2),
                        nn.ELU()
        )
        
    def forward(self,x):
        x=self.layer1(x)
        x=x.view(batch_size,300)
        x=self.layer2(x)
        
        return x

en=Encoder()



# In[ ]:


class Encoder1(nn.Module):
    def __init__(self):
        super(Encoder1, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(1, 16, 3, stride=2),
                                    )


# In[132]:


criterion=nn.BCEWithLogitsLoss()
optimizer=torch.optim.Adam(en.parameters(),lr=0.00001)
epoch_loss_data=[]
total_epochs = 30


if torch.cuda.is_available():
    en = nn.DataParallel(en)
    en.cuda()
    
def Preprocess(matrix):
    for i in range(matrix.size()[0]):
        mean, std = matrix[i].mean(), matrix[i].std()
        matrix[i]-=mean
        matrix[i]/=std
    return matrix


# In[133]:


for epoch in range(total_epochs):
    loss=0.0
    print('Epoch',epoch)
    for i, element in enumerate(dataloader):
        sample = element
        if torch.cuda.is_available():
            inputs = Variable((Preprocess(sample['audio']).transpose(1,2)).type(torch.FloatTensor)).cuda()
            target = Variable(sample['target'].type(torch.FloatTensor)).cuda()
        else:        
            inputs = Variable((Preprocess(sample['audio']).transpose(1,2)).type(torch.FloatTensor))
            target = Variable(sample['target'].type(torch.FloatTensor))
        optimizer.zero_grad()

        output = en(inputs)

#        output = output.view(batch_size, 2)

        loss = criterion(output, target)

        loss.backward()

        optimizer.step()

    epoch_loss_data.append(loss.data[0])
            
    print('Loss for epoch {} is {}'.format(epoch + 1, loss.data[0]))


# In[156]:


test_data = AudioDataset('segment_valid_train_1.p', '', mfcc_features = 50)
batch_size = 32
testloader = DataLoader(test_data, num_workers=2, batch_size=batch_size, drop_last=True)


correct = 0
total = 0

for i, element in enumerate(testloader):
    sample = element
    if torch.cuda.is_available():
        inputs = Variable((Preprocess(sample['audio']).transpose(1,2)).type(torch.FloatTensor)).cuda()
        target = (sample['target'].type(torch.FloatTensor)).cuda()
    else:        
        inputs = Variable((Preprocess(sample['audio']).transpose(1,2)).type(torch.FloatTensor))
        target = (sample['target'].type(torch.FloatTensor))
    output = en(inputs)
    _, predicted = torch.max(output.data, 1)
    predicted = predicted.type(torch.FloatTensor)
    _1, given_target = torch.max(target, 1)
    total += target.size(0)
    correct += (predicted.type(torch.FloatTensor) == given_target.type(torch.FloatTensor)).sum()
    
print('Accuracy of the network on the p1 dataset: %d %%' % (
    100 * correct / total))


# In[144]:


test_data = AudioDataset('segment_valid_train_1.p', '', mfcc_features = 50)
batch_size = 32
testloader = DataLoader(test_data, num_workers=2, batch_size=batch_size, drop_last=True)



# In[148]:


plt.plot(np.arange(1, 31), epoch_loss_data[:])

