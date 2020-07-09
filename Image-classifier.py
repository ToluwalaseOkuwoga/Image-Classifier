#!/usr/bin/env python
# coding: utf-8

# # Classifying images of everyday objects using a neural network
# 

# In[74]:


get_ipython().system('conda install numpy pandas pytorch torchvision cpuonly -c pytorch -y')
get_ipython().system('pip install matplotlib --upgrade --quiet')


# In[3]:


import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


project_name = 'Image-classifier'


# ## Exploring the CIFAR10 dataset

# In[5]:


dataset = CIFAR10(root='data/', download=True, transform=ToTensor())
test_dataset = CIFAR10(root='data/', train=False, transform=ToTensor())


# ##### How many images does the training dataset contain?

# In[6]:


dataset_size = len(dataset)
dataset_size


# ##### How many images does the training dataset contain?

# In[7]:


test_dataset_size = len(test_dataset)
test_dataset_size


# ##### How many output classes does the dataset contain?

# In[8]:


classes = dataset.classes
classes


# In[9]:


num_classes = len(dataset.classes)
num_classes


# ##### What is the shape of an image tensor from the dataset?

# In[10]:


img, label = dataset[0]
img_shape = img.shape
img_shape


# Note that this dataset consists of 3-channel color images (RGB). Let us look at a sample image from the dataset. `matplotlib` expects channels to be the last dimension of the image tensors (whereas in PyTorch they are the first dimension), so we'll the `.permute` tensor method to shift channels to the last dimension. Let's also print the label for the image.

# In[11]:


img, label = dataset[0]
plt.imshow(img.permute((1, 2, 0)))
print('Label (numeric):', label)
print('Label (textual):', classes[label])


# #####  Determine the number of images belonging to each class

# In[43]:


x = []
for i in range(dataset_size):
    x.append(dataset[i][1])
uimg = torch.tensor(x).unique(sorted=True)
uimg_count = torch.stack([(torch.tensor(x)==i).sum() for i in uimg])
for i in range(len(uimg)):
    print(f'{classes[i]}: {uimg_count[i].item()}')   


# ## Preparing the data for training
# 
# We'll use a validation set with 5000 images (10% of the dataset). To ensure we get the same validation set each time, we'll set PyTorch's random number generator to a seed value of 43.

# In[18]:


torch.manual_seed(43)
val_size = 5000
train_size = len(dataset) - val_size


# Let's use the `random_split` method to create the training & validation sets

# In[19]:


train_ds, val_ds = random_split(dataset, [train_size, val_size])
len(train_ds), len(val_ds)


# We can now create data loaders to load the data in batches.

# In[20]:


batch_size=128


# In[21]:


train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size*2, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size*2, num_workers=4, pin_memory=True)


# Let's visualize a batch of data using the `make_grid` helper function from Torchvision.

# In[22]:


for images, _ in train_loader:
    print('images.shape:', images.shape)
    plt.figure(figsize=(16,8))
    plt.axis('off')
    plt.imshow(make_grid(images, nrow=16).permute((1, 2, 0)))
    break


# ## Base Model class & Training on GPU
# 
# Let's create a base model class, which contains everything except the model architecture i.e. it wil not contain the `__init__` and `__forward__` methods. We will later extend this class to try out different architectures. In fact, this model can be extended to solve any image classification problem.

# In[23]:


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


# In[24]:


class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))


# In[25]:


def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase 
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result)
        history.append(result)
    return history


# Finally, let's also define some utilities for moving out data & labels to the GPU, if one is available.

# In[26]:


torch.cuda.is_available()


# In[27]:


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


# In[28]:


device = get_default_device()
device


# In[29]:


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


# Let us also define a couple of helper functions for plotting the losses & accuracies.

# In[30]:


def plot_losses(history):
    losses = [x['val_loss'] for x in history]
    plt.plot(losses, '-x')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss vs. No. of epochs');


# In[31]:


def plot_accuracies(history):
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs');


# Let's move our data loaders to the appropriate device.

# In[32]:


train_loader = DeviceDataLoader(train_loader, device)
val_loader = DeviceDataLoader(val_loader, device)
test_loader = DeviceDataLoader(test_loader, device)


# ## Training the model
# 
# We will make several attempts at training the model. Each time, try a different architecture and a different set of learning rates. Here are some ideas to try:
# - Increase or decrease the number of hidden layers
# - Increase of decrease the size of each hidden layer
# - Try different activation functions
# - Try training for different number of epochs
# - Try different learning rates in every epoch

# In[33]:


input_size = 3*32*32
output_size = 10


# 
# ##### Extending the `ImageClassificationBase` class to complete the model definition.

# In[55]:


class CIFAR10Model(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 32)
        self.fc4 = nn.Linear(32, output_size)
        
    def forward(self, xb):
        # Flatten images into vectors
        out = xb.view(xb.size(0), -1)
        # Apply layers & activation functions
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        out = F.relu(out)
        out = self.fc4(out)
        return out


# You can now instantiate the model, and move it the appropriate device.

# In[56]:


model = to_device(CIFAR10Model(), device)


# Before training the model, it's a good idea to check the validation loss & accuracy with the initial set of weights.

# In[57]:


history = [evaluate(model, val_loader)]
history


# ##### Training the model using the `fit` function to reduce the validation loss & improve accuracy.

# In[58]:


history += fit(10, 0.1, model, train_loader, val_loader)


# In[59]:


history += fit(5, 0.01, model, train_loader, val_loader)


# In[60]:


history += fit(5, 0.001, model, train_loader, val_loader)


# Plot the losses and the accuracies to check if you're starting to hit the limits of how well your model can perform on this dataset. You can train some more if you can see the scope for further improvement.

# In[61]:


plot_losses(history)


# In[62]:


plot_accuracies(history)


# Finally, evaluate the model on the test dataset report its final performance.

# In[63]:


evaluate(model, test_loader)


# ## Recoding your results
# 
# As your perform multiple experiments, it's important to record the results in a systematic fashion, so that you can review them later and identify the best approaches that you might want to reproduce or build upon later. 
# 
# ##### Describing the model's architecture.

# In[64]:


arch = "4 layers (1024, 256, 32, 10)"


# ##### List of learning rates used while training.

# In[65]:


lrs = [0.1,0.01,0.001]


# ##### List of no. of epochs used while training.

# In[66]:


epochs = [10,5,5]


# ##### What were the final test accuracy & test loss?

# In[68]:


test_acc = 0.529296875
test_loss = 1.3336061239242554


# Finally, let's save the trained model weights to disk, so we can use this model later.

# In[69]:


torch.save(model.state_dict(), 'Image-classifier.pth')


# In[ ]:




