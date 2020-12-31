import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import colorConverter, ListedColormap
import torch
import torchvision
from torchvision import models
import random
import sys
from torch.utils.data import SubsetRandomSampler
from sklearn.metrics import classification_report
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

batch_size_train = 32
batch_size_test = 32
learning_rate = float(sys.argv[3])#1e-3
momentum = float(sys.argv[6])
wd = float(sys.argv[7])
log_interval = 10

torch.backends.cudnn.deterministic = True
random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)

imb_a = int(sys.argv[4])
imb_b = int(sys.argv[5])
import itertools

# Function gets a new train_set and gets a new set of random indicies to noisify
def get_data():

    train_set = torchvision.datasets.CIFAR10('./../../files/', train=True, download=False,
                            transform=torchvision.transforms.Compose([
                                torchvision.transforms.RandomCrop(32, padding=4),
                                torchvision.transforms.RandomHorizontalFlip(),
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]))

    noisy_prob = float(sys.argv[1])
    noisy_indices = list(range(len(train_set.targets)))
    random.shuffle(noisy_indices)
    noisy_indices = noisy_indices[:int(noisy_prob*len(noisy_indices))]

    for idx in noisy_indices:
        label = train_set.targets[idx]
        choices = list(range(10))
        choices.remove(label)
        new_label = np.random.choice(choices)
        train_set.targets[idx] = int(new_label) #torch.LongTensor([new_label])

    train_subset_loader  = torch.utils.data.DataLoader(train_set, batch_size=batch_size_train,drop_last=False)
    
    test_set = torchvision.datasets.CIFAR10('./../../files/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                             ]))

    test_subset_loader = torch.utils.data.DataLoader(test_set,batch_size=batch_size_test)
    return train_subset_loader, test_subset_loader

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

class Model(nn.Module):

    def __init__(self):
        super(Model,self).__init__()
        # Block 1
        self.conv1 = nn.Conv2d(3,32,(3,3))
        self.batch1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32,64,(3,3))
        self.batch2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d((4,4),stride=(2,2))
        self.conv3 = nn.Conv2d(64,64,(3,3))
        self.batch3 = nn.BatchNorm2d(64)
        self.linear1 = nn.Linear(7744,128)
        self.linear2 = nn.Linear(128,10)

    def forward(self,x):
        # Block
        x = self.conv1(x)
        x = self.batch1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.batch2(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.batch3(x)
        x = F.relu(x)
        x = x.view(x.size()[0],-1)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = torch.sigmoid(x)
        return x


max_epoch = 120

def my_loss(output,target):
    loss = torch.mean((output-target)**2)
    return loss

def log_loss(output,target):
    loss = torch.sum(-target*torch.nn.functional.log_softmax(output))
    loss = loss.sum()
    return loss

my_alpha = float(sys.argv[2])
def alpha_loss(output,target):
    loss = 0
    if my_alpha == 1.0:
        loss = torch.mean(torch.sum(-target*torch.log(torch.softmax(output,dim=1) + 1e-8),dim=1))
    else:
        alpha = torch.FloatTensor([my_alpha]).cuda()
        one = torch.FloatTensor([1.0]).cuda()
        loss = (alpha/(alpha-one))*torch.mean(torch.sum(target*(one - torch.softmax(output,dim=1).pow(one - (one/alpha))),dim=1))
    return loss

import time
experiments = 5
cm_average = np.zeros((2,2))
from sklearn.metrics import matthews_corrcoef
for experiment in range(experiments):
    train_loader, test_loader = get_data()
    network = Model().cuda()
    optimizer = optim.SGD(network.parameters(), lr=learning_rate)
    epoch_train_loss = []
    scheduler = MultiStepLR(optimizer, milestones=[10,30], gamma=0.1)
    for epoch in range(max_epoch):
        train_loss = []
        network.train()
        start = time.time()
        for batch_idx, (data, target) in enumerate(train_loader):
            target_hot = torch.FloatTensor(torch.zeros((target.unsqueeze(1).size()[0],10)).scatter(1,target.unsqueeze(1),1.0)).cuda()
            data = data.cuda()
            output = network(data)
            optimizer.zero_grad()
            loss = alpha_loss(output,target_hot)
            train_loss.append(loss.cpu().item())
            loss.backward()
            optimizer.step()
        #scheduler.step()

    # Testing
    test_loss = []
    test_total, test_acc = 0, 0
    predictions, actual = [], []
    network.eval()
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import matthews_corrcoef
    for batch_idx, (data, target) in enumerate(test_loader):
        target_hot = torch.FloatTensor(torch.zeros((target.unsqueeze(1).size()[0],10)).scatter(1,target.unsqueeze(1),1.0)).cuda()
        data = data.cuda()
        output = network(data)
        _,test_pred = torch.max(output,dim=1)
        test_total += target.size()[0]
        predictions += test_pred.cpu().tolist()
        test_acc += (target == test_pred.cpu()).sum().item()
        t_loss = alpha_loss(output,target_hot)
        test_loss.append(t_loss.cpu().item())
        actual += target.tolist()
    print('Test Accuracy:\t' + str((test_acc*100)/test_total))
