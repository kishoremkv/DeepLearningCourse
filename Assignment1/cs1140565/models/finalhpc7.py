import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.init as nninit

from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Scale, Compose, CenterCrop, RandomCrop, RandomHorizontalFlip
from torch.utils.data import DataLoader

my_transform1 = Compose([Scale(256),CenterCrop(256), RandomCrop(224),RandomHorizontalFlip(),ToTensor()])

train_data1 = ImageFolder(root='data/train', transform=my_transform1 )
validation_data = ImageFolder(root='data/validation', transform=my_transform1 )


#33870 images in train
#4881 images in validation
#4814 test images 
batch_size = 128
num_epochs = 84


train_loader1 = torch.utils.data.DataLoader(dataset=train_data1, 
                                           batch_size=batch_size, 
                                           shuffle=True)


validation_loader = torch.utils.data.DataLoader(dataset=validation_data, 
                                          batch_size=batch_size, 
                                          shuffle=True)

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()     
        self.cnn1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0)
        #self.batchnorm1 = BatchNorm2d(128*96*55*55, eps=1e-05, affine=True)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=3,stride=2)
        self.cnn2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2,groups=2)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=3,stride=2)
        self.cnn3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.cnn4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1,groups=2)
        self.relu4 = nn.ReLU()
        self.cnn5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=2,groups=2)
        self.relu5 = nn.ReLU()
        self.maxpool5 = nn.MaxPool2d(kernel_size=3,stride=2)
        self.fc6 = nn.Linear(256 * 6 * 6, 256) 
        self.relu6 = nn.ReLU()
        self.fc7 = nn.Linear(256, 128) 
        self.relu7 = nn.ReLU()
        self.fc8 = nn.Linear(128, 35)
        nninit.normal(self.cnn1.weight, mean=0, std=0.01)
        nninit.normal(self.cnn2.weight, mean=0, std=0.01)
        nninit.normal(self.cnn3.weight, mean=0, std=0.01)
        nninit.normal(self.cnn4.weight, mean=0, std=0.01)
        nninit.normal(self.cnn5.weight, mean=0, std=0.01)
        nninit.normal(self.fc6.weight, mean=0, std=0.01)
        nninit.normal(self.fc7.weight, mean=0, std=0.01)
        nninit.normal(self.fc8.weight, mean=0, std=0.01)
        nninit.normal(self.cnn2.bias, 1)
        nninit.normal(self.cnn4.bias, 1)
        nninit.normal(self.cnn5.bias, 1)
        nninit.normal(self.fc6.bias, 1)
        nninit.normal(self.fc7.bias, 1)
        nninit.normal(self.fc8.bias, 1)
        self.dp1 = nn.Dropout2d(p=0.5)
        self.dp2 = nn.Dropout2d(p=0.5)
    def forward(self, x):
        out = self.cnn1(x)
        out = self.relu1(out)
        out = self.maxpool1(out)
        out = self.cnn2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)
        out = self.cnn3(out)
        out = self.relu3(out)
        out = self.cnn4(out)
        out = self.relu4(out)
        out = self.cnn5(out)
        out = self.relu5(out)
        out = self.maxpool5(out)
        out = out.view(out.size(0), -1)
        out = self.fc6(out)
        out = self.relu6(out)
        out = self.dp1(out)
        out = self.fc7(out)
        out = self.relu7(out)
        out = self.dp1(out)
        out = self.fc8(out)          
        return out



model = CNNModel()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',factor=0.1)

if torch.cuda.is_available():
    model.cuda()




def takeone_epochstep(trainloader):
    for i, (images, labels) in enumerate(trainloader):
        if torch.cuda.is_available():
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())
        else:
            images = Variable(images)
            labels = Variable(labels)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    return loss


def checkaccuracy(myloader):
    correct = 0
    total = 0
    for images, labels in myloader:
        if torch.cuda.is_available():
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())
        else:
            images = Variable(images)
            labels = Variable(labels)
        outputs = model(images)
        predicted_values, predicted_index = torch.max(outputs.data, 1)
        total += labels.size(0)
        if torch.cuda.is_available():
            predicted_index = Variable(predicted_index.cuda())
        else:
            predicted_index = Variable(predicted_index)
        correct += (predicted_index == labels).sum()
    accuracy = 100.0 * correct.data[0]/ total
    return accuracy


for epoch in range(0,num_epochs):    
    loss=takeone_epochstep(train_loader1)
    trainacc = checkaccuracy(train_loader1)
    accuracy = checkaccuracy(validation_loader)
    scheduler.step(accuracy)
    print('Epoch {} Loss {} TrainAccuracy {} ValAccuracy {}',epoch+1,loss.data[0],trainacc,accuracy)

torch.save(model.state_dict(), 'newmodel7.pt')
