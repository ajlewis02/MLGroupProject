#Load libraries
import os
import numpy as np
import torch
import glob
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable
import torchvision
import pathlib
import matplotlib.pyplot as plt
from torch.nn.utils import prune

#checking for device
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

#Transforms
transformer=transforms.Compose([
    transforms.Resize((180,180)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),  #0-255 to 0-1, numpy to tensors
    transforms.Normalize([0.5,0.5,0.5], # 0-1 to [-1,1] , formula (x-mean)/std
                        [0.5,0.5,0.5])
])

#Dataloader
#Path for training and testing directory
train_path='C:\THIS PC\WPI\FALL 2021\CS 539 Machine Learning\Project\dataverse_files\\flowers\\flowers\\flower_photos\\train'
validation_path = 'C:\THIS PC\WPI\FALL 2021\CS 539 Machine Learning\Project\dataverse_files\\flowers\\flowers\\flower_photos\\validation'
test_path='C:\THIS PC\WPI\FALL 2021\CS 539 Machine Learning\Project\dataverse_files\\flowers\\flowers\\flower_photos\\test'

train_loader=DataLoader(
    torchvision.datasets.ImageFolder(train_path,transform=transformer),
    batch_size=32, shuffle=True
)
validation_loader=DataLoader(
    torchvision.datasets.ImageFolder(validation_path,transform=transformer),
    batch_size=32, shuffle=True
)
test_loader=DataLoader(
    torchvision.datasets.ImageFolder(test_path,transform=transformer),
    batch_size=32, shuffle=True
)

#categories
root=pathlib.Path(train_path)
classes=sorted([j.name.split('/')[-1] for j in root.iterdir()])
print(classes)


# CNN Network
class ConvNet(nn.Module):
    def __init__(self, num_classes=6):
        super(ConvNet, self).__init__()

        # Output size after convolution filter
        # ((w-f+2P)/s) +1

        # Input shape= (32,3,180,180)

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        # Shape= (32,12,180,180)
        self.bn1 = nn.BatchNorm2d(num_features=12)
        # Shape= (32,12,180,180)
        self.relu1 = nn.ReLU()
        # Shape= (32,12,180,180)

        self.pool = nn.MaxPool2d(kernel_size=2)
        # Reduce the image size be factor 2
        # Shape= (32,12,90,90)

        self.conv2 = nn.Conv2d(in_channels=12, out_channels=20, kernel_size=3, stride=1, padding=1)
        # Shape= (32,20,90,90)
        self.relu2 = nn.ReLU()
        # Shape= (32,20,90,90)

        self.conv3 = nn.Conv2d(in_channels=20, out_channels=32, kernel_size=3, stride=1, padding=1)
        # Shape= (32,32,90,90)
        self.bn3 = nn.BatchNorm2d(num_features=32)
        # Shape= (32,32,90,90)
        self.relu3 = nn.ReLU()
        # Shape= (32,32,90,90)

        self.stack1 = nn.Linear(in_features=90 * 90 * 32, out_features=32)
        self.stack2 = nn.Linear(in_features=32, out_features=num_classes)

        self.relu_stack1 = nn.ReLU()
        self.tan_stack2 = nn.Tanh()

    # Feed forwad function
    def forward(self, input):
        output = self.conv1(input)
        output = self.bn1(output)
        output = self.relu1(output)

        output = self.pool(output)

        output = self.conv2(output)
        output = self.relu2(output)

        output = self.conv3(output)
        output = self.bn3(output)
        output = self.relu3(output)

        # Above output will be in matrix form, with shape (256,32,75,75)

        output = output.view(-1, 32 * 90 * 90)

        output = self.tan_stack2(self.stack2(self.relu_stack1(self.stack1(output))))

        return output

model = ConvNet(num_classes=5).to(device)

# Optmizer and loss function
optimizer = Adam(model.parameters(), lr=0.0001, weight_decay=0.00001)
loss_function = nn.CrossEntropyLoss()

num_epochs = 600
EPOCHS = np.arange(0,num_epochs,1)
# calculating the size of training and testing images

train_count = len(glob.glob(train_path + '/**/*.jpg'))
validation_count = len(glob.glob(validation_path + '/**/*.jpg'))
test_count = len(glob.glob(test_path + '/**/*.jpg'))

print(train_count, validation_count, test_count)

# Model training and saving best model

best_accuracy = 0.0
Train_Performance = np.zeros(num_epochs)
Validation_Performance = np.zeros(num_epochs)


for epoch in range(num_epochs):

    # Evaluation and training on training dataset
    model.train()
    train_accuracy = 0.0
    train_loss = 0.0

    for i, (images, labels) in enumerate(train_loader):
        if torch.cuda.is_available():
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

        optimizer.zero_grad()

        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.cpu().data * images.size(0)
        _, prediction = torch.max(outputs.data, 1)

        train_accuracy += int(torch.sum(prediction == labels.data))

    train_accuracy = train_accuracy / train_count
    train_loss = train_loss / train_count
    Train_Performance[epoch] = train_accuracy

    # Evaluation on validation dataset
    model.eval()

    validation_accuracy = 0.0


    for i, (images, labels) in enumerate(validation_loader):
        if torch.cuda.is_available():
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

        outputs = model(images)
        _, prediction = torch.max(outputs.data, 1)
        validation_accuracy += int(torch.sum(prediction == labels.data))

    validation_accuracy = validation_accuracy / validation_count

    Validation_Performance[epoch] = validation_accuracy

    print('Epoch: ' + str(epoch) + ' Train Loss: ' + str(train_loss) + ' Train Accuracy: ' + str(
        train_accuracy) + ' Validation Accuracy: ' + str(validation_accuracy))

    # Save the best model
    if validation_accuracy > best_accuracy:
        torch.save(model,'C:\THIS PC\WPI\FALL 2021\CS 539 Machine Learning\Project\Results\Pruning\\best_model.pth')
        best_accuracy = validation_accuracy

###### ONE SHOT PRUNING OF 30% LOWEST WEIGHTS IN THE FULLY CONNECTED NN
best_model = torch.load('C:\THIS PC\WPI\FALL 2021\CS 539 Machine Learning\Project\Results\Pruning\\best_model.pth')
to_prune = ((best_model.stack1, "weight"), (best_model.stack2, "weight"))
prune.global_unstructured(to_prune, pruning_method=prune.L1Unstructured, amount=0.3)
torch.save(model,'C:\THIS PC\WPI\FALL 2021\CS 539 Machine Learning\Project\Results\Pruning\\pruned_model.pth')

pruned_model = torch.load('C:\THIS PC\WPI\FALL 2021\CS 539 Machine Learning\Project\Results\Pruning\\pruned_model.pth')
pruned_model.eval()
test_accuracy = 0.0

for i, (images, labels) in enumerate(test_loader):
    if torch.cuda.is_available():
        images = Variable(images.cuda())
        labels = Variable(labels.cuda())

    outputs = pruned_model(images)
    _, prediction = torch.max(outputs.data, 1)
    test_accuracy += int(torch.sum(prediction == labels.data))

test_accuracy = test_accuracy / test_count

print('Best Validation Accuracy: ' + str(best_accuracy))
print('Test Accuracy: ' + str(test_accuracy))

plt.plot(EPOCHS,Validation_Performance, label="Validation Accuracy")
plt.plot(EPOCHS,Train_Performance, label="Training Accuracy")
plt.legend()
plt.title('Model Training (One Shot Pruning)')
plt.show()