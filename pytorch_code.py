# pylint: disable=no-member
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
from tqdm import tqdm
import time

from torch.autograd import Variable

BATCH_SIZE = 128
NUM_EPOCHS = 10
learning_rate = 0.02

#First, we read the mnist data, preprocess them and encapsulate them into dataloader form
#首先，我们读取mnist数据，对其进行预处理并将其封装到dataloader表单中
# preprocessing
# 预处理
normalize = transforms.Normalize(mean=[.5], std=[.5])
transform = transforms.Compose([transforms.ToTensor(), normalize])

# download and load the data
# 下载数据
train_dataset = torchvision.datasets.MNIST(root='./mnist/', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./mnist/', train=False, transform=transform, download=False)

print(train_dataset)

# encapsulate them into dataloader form
# 将它们封装到dataloader表单中
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

#Then, we define the model, object function and optimizer that we use to classify.
#然后，我们定义了用于分类的模型、对象函数和优化器。
class SimpleNet(nn.Module):
# TODO:define model
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(SimpleNet, self).__init__()
        self.layer1 = nn.Linear(in_dim, n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.layer3 = nn.Linear(n_hidden_2, out_dim)
 
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

model = SimpleNet(28*28,300,100,10)

if torch.cuda.is_available():
    model = model.cuda()

# TODO:define loss function and optimiter
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

#Next, we can start to train and evaluate!
#接下来，我们可以开始训练和评估！
for epoch in range(NUM_EPOCHS):
    for images, labels in train_loader:
        # TODO:forward + backward + optimize
        images = images.view(images.size(0), -1)
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
        else:
            images = Variable(images)
            labels = Variable(labels)
        out = model(images)
        loss = criterion(out, labels)
        print_loss = loss.data.item()
 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('.')

# evaluate
# TODO:calculate the accuracy using traning and testing dataset
# 模型评估
model.eval()
train_accuracy=0
test_accuracy=0
#calculate the train accuracy
#计算训练正确率
for images, labels in train_loader:
    images = images.view(images.size(0), -1)
    if torch.cuda.is_available():
        images = images.cuda()
        labels = labels.cuda()
 
    out = model(images)
    _, pred = torch.max(out, 1)
    num_correct = (pred == labels).sum()
    train_accuracy += num_correct.item()
train_accuracy/=(len(train_dataset))

#calculate the test accuracy
#计算测试正确率
for images, labels in test_loader:
    images = images.view(images.size(0), -1)
    if torch.cuda.is_available():
        images = images.cuda()
        labels = labels.cuda()
 
    out = model(images)
    _, pred = torch.max(out, 1)
    num_correct = (pred == labels).sum()
    test_accuracy += num_correct.item()
test_accuracy/=(len(test_dataset))

print('Training accuracy of torch: %0.2f%%' % (train_accuracy*100))
print('Testing accuracy of torch: %0.2f%%' % (test_accuracy*100))
