import torch
import pandas as pd
import numpy as np
import random
import plotly.graph_objects as go

from torch import nn
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
from torchvision import datasets
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


test_df = pd.read_csv('test.csv')
train_df = pd.read_csv('train.csv')
Y = train_df.label.values
X = train_df.loc[:, train_df.columns != 'label'].values / 255
X_test = test_df.values / 255

train_x, val_x, train_y, val_y = train_test_split(X, Y, test_size = 0.2, random_state = 42)

trainTorch_x = torch.from_numpy(train_x).type(torch.FloatTensor)
trainTorch_y = torch.from_numpy(train_y).type(torch.LongTensor)


valTorch_x = torch.from_numpy(val_x).type(torch.FloatTensor)
valTorch_y = torch.from_numpy(val_y).type(torch.LongTensor)

testTorch_x = torch.from_numpy(np.array(X_test)).type(torch.FloatTensor)

train = torch.utils.data.TensorDataset(trainTorch_x, trainTorch_y)
val = torch.utils.data.TensorDataset(valTorch_x, valTorch_y)
test = torch.utils.data.TensorDataset(testTorch_x)

train_loader = DataLoader(train, batch_size = 100, shuffle = False)
val_loader = DataLoader(val, batch_size = 100, shuffle = False)
test_loader = DataLoader(test, batch_size = 100, shuffle = False)


class mnist_net2(nn.Module):
    def __init__(self):
        super(mnist_net2, self).__init__()
        self.c1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3,3), stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.c11 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3,3), stride=1, padding=0)
        self.relu11 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2,2))
        self.dropout1 = nn.Dropout(0.2)
        self.c2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3), stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2,2))
        self.dropout2 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(800, 256)
        self.dropout3 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):

        output = self.c1(x)
        output = self.relu1(output)
        output = self.c11(output)
        output = self.relu11(output)
        output = self.maxpool1(output)
        output = self.dropout1(output)
        output = self.c2(output)
        output = self.relu2(output)
        output = self.maxpool2(output)
        output = self.dropout2(output)
        output = output.view(output.size(0), -1)
        output = self.fc1(output)
        output = self.dropout3(output)
        output = self.fc2(output)
        return output

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = mnist_net2()
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, weight_decay = 0.0002)
criterion = nn.CrossEntropyLoss()


from torch.autograd import Variable
model.train()
n_iterations = 0
print_every = 50
steps = 0
train_losses, val_losses = [], []
total_epochs = 50
for epoch_number in range(total_epochs):
    running_loss = 0
    for i, (images, labels) in enumerate(train_loader):
        steps += 1
        data, target = Variable(images.view(100,1,28,28)), Variable(labels)
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()
        optimizer.zero_grad()
        output = model(data)
        model.zero_grad()
        loss = criterion(output, target)

        n_iterations += 1

        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        if steps % print_every == 0:
            val_loss = 0
            accuracy = 0


            # Turn off gradients for validation
            with torch.no_grad():
                model.eval()
                for images, labels in val_loader:
                    data, target = Variable(images.view(100,1,28,28), volatile=True), Variable(labels)

                    if torch.cuda.is_available():
                        data = data.cuda()
                        target = target.cuda()

                    log_ps = model(data)
                    loss = criterion(log_ps, target)
                    val_loss += loss.item()

                    ps = torch.exp(log_ps)

                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == target.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor))

            model.train()

            train_losses.append(running_loss/len(train_loader))
            val_losses.append(val_loss / len(val_loader))

    print("Epoch: {}/{}.. ".format(epoch_number + 1, total_epochs),
                  "Training Loss: {:.3f}.. ".format(train_losses[-1]),
                  "Val Loss: {:.3f}.. ".format(val_losses[-1]),
                  "Val Accuracy: {:.3f}".format(accuracy/len(val_loader)))

fig = go.Figure()

fig.add_trace(go.Scatter(y = np.array(train_losses),
                    mode='lines+markers',
                    name='Training loss'))
fig.add_trace(go.Scatter(y = np.array(val_losses),
                    mode='lines+markers',
                    name='Validation loss'))

fig.update_layout(title_text = 'Loss of model', xaxis = dict(tickmode = 'linear', dtick = 1))
fig.update_xaxes(
    range=(1, total_epochs),
    constrain='domain'
)

fig.show()
