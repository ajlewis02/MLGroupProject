import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torch.nn as nn
from torch.nn.utils import prune

device = "cuda"


class PenguinsDataset(Dataset):
    def __init__(self, start, end):
        f = open("penguins_processed.txt")
        lines = f.readlines()
        linesplit = [lines[n].split("|") for n in range(start, end)]
        lab = [n[0] for n in linesplit]
        dat = [n[1] for n in linesplit]

        labsplit = [[float(m) for m in n.split(",")] for n in lab]
        datsplit = [[float(m) for m in n.split(",")] for n in dat]

        self.labels = labsplit
        self.data = datsplit

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], device=device).float(), torch.tensor(self.labels[idx],
                                                                                 device=device).float()


class PenguinModel(nn.Module):
    def __init__(self):
        super(PenguinModel, self).__init__()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.stack1 = nn.Linear(12, 12)
        self.stack2 = nn.Linear(12, 3)

    def forward(self, x):
        return self.tanh(self.stack2(self.relu(self.stack1(x))))


train_dataset = PenguinsDataset(0, 200)
val_dataset = PenguinsDataset(200, 300)

model = PenguinModel()
model = model.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=50, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=100, shuffle=False)

train_performance = []
val_performance = []
epochs = 1000
for t in range(epochs):
    if t % 100 == 0:
        print(t)
    for batch, (X, y) in enumerate(train_dataloader):
        # print("\t"+str(batch))
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        val_X, val_y = next(iter(val_dataloader))
        val_pred = model(val_X)
        val_loss = loss_fn(val_pred, val_y)

        train_performance.append(loss.item())
        val_performance.append(val_loss.item())

to_prune = ((model.stack1, "weight"), (model.stack2, "weight"))
prune.global_unstructured(to_prune, pruning_method=prune.L1Unstructured, amount=0.3)

plt.plot(train_performance, label="Training Loss")
plt.plot(val_performance, label="Validation Loss")
plt.legend()
plt.title("Model Training (Pruned After Training)")
plt.show()

test_dataset = PenguinsDataset(300, 333)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=33, shuffle=False)
test_X, test_y = next(iter(test_dataloader))
test_pred = model(test_X)
test_loss = loss_fn(test_pred, test_y)
print(test_loss.item())

