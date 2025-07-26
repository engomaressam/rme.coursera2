# Fashion-MNIST Project
# Converted from 06c.FashionMNISTProject-v1.ipynb
# Submission requirements: (1) Screenshot of first 3 validation images, (2) Screenshot of cost/accuracy plots

# --- Preparation ---
# Install required packages if not already installed
# !pip install torch torchvision matplotlib  # Uncomment if running in a new environment

# PyTorch Modules
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import matplotlib.pylab as plt
from PIL import Image

# --- Helper Function ---
IMAGE_SIZE = 16

def show_data(data_sample):
    plt.imshow(data_sample[0].numpy().reshape(IMAGE_SIZE, IMAGE_SIZE), cmap='gray')
    plt.title('y = ' + str(data_sample[1]))

# --- Compose Transforms ---
composed = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

# --- Create Datasets ---
dataset_train = dsets.FashionMNIST(root='.fashion/data', train=True, transform=composed, download=True)
dataset_val = dsets.FashionMNIST(root='.fashion/data', train=False, transform=composed, download=True)

# --- Show and Save First 3 Validation Images Together in a PDF (for Coursera) ---
from matplotlib.backends.backend_pdf import PdfPages
val_images_pdf = "val_images.pdf"
with PdfPages(val_images_pdf) as pdf:
    for n, data_sample in enumerate(dataset_val):
        plt.imshow(data_sample[0].numpy().reshape(IMAGE_SIZE, IMAGE_SIZE), cmap='gray')
        plt.title('y = ' + str(data_sample[1]))
        pdf.savefig()  # saves the current figure into the PDF
        print(f"Added validation image {n+1} to {val_images_pdf}")
        plt.close()
        if n == 2:
            break
print(f"Saved first 3 validation images together in {val_images_pdf}")

# --- CNN Model Definitions ---
class CNN_batch(nn.Module):
    def __init__(self, out_1=16, out_2=32, number_of_classes=10):
        super(CNN_batch, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=out_1, kernel_size=5, padding=2)
        self.conv1_bn = nn.BatchNorm2d(out_1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.cnn2 = nn.Conv2d(in_channels=out_1, out_channels=out_2, kernel_size=5, stride=1, padding=2)
        self.conv2_bn = nn.BatchNorm2d(out_2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(out_2 * 4 * 4, number_of_classes)
        self.bn_fc1 = nn.BatchNorm1d(10)
    def forward(self, x):
        x = self.cnn1(x)
        x = self.conv1_bn(x)
        x = torch.relu(x)
        x = self.maxpool1(x)
        x = self.cnn2(x)
        x = self.conv2_bn(x)
        x = torch.relu(x)
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.bn_fc1(x)
        return x

class CNN(nn.Module):
    def __init__(self, out_1=16, out_2=32, number_of_classes=10):
        super(CNN, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=out_1, kernel_size=5, padding=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.cnn2 = nn.Conv2d(in_channels=out_1, out_channels=out_2, kernel_size=5, stride=1, padding=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(out_2 * 4 * 4, number_of_classes)
    def forward(self, x):
        x = self.cnn1(x)
        x = torch.relu(x)
        x = self.maxpool1(x)
        x = self.cnn2(x)
        x = torch.relu(x)
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

# --- Data Loaders ---
train_loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=100)
test_loader = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=100)

# --- Model, Criterion, Optimizer ---
model = CNN_batch(out_1=16, out_2=32, number_of_classes=10)  # or use CNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# --- Training Loop ---
import time
start_time = time.time()
cost_list = []
accuracy_list = []
N_test = len(dataset_val)
n_epochs = 5
for epoch in range(n_epochs):
    cost = 0
    model.train()
    for x, y in train_loader:
        optimizer.zero_grad()
        z = model(x)
        loss = criterion(z, y)
        loss.backward()
        optimizer.step()
        cost += loss.item()
    correct = 0
    model.eval()
    for x_test, y_test in test_loader:
        z = model(x_test)
        _, yhat = torch.max(z.data, 1)
        correct += (yhat == y_test).sum().item()
    accuracy = correct / N_test
    accuracy_list.append(accuracy)
    cost_list.append(cost)
    print(f"Epoch {epoch+1}: Cost={cost:.4f}, Accuracy={accuracy*100:.2f}%")

# --- Plot Cost and Accuracy (Auto-save for Coursera) ---
fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.plot(cost_list, color=color)
ax1.set_xlabel('epoch', color=color)
ax1.set_ylabel('Cost', color=color)
ax1.tick_params(axis='y', color=color)
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('accuracy', color=color)
ax2.set_xlabel('epoch', color=color)
ax2.plot(accuracy_list, color=color)
ax2.tick_params(axis='y', color=color)
fig.tight_layout()
plot_png = "cost_accuracy_plot.png"
plot_pdf = "cost_accuracy_plot.pdf"
plt.savefig(plot_png)
plt.savefig(plot_pdf)
print(f"Saved cost/accuracy plot as {plot_png} and {plot_pdf}")
plt.close()
# Take a screenshot of the above plot for submission

# --- End of Script --- 