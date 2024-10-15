import random
import torch.optim as optim
import torch.utils.data
import numpy as np
from model import CNNModel
from test import test
from data_loader import get_data_loaders

# Configurations
model_root = 'models'
cuda = torch.cuda.is_available()
lr = 1e-3
batch_size = 32
image_size = 128
n_epoch = 20

# Set random seed
manual_seed = random.randint(1, 10000)
random.seed(manual_seed)
torch.manual_seed(manual_seed)

# Load datasets
print("Preparing data loaders...")
source_loader, target_loader = get_data_loaders(batch_size, image_size)

# Initialize model, optimizer, and loss function
my_net = CNNModel()
optimizer = optim.Adam(my_net.parameters(), lr=lr)
loss_class = torch.nn.CrossEntropyLoss()

if cuda:
    my_net = my_net.cuda()
    loss_class = loss_class.cuda()

# Training loop
print("Starting training...")
for epoch in range(n_epoch):
    my_net.train()
    len_loader = min(len(source_loader), len(target_loader))
    source_iter = iter(source_loader)
    target_iter = iter(target_loader)

    for i in range(len_loader):
        p = float(i + epoch * len_loader) / (n_epoch * len_loader)
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        source_images, source_labels = next(source_iter)
        if cuda:
            source_images, source_labels = source_images.cuda(), source_labels.cuda()

        optimizer.zero_grad()
        class_output, _ = my_net(source_images, alpha=alpha)
        loss = loss_class(class_output, source_labels)
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{n_epoch}], Step [{i + 1}/{len_loader}], Loss: {loss.item():.4f}')

    torch.save(my_net.state_dict(), f'{model_root}/adaptiope_epoch_{epoch + 1}.pth')

# Test model
print("Evaluating the model...")
test_accuracy = test(my_net, target_loader, cuda)
print(f'Test Accuracy: {test_accuracy:.2f}%')
