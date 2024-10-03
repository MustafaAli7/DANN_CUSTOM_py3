import random
import os
import sys
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import numpy as np
from torchvision import datasets
from torchvision import transforms
from model import CNNModel
from test import test

# Updated for EMNIST dataset
dataset_name = 'EMNIST_letters'
source_image_root = os.path.join('dataset', dataset_name)
model_root = 'models'
cuda = True
cudnn.benchmark = True
lr = 1e-3
batch_size = 128
image_size = 28
n_epoch = 100

manual_seed = random.randint(1, 10000)
random.seed(manual_seed)
torch.manual_seed(manual_seed)

# load data
img_transform_source = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.1307,), std=(0.3081,))
])

# Load EMNIST Letters dataset
dataset_source = datasets.EMNIST(
    root='dataset',
    split='letters',  # Use 'letters' subset of EMNIST
    train=True,
    transform=img_transform_source,
    download=True
)

dataloader_source = torch.utils.data.DataLoader(
    dataset=dataset_source,
    batch_size=batch_size,
    shuffle=True,
    num_workers=8
)

# load model
my_net = CNNModel()

# setup optimizer
optimizer = optim.Adam(my_net.parameters(), lr=lr)

loss_class = torch.nn.NLLLoss()

if cuda:
    my_net = my_net.cuda()
    loss_class = loss_class.cuda()

for p in my_net.parameters():
    p.requires_grad = True

# training
best_accu = 0.0
for epoch in range(n_epoch):

    len_dataloader = len(dataloader_source)
    data_source_iter = iter(dataloader_source)

    for i in range(len_dataloader):
        my_net.zero_grad()
        data_source = next(data_source_iter)
        s_img, s_label = data_source

        if cuda:
            s_img = s_img.cuda()
            s_label = s_label.cuda()

        class_output, _ = my_net(input_data=s_img, alpha=0)
        err_s_label = loss_class(class_output, s_label)
        
        err_s_label.backward()
        optimizer.step()

        sys.stdout.write('\r epoch: %d, [iter: %d / all %d], err_s_label: %f' \
              % (epoch, i + 1, len_dataloader, err_s_label.data.cpu().numpy()))
        sys.stdout.flush()

    print('\n')
    accu = test('EMNIST_letters')
    print('Accuracy of the EMNIST Letters dataset: %f' % accu)
    if accu > best_accu:
        best_accu = accu
        torch.save(my_net, '{0}/emnist_model_epoch_best.pth'.format(model_root))

print('============ Summary ============= \n')
print('Best Accuracy of the EMNIST Letters dataset: %f' % best_accu)
print('Corresponding model was saved in ' + model_root + '/emnist_model_epoch_best.pth')
