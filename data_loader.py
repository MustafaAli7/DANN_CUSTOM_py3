import os
import requests
import zipfile
import torch.utils.data as data
from torchvision import datasets, transforms

DATASET_URL = 'https://drive.usercontent.google.com/download?id=1FmdsvetC0oVyrFJ9ER7fcN-cXPOWx2gq&export=download'
DATASET_ZIP = 'adaptiope.zip'
DATASET_DIR = 'adaptiope/Adaptiope'

def download_and_extract():
    if not os.path.exists(DATASET_DIR):
        print("Downloading the Adaptiope dataset...")
        response = requests.get(DATASET_URL, stream=True)
        with open(DATASET_ZIP, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete! Extracting...")
        with zipfile.ZipFile(DATASET_ZIP, 'r') as zip_ref:
            zip_ref.extractall('adaptiope')
        print("Dataset extracted successfully!")
    else:
        print("Dataset already exists. Skipping download.")

def get_data_loaders(batch_size=32, image_size=128):
    download_and_extract()

    img_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    source_dataset = datasets.ImageFolder(root=f'{DATASET_DIR}/synthetic', transform=img_transform)
    target_dataset = datasets.ImageFolder(root=f'{DATASET_DIR}/product_images', transform=img_transform)

    source_loader = data.DataLoader(source_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    target_loader = data.DataLoader(target_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    return source_loader, target_loader
