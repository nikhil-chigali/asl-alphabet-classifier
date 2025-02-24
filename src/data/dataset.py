import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision import transforms
from sklearn.model_selection import train_test_split

from lightning import LightningDataModule
from loguru import logger

from src.utils.helper import Paths

class ASLAlphabetDataset(LightningDataModule):
    def __init__(self, batch_size: int = 32):
        super().__init__()
        self.data_dir = Paths.data_dir / 'asl-alphabets' / 'train'
        self.batch_size = batch_size
        self.transform = None
        self.dataset = None
        self.trainset = None
        self.valset = None
        self.class_to_label = None
        self.label_to_class = None
        
    def prepare_data(self):
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        self.dataset = ImageFolder(self.data_dir, transform=self.transform)        

    def setup(self, stage=None):
        samples = self.dataset.samples
        idx_class_pairs = [[i, x[1]] for i, x in enumerate(samples)]
        train_idx, val_idx = train_test_split(idx_class_pairs, test_size=0.2, stratify=[x[1] for x in idx_class_pairs])
        train_idx, val_idx = [x[0] for x in train_idx], [x[0] for x in val_idx]
        
        self.trainset = Subset(self.dataset, train_idx)
        self.valset = Subset(self.dataset, val_idx)
        
        self.class_to_label = {class_name: idx for idx, class_name in enumerate(self.dataset.classes)}
        self.label_to_class = {idx: class_name for class_name, idx in self.class_to_label.items()}

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=self.batch_size, shuffle=False)
