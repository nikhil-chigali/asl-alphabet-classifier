import torch
import torch.nn as nn 
from torchvision import models


from lightning import LightningModule

class ResNet18(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # self.model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
        # Freeze all layers except the last one
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)

class ASLAlphabetModel(LightningModule):
    def __init__(self, config: dict):
        super().__init__()
        self.save_hyperparameters(config.to_dict())
        self.config = config
        self.model = ResNet18(num_classes=self.config.data.num_classes)
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = None
        self.scheduler = None
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        return loss
    
    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.config.train.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 
                mode='min', 
                factor=self.config.train.factor, 
                patience=self.config.train.patience, 
                verbose=True,
            )
        return {
            'optimizer': self.optimizer,
            'lr_scheduler': self.scheduler,
            'monitor': 'val_loss'
        }    
    