import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from argparse import ArgumentParser
from lightning import Trainer
from lightning.pytorch.loggers import CSVLogger
from loguru import logger

from src.models.model import ASLAlphabetModel
from src.data.dataset import ASLAlphabetDataset
from src.utils.helper import get_config, Paths

def train(config: dict):
    model = ASLAlphabetModel(config)
    data = ASLAlphabetDataset(config.train.batch_size)
    trainer = Trainer(
        max_epochs=config.train.epochs, 
        accelerator=config.train.accelerator,
        devices=config.train.devices,
        logger=CSVLogger(Paths.logs_dir),
    )
    
    trainer.fit(model, data) 
    trianer.validate(model, data)   
    trainer.save_checkpoint(Paths.models_dir / 'model_last.ckpt')
    
if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument('--config_file', type=str, default="config.yaml", help='Name of the configuration file')
    args = argparser.parse_args()
    
    try:
        config = get_config(args.config_file)
    except FileNotFoundError as e:
        logger.error("Configuration file not found: {}", e)
        sys.exit(1)

    train(config)