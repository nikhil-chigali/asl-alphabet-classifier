import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import yaml
from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt
from ml_collections import ConfigDict


@dataclass
class Paths:
    root_dir: Path = Path(__file__).resolve().parents[2]
    data_dir: Path = root_dir / 'data'
    models_dir: Path = root_dir / 'models'
    logs_dir: Path = root_dir / 'logs'
    src_dir: Path = root_dir / 'src'
    config_dir: Path = root_dir / 'config'


def get_config(config_file: str) -> dict:
    config_path = Paths.config_dir / config_file
    with open(config_path, 'r') as file:
        config = ConfigDict(yaml.safe_load(file))
    return config

def show_images(batch):
    x, y = batch
    label_to_class = {
        0: 'A',
        1: 'B',
        2: 'C',
        3: 'D',
        4: 'E',
        5: 'F',
        6: 'G',
        7: 'H',
        8: 'I',
        9: 'J',
        10: 'K',
        11: 'L',
        12: 'M',
        13: 'N',
        14: 'O',
        15: 'P',
        16: 'Q',
        17: 'R',
        18: 'S',
        19: 'T',
        20: 'U',
        21: 'V',
        22: 'W',
        23: 'X',
        24: 'Y',
        25: 'Z',
        26: 'del',
        27: 'nothing',
        28: 'space'
    }

    plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(x[i].permute(1, 2, 0).cpu().numpy())
        plt.title(label_to_class[y[i].item()])
        plt.axis("off")
        
    plt.show()