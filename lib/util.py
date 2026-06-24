#!/usr/bin/env python3

import yaml
import torch.utils.data as data
from sklearn.metrics import roc_auc_score
from pathlib import Path


def load_config(config_path):
    """
    Load YAML configuration file with proper UTF-8 encoding.
    
    Args:
        config_path: Path to config file (.cfg or .yaml)
        
    Returns:
        dict: Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    try:
        # Try UTF-8 first
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except UnicodeDecodeError:
        # Fallback: try different encodings
        for encoding in ['utf-8-sig', 'latin-1', 'cp1252']:
            try:
                with open(config_path, 'r', encoding=encoding) as f:
                    config = yaml.safe_load(f)
                print(f"Warning: Loaded config with {encoding} encoding")
                return config
            except UnicodeDecodeError:
                continue
        
        # If all fail, raise error
        raise UnicodeDecodeError(
            'utf-8', b'', 0, 1,
            f"Could not decode {config_path} with any standard encoding. "
            f"Please save the file with UTF-8 encoding."
        )


def update_learning_rate(epoch):
    lr = None
    if epoch < 4:
        lr = 3.6e-4
    elif epoch < 10:
        lr = 1e-4  # 2e-4 * 2
    elif epoch < 20:
        lr = 5e-5  # 5e-5 * 2
    else:
        lr = 5e-5
    return lr


def my_collate(batch):
    batch = filter(lambda img: img[0] is not None, batch)
    return data.dataloader.default_collate(list(batch))


def get_video_auc(f_label_list, v_name_list, f_pred_list):
    video_res_dict = dict()
    video_pred_list = list()
    video_label_list = list()
    # summarize all the results for each video
    for label, video, score in zip(f_label_list, v_name_list, f_pred_list):
        if video not in video_res_dict.keys():
            video_res_dict[video] = {"scores": [score], "label": label}
        else:
            video_res_dict[video]["scores"].append(score)
    # get the score and label for each video
    for video, res in video_res_dict.items():
        score = sum(res['scores']) / len(res['scores'])
        label = res['label']
        video_pred_list.append(score)
        video_label_list.append(label)

    v_auc = roc_auc_score(video_label_list, video_pred_list)
    return v_auc


# vim: ts=4 sw=4 sts=4 expandtab
