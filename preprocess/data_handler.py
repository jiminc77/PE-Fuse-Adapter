import os
import torch
import json
import logging
import itertools
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as T
from typing import Dict

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as pe_transforms

def get_text_features(pe_model: pe.CLIP, config: Dict, device: str) -> torch.Tensor:

    data_cfg = config['data']
    output_dir = data_cfg['text_feature_avg_root']
    class_names = data_cfg['class_names']
    
    os.makedirs(output_dir, exist_ok=True)
    
    feature_files = [os.path.join(output_dir, f"{name}_avg.pt") for name in class_names]
    if all(os.path.exists(f) for f in feature_files):
        logging.info(f"Loading pre-computed text features from {output_dir}...")
        features = [torch.load(f, map_location=device) for f in feature_files]
        return torch.stack(features, dim=0)

    logging.info(f"Pre-computed text features not found. Generating from scratch...")
    caption_json_path = data_cfg.get('prompt_json_path')
    if not caption_json_path or not os.path.exists(caption_json_path):
        raise FileNotFoundError(f"Caption JSON file not found at path: {caption_json_path}")

    with open(caption_json_path, 'r') as f:
        caption_data = json.load(f)

    templates = caption_data['template']['template_caption']
    subjects = caption_data['placeholder']['syn_subject']
    actions_by_class = {k: v for k, v in caption_data['placeholder']['syn_aug_action'].items() if k in class_names}
    
    tokenizer = pe_transforms.get_text_tokenizer(pe_model.context_length)
    all_class_features = []

    for class_name in tqdm(class_names, desc="Generating text features"):
        if class_name not in actions_by_class:
            logging.warning(f"Class '{class_name}' not found in caption JSON. Generating a simple prompt.")
            prompts = [f"a photo of a {class_name}."]
        else:
            class_actions = actions_by_class[class_name]
            prompts = [
                template.replace("SYN_SUBJECT", subject).replace("SYN_ACTION", action)
                for template, subject, action in itertools.product(templates, subjects, class_actions)
            ]
        
        with torch.no_grad():
            class_feature_sum = torch.zeros(pe_model.text_projection.shape[1]).to(device)
            prompt_batches = [prompts[i:i + 256] for i in range(0, len(prompts), 256)]
            
            for batch in prompt_batches:
                text_tokens = tokenizer(batch).to(device)
                batch_features = pe_model.encode_text(text_tokens, normalize=True)
                class_feature_sum += batch_features.sum(dim=0)
            
            class_avg_feature = class_feature_sum / len(prompts)
            class_avg_feature = class_avg_feature / class_avg_feature.norm()
            all_class_features.append(class_avg_feature)
            
            torch.save(class_avg_feature.cpu(), os.path.join(output_dir, f"{class_name}_avg.pt"))
            logging.info(f"Saved average text feature for '{class_name}'.")

    return torch.stack(all_class_features, dim=0)


def create_dataloaders(config: Dict, pe_model: pe.CLIP) -> Dict[str, DataLoader]:
    image_root = config['paths']['raw_image_root']
    
    eval_preprocess = pe_transforms.get_image_transform(pe_model.image_size)
    
    resize_size = eval_preprocess.transforms[0].size
    to_tensor_transform = eval_preprocess.transforms[-2]
    normalize_transform = eval_preprocess.transforms[-1]

    train_preprocess = T.Compose([
        T.RandomResizedCrop(size=resize_size, scale=(0.8, 1.0), interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(p=0.5),
        to_tensor_transform,
        normalize_transform,
    ])
    logging.info("Data augmentation for training is enabled.")

    train_path = os.path.join(image_root, config['paths']['train_dir'])
    test_path = os.path.join(image_root, config['paths']['test_dir'])
    
    train_ds_raw = datasets.ImageFolder(train_path, transform=train_preprocess)
    test_ds_raw = datasets.ImageFolder(test_path, transform=eval_preprocess)

    name_to_idx_map = {name: i for i, name in enumerate(config['data']['class_names'])}
    def remap_targets(dataset, transform_pipeline):
        original_idx_to_name = {v: k for k, v in dataset.class_to_idx.items()}
        target_transform = lambda y: name_to_idx_map[original_idx_to_name[y]]
        return datasets.ImageFolder(dataset.root, transform=transform_pipeline, target_transform=target_transform)

    train_ds = remap_targets(train_ds_raw, train_preprocess)
    test_ds = remap_targets(test_ds_raw, eval_preprocess)

    bs = config['training']['batch_size']
    nw = config['training']['num_workers']

    return {
        'train': DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=True),
        'test': DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
    }