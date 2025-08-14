import os
import yaml
import json
import logging
import torch
import pandas as pd

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import core.vision_encoder.pe as pe
from core.model import Fuse_Adapter
from core.trainer import Trainer
from preprocess.data_handler import get_text_features, create_dataloaders
from utils import create_train_grid, create_calib_grid

os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/train.log", mode='a'),
        logging.StreamHandler()
    ]
)

def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def append_to_log(log_path: str, record: dict):
    df = pd.DataFrame([record])
    if os.path.exists(log_path):
        df.to_csv(log_path, mode='a', header=False, index=False)
    else:
        df.to_csv(log_path, mode='w', header=True, index=False)

def main():
    config = load_config("configs/image_benchmark_config.yaml")
    device = config['training']['device']
    log_file = "logs/training_log.csv"

    logging.info(f"Loading Perception Encoder: {config['model']['encoder_model_name']}")
    pe_model = pe.CLIP.from_config(config['model']['encoder_model_name'], pretrained=True).to(device)
    pe_model.eval()
    
    logging.info("Preparing text features...")
    text_features = get_text_features(pe_model, config, device)
    
    logging.info("Preparing dataloaders...")
    dataloaders = create_dataloaders(config, pe_model)
    
    encoder_tag = config['model']['encoder_model_name'].replace('/', '_')
    output_dir = config['paths']['output_model_dir']
    os.makedirs(output_dir, exist_ok=True)

    train_grid = create_train_grid(config)
    calib_grid = create_calib_grid(config['calibration'])
    
    total_runs = len(train_grid) * len(calib_grid)
    logging.info(f"Starting grid search: {total_runs} runs")

    run_index = 0
    for hp_params in train_grid:
        for calib_params in calib_grid:
            run_index += 1
            
            tag = (f"{encoder_tag}_lr{hp_params['lr']}_m{hp_params['mult']}"
                   f"_wd{hp_params['wd']}_d{hp_params['drop']}"
                   f"_w{calib_params['warmup_epochs']}_b{calib_params['beta']}")
            
            model_path = os.path.join(output_dir, f"{tag}.pt")
            
            adapter_model = Fuse_Adapter(
                input_dim=config['model']['input_dim'],
                num_classes=len(config['data']['class_names']),
                hidden_dim=config['model']['hidden_dim'],
                dropout=hp_params['drop']
            ).to(device)

            run_config = json.loads(json.dumps(config))
            run_config['training'].update({
                'lr': hp_params['lr'],
                'interaction_lr_multiplier': hp_params['mult'],
                'weight_decay': hp_params['wd']
            })
            run_config['model']['dropout'] = hp_params['drop']
            run_config['calibration'].update({
                'warmup_epochs': calib_params['warmup_epochs'],
                'beta': calib_params['beta']
            })

            trainer = Trainer(pe_model, adapter_model, dataloaders, text_features, run_config, model_path)
            
            logging.info(f"[{run_index}/{total_runs}] START: {tag}")
            final_metrics = trainer.train()
            logging.info(f"[{run_index}/{total_runs}] END:   {tag} | Best Sel Score = {final_metrics.get('best_sel_score', 0.0):.4f}")
            
            log_record = {
                'tag': tag,
                'model_path': model_path,
                'calib_path': model_path.replace(".pt", "_calib_ovn.json"),
                'registry_path': model_path.replace(".pt", "_registry.json"),
                'best_sel_score': final_metrics.get('best_sel_score', 0.0),
                'best_epoch': final_metrics.get('best_epoch', -1),
                'lr': hp_params['lr'],
                'mult': hp_params['mult'],
                'wd': hp_params['wd'],
                'drop': hp_params['drop'],
                'warmup_epochs': calib_params['warmup_epochs'],
                'beta': calib_params['beta'],
            }
            append_to_log(log_file, log_record)

    logging.info("Grid search complete.")

if __name__ == "__main__":
    main()