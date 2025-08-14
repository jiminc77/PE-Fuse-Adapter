import os
import json
import yaml
import cv2
import torch
import logging
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
from torch.amp import autocast
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import argparse

import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as pe_transforms
from core.model import Fuse_Adapter

os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/benchmark_evaluation.log", mode='a'),
        logging.StreamHandler()
    ]
)

def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def load_average_text_features(cfg: dict, device: str) -> torch.Tensor:
    root = cfg['data']['text_feature_avg_root']
    names = cfg['data']['class_names']
    feats = [torch.load(os.path.join(root, f"{n}_avg.pt"), map_location=device) for n in names]
    return torch.stack(feats, dim=0)

class VideoFrameDataset(torch.utils.data.Dataset):
    def __init__(self, video_path: str, transform):
        self.video_path = video_path
        self.transform = transform
        self.video_handles: Dict[int, cv2.VideoCapture] = {}
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise IOError(f"Cannot open video: {video_path}")
            self.num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        except Exception as e:
            logging.error(f"Failed to initialize VideoCapture for {video_path}: {e}")
            self.num_frames = 0

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx: int):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        if worker_id not in self.video_handles:
            self.video_handles[worker_id] = cv2.VideoCapture(self.video_path)
        
        cap = self.video_handles[worker_id]
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError(f"Failed to read frame {idx} from {self.video_path}")
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.transform(Image.fromarray(frame_rgb))

def compute_margins(logits: torch.Tensor, idx_map: Dict[str, int]) -> Tuple[torch.Tensor, torch.Tensor]:
    return logits[:, idx_map["fire"]] - logits[:, idx_map["normal"]], logits[:, idx_map["falldown"]] - logits[:, idx_map["normal"]]

def apply_affine(values: np.ndarray, s: float, b: float, mode: str) -> np.ndarray:
    z = s * values + b
    return 1.0 / (1.0 + np.exp(-z)) if mode == "prob" else z

def load_calibration(calib_json_path: str) -> Dict:
    with open(calib_json_path, "r") as f:
        data = json.load(f)
    return {
        "mode": data.get("mode", "margin"),
        "affine": data["affine"],
        "thresholds": data["thresholds"],
        "idx_map": data.get("idx_map", {"fire": 0, "falldown": 1, "normal": 2}),
    }

def infer_video_margins(video_path: str, feature_cache_path: str, pe_model: pe.CLIP, adapter: Fuse_Adapter,
                        text_features: torch.Tensor, transform, cfg: dict, device: str, idx_map: Dict[str, int]) -> Dict[str, np.ndarray]:
    if os.path.exists(feature_cache_path):
        image_features = torch.load(feature_cache_path, map_location=device)
    else:
        dataset = VideoFrameDataset(video_path, transform)
        if len(dataset) == 0:
            logging.warning(f"Skipping empty or invalid video: {video_path}")
            return {}
        
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=cfg['training']['batch_size'] * 16, shuffle=False,
            num_workers=cfg['training'].get('num_workers', 4), pin_memory=True, prefetch_factor=2)
        
        feature_list = []
        with torch.no_grad():
            for batch in tqdm(loader, desc="Extracting features", leave=False):
                batch = batch.to(device, non_blocking=True)
                with autocast(device):
                    features = pe_model.encode_image(batch, normalize=True)
                feature_list.append(features.cpu())
        
        if not feature_list: return {}
        image_features = torch.cat(feature_list, dim=0).to(device)
        os.makedirs(os.path.dirname(feature_cache_path), exist_ok=True)
        torch.save(image_features, feature_cache_path)

    margins = {"m_fire": [], "m_fall": []}
    with torch.no_grad():
        for chunk in torch.split(image_features, cfg['training']['batch_size'] * 4):
            with autocast(device):
                logits = adapter(chunk, text_features)
            m_fire, m_fall = compute_margins(logits, idx_map)
            margins["m_fire"].append(m_fire.cpu())
            margins["m_fall"].append(m_fall.cpu())
            
    return {k: torch.cat(v, dim=0).numpy() for k, v in margins.items()}

def list_top_checkpoints(registry_path: str) -> List[Dict]:
    if os.path.exists(registry_path):
        try:
            with open(registry_path, "r") as f:
                items = json.load(f)
            if isinstance(items, list):
                return sorted(items, key=lambda d: d.get("sel_score", 0.0), reverse=True)
        except (json.JSONDecodeError, IOError) as e:
            logging.warning(f"Failed to read registry {registry_path}: {e}")
    return []

def get_evaluation_targets(run_log: pd.Series) -> List[Dict]:
    tag = run_log['tag']
    registry_path = run_log['registry_path']
    
    items = list_top_checkpoints(registry_path)
    if not items:
        model_path = run_log['model_path']
        calib_path = run_log['calib_path']
        if os.path.exists(model_path) and os.path.exists(calib_path):
            return [{
                "rank": 1, "epoch": -1, "sel_score": np.nan,
                "model": model_path, "calib": calib_path, "tag": f"{tag}#best",
            }]
        return []

    targets = []
    for rank, item in enumerate(items, start=1):
        model_path, calib_path = item.get("model", ""), item.get("calib", "")
        if os.path.exists(model_path) and os.path.exists(calib_path):
            targets.append({
                "rank": rank, "epoch": item.get("epoch", -1),
                "sel_score": float(item.get("sel_score", 0.0)),
                "model": model_path, "calib": calib_path,
                "tag": f"{tag}#top{rank}",
            })
    return targets

def calculate_final_metrics(overall_true, overall_pred):
    results = {}
    for cname in ['fire', 'falldown']:
        if not overall_true[cname]: continue
        y_true = pd.concat(overall_true[cname], ignore_index=True).astype(int)
        y_pred = pd.concat(overall_pred[cname], ignore_index=True).astype(int)
        results[cname] = {
            'Accuracy': accuracy_score(y_true, y_pred),
            'F1-score': f1_score(y_true, y_pred, average='binary', zero_division=0),
            'Precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
            'Recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
        }
    return results

def print_results_summary(tag, results):
    print("\n" + "-"*30 + f" RESULTS ({tag}) " + "-"*30)
    for cname, metrics in results.items():
        print(f" Category: {cname.upper()}")
        for name, value in metrics.items():
            print(f"  - {name:<20} {value:.4f}")
        print("-"*20)

def main(args):
    cfg = load_config('configs/image_benchmark_config.yaml')
    device = cfg['training']['device']
    training_log_path = "logs/training_log.csv"

    if not os.path.exists(training_log_path):
        logging.error(f"Training log file not found: {training_log_path}. Please run training first.")
        return

    training_log_df = pd.read_csv(training_log_path)
    
    if args.tag:
        if args.tag not in training_log_df['tag'].values:
            logging.error(f"Tag '{args.tag}' not found in {training_log_path}.")
            logging.info("Available tags are:")
            for t in training_log_df['tag']:
                logging.info(f"- {t}")
            return

        training_log_df = training_log_df[training_log_df['tag'] == args.tag]
        logging.info(f"Filtered for specific model with tag: {args.tag}")
    else:
        logging.info(f"Loaded {len(training_log_df)} completed training runs from {training_log_path}.")

    pe_model = pe.CLIP.from_config(cfg['model']['encoder_model_name'], pretrained=True).to(device).eval()
    transform = pe_transforms.get_image_transform(pe_model.image_size)
    text_features = load_average_text_features(cfg, device)

    class_names = cfg['data']['class_names']
    benchmark_root = cfg['paths']['benchmark_dataset_root']
    feat_cache_root = cfg['paths']['image_frame_features_root']
    csv_out_root = cfg['paths']['prediction_csv_output_dir']
    os.makedirs(csv_out_root, exist_ok=True)

    summary_csv = "logs/benchmark_evaluation_summary.csv"
    try:
        summary_df = pd.read_csv(summary_csv) if os.path.exists(summary_csv) else pd.DataFrame()
    except pd.errors.EmptyDataError:
        summary_df = pd.DataFrame()
    if not summary_df.empty:
        summary_df.columns = summary_df.columns.str.strip()

    for _, run_log in training_log_df.iterrows():
        targets = get_evaluation_targets(run_log)
        
        if args.rank:
            original_target_count = len(targets)
            targets = [t for t in targets if t['rank'] == args.rank]
            if not targets:
                logging.warning(f"Rank {args.rank} not found for tag {run_log['tag']}. "
                                f"Available ranks are 1 to {original_target_count}. Skipping.")
                continue
            logging.info(f"Filtered for checkpoint with rank: {args.rank}")

        if not targets:
            logging.warning(f"No valid checkpoints found for tag: {run_log['tag']}. Skipping.")
            continue

        for target in targets:
            logging.info(f"Evaluating: {target['tag']} (epoch={target['epoch']}, sel_score={target.get('sel_score', 0.0):.4f})")
            
            adapter = Fuse_Adapter(
                input_dim=cfg['model']['input_dim'], num_classes=len(class_names),
                hidden_dim=cfg['model']['hidden_dim'], dropout=run_log['drop'],
            ).to(device)
            adapter.load_state_dict(torch.load(target["model"], map_location=device))
            adapter.eval()

            cal_data = load_calibration(target["calib"])
            mode, aff, th, idx_map = cal_data["mode"], cal_data["affine"], cal_data["thresholds"], cal_data["idx_map"]

            overall_true = {cn: [] for cn in class_names if cn != 'normal'}
            overall_pred = {cn: [] for cn in class_names if cn != 'normal'}

            for cls_dir in tqdm(sorted(os.listdir(benchmark_root)), desc=f"{target['tag']}: Categories"):
                cname = cls_dir.split('_')[-1].lower()
                if cname not in ['fire', 'falldown']: continue
                gt_dir = os.path.join(benchmark_root, cls_dir, "dataset", cname)
                if not os.path.isdir(gt_dir): continue

                for gt_csv in tqdm(sorted(os.listdir(gt_dir)), desc=f"Infer {cname}", leave=False):
                    if not gt_csv.endswith('.csv'): continue
                    stem = os.path.splitext(gt_csv)[0]
                    video_path = next((os.path.join(gt_dir, f"{stem}{ext}") for ext in ['.mp4', '.MOV', '.mov', '.MP4'] if os.path.exists(os.path.join(gt_dir, f"{stem}{ext}"))), None)
                    if not video_path: continue

                    feat_path = os.path.join(feat_cache_root, cname, f"{stem}.pt")
                    margins = infer_video_margins(video_path, feat_path, pe_model, adapter, text_features, transform, cfg, device, idx_map)
                    if not margins: continue

                    margin_key, th_key = ('m_fire', 'fire') if cname == 'fire' else ('m_fall', 'falldown')
                    vals = apply_affine(margins[margin_key], aff[th_key]['s'], aff[th_key]['b'], mode)
                    preds = (vals >= th[th_key]).astype(int)

                    rank_folder = f"rank_{target['rank']}"
                    out_dir_cls = os.path.join(csv_out_root, run_log['tag'], rank_folder, cname)
                    os.makedirs(out_dir_cls, exist_ok=True)
                    pred_csv_path = os.path.join(out_dir_cls, f"{stem}.csv")
                    pd.DataFrame({'frame': np.arange(len(preds)), cname: preds}).to_csv(pred_csv_path, index=False)

                    gt = pd.read_csv(os.path.join(gt_dir, gt_csv))
                    if cname not in gt.columns: continue
                    y_true = gt[cname].values.astype(int)
                    m = min(len(y_true), len(preds))
                    if m > 0:
                        overall_true[cname].append(pd.Series(y_true[:m]))
                        overall_pred[cname].append(pd.Series(preds[:m]))

            results = calculate_final_metrics(overall_true, overall_pred)
            
            row = {
                'tag': run_log['tag'], 'ckpt_rank': target['rank'], 'ckpt_epoch': target['epoch'], 'ckpt_sel_score': target['sel_score'],
                'lr': run_log['lr'], 'interaction_lr_multiplier': run_log['mult'],
                'weight_decay': run_log['wd'], 'dropout': run_log['drop'],
                'warmup_epochs': run_log['warmup_epochs'], 'beta': run_log['beta'],
            }
            
            for cname, metrics in results.items():
                for k, v in metrics.items():
                    row[f"{cname}_{k}"] = v
            
            if not summary_df.empty and {'tag', 'ckpt_rank'}.issubset(summary_df.columns):
                mask = (summary_df['tag'] == row['tag']) & (summary_df['ckpt_rank'] == row['ckpt_rank'])
                if mask.any():
                    summary_df = summary_df.loc[~mask].copy()

            summary_df = pd.concat([summary_df, pd.DataFrame([row])], ignore_index=True)
            summary_df.to_csv(summary_csv, index=False)
            print_results_summary(target['tag'], results)

    logging.info("All evaluations completed.")
    if os.path.exists(summary_csv) and not summary_df.empty:
        f1_cols = [c for c in summary_df.columns if 'F1-score' in c]
        if f1_cols:
            summary_df['avg_f1'] = summary_df[f1_cols].mean(axis=1)
            summary_df = summary_df.sort_values(by=['tag', 'ckpt_rank', 'avg_f1'], ascending=[True, True, False]).drop(columns=['avg_f1'])
        print("\n--- FINAL EVALUATION SUMMARY ---")
        print(summary_df.to_string(index=False))
    else:
        print("No evaluation summary file was generated.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate benchmark models.")
    parser.add_argument(
        '--tag', 
        type=str, 
        default=None,
        help="Specify a model tag from training_log.csv to evaluate. If not provided, all models will be evaluated."
    )
    parser.add_argument(
        '--rank',
        type=int,
        default=None,
        choices=range(1, 6),
        help="Specify the rank of the checkpoint to evaluate (1 to 5). Requires --tag to be set."
    )
    args = parser.parse_args()
    
    if args.rank and not args.tag:
        parser.error("--rank requires --tag to be specified.")

    main(args)