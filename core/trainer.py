import os
import json
import logging
import shutil
from typing import Dict, Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score
import pandas as pd

import core.vision_encoder.pe as pe

def _infer_counts(dataset, num_classes: int) -> List[int]:
    labels = [y for _, y in dataset.samples]
    
    if labels is None:
        labels = []
        try:
            for i in range(len(dataset)):
                _, y = dataset[i]
                labels.append(int(y))
        except Exception:
            logging.warning("Failed to infer class counts by iterating dataset.")

    counts = [0] * num_classes
    for label in labels:
        if 0 <= int(label) < num_classes:
            counts[int(label)] += 1
    return counts

def _safe_get_dataset_paths(dataset) -> Optional[List[str]]:
    if hasattr(dataset, "samples"):
        try: return [p for p, _ in dataset.samples]
        except Exception: pass
    return None

class Trainer:
    def __init__(self, pe_model: pe.CLIP, adapter_model: nn.Module, dataloaders: Dict[str, DataLoader],
                 text_features: torch.Tensor, config: Dict, model_save_path: str):
        
        self.pe_model = pe_model
        self.adapter_model = adapter_model
        self.dataloaders = dataloaders
        self.config = config
        self.train_cfg = config["training"]
        self.calib_cfg = config.get("calibration", {})
        self.sel_cfg = config.get("selection", {})
        self.device = self.train_cfg["device"]
        self.text_features = text_features.to(self.device)
        self.model_save_path = model_save_path
        self.calib_save_path = model_save_path.replace(".pt", "_calib_ovn.json")

        cudnn.benchmark = True

        self.class_names = config["data"]["class_names"]
        self.idx_fire = self.class_names.index("fire")
        self.idx_fall = self.class_names.index("falldown")
        self.idx_norm = self.class_names.index("normal")
        self.num_classes = len(self.class_names)

        self.optimizer = self._create_optimizer()
        self.scaler = GradScaler()
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="max", factor=0.5, patience=5)

        counts = _infer_counts(self.dataloaders["train"].dataset, self.num_classes)
        n_fire, n_fall, n_norm = counts[self.idx_fire], counts[self.idx_fall], counts[self.idx_norm]
        pw_fire = torch.tensor([max(n_norm, 1) / max(n_fire, 1)], device=self.device)
        pw_fall = torch.tensor([max(n_norm, 1) / max(n_fall, 1)], device=self.device)
        use_pos_weight = self.train_cfg.get("use_pos_weight", True)
        self.crit_fire = nn.BCEWithLogitsLoss(pos_weight=pw_fire if use_pos_weight else None)
        self.crit_fall = nn.BCEWithLogitsLoss(pos_weight=pw_fall if use_pos_weight else None)

        self.calib = {"fire": {"s": 1.0, "b": 0.0}, "falldown": {"s": 1.0, "b": 0.0}, "mode": self.calib_cfg.get("gating_mode", "margin")}
        self.thresholds = {"fire": 0.0, "falldown": 0.0}

        self.lock_after_warmup = bool(self.calib_cfg.get("lock_after_warmup", True))
        self.warmup_epochs = int(self.calib_cfg.get("warmup_epochs", 12))
        self.post_lock_max_epochs = int(self.calib_cfg.get("post_lock_max_epochs", 5))
        self.calib_locked = False
        self.lock_epoch = None

        self.save_window_pre_lock = int(self.sel_cfg.get("save_window_pre_lock", 2))
        self.save_window_start_epoch = max(1, self.warmup_epochs - self.save_window_pre_lock)
        self.max_checkpoints = int(self.sel_cfg.get("max_checkpoints", 3))

        base = os.path.splitext(os.path.basename(self.model_save_path))[0]
        self.ckpt_dir = os.path.dirname(self.model_save_path)
        self.registry_path = os.path.join(self.ckpt_dir, f"{base}_registry.json")
        self.top_ckpts: List[Dict] = self._load_registry()

        errs_cfg = self.calib_cfg.get("error_logging", {})
        self.err_top_k = int(errs_cfg.get("top_k", 5))
        self.err_save_dir = errs_cfg.get("save_dir", "logs/errors")
        os.makedirs(self.err_save_dir, exist_ok=True)

        self.bootstrap = int(self.calib_cfg.get("bootstrap", 7))
        self.quantile_q = float(self.calib_cfg.get("quantile_q", 0.7))
        self.affine_reg = float(self.calib_cfg.get("affine_reg", 0.0))
        self.grid_points = int(self.calib_cfg.get("grid_points", 201))
        self.obj = self.calib_cfg.get("sweep_objective", "fbeta")
        self.beta = float(self.calib_cfg.get("beta", 0.5))
        
        self.sel_policy = self.sel_cfg.get("policy", "avg")

    def _create_optimizer(self) -> optim.Optimizer:
        interaction_param_names = ["interaction_weight", "class_specific_adjustments"]
        inter, base = [], []
        for name, p in self.adapter_model.named_parameters():
            if not p.requires_grad:
                continue
            (inter if any(k in name for k in interaction_param_names) else base).append(p)

        base_lr = float(self.train_cfg.get("lr", 5e-4))
        mult    = float(self.train_cfg.get("interaction_lr_multiplier", 1.0))
        wd      = float(self.train_cfg.get("weight_decay", 1e-4))

        return optim.AdamW(
            [
                {"params": base, "lr": base_lr, "weight_decay": wd},
                {"params": inter, "lr": base_lr * mult, "weight_decay": wd},
            ]
        )

    def _load_registry(self) -> List[Dict]:
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, "r") as f:
                    data = json.load(f)
                if isinstance(data, list): return data
            except Exception:
                pass
        return []

    def _save_registry(self):
        with open(self.registry_path, "w") as f:
            json.dump(self.top_ckpts, f, indent=2)

    def _canonicalize_best(self):
        if not self.top_ckpts: return
        best = max(self.top_ckpts, key=lambda d: d["sel_score"])
        if os.path.exists(best["model"]):
            shutil.copyfile(best["model"], self.model_save_path)
        if os.path.exists(best["calib"]):
            shutil.copyfile(best["calib"], self.calib_save_path)

    def _maybe_update_top_checkpoints(self, epoch: int, sel_score: float):
        if (epoch < self.save_window_start_epoch):
            return
        base = os.path.splitext(os.path.basename(self.model_save_path))[0]
        model_fname = f"{base}_ep{epoch:03d}_sel{sel_score:.4f}.pt"
        calib_fname = f"{base}_ep{epoch:03d}_sel{sel_score:.4f}_calib_ovn.json"
        model_out = os.path.join(self.ckpt_dir, model_fname)
        calib_out = os.path.join(self.ckpt_dir, calib_fname)
        if len(self.top_ckpts) < self.max_checkpoints or sel_score > min(c["sel_score"] for c in self.top_ckpts):
            torch.save(self.adapter_model.state_dict(), model_out)
            with open(calib_out, "w") as f:
                json.dump({
                    "mode": self.calib.get("mode","margin"),
                    "affine": self.calib,
                    "thresholds": self.thresholds,
                    "class_names": self.class_names,
                    "idx_map": {"fire": self.idx_fire, "falldown": self.idx_fall, "normal": self.idx_norm}
                }, f, indent=2)
            self.top_ckpts.append({"epoch": epoch, "sel_score": float(sel_score), "model": model_out, "calib": calib_out})
            self.top_ckpts = sorted(self.top_ckpts, key=lambda d: d["sel_score"], reverse=True)
            while len(self.top_ckpts) > self.max_checkpoints:
                victim = self.top_ckpts.pop(-1)
                try:
                    if os.path.exists(victim["model"]): os.remove(victim["model"])
                    if os.path.exists(victim["calib"]): os.remove(victim["calib"])
                except Exception:
                    pass
            self._save_registry()
            self._canonicalize_best()

    def _compute_margins(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        zf, zd, zn = logits[:, self.idx_fire], logits[:, self.idx_fall], logits[:, self.idx_norm]
        return zf - zn, zd - zn

    def _fit_affine(self, m: np.ndarray, y: np.ndarray, head: str):
        s = torch.tensor([self.calib[head]["s"]], dtype=torch.float32, requires_grad=True)
        b = torch.tensor([self.calib[head]["b"]], dtype=torch.float32, requires_grad=True)
        opt = torch.optim.LBFGS([s, b], lr=0.1, max_iter=80, line_search_fn="strong_wolfe")
        x = torch.from_numpy(m).float()
        t = torch.from_numpy(y).float().unsqueeze(1)
        bce = nn.BCEWithLogitsLoss()
        lam = self.affine_reg

        def closure():
            opt.zero_grad(set_to_none=True)
            ylog = s * x.unsqueeze(1) + b
            loss = bce(ylog, t)
            if lam > 0:
                loss = loss + lam * ((s - 1.0) ** 2 + b ** 2).sum()
            loss.backward()
            return loss

        prev = None
        for _ in range(80):
            loss = opt.step(closure)
            cur = float(loss.item())
            if prev is not None and abs(prev - cur) < 1e-7: break
            prev = cur
        self.calib[head]["s"] = float(s.detach().item())
        self.calib[head]["b"] = float(b.detach().item())

    def _apply_affine(self, m: np.ndarray, head: str, to_prob: bool):
        s, b = self.calib[head]["s"], self.calib[head]["b"]
        z = s * m + b
        return 1.0/(1.0+np.exp(-z)) if to_prob else z

    def _best_threshold_single(self, vals: np.ndarray, y: np.ndarray) -> float:
        qs = np.linspace(0.0, 1.0, self.grid_points)
        cand = np.unique(np.quantile(vals, qs))
        if len(cand) == 0: return 0.0

        best, best_th = -1.0, float(cand[0])

        for th in cand:
            pred = (vals >= th).astype(int)

            p = precision_score(y, pred, zero_division=0)
            if self.obj == "macro_f1":
                score = f1_score(y, pred, average="binary", zero_division=0)
            else:
                r = recall_score(y, pred, zero_division=0)
                beta_sq = self.beta * self.beta
                score = (1 + beta_sq) * p * r / (beta_sq * p + r + 1e-12)

            if score > best:
                best, best_th = score, float(th)

        return float(np.max(cand)) if best < 0 else best_th

    def _tune_threshold_conservative(self, vals: np.ndarray, y: np.ndarray) -> float:
        if self.bootstrap <= 1:
            return self._best_threshold_single(vals, y)
        
        rng = np.random.RandomState(42)
        thresholds = []
        pos_idx, neg_idx = np.where(y == 1)[0], np.where(y == 0)[0]
        
        for _ in range(self.bootstrap):
            pos_bs = rng.choice(pos_idx, len(pos_idx), replace=True)
            neg_bs = rng.choice(neg_idx, len(neg_idx), replace=True)
            idx = np.concatenate([pos_bs, neg_bs])
            
            th = self._best_threshold_single(vals[idx], y[idx])
            thresholds.append(th)
            
        return float(np.quantile(np.sort(np.array(thresholds, dtype=float)), np.clip(self.quantile_q, 0.0, 1.0)))

    def _save_top_errors_csv(self, y_bin: np.ndarray, vals: np.ndarray, th: float,
                             head: str, paths: Optional[List[str]], indices: List[int], top_k: int,
                             epoch_tag: str):
        pred = (vals >= th).astype(int)
        wrong = (pred != y_bin)
        if wrong.sum() == 0: return
        wrongness = np.where(pred > y_bin, vals - th, th - vals)
        wrong_idx = np.where(wrong)[0]
        
        top_idx = wrong_idx[np.argsort(-wrongness[wrong_idx])[:top_k]]
        
        recs = []
        for j in top_idx:
            ds_idx = indices[j] if j < len(indices) else j
            recs.append({
                "index": int(ds_idx),
                "path": (paths[ds_idx] if (paths is not None and ds_idx < len(paths)) else ""),
                "head": head, "label": int(y_bin[j]), "pred": int(pred[j]),
                "value": float(vals[j]), "threshold": float(th),
                "wrongness": float(wrongness[j]),
                "error_type": "FP" if pred[j]==1 else "FN",
            })
        os.makedirs(self.err_save_dir, exist_ok=True)
        pd.DataFrame.from_records(recs).sort_values("wrongness", ascending=False)\
            .to_csv(os.path.join(self.err_save_dir, f"{epoch_tag}_{head}_top{top_k}.csv"), index=False)

    def _selection_score(self, fire_metrics: Dict[str,float], fall_metrics: Dict[str,float]) -> float:
        if self.sel_policy == "min":
            return float(min(fire_metrics["f1"], fall_metrics["f1"]))
        return float(0.5 * (fire_metrics["f1"] + fall_metrics["f1"]))

    def train(self) -> Dict:
        os.makedirs(self.ckpt_dir, exist_ok=True)

        best_sel_score = -1e9
        best_epoch = -1
        epochs_no_improve = 0
        patience = int(self.train_cfg.get("early_stop_patience", 20))

        tr_loader = self.dataloaders["train"]
        val_loader = self.dataloaders["test"]

        val_dataset = val_loader.dataset
        val_paths = _safe_get_dataset_paths(val_dataset)
        is_sequential = isinstance(val_loader.sampler, SequentialSampler)

        for ep in range(self.train_cfg["epochs"]):
            epoch_num = ep + 1

            self.adapter_model.train(); self.pe_model.eval()
            tot_loss = 0.0
            for imgs, labels in tqdm(tr_loader, desc=f"Epoch {epoch_num}/{self.train_cfg['epochs']} [Training]"):
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad(set_to_none=True)
                with autocast(self.device):
                    with torch.no_grad():
                        feats = self.pe_model.encode_image(imgs, normalize=True)
                    logits = self.adapter_model(feats, self.text_features)
                    m_fire, m_fall = self._compute_margins(logits)
                    mask_fire = (labels == self.idx_fire) | (labels == self.idx_norm)
                    mask_fall = (labels == self.idx_fall) | (labels == self.idx_norm)
                    y_fire = (labels == self.idx_fire).float()
                    y_fall = (labels == self.idx_fall).float()
                    loss_fire = self.crit_fire(m_fire[mask_fire].unsqueeze(1), y_fire[mask_fire].unsqueeze(1)) if mask_fire.any() else m_fire.sum()*0
                    loss_fall = self.crit_fall(m_fall[mask_fall].unsqueeze(1), y_fall[mask_fall].unsqueeze(1)) if mask_fall.any() else m_fall.sum()*0
                    loss = loss_fire + loss_fall
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                tot_loss += float(loss.item())
            avg_loss = tot_loss / max(1, len(tr_loader))

            self.adapter_model.eval()
            logits_list, labels_list = [], []
            with torch.no_grad():
                for imgs, labels in tqdm(val_loader, desc="Collecting logits (Val)"):
                    imgs, labels = imgs.to(self.device), labels.to(self.device)
                    with autocast(self.device):
                        feats = self.pe_model.encode_image(imgs, normalize=True)
                        logits = self.adapter_model(feats, self.text_features)
                    logits_list.append(logits.float().cpu()); labels_list.append(labels.cpu())
            val_logits = torch.cat(logits_list); val_labels = torch.cat(labels_list)
            
            indices = list(range(len(val_labels))) if is_sequential else list(val_loader.sampler)
            if not is_sequential and val_paths:
                logging.warning("Val sampler is not sequential; error CSV paths may misalign.")

            m_fire, m_fall = self._compute_margins(val_logits)
            m_fire_np, m_fall_np = m_fire.numpy(), m_fall.numpy()
            y_all = val_labels.numpy()

            mask_fire = (y_all == self.idx_fire) | (y_all == self.idx_norm)
            mask_fall = (y_all == self.idx_fall) | (y_all == self.idx_norm)
            y_fire = (y_all == self.idx_fire).astype(int)[mask_fire]
            y_fall = (y_all == self.idx_fall).astype(int)[mask_fall]
            v_idx_fire = np.array(indices, dtype=int)[mask_fire]
            v_idx_fall = np.array(indices, dtype=int)[mask_fall]

            mode = self.calib.get("mode", "margin")
            if not self.calib_locked:
                self._fit_affine(m_fire_np[mask_fire], y_fire, "fire")
                self._fit_affine(m_fall_np[mask_fall], y_fall, "falldown")
                
                v_fire = self._apply_affine(m_fire_np[mask_fire], "fire", to_prob=(mode=="prob"))
                v_fall = self._apply_affine(m_fall_np[mask_fall], "falldown", to_prob=(mode=="prob"))
                
                self.thresholds["fire"] = self._tune_threshold_conservative(v_fire, y_fire)
                self.thresholds["falldown"] = self._tune_threshold_conservative(v_fall, y_fall)

                if self.lock_after_warmup and (epoch_num >= self.warmup_epochs):
                    self.calib_locked = True
                    self.lock_epoch = epoch_num
                    logging.info(f"Calibration LOCKED at epoch {epoch_num}. Post-lock training limited to {self.post_lock_max_epochs} epochs.")

            v_fire_all = self._apply_affine(m_fire_np, "fire", to_prob=(mode=="prob"))
            v_fall_all = self._apply_affine(m_fall_np, "falldown", to_prob=(mode=="prob"))
            vals_fire = v_fire_all[mask_fire]; vals_fall = v_fall_all[mask_fall]
            pred_fire = (vals_fire >= self.thresholds["fire"]).astype(int)
            pred_fall = (vals_fall >= self.thresholds["falldown"]).astype(int)

            fire_metrics = {"f1": f1_score(y_fire, pred_fire, zero_division=0), "precision": precision_score(y_fire, pred_fire, zero_division=0), "recall": recall_score(y_fire, pred_fire, zero_division=0)}
            fall_metrics = {"f1": f1_score(y_fall, pred_fall, zero_division=0), "precision": precision_score(y_fall, pred_fall, zero_division=0), "recall": recall_score(y_fall, pred_fall, zero_division=0)}
            
            ep_tag = f"epoch_{epoch_num:03d}"
            if self.err_top_k > 0:
                self._save_top_errors_csv(y_fire, vals_fire, self.thresholds["fire"], "fire", val_paths, list(v_idx_fire), self.err_top_k, ep_tag)
                self._save_top_errors_csv(y_fall, vals_fall, self.thresholds["falldown"], "falldown", val_paths, list(v_idx_fall), self.err_top_k, ep_tag)

            logging.info("="*105)
            logging.info(f"Epoch {epoch_num} | Loss: {avg_loss:.4f} | [FIRE] P={fire_metrics['precision']:.3f} R={fire_metrics['recall']:.3f} F1={fire_metrics['f1']:.3f} | [FALL] P={fall_metrics['precision']:.3f} R={fall_metrics['recall']:.3f} F1={fall_metrics['f1']:.3f} | locked={self.calib_locked}")
            logging.info("="*105)

            sel_score = self._selection_score(fire_metrics, fall_metrics)
            self.scheduler.step(sel_score if sel_score > -1e8 else 0.0)
            self._maybe_update_top_checkpoints(epoch=epoch_num, sel_score=sel_score)

            if sel_score > best_sel_score:
                best_sel_score = sel_score
                best_epoch = epoch_num
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    logging.info("Early stopping by selection criterion (patience).")
                    break

            if self.calib_locked and self.lock_epoch is not None:
                if epoch_num >= (self.lock_epoch + self.post_lock_max_epochs):
                    logging.info(f"Reached post-lock epoch budget. Stopping.")
                    break

        self._canonicalize_best()
        logging.info(f"Training done. Best selection score: {best_sel_score:.4f} at epoch {best_epoch}")
        logging.info(f"Top-{self.max_checkpoints} checkpoints registry: {self.registry_path}")
        
        return {"best_sel_score": best_sel_score, "best_epoch": best_epoch}