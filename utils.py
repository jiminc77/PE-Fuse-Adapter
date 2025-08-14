import itertools
import json
import hashlib
from typing import List, Dict, Any

def _as_list(value: Any) -> list:
    return value if isinstance(value, (list, tuple)) else [value]

def create_hash_suffix(obj: Dict) -> str:
    serialized_obj = json.dumps(obj, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.md5(serialized_obj).hexdigest()[:8]

def create_calib_grid(calib_cfg: Dict) -> List[Dict]:
    betas = _as_list(calib_cfg.get('beta', 0.5))
    warmups = _as_list(calib_cfg.get('warmup_epochs', 12))
    return [{'beta': float(b), 'warmup_epochs': int(w)} for b, w in itertools.product(betas, warmups)]

def create_train_grid(cfg: Dict) -> List[Dict]:
    train_cfg = cfg['training']
    model_cfg = cfg['model']
    
    learning_rates = _as_list(train_cfg['lr'])
    multipliers = _as_list(train_cfg['interaction_lr_multiplier'])
    weight_decays = _as_list(train_cfg['weight_decay'])
    dropouts = _as_list(model_cfg.get('dropout', [0.4]))
    
    grid = []
    for lr, mult, wd, drop in itertools.product(learning_rates, multipliers, weight_decays, dropouts):
        grid.append({'lr': float(lr), 'mult': float(mult), 'wd': float(wd), 'drop': float(drop)})
    return grid