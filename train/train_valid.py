import time

import torch
import numpy as np

from network.metrics import name2key_metrics
from train.train_tools import to_cuda


class ValidationEvaluator:
    default_cfg = {}

    def __init__(self, cfg):
        self.cfg = {**self.default_cfg, **cfg}
        self.key_metric_name = cfg['key_metric_name']
        self.key_metric = name2key_metrics[self.key_metric_name]

    def __call__(self, model, losses, eval_dataset, step, model_name, val_set_name=None):
        if val_set_name is not None: model_name = f'{model_name}-{val_set_name}'
        model.eval()
        eval_results = {}
        eval_images = {}
        # Get actual dataset length (DataLoader might wrap it)
        if hasattr(eval_dataset, 'dataset'):
            dataset_len = len(eval_dataset.dataset)
        else:
            dataset_len = len(eval_dataset)
        
        # Pick random index BEFORE iterating (will be validated during iteration)
        seed = step + hash(model_name) % 1000000
        if dataset_len > 1:
            rng = np.random.RandomState(seed)
            random_save_index = rng.randint(0, dataset_len)
            print(f'[Validation] dataset_len={dataset_len}, random_save_index={random_save_index} (step={step})')
        elif dataset_len == 1:
            random_save_index = 0
            print(f'[Validation] WARNING: dataset_len=1, using index 0')
        else:
            random_save_index = 0
            print(f'[Validation] WARNING: dataset_len=0, using index 0')
        
        # Validate only the single randomly chosen image (faster; metrics are for that image only)
        dataset = eval_dataset.dataset if hasattr(eval_dataset, 'dataset') else eval_dataset
        data = dataset[random_save_index]
        data = to_cuda(data)
        data['eval'] = True
        data['step'] = step

        begin = time.time()
        with torch.no_grad():
            outputs = model(data)

        for loss in losses:
            loss_results = loss(
                outputs,
                data,
                step,
                data_index=random_save_index,
                model_name=model_name,
                random_save_index=random_save_index,
            )
            for k, v in loss_results.items():
                if type(v) == torch.Tensor:
                    v = v.detach().cpu().numpy()
                # Keep images for wandb (gt/pred)
                if isinstance(v, np.ndarray) and v.ndim >= 2 and k in ('val_gt_rgb', 'val_pred_rgb', 'val_gt_pred'):
                    if k not in eval_images:
                        eval_images[k] = v
                    continue
                if k in eval_results:
                    eval_results[k].append(v)
                else:
                    eval_results[k] = [v]

        for k, v in eval_results.items():
            eval_results[k] = np.concatenate(v, axis=0)

        key_metric_val = self.key_metric(eval_results)
        eval_results[self.key_metric_name] = key_metric_val
        print(f'eval cost {time.time() - begin:.2f} s (1 image, index={random_save_index})')
        torch.cuda.empty_cache()
        return eval_results, key_metric_val, eval_images
