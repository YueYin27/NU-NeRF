import argparse
import os
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch
from tqdm import tqdm

from utils.base_utils import color_map_backward, load_cfg


def _get_renderer_and_utils(cfg):
    if cfg.get('zero_thickness', False):
        from network.renderer_zerothick import name2renderer, build_imgs_info, imgs_info_to_torch
    else:
        from network.renderer import name2renderer, build_imgs_info, imgs_info_to_torch
    return name2renderer, build_imgs_info, imgs_info_to_torch


def _load_checkpoint(model, cfg, ckpt_path=None):
    if ckpt_path is None:
        model_dir = Path('data/model') / cfg['name']
        best_ckpt = model_dir / 'model_best.pth'
        last_ckpt = model_dir / 'model.pth'
        ckpt_path = best_ckpt if best_ckpt.exists() else last_ckpt

    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

    checkpoint = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['network_state_dict'], strict=False)
    step = int(checkpoint.get('step', 0))
    print(f'[eval] Loaded checkpoint: {ckpt_path} (step={step})')
    return step


def _resolve_scene_name(cfg):
    db_name = cfg.get('database_name', '')
    if not db_name:
        return cfg.get('name', 'scene')
    return db_name.split('/')[-1]


def _choose_test_ids(database):
    split_ids = getattr(database, 'split_ids', None)
    if isinstance(split_ids, dict) and 'test' in split_ids:
        test_ids = split_ids['test']
        print(f'[eval] Using explicit test split from transforms_test.json: {len(test_ids)} images')
        return test_ids

    from dataset.database import get_database_split

    try:
        _, test_ids = get_database_split(database, 'test')
        print(f'[eval] Using dataset test split from get_database_split(..., "test"): {len(test_ids)} images')
        return test_ids
    except Exception:
        _, test_ids = get_database_split(database, 'validation')
        print(f'[eval] Falling back to validation split: {len(test_ids)} images')
        return test_ids


def _maybe_rebuild_full_test_database(cfg, network):
    database_name = cfg.get('database_name', '')
    if not database_name.startswith('nerf/'):
        return

    from dataset.database import NeRFSyntheticDatabase

    pose_scale = cfg.get('pose_scale', 1.0)
    network.database = NeRFSyntheticDatabase(
        database_name,
        cfg['dataset_dir'],
        testskip=1,
        pose_scale=pose_scale,
    )
    print('[eval] Rebuilt nerf database with testskip=1 (use all test images).')


def _force_network_test_split(network, test_ids, build_imgs_info, imgs_info_to_torch):
    get_mask_flag = getattr(network, 'get_mask', network.cfg.get('get_mask', True))
    network.test_ids = list(test_ids)
    try:
        network.test_imgs_info = build_imgs_info(
            network.database,
            network.test_ids,
            network.is_nerf,
            get_mask_flag,
        )
    except TypeError:
        network.test_imgs_info = build_imgs_info(
            network.database,
            network.test_ids,
            network.is_nerf,
        )
    network.test_imgs_info = imgs_info_to_torch(network.test_imgs_info, 'cpu')
    network.test_num = len(network.test_ids)


def _image_name_from_id(database, img_id, fallback_index):
    image_names = getattr(database, 'image_names', None)
    if isinstance(image_names, dict) and img_id in image_names:
        return Path(image_names[img_id]).name

    if isinstance(image_names, list):
        try:
            idx = int(img_id)
            if 0 <= idx < len(image_names):
                return Path(image_names[idx]).name
        except Exception:
            pass

    return f'{fallback_index:06d}.png'


def _relative_image_name(database, img_id):
    image_names = getattr(database, 'image_names', None)
    if isinstance(image_names, dict) and img_id in image_names:
        return image_names[img_id]
    if isinstance(image_names, list):
        try:
            idx = int(img_id)
            if 0 <= idx < len(image_names):
                return image_names[idx]
        except Exception:
            return None
    return None


def _load_gt_distance_map(database, img_id):
    root = getattr(database, 'root', None)
    rel_rgb = _relative_image_name(database, img_id)
    if root is None or rel_rgb is None:
        return None

    rgb_path = Path(root) / rel_rgb
    stem = rgb_path.stem
    candidates = [
        rgb_path.with_name(f'{stem}_depth_0000.png'),
        rgb_path.with_name(f'{stem}_depth_0001.png'),
    ]

    for candidate in candidates:
        if candidate.exists():
            dist = imageio.imread(str(candidate))
            if dist.ndim == 3:
                dist = dist[..., 0]
            src_dtype = dist.dtype
            dist = dist.astype(np.float32)
            if src_dtype == np.uint16 or dist.max() > 255:
                dist = dist / 65535.0 * 15.0
            return dist

    wildcard = list(rgb_path.parent.glob(f'{stem}_depth_*.png'))
    if wildcard:
        dist = imageio.imread(str(wildcard[0]))
        if dist.ndim == 3:
            dist = dist[..., 0]
        src_dtype = dist.dtype
        dist = dist.astype(np.float32)
        if src_dtype == np.uint16 or dist.max() > 255:
            dist = dist / 65535.0 * 15.0
        return dist

    return None


def _to_uint8_rgb(img):
    if img.dtype == np.uint8:
        return img
    if img.max() <= 1.0:
        return color_map_backward(img)
    return np.clip(img, 0, 255).astype(np.uint8)


def _depth_to_vis(depth, min_val, max_val):
    d = np.nan_to_num(depth.astype(np.float32), nan=max_val, posinf=max_val, neginf=min_val)
    denom = max(max_val - min_val, 1e-8)
    d = np.clip((d - min_val) / denom, 0.0, 1.0)
    d = (d * 255.0).astype(np.uint8)
    return np.repeat(d[..., None], 3, axis=-1)


def _distance_to_vis_fixed(distance, new_min=1.0, new_max=11.5):
    distance = np.nan_to_num(distance.astype(np.float32), nan=new_max, posinf=new_max, neginf=new_min)
    norm = (new_max - distance) / max(new_max - new_min, 1e-8)
    norm = np.clip(norm, 0.0, 1.0)
    img = (norm * 255.0).astype(np.uint8)
    return np.repeat(img[..., None], 3, axis=-1)


def _build_distance_pair(gt_depth, pred_depth, gt_mask=None):
    gt_depth = gt_depth.astype(np.float32)
    pred_depth = pred_depth.astype(np.float32)

    valid = np.isfinite(gt_depth) & np.isfinite(pred_depth)
    if gt_mask is not None:
        valid = valid & (gt_mask > 0)

    if np.any(valid):
        min_val = float(np.minimum(gt_depth[valid].min(), pred_depth[valid].min()))
        max_val = float(np.maximum(gt_depth[valid].max(), pred_depth[valid].max()))
    else:
        min_val = float(np.nanmin(np.nan_to_num(gt_depth, nan=0.0)))
        max_val = float(np.nanmax(np.nan_to_num(pred_depth, nan=1.0)))
        if max_val <= min_val:
            max_val = min_val + 1.0

    gt_vis = _depth_to_vis(gt_depth, min_val, max_val)
    pred_vis = _depth_to_vis(pred_depth, min_val, max_val)
    return np.concatenate([gt_vis, pred_vis], axis=1)


def _build_distance_pair_r3f(gt_distance, pred_distance, new_min=1.0, new_max=11.5):
    gt_vis = _distance_to_vis_fixed(gt_distance, new_min=new_min, new_max=new_max)
    pred_vis = _distance_to_vis_fixed(pred_distance, new_min=new_min, new_max=new_max)
    return np.concatenate([gt_vis, pred_vis], axis=1)


def _extract_pred_depth(outputs):
    if 'depth' not in outputs:
        return None

    depth = outputs['depth'].detach().cpu().numpy()
    if depth.ndim == 3 and depth.shape[-1] == 1:
        return depth[..., 0]

    if depth.ndim == 2 and depth.shape[-1] == 1:
        h, w, _ = outputs['gt_rgb'].shape
        return depth.reshape(h, w)

    if depth.ndim == 2:
        h, w, _ = outputs['gt_rgb'].shape
        return depth.reshape(h, w)

    return None


def _effective_K(model, local_idx, out_h, out_w):
    K = model.test_imgs_info['Ks'][local_idx].detach().cpu().numpy().astype(np.float32)
    if model.cfg.get('test_downsample_ratio', False):
        _, h0, w0 = model.test_imgs_info['imgs'][local_idx].shape
        ratio = float(model.cfg.get('downsample_ratio', 1.0))
        dh, dw = int(ratio * h0), int(ratio * w0)
        if dh > 0 and dw > 0:
            scale = np.diag([dw / w0, dh / h0, 1.0]).astype(np.float32)
            K = scale @ K
    return K


def _ray_to_camera_z(ray_distance, K):
    h, w = ray_distance.shape
    ys, xs = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing='ij')
    x_cam = (xs + 0.5 - K[0, 2]) / K[0, 0]
    y_cam = (ys + 0.5 - K[1, 2]) / K[1, 1]
    norm = np.sqrt(x_cam * x_cam + y_cam * y_cam + 1.0)
    z_distance = ray_distance / np.maximum(norm, 1e-8)
    return z_distance.astype(np.float32)


def _directions_norm_from_K(K, h, w):
    ys, xs = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing='ij')
    x_cam = (xs + 0.5 - K[0, 2]) / K[0, 0]
    y_cam = (ys + 0.5 - K[1, 2]) / K[1, 1]
    return np.sqrt(x_cam * x_cam + y_cam * y_cam + 1.0).astype(np.float32)


def _to_unit_depth(raw_depth):
    raw_depth = raw_depth.astype(np.float32)
    max_val = float(np.nanmax(raw_depth)) if raw_depth.size > 0 else 0.0
    if max_val <= 1.0:
        return np.clip(raw_depth, 0.0, 1.0)
    if max_val > 255.0:
        return np.clip(raw_depth / 65535.0, 0.0, 1.0)
    return np.clip(raw_depth / 255.0, 0.0, 1.0)


def _convert_pred_depth_to_distance(pred_depth_raw, directions_norm, pose_scale=1.0, depth_type='ray', pred_scale=1.0):
    pred = pred_depth_raw.astype(np.float64)
    if depth_type == 'z':
        pred = pred * directions_norm
    pred /= max(float(pose_scale), 1e-8)
    pred *= float(pred_scale)
    return pred.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--ckpt', type=str, default=None, help='Optional checkpoint path. Defaults to model_best.pth then model.pth')
    parser.add_argument('--output_root', type=str, default='outputs', help='Root output folder')
    parser.add_argument('--scene_name', type=str, default=None, help='Optional override scene name for outputs/<scene_name>')
    parser.add_argument('--step', type=int, default=0, help='Eval step value passed into the renderer')
    parser.add_argument('--pred_depth_type', type=str, choices=['ray', 'z'], default='ray',
                        help='Interpretation of model output depth: ray distance (ray) or camera-z (z)')
    parser.add_argument('--pred_depth_scale', type=float, default=1.0,
                        help='Extra multiplicative scale on predicted depth after pose_scale correction')
    args = parser.parse_args()

    cfg = load_cfg(args.cfg)
    name2renderer, build_imgs_info, imgs_info_to_torch = _get_renderer_and_utils(cfg)

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this repo eval path, but no CUDA device is available.')

    model = name2renderer[cfg['network']](cfg).cuda().eval()
    ckpt_step = _load_checkpoint(model, cfg, args.ckpt)

    _maybe_rebuild_full_test_database(cfg, model)

    # Ensure test set uses transforms_test.json when available.
    test_ids = _choose_test_ids(model.database)
    _force_network_test_split(model, test_ids, build_imgs_info, imgs_info_to_torch)

    scene_name = args.scene_name or _resolve_scene_name(cfg)
    scene_root = Path(args.output_root) / scene_name
    save_rgb = False  # RGB pairs are already good; disable to speed up reruns.
    rgb_dir = scene_root / 'rgb_images'
    dist_dir = scene_root / 'distance_maps'
    if save_rgb:
        rgb_dir.mkdir(parents=True, exist_ok=True)
    dist_dir.mkdir(parents=True, exist_ok=True)

    print(f'[eval] Output folder: {scene_root}')
    if not save_rgb:
        print('[eval] RGB export disabled (save_rgb=False).')

    eval_step = args.step if args.step > 0 else ckpt_step
    new_max = 11.5
    new_min = 1.0
    old_max = 1.0
    old_min = 0.0
    scale_factor = float(cfg.get('pose_scale', 1.0))
    print(f'[eval] pred_depth_type={args.pred_depth_type}, pred_depth_scale={args.pred_depth_scale}, pose_scale={scale_factor}')

    for local_idx, img_id in enumerate(tqdm(model.test_ids, desc='Evaluating test set')):
        data = {'eval': True, 'index': local_idx, 'step': eval_step}
        with torch.no_grad():
            outputs = model(data)

        file_name = _image_name_from_id(model.database, img_id, local_idx)
        if save_rgb:
            gt_rgb = outputs['gt_rgb'].detach().cpu().numpy()
            pred_rgb = outputs['ray_rgb'].detach().cpu().numpy()
            gt_rgb_u8 = _to_uint8_rgb(gt_rgb)
            pred_rgb_u8 = _to_uint8_rgb(pred_rgb)
            rgb_pair = np.concatenate([gt_rgb_u8, pred_rgb_u8], axis=1)
            rgb_out = rgb_dir / file_name
            imageio.imwrite(str(rgb_out), rgb_pair)

        gt_depth_raw = _load_gt_distance_map(model.database, img_id)
        if gt_depth_raw is None:
            gt_depth_raw = outputs['gt_depth'].detach().cpu().numpy().squeeze(-1)

        gt_mask = outputs['gt_mask'].detach().cpu().numpy().squeeze(-1) if 'gt_mask' in outputs else None
        pred_ray_depth = _extract_pred_depth(outputs)

        out_h, out_w, _ = outputs['gt_rgb'].shape
        if gt_depth_raw.shape != (out_h, out_w):
            gt_depth_raw = cv2.resize(gt_depth_raw, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
            if gt_mask is not None and gt_mask.shape != (out_h, out_w):
                gt_mask = cv2.resize(gt_mask.astype(np.uint8), (out_w, out_h), interpolation=cv2.INTER_NEAREST)

        K_eff = _effective_K(model, local_idx, out_h, out_w)
        directions_norm = _directions_norm_from_K(K_eff, out_h, out_w)

        depth_gt_unit = _to_unit_depth(gt_depth_raw)
        depth_gt_world = new_min + (new_max - new_min) * (old_max - depth_gt_unit) / (old_max - old_min)
        distance_gt = depth_gt_world * directions_norm

        pred_depth = None
        if pred_ray_depth is not None:
            pred_depth = _convert_pred_depth_to_distance(
                pred_ray_depth,
                directions_norm,
                pose_scale=scale_factor,
                depth_type=args.pred_depth_type,
                pred_scale=args.pred_depth_scale,
            )

        if pred_depth is None:
            pred_depth = np.zeros_like(distance_gt, dtype=np.float32)

        dist_pair = _build_distance_pair_r3f(distance_gt, pred_depth, new_min=new_min, new_max=new_max)
        dist_out = dist_dir / file_name
        imageio.imwrite(str(dist_out), dist_pair)

    print('[eval] Done.')
    if save_rgb:
        print(f'[eval] RGB pairs: {rgb_dir}')
    print(f'[eval] Distance pairs: {dist_dir}')


if __name__ == '__main__':
    main()
