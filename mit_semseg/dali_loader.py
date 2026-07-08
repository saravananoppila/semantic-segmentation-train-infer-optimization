"""DALI-based training data loader (experimental — for the batch=20 + checkpointing
data-bottleneck re-test). GPU-accelerated decode + resize + flip + normalize, replacing
the CPU PIL pipeline. Yields the same [{'img_data','seg_label'}] structure as CudaPrefetcher
so the training loop is unchanged.

Note: augmentation is representative, not bit-identical to the PIL pipeline (DALI batches are
dense/uniform-size, so images are resized to a fixed crop rather than per-batch padded). This
is fine for the throughput/data_time measurement it's built for, but loss/acc are not directly
comparable to the PIL-pipeline experiments.
"""
import os
import json
import torch
from nvidia.dali import pipeline_def, fn, types
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

_SCALES = [300., 375., 450., 525., 600.]                  # match DATASET.imgSizes (short edge)
_MEAN = [0.485 * 255, 0.456 * 255, 0.406 * 255]
_STD = [0.229 * 255, 0.224 * 255, 0.225 * 255]


@pipeline_def
def _seg_pipeline(img_files, seg_files, crop_h, crop_w, seg_rate):
    jpegs, _ = fn.readers.file(files=img_files, random_shuffle=True, seed=42, name="img_reader")
    pngs, _ = fn.readers.file(files=seg_files, random_shuffle=True, seed=42, name="seg_reader")
    img = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)        # GPU decode
    seg = fn.decoders.image(pngs, device="mixed", output_type=types.GRAY)        # GPU decode
    short = fn.random.uniform(values=_SCALES)                                     # shared scale
    mirror = fn.random.coin_flip(probability=0.5)                                 # shared flip
    img = fn.resize(img, resize_shorter=short, interp_type=types.INTERP_LINEAR)
    seg = fn.resize(seg, resize_shorter=short, interp_type=types.INTERP_NN)
    img = fn.crop_mirror_normalize(
        img, dtype=types.FLOAT, output_layout="CHW", crop=(crop_h, crop_w),
        crop_pos_x=0.5, crop_pos_y=0.5, mirror=mirror, mean=_MEAN, std=_STD,
        out_of_bounds_policy="pad", fill_values=[0., 0., 0.])  # 0 = normalized mean (gray pad)
    seg = fn.crop_mirror_normalize(
        seg, dtype=types.FLOAT, output_layout="CHW", crop=(crop_h, crop_w),
        crop_pos_x=0.5, crop_pos_y=0.5, mirror=mirror, mean=[0.], std=[1.],
        out_of_bounds_policy="pad", fill_values=[0.])
    seg = fn.resize(seg, size=[crop_h // seg_rate, crop_w // seg_rate], interp_type=types.INTERP_NN)
    return img, seg


class DaliTrainLoader:
    """Yields [{'img_data': (B,3,H,W) float cuda channels_last,
               'seg_label': (B,H/r,W/r) long cuda}] to match the existing loop."""
    def __init__(self, cfg, crop_h=448, crop_w=576):
        recs = [json.loads(x) for x in open(cfg.DATASET.list_train)]
        root = cfg.DATASET.root_dataset
        img_files = [os.path.join(root, r['fpath_img']) for r in recs]
        seg_files = [os.path.join(root, r['fpath_segm']) for r in recs]
        self.seg_rate = cfg.DATASET.segm_downsampling_rate
        pipe = _seg_pipeline(
            img_files, seg_files, crop_h, crop_w, self.seg_rate,
            batch_size=cfg.TRAIN.batch_size_per_gpu, num_threads=cfg.TRAIN.workers,
            device_id=0)
        pipe.build()
        self._it = DALIGenericIterator(
            [pipe], ['img_data', 'seg_label'], reader_name='img_reader',
            last_batch_policy=LastBatchPolicy.DROP, auto_reset=True)

    def __iter__(self):
        return self

    def __next__(self):
        data = next(self._it)[0]
        img = data['img_data'].to(memory_format=torch.channels_last)
        seg = (data['seg_label'].squeeze(1).long() - 1)   # 0..150 -> -1..149 (ignore_index=-1)
        return [{'img_data': img, 'seg_label': seg.contiguous()}]
