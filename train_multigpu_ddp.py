# Multi-GPU training with DistributedDataParallel (DDP) — one process per GPU.
#
# Launch with torchrun, e.g. for 4 GPUs:
#   torchrun --standalone --nproc_per_node=4 train_multigpu_ddp.py \
#       --cfg config/ade20k-hrnetv2.yaml \
#       TRAIN.amp True TRAIN.fused_loss True TRAIN.batch_size_per_gpu 11 TRAIN.workers 8
#
# Carries over the single-GPU optimization stack PER GPU (BF16, channels_last, fused SGD,
# fused loss, GPU-normalizing CUDA prefetcher). DDP overlaps gradient all-reduce with the
# backward pass, so multi-GPU compute scales well; the new things to watch are the data
# pipeline (now feeding N GPUs) and the all-reduce communication.
#
# Key differences vs train_single_gpu.py:
#   - One process per GPU (torchrun sets RANK/LOCAL_RANK/WORLD_SIZE); no UserScatteredDataParallel.
#   - Model wrapped in torch DDP; each rank has its own DataLoader + prefetcher feeding its GPU.
#   - Per-rank data sharding via different seeds; rank-0-only logging and checkpointing.
#   - Linear LR scaling by world_size (the effective batch is world_size x larger).
#   - Optional torch-native SyncBatchNorm (--sync-bn); default is local BN (fine at batch=11, faster).
import os
import time
import random
import argparse
from distutils.version import LooseVersion

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from mit_semseg.config import cfg
from mit_semseg.dataset import TrainDataset
from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.utils import AverageMeter, setup_logger
from mit_semseg.lib.nn import user_scattered_collate


class CudaPrefetcher:
    """Per-rank GPU-normalizing prefetcher: workers ship uint8; we do float/255 + (x-mean)/std
    on the local GPU's side stream and convert to channels_last (overlaps H2D with compute)."""
    def __init__(self, loader, device):
        self.loader = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        self.next_batch = None
        self._preload()

    def _preload(self):
        try:
            self.next_batch = next(self.loader)
        except StopIteration:
            self.next_batch = None
            return
        with torch.cuda.stream(self.stream):
            for d in self.next_batch:
                img = d['img_data'].to(self.device, non_blocking=True).float().div_(255)
                img = img.sub_(self.mean).div_(self.std)
                d['img_data'] = img.to(memory_format=torch.channels_last)
                d['seg_label'] = d['seg_label'].to(self.device, non_blocking=True)

    def __iter__(self):
        return self

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.next_batch
        if batch is None:
            raise StopIteration
        for d in batch:
            d['img_data'].record_stream(torch.cuda.current_stream())
            d['seg_label'].record_stream(torch.cuda.current_stream())
        self._preload()
        return batch


def group_weight(module):
    group_decay, group_no_decay = [], []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.conv._ConvNd):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.batchnorm._BatchNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
    assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
    return [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]


def create_optimizers(net_encoder, net_decoder, cfg):
    opt_enc = torch.optim.SGD(group_weight(net_encoder), lr=cfg.TRAIN.lr_encoder,
                              momentum=cfg.TRAIN.beta1, weight_decay=cfg.TRAIN.weight_decay, fused=True)
    opt_dec = torch.optim.SGD(group_weight(net_decoder), lr=cfg.TRAIN.lr_decoder,
                              momentum=cfg.TRAIN.beta1, weight_decay=cfg.TRAIN.weight_decay, fused=True)
    return (opt_enc, opt_dec)


def adjust_learning_rate(optimizers, cur_iter, cfg):
    scale = ((1. - float(cur_iter) / cfg.TRAIN.max_iters) ** cfg.TRAIN.lr_pow)
    cfg.TRAIN.running_lr_encoder = cfg.TRAIN.lr_encoder * scale
    cfg.TRAIN.running_lr_decoder = cfg.TRAIN.lr_decoder * scale
    for pg in optimizers[0].param_groups:
        pg['lr'] = cfg.TRAIN.running_lr_encoder
    for pg in optimizers[1].param_groups:
        pg['lr'] = cfg.TRAIN.running_lr_decoder


def checkpoint(net_encoder, net_decoder, history, cfg, epoch):
    print('Saving checkpoints...')
    torch.save(history, '{}/history_epoch_{}.pth'.format(cfg.DIR, epoch))
    torch.save(net_encoder.state_dict(), '{}/encoder_epoch_{}.pth'.format(cfg.DIR, epoch))
    torch.save(net_decoder.state_dict(), '{}/decoder_epoch_{}.pth'.format(cfg.DIR, epoch))


def reduce_mean(t, world_size):
    """All-reduce a scalar tensor and average — for correct cross-rank logging."""
    rt = t.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt / world_size


def train(model, iterator, optimizers, history, epoch, cfg, rank, world_size):
    batch_time, data_time = AverageMeter(), AverageMeter()
    ave_loss, ave_acc = AverageMeter(), AverageMeter()
    model.train()
    device = torch.cuda.current_device()

    tic = time.time()
    for i in range(cfg.TRAIN.epoch_iters):
        batch_data = next(iterator)[0]   # prefetcher yields [dict]; one dict per rank
        data_time.update(time.time() - tic)

        model.zero_grad(set_to_none=True)
        cur_iter = i + (epoch - 1) * cfg.TRAIN.epoch_iters
        adjust_learning_rate(optimizers, cur_iter, cfg)

        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=cfg.TRAIN.amp):
            loss, acc = model(batch_data)
            loss = loss.mean()
            acc = acc.mean()

        loss.backward()                  # DDP all-reduces grads here (overlapped with backward)
        for opt in optimizers:
            opt.step()

        batch_time.update(time.time() - tic)
        tic = time.time()

        # average metrics across ranks for honest logging
        gloss = reduce_mean(loss.detach(), world_size).item()
        gacc = reduce_mean(acc.detach(), world_size).item() * 100
        ave_loss.update(gloss)
        ave_acc.update(gacc)

        if rank == 0 and i % cfg.TRAIN.disp_iter == 0:
            imgs_per_s = world_size * cfg.TRAIN.batch_size_per_gpu / max(batch_time.average(), 1e-6)
            print('Epoch: [{}][{}/{}], Time: {:.2f}, Data: {:.2f}, lr_enc: {:.6f}, '
                  'Acc: {:4.2f}, Loss: {:.6f}, {:.1f} img/s (aggregate)'
                  .format(epoch, i, cfg.TRAIN.epoch_iters, batch_time.average(),
                          data_time.average(), cfg.TRAIN.running_lr_encoder,
                          ave_acc.average(), ave_loss.average(), imgs_per_s))
            fe = epoch - 1 + 1. * i / cfg.TRAIN.epoch_iters
            history['train']['epoch'].append(fe)
            history['train']['loss'].append(gloss)
            history['train']['acc'].append(gacc / 100)


def main(cfg, args, rank, local_rank, world_size):
    device = torch.device('cuda', local_rank)

    # ---- model ----
    net_encoder = ModelBuilder.build_encoder(
        arch=cfg.MODEL.arch_encoder.lower(), fc_dim=cfg.MODEL.fc_dim,
        weights=cfg.MODEL.weights_encoder)
    net_encoder.grad_checkpoint = cfg.TRAIN.grad_checkpoint
    net_decoder = ModelBuilder.build_decoder(
        arch=cfg.MODEL.arch_decoder.lower(), fc_dim=cfg.MODEL.fc_dim,
        num_class=cfg.DATASET.num_class, weights=cfg.MODEL.weights_decoder)
    crit = nn.NLLLoss(ignore_index=-1)

    if cfg.MODEL.arch_decoder.endswith('deepsup'):
        seg_module = SegmentationModule(net_encoder, net_decoder, crit, cfg.TRAIN.deep_sup_scale)
    else:
        seg_module = SegmentationModule(net_encoder, net_decoder, crit)
    net_decoder.fused_loss = cfg.TRAIN.fused_loss
    seg_module.fused_loss = cfg.TRAIN.fused_loss

    # optional correct cross-GPU BatchNorm. Default OFF: local BN is fine at batch=11 and
    # avoids the cross-GPU BN sync. Turn on only if you need a global BN over all GPUs.
    if args.sync_bn:
        seg_module = nn.SyncBatchNorm.convert_sync_batchnorm(seg_module)
        if rank == 0:
            print('Using torch-native SyncBatchNorm (global BN across GPUs)')

    seg_module = seg_module.to(device).to(memory_format=torch.channels_last)
    # find_unused_parameters=False is correct for c1 (all params contribute); deepsup may need True.
    model = DDP(seg_module, device_ids=[local_rank], output_device=local_rank,
                find_unused_parameters=cfg.MODEL.arch_decoder.endswith('deepsup'))

    # ---- data (each rank gets its own loader; per-rank seed shards the random stream) ----
    if rank == 0:
        print('1 Epoch = {} iters/GPU; effective batch = {} (={} GPUs x {})'.format(
            cfg.TRAIN.epoch_iters, world_size * cfg.TRAIN.batch_size_per_gpu,
            world_size, cfg.TRAIN.batch_size_per_gpu))
    dataset_train = TrainDataset(cfg.DATASET.root_dataset, cfg.DATASET.list_train,
                                 cfg.DATASET, batch_per_gpu=cfg.TRAIN.batch_size_per_gpu)
    loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=1, shuffle=False, collate_fn=user_scattered_collate,
        num_workers=cfg.TRAIN.workers, drop_last=True, pin_memory=True, persistent_workers=True)
    iterator_train = CudaPrefetcher(loader_train, device)

    # ---- optimizers (operate on the raw submodule params; DDP all-reduces their grads) ----
    optimizers = create_optimizers(net_encoder, net_decoder, cfg)
    history = {'train': {'epoch': [], 'loss': [], 'acc': []}}

    for epoch in range(cfg.TRAIN.start_epoch, cfg.TRAIN.num_epoch):
        train(model, iterator_train, optimizers, history, epoch + 1, cfg, rank, world_size)
        dist.barrier()
        if rank == 0:
            checkpoint(net_encoder, net_decoder, history, cfg, epoch + 1)

    if rank == 0:
        print('Training Done!')


if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('1.10.0'), 'PyTorch>=1.10 required for DDP path'
    parser = argparse.ArgumentParser(description="PyTorch Semantic Segmentation — multi-GPU DDP")
    parser.add_argument("--cfg", default="config/ade20k-hrnetv2.yaml", metavar="FILE", type=str)
    parser.add_argument("--sync-bn", action="store_true", help="use global SyncBatchNorm across GPUs")
    parser.add_argument("--no-lr-scale", action="store_true",
                        help="disable linear LR scaling by world_size")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    # ---- distributed init (torchrun sets these env vars) ----
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")

    # per-rank seeding -> ranks draw different data from the (fake-length) TrainDataset
    seed = cfg.TRAIN.seed + rank
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False

    # linear LR scaling for the world_size-larger effective batch (standard large-batch rule)
    if not args.no_lr_scale and world_size > 1:
        cfg.TRAIN.lr_encoder *= world_size
        cfg.TRAIN.lr_decoder *= world_size
        if rank == 0:
            print(f'[LR] linear-scaled by world_size={world_size}: '
                  f'lr_encoder={cfg.TRAIN.lr_encoder}, lr_decoder={cfg.TRAIN.lr_decoder} '
                  f'(disable with --no-lr-scale; consider adding warmup)')

    cfg.TRAIN.batch_size = world_size * cfg.TRAIN.batch_size_per_gpu
    cfg.TRAIN.max_iters = cfg.TRAIN.epoch_iters * cfg.TRAIN.num_epoch
    cfg.TRAIN.running_lr_encoder = cfg.TRAIN.lr_encoder
    cfg.TRAIN.running_lr_decoder = cfg.TRAIN.lr_decoder

    if rank == 0:
        logger = setup_logger(distributed_rank=0)
        logger.info("Loaded configuration file {}".format(args.cfg))
        logger.info("World size: {} GPUs".format(world_size))
        if not os.path.isdir(cfg.DIR):
            os.makedirs(cfg.DIR)

    main(cfg, args, rank, local_rank, world_size)
    dist.destroy_process_group()
