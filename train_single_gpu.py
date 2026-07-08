# System libs
import os
import time
# import math
import random
import argparse
from distutils.version import LooseVersion
# Numerical libs
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.io import decode_jpeg, ImageReadMode  # Exp 20: nvJPEG GPU decode
# Our libs
from mit_semseg.config import cfg
from mit_semseg.dataset import TrainDataset
from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.utils import AverageMeter, parse_devices, setup_logger
from mit_semseg.lib.nn import UserScatteredDataParallel, user_scattered_collate, patch_replication_callback


class CudaPrefetcher:
    """Overlaps the next batch's CPU→GPU transfer with current-batch compute using a
    side CUDA stream + non_blocking copies (requires pin_memory=True). Added in Exp 4."""
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        # Exp 17: ImageNet mean/std as GPU constants for on-device normalization.
        # Workers ship uint8; we do float/255 + (x-mean)/std here, off the data path.
        self.mean = torch.tensor([0.485, 0.456, 0.406], device='cuda').view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device='cuda').view(1, 3, 1, 1)
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
                img = d['img_data'].cuda(non_blocking=True).float().div_(255)
                img = img.sub_(self.mean).div_(self.std)
                d['img_data'] = img.to(memory_format=torch.channels_last)
                d['seg_label'] = d['seg_label'].cuda(non_blocking=True)

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


class NvjpegPrefetcher:
    """Exp 20: GPU JPEG decode. Workers ship raw JPEG bytes + per-image geometry;
    nvJPEG decodes on the GPU hardware decoder, then flip/resize/normalize run
    on-device in a side stream (overlapping with current-batch compute). This
    offloads decode + resize from the 4 CPU workers and shrinks H2D traffic to
    the compressed bytes instead of the full-resolution uint8 image."""
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device='cuda').view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device='cuda').view(3, 1, 1)
        self.next_batch = None
        self._preload()

    def _build(self, d):
        bh, bw = d['batch_shape']
        n = len(d['img_bytes'])
        # zero-init in normalized space == the (124,116,104) gray pad of the uint8 path
        batch = torch.zeros(n, 3, bh, bw, device='cuda')
        for i in range(n):
            raw = torch.from_numpy(d['img_bytes'][i])  # numpy uint8 -> CPU tensor for nvJPEG
            img = decode_jpeg(raw, mode=ImageReadMode.RGB, device='cuda').float()
            if d['img_flip'][i]:
                img = torch.flip(img, dims=[2])  # horizontal flip (W dim)
            th, tw = d['img_resize'][i]
            img = F.interpolate(img.unsqueeze(0), size=(th, tw),
                                mode='bilinear', align_corners=False).squeeze(0)
            img.div_(255).sub_(self.mean).div_(self.std)
            batch[i, :, :th, :tw] = img
        d['img_data'] = batch.to(memory_format=torch.channels_last)
        d['seg_label'] = d['seg_label'].cuda(non_blocking=True)
        return d

    def _preload(self):
        try:
            self.next_batch = next(self.loader)
        except StopIteration:
            self.next_batch = None
            return
        with torch.cuda.stream(self.stream):
            self.next_batch = [self._build(d) for d in self.next_batch]

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


# ============================================================================
# OFAT baseline (Exp 0): clean fp32, no optimizations.
# Techniques are layered in cumulatively, one per experiment. See
# experiment_template.md for the order and the accepted-stack methodology.
# NVTX ranges + cudaProfilerStart/Stop are kept so nsys capture-range profiling
# works on every run regardless of which optimizations are active.
# ============================================================================


# train one epoch
def train(segmentation_module, iterator, optimizers, history, epoch, cfg, scaler):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    ave_total_loss = AverageMeter()
    ave_acc = AverageMeter()

    segmentation_module.train(not cfg.TRAIN.fix_bn)

    # main loop
    tic = time.time()
    for i in range(cfg.TRAIN.epoch_iters):
        torch.cuda.nvtx.range_push(f"iter_{i}")

        # load a batch of data
        torch.cuda.nvtx.range_push("data_loading")
        batch_data = next(iterator)
        torch.cuda.nvtx.range_pop()
        data_time.update(time.time() - tic)

        accum = cfg.TRAIN.accum_steps
        if i % accum == 0:
            segmentation_module.zero_grad(set_to_none=True)  # Exp 15: ~+2% vs set_to_none=False; torch 2.4 default, kept explicit

        # adjust learning rate
        cur_iter = i + (epoch - 1) * cfg.TRAIN.epoch_iters
        adjust_learning_rate(optimizers, cur_iter, cfg)

        # forward pass
        torch.cuda.nvtx.range_push("forward")
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=cfg.TRAIN.amp):
            loss, acc = segmentation_module(batch_data)
            loss = loss.mean() / accum  # scale so accumulated grads match a single large batch
            acc = acc.mean()
        torch.cuda.nvtx.range_pop()

        # backward (BF16 has FP32 dynamic range — no GradScaler needed; scaler kept disabled)
        torch.cuda.nvtx.range_push("backward")
        loss.backward()
        if (i + 1) % accum == 0:
            for optimizer in optimizers:
                optimizer.step()
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_pop()  # iter_{i}

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss and acc
        ave_total_loss.update(loss.data.item())
        ave_acc.update(acc.data.item()*100)

        # calculate accuracy, and display
        if i % cfg.TRAIN.disp_iter == 0:
            print('Epoch: [{}][{}/{}], Time: {:.2f}, Data: {:.2f}, '
                  'lr_encoder: {:.6f}, lr_decoder: {:.6f}, '
                  'Accuracy: {:4.2f}, Loss: {:.6f}'
                  .format(epoch, i, cfg.TRAIN.epoch_iters,
                          batch_time.average(), data_time.average(),
                          cfg.TRAIN.running_lr_encoder, cfg.TRAIN.running_lr_decoder,
                          ave_acc.average(), ave_total_loss.average()))

            fractional_epoch = epoch - 1 + 1. * i / cfg.TRAIN.epoch_iters
            history['train']['epoch'].append(fractional_epoch)
            history['train']['loss'].append(loss.data.item())
            history['train']['acc'].append(acc.data.item())


def checkpoint(nets, history, cfg, epoch):
    print('Saving checkpoints...')
    (net_encoder, net_decoder, crit) = nets

    dict_encoder = net_encoder.state_dict()
    dict_decoder = net_decoder.state_dict()

    torch.save(
        history,
        '{}/history_epoch_{}.pth'.format(cfg.DIR, epoch))
    torch.save(
        dict_encoder,
        '{}/encoder_epoch_{}.pth'.format(cfg.DIR, epoch))
    torch.save(
        dict_decoder,
        '{}/decoder_epoch_{}.pth'.format(cfg.DIR, epoch))


def group_weight(module):
    group_decay = []
    group_no_decay = []
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
    groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
    return groups


def create_optimizers(nets, cfg):
    (net_encoder, net_decoder, crit) = nets
    fused = not cfg.TRAIN.baseline  # Exp 14: fuse param-update launches (off for fp32 baseline)
    optimizer_encoder = torch.optim.SGD(
        group_weight(net_encoder),
        lr=cfg.TRAIN.lr_encoder,
        momentum=cfg.TRAIN.beta1,
        weight_decay=cfg.TRAIN.weight_decay,
        fused=fused)
    optimizer_decoder = torch.optim.SGD(
        group_weight(net_decoder),
        lr=cfg.TRAIN.lr_decoder,
        momentum=cfg.TRAIN.beta1,
        weight_decay=cfg.TRAIN.weight_decay,
        fused=fused)
    return (optimizer_encoder, optimizer_decoder)


def adjust_learning_rate(optimizers, cur_iter, cfg):
    scale_running_lr = ((1. - float(cur_iter) / cfg.TRAIN.max_iters) ** cfg.TRAIN.lr_pow)
    cfg.TRAIN.running_lr_encoder = cfg.TRAIN.lr_encoder * scale_running_lr
    cfg.TRAIN.running_lr_decoder = cfg.TRAIN.lr_decoder * scale_running_lr

    (optimizer_encoder, optimizer_decoder) = optimizers
    for param_group in optimizer_encoder.param_groups:
        param_group['lr'] = cfg.TRAIN.running_lr_encoder
    for param_group in optimizer_decoder.param_groups:
        param_group['lr'] = cfg.TRAIN.running_lr_decoder


def main(cfg, gpus):
    # Network Builders
    net_encoder = ModelBuilder.build_encoder(
        arch=cfg.MODEL.arch_encoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        weights=cfg.MODEL.weights_encoder)
    net_encoder.grad_checkpoint = cfg.TRAIN.grad_checkpoint  # Exp 18
    net_decoder = ModelBuilder.build_decoder(
        arch=cfg.MODEL.arch_decoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        num_class=cfg.DATASET.num_class,
        weights=cfg.MODEL.weights_decoder)

    crit = nn.NLLLoss(ignore_index=-1)

    if cfg.MODEL.arch_decoder.endswith('deepsup'):
        segmentation_module = SegmentationModule(
            net_encoder, net_decoder, crit, cfg.TRAIN.deep_sup_scale)
    else:
        segmentation_module = SegmentationModule(
            net_encoder, net_decoder, crit)

    # Exp 19: fused loss (decoder emits logits, loss uses F.cross_entropy)
    net_decoder.fused_loss = cfg.TRAIN.fused_loss
    segmentation_module.fused_loss = cfg.TRAIN.fused_loss

    # Dataset and Loader
    print('1 Epoch = {} iters'.format(cfg.TRAIN.epoch_iters))
    if cfg.TRAIN.use_dali:
        # GPU data pipeline: DALI mixed-decode + resize/flip/normalize (experimental).
        from mit_semseg.dali_loader import DaliTrainLoader
        iterator_train = DaliTrainLoader(cfg)
        print('Using DALI GPU data loader')
    else:
        dataset_train = TrainDataset(
            cfg.DATASET.root_dataset,
            cfg.DATASET.list_train,
            cfg.DATASET,
            batch_per_gpu=cfg.TRAIN.batch_size_per_gpu,
            use_nvjpeg=cfg.TRAIN.use_nvjpeg,
            cpu_normalize=cfg.TRAIN.baseline)  # baseline: CPU float normalize (no GPU-normalize)

        loader_train = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=len(gpus),  # we have modified data_parallel
            shuffle=False,  # we do not use this param
            collate_fn=user_scattered_collate,
            num_workers=cfg.TRAIN.workers,
            drop_last=True,
            pin_memory=not cfg.TRAIN.baseline,        # baseline: off
            persistent_workers=not cfg.TRAIN.baseline)  # baseline: off

        if cfg.TRAIN.baseline:
            # Clean baseline (Exp 0): plain iterator, no prefetcher/GPU-normalize. The
            # UserScatteredDataParallel wrapper handles the H2D copy of CPU float batches.
            iterator_train = iter(loader_train)
            print('Using plain loader (fp32 baseline — no prefetcher)')
        elif cfg.TRAIN.use_nvjpeg:
            iterator_train = NvjpegPrefetcher(loader_train)  # Exp 20: GPU JPEG decode
            print('Using nvJPEG GPU decode prefetcher')
        else:
            # create loader iterator with CUDA prefetcher (overlaps next-batch H2D copy w/ compute)
            iterator_train = CudaPrefetcher(loader_train)

    # load nets into gpu
    segmentation_module.cuda()
    if not cfg.TRAIN.baseline:
        segmentation_module = segmentation_module.to(memory_format=torch.channels_last)
    segmentation_module = UserScatteredDataParallel(
        segmentation_module,
        device_ids=gpus)
    if len(gpus) > 1:
        # For sync bn
        patch_replication_callback(segmentation_module)

    # Set up optimizers
    nets = (net_encoder, net_decoder, crit)
    optimizers = create_optimizers(nets, cfg)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.TRAIN.amp)

    # Main loop
    history = {'train': {'epoch': [], 'loss': [], 'acc': []}}

    profile_epoch = cfg.TRAIN.num_epoch  # profile on the last epoch

    for epoch in range(cfg.TRAIN.start_epoch, cfg.TRAIN.num_epoch):
        is_profile_epoch = (epoch + 1 == profile_epoch)

        if is_profile_epoch:
            print(f'[Profiler] Starting CUDA profiler capture on epoch {epoch+1}')
            torch.cuda.cudart().cudaProfilerStart()

        train(segmentation_module, iterator_train, optimizers, history, epoch+1, cfg, scaler)

        if is_profile_epoch:
            torch.cuda.cudart().cudaProfilerStop()
            print(f'[Profiler] Stopped CUDA profiler capture after epoch {epoch+1}')
            checkpoint(nets, history, cfg, epoch+1)
            break

        # checkpointing
        checkpoint(nets, history, cfg, epoch+1)

    print('Training Done!')


if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser(
        description="PyTorch Semantic Segmentation Training"
    )
    parser.add_argument(
        "--cfg",
        default="config/ade20k-resnet50dilated-ppm_deepsup.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--gpus",
        default="0-3",
        help="gpus to use, e.g. 0-3 or 0,1,2,3"
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    # cfg.freeze()

    logger = setup_logger(distributed_rank=0)   # TODO
    logger.info("Loaded configuration file {}".format(args.cfg))
    logger.info("Running with config:\n{}".format(cfg))

    # Output directory
    if not os.path.isdir(cfg.DIR):
        os.makedirs(cfg.DIR)
    logger.info("Outputing checkpoints to: {}".format(cfg.DIR))
    with open(os.path.join(cfg.DIR, 'config.yaml'), 'w') as f:
        f.write("{}".format(cfg))

    # Start from checkpoint
    if cfg.TRAIN.start_epoch > 0:
        cfg.MODEL.weights_encoder = os.path.join(
            cfg.DIR, 'encoder_epoch_{}.pth'.format(cfg.TRAIN.start_epoch))
        cfg.MODEL.weights_decoder = os.path.join(
            cfg.DIR, 'decoder_epoch_{}.pth'.format(cfg.TRAIN.start_epoch))
        assert os.path.exists(cfg.MODEL.weights_encoder) and \
            os.path.exists(cfg.MODEL.weights_decoder), "checkpoint does not exitst!"

    # Parse gpu ids
    gpus = parse_devices(args.gpus)
    gpus = [x.replace('gpu', '') for x in gpus]
    gpus = [int(x) for x in gpus]
    num_gpus = len(gpus)
    cfg.TRAIN.batch_size = num_gpus * cfg.TRAIN.batch_size_per_gpu

    cfg.TRAIN.max_iters = cfg.TRAIN.epoch_iters * cfg.TRAIN.num_epoch
    cfg.TRAIN.running_lr_encoder = cfg.TRAIN.lr_encoder
    cfg.TRAIN.running_lr_decoder = cfg.TRAIN.lr_decoder

    random.seed(cfg.TRAIN.seed)
    torch.manual_seed(cfg.TRAIN.seed)
    torch.backends.cudnn.benchmark = False  # OFAT Exp 11/13: dropped (regressed alone and paired with channels_last)

    main(cfg, gpus)
