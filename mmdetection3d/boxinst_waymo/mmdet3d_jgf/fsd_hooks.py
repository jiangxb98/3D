import logging
import os.path as osp
from typing import Optional
from torch import Tensor
from torch.nn.utils import clip_grad
from mmcv.parallel import is_module_wrapper
from mmcv.runner.hooks import HOOKS, Hook
from mmcv.utils import TORCH_VERSION, digit_version
from mmcv.runner.dist_utils import master_only
from mmcv.runner.hooks.logger.base import LoggerHook
from mmcv.runner.hooks.logger import TensorboardLoggerHook
from mmcv.runner.hooks.optimizer import OptimizerHook

from .utils import pcgrad_fn, PCGrad_backward
from cowa3d_common import global_container as gol

@HOOKS.register_module()
class DisableAugmentationHook(Hook):
    """Switch the mode of YOLOX during training.
    This hook turns off the mosaic and mixup data augmentation and switches
    to use L1 loss in bbox_head.
    Args:
        num_last_epochs (int): The number of latter epochs in the end of the
            training to close the data augmentation and switch to L1 loss.
            Default: 15.
       skip_type_keys (list[str], optional): Sequence of type string to be
            skip pipeline. Default: ('Mosaic', 'RandomAffine', 'MixUp')
    """

    def __init__(self,
                 num_last_epochs=10,
                 skip_type_keys=('ObjectSample')):
        self.num_last_epochs = num_last_epochs
        self.skip_type_keys = skip_type_keys
        self._restart_dataloader = False

    def before_train_epoch(self, runner):
        epoch = runner.epoch # begin from 0
        train_loader = runner.data_loader
        if epoch == runner.max_epochs - self.num_last_epochs:
            runner.logger.info(f'Disable augmentations: {self.skip_type_keys}')
            # The dataset pipeline cannot be updated when persistent_workers
            # is True, so we need to force the dataloader's multi-process
            # restart. This is a very hacky approach.
            #train_loader.dataset.dataset.update_skip_type_keys(self.skip_type_keys)
            train_loader.dataset.update_skip_type_keys(self.skip_type_keys)
            if hasattr(train_loader, 'persistent_workers'
                       ) and train_loader.persistent_workers is True:

                train_loader._DataLoader__initialized = False
                train_loader._iterator = None
                self._restart_dataloader = True
                print('has persistent workers')
        else:
            # Once the restart is complete, we need to restore
            # the initialization flag.
            if self._restart_dataloader:
                train_loader._DataLoader__initialized = True

@HOOKS.register_module()
class EnableFSDDetectionHook(Hook):

    def __init__(self,
                 enable_after_epoch=1,
                 ):
        self.enable_after_epoch = enable_after_epoch

    def before_train_epoch(self, runner):
        epoch = runner.epoch # begin from 0
        if epoch == self.enable_after_epoch:
            runner.logger.info(f'Enable FSD Detection from now.')
            runner.model.module.runtime_info['enable_detection'] = True

@HOOKS.register_module()
class EnableFSDDetectionHookIter(Hook):

    def __init__(self,
                 enable_after_iter=5000,
                 threshold_buffer=0,
                 buffer_iter=2000,
                 ):
        self.enable_after_iter = enable_after_iter
        self.buffer_iter = buffer_iter
        self.delta = threshold_buffer / buffer_iter
        self.threshold_buffer = threshold_buffer

    def before_train_iter(self, runner):
        cur_iter = runner.iter # begin from 0
        if cur_iter == self.enable_after_iter:
            runner.logger.info(f'Enable FSD Detection from now.')
        if cur_iter >= self.enable_after_iter: # keep the sanity when resuming model
            runner.model.module.runtime_info['enable_detection'] = True
        if self.threshold_buffer > 0 and cur_iter > self.enable_after_iter and cur_iter < self.enable_after_iter + self.buffer_iter:
            runner.model.module.runtime_info['threshold_buffer'] = (self.enable_after_iter + self.buffer_iter - cur_iter) * self.delta
        else:
            runner.model.module.runtime_info['threshold_buffer'] = 0


@HOOKS.register_module()
class TensorboardLoggerFullHook(TensorboardLoggerHook):
    def __init__(self,
                 log_dir: Optional[str] = None,
                 interval: int = 10,
                 ignore_last: bool = True,
                 reset_flag: bool = False,
                 by_epoch: bool = True):
        super().__init__(log_dir, interval, ignore_last, reset_flag, by_epoch)


    @master_only
    def log(self, runner) -> None:
        tags = self.get_loggable_tags(runner, allow_text=True)
        for tag, val in tags.items():
            if isinstance(val, str):
                self.writer.add_text(tag, val, self.get_iter(runner))
            else:
                self.writer.add_scalar(tag, val, self.get_iter(runner))
        
        for key in gol.get_keys():
            if 'histogram' in key:
                self.writer.add_histogram("train/" + key, gol.get_value(key), self.get_iter(runner))


@HOOKS.register_module()
class SegmentationDataSwitchHook(Hook):
    def __init__(self, ratio=1):
        self.ratio = int(ratio)
        self.round = self.ratio + 1


    def before_train_epoch(self, runner):
        epoch = runner.epoch
        train_loader = runner.data_loader
        if epoch % self.round == self.ratio:
            train_loader.dataset.only_load_seg_frames = False
            runner.logger.info(f'Switch dataset.only_load_seg_frames to False')
        else:
            train_loader.dataset.only_load_seg_frames = True
            runner.logger.info(f'Switch dataset.only_load_seg_frames to True')
        # if hasattr(train_loader, 'persistent_workers'
        #                ) and train_loader.persistent_workers is True:

        #         train_loader._DataLoader__initialized = False
        #         train_loader._iterator = None
        #         runner.logger.info('has persistent workers')


@HOOKS.register_module()
class MultiTaskOptimizerHook(OptimizerHook):
    """A hook contains custom operations for the optimizer.

    Args:
        grad_clip (dict, optional): A config dict to control the clip_grad.
            Default: None.
        detect_anomalous_params (bool): This option is only used for
            debugging which will slow down the training speed.
            Detect anomalous parameters that are not included in
            the computational graph with `loss` as the root.
            There are two cases

                - Parameters were not used during
                  forward pass.
                - Parameters were not used to produce
                  loss.
            Default: False.
    """
    def __init__(self,
                 grad_clip: Optional[dict] = None,
                 detect_anomalous_params: bool = False,
                 apply_multi_task: bool = False):
        self.grad_clip = grad_clip
        self.detect_anomalous_params = detect_anomalous_params
        self.apply_multi_task = apply_multi_task

    def after_train_iter(self, runner):
        runner.optimizer.zero_grad()
        loss_scalar = sum(_value for _key, _value in runner.outputs['loss'].items()
                   if 'loss' in _key)
        if self.detect_anomalous_params:
            self.detect_anomalous_parameters(loss_scalar, runner)
        if self.apply_multi_task:
            loss_det = sum(_value for _key, _value in runner.outputs['loss'].items()
                           if not 'loss_sem_seg_full' in _key)
            loss_seg = sum(_value for _key, _value in runner.outputs['loss'].items()
                           if 'loss_sem_seg_full' in _key)
            PCGrad_backward(runner.optimizer, [loss_seg, loss_det])
            #pcgrad_fn(runner.model, [loss_det, loss_seg], runner.optimizer)
            #runner.optimizer.backward([loss_det, loss_seg])
        else:
            loss_scalar.backward()

        if self.grad_clip is not None:
            grad_norm = self.clip_grads(runner.model.parameters())
            if grad_norm is not None:
                # Add grad norm to the logger
                runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                         runner.outputs['num_samples'])
        runner.optimizer.step()
