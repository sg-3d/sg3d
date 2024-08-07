import os
import glob
import importlib
import functools
import torch
from typing import Any
from accelerate.logging import get_logger
from accelerate.state import PartialState
from accelerate.utils import recursively_apply
from accelerate.utils.constants import TORCH_DISTRIBUTED_OPERATION_TYPES
from accelerate.utils.dataclasses import DistributedType
from accelerate import Accelerator
try:
    from torch.optim.lr_scheduler import LRScheduler
except ImportError:
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from accelerate.scheduler import AcceleratedScheduler


logger = get_logger(__name__)


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)

# using wonder's beautiful simplification: https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects/31174427?noredirect=1#comment86638618_31174427

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


# def import_all(exclude_list=None):
#     if exclude_list is None:
#         exclude_list = ["__init__.py", "build.py"]
#     print(f"file: {__file__}")
#     current_directory = os.path.dirname(__file__)
#     module_names = [
#         os.path.splitext(file)[0] for file in os.listdir(current_directory)
#         if file.endswith(".py") and file not in exclude_list
#     ]
#     for module_name in module_names:
#         module = importlib.import_module(f".{module_name}", package=__name__)
#         globals().update({name: getattr(module, name) for name in getattr(module, '__all__', [])})
#     __all__ = [name for name in globals() if not name.startswith("_")]


def _gpu_gather_object(object: Any):
    # by JY Huang: re-implement the method for gathering non-tensor objects
    output_objects = [None for _ in range(PartialState().num_processes)]
    torch.distributed.all_gather_object(output_objects, object)
    if isinstance(object, (list, tuple)):
        output_list = []
        for item in output_objects:
            output_list.extend(item)
        return output_list
    elif isinstance(object, dict):
        template = output_objects[0]
        output_dict = {}
        for k, v in template.items():
            output_dict[k] = []
            for item in output_objects:
                if isinstance(item[k], list):
                    output_dict[k].extend(item[k])
                else:
                    output_dict[k].append(item[k])
        return output_dict


def gather_object(object: Any):
    """
    Recursively gather object in a nested list/tuple/dictionary of objects from all devices.

    Args:
        object (nested list/tuple/dictionary of picklable object):
            The data to gather.

    Returns:
        The same data structure as `object` with all the objects sent to every device.
    """
    if PartialState().distributed_type == DistributedType.TPU:
        raise NotImplementedError("gather objects in TPU is not supported")
    elif PartialState().distributed_type in TORCH_DISTRIBUTED_OPERATION_TYPES:
        return _gpu_gather_object(object)
    else:
        return object


def gather_for_metrics(accelerator, input_data):
    """
    by JY Huang: re-implement this method for gathering non-tensor objects
    Refer source code to https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.gather_for_metrics
    """

    try:
        recursively_apply(lambda x: x, input_data, error_on_other_type=True)
        all_tensors = True
    except TypeError:
        all_tensors = False

    if not all_tensors:
        data = gather_object(input_data)
    else:
        data = accelerator.gather(input_data)

    try:
        if accelerator.gradient_state.end_of_dataloader:
            # at the end of a dataloader, `gather_for_metrics` regresses to
            # `gather` unless the dataset has a remainder so log.
            if accelerator.gradient_state.remainder == -1:
                logger.info(
                    "The used dataset had no length, returning gathered tensors. You should drop the remainder yourself."
                )
                return data
            elif accelerator.gradient_state.remainder > 0:
                # Last batch needs to be truncated on distributed systems as it contains additional samples
                def _adjust_samples(tensor):
                    return tensor[: accelerator.gradient_state.remainder] if tensor is not None else None
                if all_tensors:
                    # This only applies to tensors, as defined in `recursively_apply`
                    return recursively_apply(_adjust_samples, data)
                else:
                    if isinstance(data, (list, tuple)):
                        return _adjust_samples(data)
                    elif isinstance(data, dict):
                        return {k: _adjust_samples(v) for k, v in data.items()}
                    else:
                        raise NotImplementedError(f"Non-tensor gather only supports list, tuple or dict")
            else:  # remainder is 0
                # no remainder even though at end of dataloader, so nothing to do.
                return data
        else:
            # Not at the end of the dataloader, no need to adjust the tensors
            return data
    except Exception:
        # Dataset had no length or raised an error
        return data
    
def gather_dict(accelerator, data_dict):
    data_dict_non_tensor = {k : v for k, v in data_dict.items() if not isinstance(v, torch.Tensor)}
    data_dict_non_tensor = gather_for_metrics(accelerator, data_dict_non_tensor)
    data_dict = {k : v for k, v in data_dict.items() if isinstance(v, torch.Tensor)}
    data_dict = gather_for_metrics(accelerator, data_dict)
    data_dict.update(data_dict_non_tensor)
    return data_dict

"""
Customize Accelerator to support:
    1. advanced gather_for_metrics
    2. only saving partial model weights when calling save_state
"""
class CustomAccelerator(Accelerator):

    def gather_for_metrics(self, input_data):
        # by JY Huang: re-implement this method for gathering non-tensor objects
        try:
            recursively_apply(lambda x: x, input_data, error_on_other_type=True)
            all_tensors = True
        except TypeError:
            all_tensors = False

        if not all_tensors:
            """ custom part 1 """
            data = gather_object(input_data)
            """ custom part 1 """
        else:
            data = self.gather(input_data)

        try:
            if self.gradient_state.end_of_dataloader:
                # at the end of a dataloader, `gather_for_metrics` regresses to
                # `gather` unless the dataset has a remainder so log.
                if self.gradient_state.remainder == -1:
                    logger.info(
                        "The used dataset had no length, returning gathered tensors. You should drop the remainder yourself."
                    )
                    return data
                elif self.gradient_state.remainder > 0:
                    """ custom part 2 """
                    # Last batch needs to be truncated on distributed systems as it contains additional samples
                    def _adjust_samples(tensor):
                        return tensor[: self.gradient_state.remainder] if tensor is not None else None
                    if all_tensors:
                        # This only applies to tensors, as defined in `recursively_apply`
                        return recursively_apply(_adjust_samples, data)
                    else:
                        if isinstance(data, (list, tuple)):
                            return _adjust_samples(data)
                        elif isinstance(data, dict):
                            return {k: _adjust_samples(v) for k, v in data.items()}
                        else:
                            raise NotImplementedError(f"Non-tensor gather only supports list, tuple or dict")
                    """ custom part 2 """
                else:  # remainder is 0
                    # no remainder even though at end of dataloader, so nothing to do.
                    return data
            else:
                # Not at the end of the dataloader, no need to adjust the tensors
                return data
        except Exception:
            # Dataset had no length or raised an error
            return data

    def get_state_dict(self, model, unwrap=True):
        # only save learnable parameters
        if self.distributed_type == DistributedType.DEEPSPEED:
            if self.deepspeed_config["zero_optimization"]["stage"] == 3:
                if model.zero_gather_16bit_weights_on_model_save():
                    state_dict = model._zero3_consolidated_16bit_state_dict()
                else:
                    raise ValueError(
                        "Cannot get 16bit model weights because `stage3_gather_16bit_weights_on_model_save` in DeepSpeed config is False. "
                        "To save the model weights in 16bit, set `stage3_gather_16bit_weights_on_model_save` to True in DeepSpeed config file or "
                        "set `zero3_save_16bit_model` to True when using `accelerate config`. "
                        "To save the full checkpoint, run `model.save_checkpoint(save_dir)` and use `zero_to_fp32.py` to recover weights."
                    )
            else:
                from deepspeed.checkpoint.utils import clone_tensors_for_torch_save

                state_dict = clone_tensors_for_torch_save(self.unwrap_model(model).state_dict())
        elif self.distributed_type == DistributedType.FSDP:
            from torch.distributed.fsdp import FullStateDictConfig, StateDictType
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

            full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_state_dict_config):
                state_dict = model.state_dict()
        else:
            if unwrap:
                model = self.unwrap_model(model)
            state_dict = model.state_dict()
        
        """ custom part """
        keys_list = list(state_dict.keys())
        for k in keys_list:
            if k not in self.learn_params_list:   # need to assign `learn_params_list` before calling this method
                del state_dict[k]
        """ custom part """

        return state_dict

    def prepare_scheduler(self, scheduler: LRScheduler):
        # Ensure we can't double wrap a scheduler due to `find_batch_size`
        if getattr(scheduler, "_is_accelerate_prepared", False):
            if scheduler not in self._schedulers:
                self._schedulers.append(scheduler)
            return scheduler
        # We try to find the optimizer associated with `scheduler`, the default is the full list.
        optimizer = self._optimizers
        for opt in self._optimizers:
            if getattr(scheduler, "optimizer", None) == opt.optimizer:
                optimizer = opt
                break
        scheduler = AcceleratedScheduler(
            scheduler,
            optimizer,
            step_with_optimizer=self.step_scheduler_with_optimizer,
            split_batches=True,   # custom, for proper scheduler.step()
        )
        self._schedulers.append(scheduler)
        return scheduler
