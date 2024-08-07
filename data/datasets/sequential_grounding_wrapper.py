from random import sample
import torch
from torch.utils.data import Dataset, default_collate

from data.datasets.dataset_wrapper import DATASETWRAPPER_REGISTRY

from ..data_utils import make_bce_label, pad_sequence_2d, pad_sequence


@DATASETWRAPPER_REGISTRY.register()
class SequentialGroundingDatasetWrapper(Dataset):
    def __init__(self, cfg, dataset):
        self.dataset = dataset
        self.useful_keys = ['tgt_object_id', 'scan_id', 'obj_labels', 'data_idx',
                          'obj_fts', 'obj_locs', 'obj_pad_masks', 'obj_ids',
                          'source', 'prompt_before_obj', 'prompt_middle_1', 'prompt_middle_2', 'prompt_after_obj', 'output_gt']

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data_dict = self.dataset[idx]
        for k in list(data_dict.keys()):
            if k not in self.useful_keys:
                del data_dict[k]
        # add new keys because of leo
        data_dict['img_fts'] = torch.zeros(3, 224, 224)
        data_dict['img_masks'] = torch.LongTensor([0]).bool()
        data_dict['anchor_locs'] = torch.zeros(3)
        data_dict['anchor_orientation'] = torch.zeros(4)
        data_dict['anchor_orientation'][-1] = 1   # xyzw
        # convert to leo format
        data_dict['obj_masks'] = data_dict['obj_pad_masks']
        del data_dict['obj_pad_masks']
        return data_dict
    
    def collate_fn(self, batch):
        new_batch = {}
        # pad
        padding_keys = ['obj_fts', 'obj_locs', 'obj_masks', 'obj_labels', 'obj_ids']
        for k in padding_keys:
            tensors = [sample.pop(k) for sample in batch]
            padded_tensor = pad_sequence(tensors, pad=0)
            new_batch[k] = padded_tensor
        # list
        list_keys = ['tgt_object_id']
        for k in list_keys:
            new_batch[k] = [sample.pop(k) for sample in batch]
        # default collate
        new_batch.update(default_collate(batch))
        return new_batch
