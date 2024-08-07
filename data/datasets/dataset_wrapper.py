import os
from time import time

import torch
import numpy as np
from fvcore.common.registry import Registry
from transformers import BertTokenizer, AutoTokenizer
from torch.utils.data import Dataset, default_collate
import random
import MinkowskiEngine as ME
import copy

from ..data_utils import random_word, random_point_cloud, pad_tensors, Vocabulary, random_caption_word
# from modules.third_party.softgroup_ops.ops import functions as sg_ops


DATASETWRAPPER_REGISTRY = Registry("dataset_wrapper")
DATASETWRAPPER_REGISTRY.__doc__ = """ """


@DATASETWRAPPER_REGISTRY.register()
class MaskDatasetWrapper(Dataset):
    def __init__(self, cfg, dataset):
        # tokenizer, max_seq_length=80, max_obj_len=80,
        #  mask_strategy='random', txt_mask_ratio=0.15, pc_mask_ratio=0.1
        assert cfg.data.args.mask_strategy in ['random']
        self.dataset = dataset
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
        self.max_seq_length = cfg.data.args.max_seq_len
        self.max_obj_len = cfg.data.args.max_obj_len
        self.txt_mask_ratio = cfg.data.args.txt_mask_ratio
        self.pc_mask_ratio = cfg.data.args.pc_mask_ratio

        self.use_voxel = cfg.data.args.get('use_voxel', None)
        if self.use_voxel:
            self.voxel_size = cfg.data.args.get('voxel_size', 0.02)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data_dict = self.dataset[idx]
        sentence = data_dict['sentence']
        encoded_input = self.tokenizer(sentence, max_length=self.max_seq_length,
                          add_special_tokens=True, truncation=True,
                          padding='max_length', return_tensors="pt")
        # build txt
        data_dict['txt_ids'] = encoded_input['input_ids'].squeeze(0) # L
        data_dict['txt_masks'] = encoded_input['attention_mask'].squeeze(0) # L

        # mask txt
        masked_txt_ids, masked_lm_labels = random_word(data_dict['txt_ids'], data_dict['txt_masks'],
                                                       self.tokenizer, self.txt_mask_ratio)
        data_dict['txt_ids'] = masked_txt_ids
        data_dict['masked_lm_labels'] = masked_lm_labels
        # build object
        data_dict['obj_masks'] = (torch.arange(self.max_obj_len) < len(data_dict['obj_locs'])) # O
        if 'obj_fts' in data_dict.keys():
            data_dict['obj_fts'] = pad_tensors(data_dict['obj_fts'], lens=self.max_obj_len,
                                                    pad=1.0).float() # O, 1024, 6
        if 'obj_pcds_masks' in data_dict.keys():
            data_dict['obj_pcds_masks'] = pad_tensors(data_dict['obj_pcds_masks'], lens=self.max_obj_len, 
                                                      pad=1.0).float()
        data_dict['obj_locs']= pad_tensors(data_dict['obj_locs'], lens=self.max_obj_len,
                                                pad=0.0).float() # O, 3
        data_dict['obj_labels'] = pad_tensors(data_dict['obj_labels'], lens=self.max_obj_len,
                                                   pad=-100).long() # O
        # mask object, 0 means mask object, 1 means keep object
        if 'obj_fts' in data_dict.keys():
            obj_sem_masks = random_point_cloud(data_dict['obj_fts'], data_dict['obj_masks'],
                                            self.pc_mask_ratio)
            data_dict['obj_sem_masks'] = obj_sem_masks
        else:
            obj_sem_masks = []
            for i in range(self.max_obj_len):
                if i >= len(data_dict['obj_locs']):
                    obj_sem_masks.append(0)
                else:
                    prob = random.random()
                    if prob < self.pc_mask_ratio:
                        obj_sem_masks.append(0)
                    else:
                        obj_sem_masks.append(1)
            data_dict['obj_sem_masks'] = torch.tensor(obj_sem_masks).long()
        if 'tgt_object_id' in data_dict.keys():
            data_dict['tgt_object_id'] = data_dict['tgt_object_id'].long() # 1 or O

        # # Scene pcds
        # data_dict["scene_pcds"] = torch.from_numpy(data_dict["scene_pcds"]).float()
        key_list = [
            'txt_ids', 'txt_masks', 'masked_lm_labels', 'obj_masks', 'obj_fts',
            'obj_locs', 'obj_labels', 'obj_sem_masks', 'tgt_object_id'
        ]
        if 'obj_fts' not in data_dict.keys():
            key_list.remove('obj_fts')
            # key_list.remove('obj_sem_masks')
        if 'obj_pcds_masks' in data_dict.keys():
            key_list.append('obj_pcds_masks')
        if 'scene_pcds' in data_dict.keys():
            key_list.append('scene_pcds')
        data_dict = {k : v for k, v in data_dict.items() if k in key_list}
        return data_dict
    
    def collate_fn(self, batch_list):
        if not self.use_voxel:
            ret = default_collate(batch_list)
        
        else:
            new_batch_list = copy.deepcopy(batch_list)
            for i in range(len(new_batch_list)):
                new_batch_list[i].pop('scene_pcds')
            ret = default_collate(new_batch_list)
            # ret.pop('scene_pcds')

            scene_point_nums = []
            for cur_sample in batch_list:
                scene_pcds = cur_sample['scene_pcds']
                scene_point_nums.append(len(scene_pcds))
            scene_point_nums = torch.tensor(scene_point_nums)
            
            # scene_begin_indices = torch.cumsum(torch.tensor(scene_point_nums), dim=0) - scene_point_nums[0]
            scene_begin_indices = []
            for i in range(len(scene_point_nums)):
                if i == 0:
                    scene_begin_indices.append(0)
                else:
                    scene_begin_indices.append(torch.sum(scene_point_nums[:i]))
            batch_size = len(batch_list)

            # ret['obj_pds_masks'] = torch.cat([ret['obj_pcds_masks'][i].unsqueeze(0) + scene_begin_indices[i] for i in range(batch_size)], dim=0)
            for i in range(batch_size):
                ret['obj_pcds_masks'][i] += scene_begin_indices[i]

            from data.voxelize import voxelize_and_inverse
            voxel_coordinates_list = []
            voxel_features_list = []
            v2p_map_list = []
            
            count = 0
            for cur_sample in batch_list:
                scene_pcds = cur_sample['scene_pcds']
                coord = scene_pcds[:, :3]
                feats = scene_pcds[:, 3:]
                coord_min = np.min(coord, 0)
                coord -= coord_min
                coord = coord.astype(np.float32)
                coord = coord / self.voxel_size
                int_coord = coord.astype(np.int32)
                p2v_map, v2p_map = voxelize_and_inverse(int_coord)
                p2v_map = torch.from_numpy(p2v_map)
                v2p_map = torch.from_numpy(v2p_map)
                voxel_coords = torch.from_numpy(coord[p2v_map]).float()
                voxel_feats = torch.from_numpy(feats[p2v_map]).float()

                voxel_coordinates_list.append(voxel_coords)
                voxel_features_list.append(voxel_feats)
                v2p_map_list.append(v2p_map + count)
                count += voxel_coords.shape[0]

            input_dict = {"coords": voxel_coordinates_list, "feats": voxel_features_list}
            voxel_coords, voxel_features = ME.utils.sparse_collate(**input_dict, dtype=torch.float32)

            ret['voxel_features'] = voxel_features
            ret['v2p_map'] = torch.cat(v2p_map_list)
            # ret['v2p_map'] = v2p_map
            ret['voxel_coords'] = voxel_coords

        return ret


@DATASETWRAPPER_REGISTRY.register()
class MaskMVDatasetWrapper(Dataset):
    def __init__(self, cfg, dataset):
        # tokenizer, max_seq_length=80, max_obj_len=80,
        #  mask_strategy='random', txt_mask_ratio=0.15, pc_mask_ratio=0.1
        assert getattr(cfg.data, cfg.task.lower()).args.mask_strategy in ['random']
        self.cfg = cfg
        self.dataset = dataset
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
        self.training_args = getattr(cfg.data, cfg.task.lower()).args

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data_dict = self.dataset[idx]
        sentence = data_dict['sentence']
        encoded_input = self.tokenizer(sentence, max_length=self.training_args.max_seq_len,
                          add_special_tokens=True, truncation=True,
                          padding='max_length', return_tensors="pt")
        # build txt
        data_dict['txt_ids'] = encoded_input['input_ids'].squeeze(0) # L
        data_dict['txt_masks'] = encoded_input['attention_mask'].squeeze(0) # L
        # mask txt
        masked_txt_ids, masked_lm_labels = random_word(data_dict['txt_ids'], data_dict['txt_masks'],
                                                       self.tokenizer, self.training_args.txt_mask_ratio)
        data_dict['txt_ids'] = masked_txt_ids
        data_dict['masked_lm_labels'] = masked_lm_labels
        # build multi view object
        if self.cfg.data.mvdatasettings.is_pool_obj_feature:
            if len(data_dict['vis_obj_feats']) > self.training_args.max_obj_len:
                data_dict['vis_obj_feats'] = random.sample(data_dict['vis_obj_feats'], self.training_args.max_obj_len)
                data_dict['vis_obj_locs'] = random.sample(data_dict['vis_obj_locs'], self.training_args.max_obj_len)
                data_dict['vis_obj_labels'] = random.sample(data_dict['vis_obj_labels'], self.training_args.max_obj_len)

            data_dict['vis_obj_feats'] = torch.from_numpy(np.array(data_dict['vis_obj_feats']))
            data_dict['vis_obj_locs'] = torch.from_numpy(np.array(data_dict['vis_obj_locs']))
            data_dict['vis_obj_labels'] = torch.from_numpy(np.array(data_dict['vis_obj_labels']))

            data_dict['vis_obj_masks'] = (torch.arange(self.training_args.max_obj_len) < len(data_dict['vis_obj_locs'])) # O
            data_dict['vis_obj_feats'] = pad_tensors(data_dict['vis_obj_feats'], lens=self.training_args.max_obj_len,
                                                    pad=1.0).float() # O, 1024, 6
            data_dict['vis_obj_locs']= pad_tensors(data_dict['vis_obj_locs'], lens=self.training_args.max_obj_len,
                                                    pad=0.0).float() # O, 3
            data_dict['vis_obj_labels'] = pad_tensors(data_dict['vis_obj_labels'], lens=self.training_args.max_obj_len,
                                                    pad=-100).long() # O
            # mask object, 0 means mask object, 1 means keep object
            mv_sem_masks = random_point_cloud(data_dict['vis_obj_feats'], data_dict['vis_obj_masks'],
                                            self.training_args.mv_mask_ratio) 
        else:
            data_dict['mv_inst_feats'] = torch.from_numpy(data_dict['mv_inst_feats']).float()
            data_dict['mv_camera_pose'] = torch.from_numpy(data_dict['mv_camera_pose']).float()
            data_dict['mv_inst_locs'] = torch.from_numpy(data_dict['mv_inst_locs']).float()
            data_dict['mv_inst_masks'] = torch.from_numpy(data_dict['mv_inst_masks'])
            data_dict['mv_inst_masks'] = (data_dict['mv_inst_masks'] > 0.5)
            data_dict['mv_inst_labels'] = torch.from_numpy(data_dict['mv_inst_labels']).long()
            # mask object, 0 means mask object, 1 means keep object
            mv_sem_masks = random_point_cloud(data_dict['mv_inst_feats'], data_dict['mv_inst_masks'],
                                            self.training_args.mv_mask_ratio)
        data_dict['mv_sem_masks'] = mv_sem_masks

        return data_dict


@DATASETWRAPPER_REGISTRY.register()
class ScanFamilyDatasetWrapperOld(Dataset):
    def __init__(self, cfg, dataset):
        # stokenizer, max_seq_length=80, max_obj_len=80
        self.dataset = dataset
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
        self.max_seq_length = cfg.data.args.max_seq_len
        self.max_obj_len = cfg.data.args.max_obj_len

        self.use_voxel = cfg.data.args.get('use_voxel', None)
        if self.use_voxel:
            self.voxel_size = cfg.data.args.get('voxel_size', 0.02)

    def __len__(self):
        return len(self.dataset)

    def pad_tensors(self, tensors, lens=None, pad=0):
        assert tensors.shape[0] <= lens
        if tensors.shape[0] == lens:
            return tensors
        shape = list(tensors.shape)
        shape[0] = lens - shape[0]
        res = torch.ones(shape, dtype=tensors.dtype) * pad
        res = torch.cat((tensors, res), dim=0)
        return res

    def __getitem__(self, idx):
        data_dict = self.dataset[idx]
        sentence = data_dict['sentence']
        encoded_input = self.tokenizer(sentence, max_length=self.max_seq_length,
                          add_special_tokens=True, truncation=True,
                          padding='max_length', return_tensors="pt")
        # build txt
        data_dict['txt_ids'] = encoded_input['input_ids'].squeeze(0) # L
        data_dict['txt_masks'] = encoded_input['attention_mask'].squeeze(0) # L
        # build object
        data_dict['obj_masks'] = (torch.arange(self.max_obj_len) < len(data_dict['obj_locs'])) # O
        if 'obj_fts' in data_dict.keys():
            data_dict['obj_fts'] = self.pad_tensors(data_dict['obj_fts'], lens=self.max_obj_len,
                                                    pad=1.0).float() # O, 1024, 6
        if 'obj_pcds_masks' in data_dict.keys():
            data_dict['obj_pcds_masks'] = pad_tensors(data_dict['obj_pcds_masks'], lens=self.max_obj_len, 
                                                      pad=1.0).float()
        data_dict['obj_locs']= self.pad_tensors(data_dict['obj_locs'], lens=self.max_obj_len,
                                                pad=0.0).float() # O, 3
        data_dict['obj_boxes']= self.pad_tensors(data_dict['obj_boxes'], lens=self.max_obj_len,
                                                 pad=0.0).float() # O, 3
        data_dict['obj_labels'] = self.pad_tensors(data_dict['obj_labels'], lens=self.max_obj_len,
                                                   pad=-100).long() # O
        # build sem mask, no mask
        data_dict['obj_sem_masks'] = (torch.arange(self.max_obj_len) < len(data_dict['obj_locs']))
        # build label for refer
        data_dict['tgt_object_label'] = data_dict['tgt_object_label'].long() # 1 or C
        data_dict['tgt_object_id'] = data_dict['tgt_object_id'].long() # 1 or O
        if len(data_dict['tgt_object_id']) > 1: # O, pad to max objet length
            data_dict['tgt_object_id'] = self.pad_tensors(data_dict['tgt_object_id'].long(),
                                                          lens=self.max_obj_len, pad=0).long() # O
        # build target
        if data_dict.get('tgt_object_id_iou25') is not None:
            data_dict['tgt_object_id_iou25'] = self.pad_tensors(data_dict['tgt_object_id_iou25'],
                                                                lens=self.max_obj_len, pad=0).long()
        if data_dict.get('tgt_object_id_iou50') is not None:
            data_dict['tgt_object_id_iou50'] = self.pad_tensors(data_dict['tgt_object_id_iou50'],
                                                                lens=self.max_obj_len, pad=0).long()
        # build label for qa
        if "answer_label" in data_dict:
            data_dict['answer_label'] = data_dict['answer_label'].long() # N, C
        return data_dict
    
    def collate_fn(self, batch_list):
        if not self.use_voxel:
            ret = default_collate(batch_list)
        
        else:
            new_batch_list = copy.deepcopy(batch_list)
            for i in range(len(new_batch_list)):
                new_batch_list[i].pop('scene_pcds')
            ret = default_collate(new_batch_list)
            # ret.pop('scene_pcds')

            scene_point_nums = []
            for cur_sample in batch_list:
                scene_pcds = cur_sample['scene_pcds']
                scene_point_nums.append(len(scene_pcds))
            scene_point_nums = torch.tensor(scene_point_nums)
            
            # scene_begin_indices = torch.cumsum(torch.tensor(scene_point_nums), dim=0) - scene_point_nums[0]
            scene_begin_indices = []
            for i in range(len(scene_point_nums)):
                if i == 0:
                    scene_begin_indices.append(0)
                else:
                    scene_begin_indices.append(torch.sum(scene_point_nums[:i]))
            batch_size = len(batch_list)

            # ret['obj_pds_masks'] = torch.cat([ret['obj_pcds_masks'][i].unsqueeze(0) + scene_begin_indices[i] for i in range(batch_size)], dim=0)
            for i in range(batch_size):
                ret['obj_pcds_masks'][i] += scene_begin_indices[i]

            from data.voxelize import voxelize_and_inverse
            voxel_coordinates_list = []
            voxel_features_list = []
            v2p_map_list = []
            
            count = 0
            for cur_sample in batch_list:
                scene_pcds = cur_sample['scene_pcds']
                coord = scene_pcds[:, :3]
                feats = scene_pcds[:, 3:]
                coord_min = np.min(coord, 0)
                coord -= coord_min
                coord = coord.astype(np.float32)
                coord = coord / self.voxel_size
                int_coord = coord.astype(np.int32)
                p2v_map, v2p_map = voxelize_and_inverse(int_coord)
                p2v_map = torch.from_numpy(p2v_map)
                v2p_map = torch.from_numpy(v2p_map)
                voxel_coords = torch.from_numpy(coord[p2v_map]).float()
                voxel_feats = torch.from_numpy(feats[p2v_map]).float()

                voxel_coordinates_list.append(voxel_coords)
                voxel_features_list.append(voxel_feats)
                v2p_map_list.append(v2p_map + count)
                count += voxel_coords.shape[0]

            input_dict = {"coords": voxel_coordinates_list, "feats": voxel_features_list}
            voxel_coords, voxel_features = ME.utils.sparse_collate(**input_dict, dtype=torch.float32)

            ret['voxel_features'] = voxel_features
            ret['v2p_map'] = torch.cat(v2p_map_list)
            # ret['v2p_map'] = v2p_map
            ret['voxel_coords'] = voxel_coords

        return ret

class CaptionDatasetWrapper(Dataset):
    def __init__(self, cfg, dataset):
        self.dataset = dataset
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
        self.max_seq_length = cfg.data.args.max_seq_length
        self.max_obj_len = cfg.data.args.max_obj_len
        self.txt_mask_ratio = cfg.data.args.txt_mask_ratio
        self.split = dataset.split

        self.vocab = Vocabulary(os.path.join(cfg.data.scan_family_base,
                                             'annotations/meta_data/scanrefer_vocab.pth'))
        self.corpus = torch.load(os.path.join(cfg.data.scan_family_base,
                                             'annotations/meta_data/scanrefer_corpus.pth'))

    def __len__(self):
        return len(self.dataset)

    def pad_tensors(self, tensors, lens=None, pad=0):
        try:
            assert tensors.shape[0] <= lens
        except:
            print(tensors.shape[0], lens)
            print(tensors.shape)
        if tensors.shape[0] == lens:
            return tensors
        shape = list(tensors.shape)
        shape[0] = lens - shape[0]
        res = torch.ones(shape, dtype=tensors.dtype) * pad
        res = torch.cat((tensors, res), dim=0)
        return res

    def __getitem__(self, idx):
        data_dict = self.dataset[idx]
        sentence = data_dict['sentence']
        encoded_input = self.tokenizer(sentence, max_length=self.max_seq_length,
                          add_special_tokens=True, truncation=True,
                          padding='max_length', return_tensors="pt")
        # build txt
        data_dict['txt_ids'] = encoded_input['input_ids'].squeeze(0) # L
        data_dict['txt_masks'] = encoded_input['attention_mask'].squeeze(0) # L
        # build object
        data_dict['obj_masks'] = (torch.arange(self.max_obj_len) < len(data_dict['obj_locs'])) # O
        data_dict['obj_fts'] = self.pad_tensors(data_dict['obj_fts'],
                                                lens=self.max_obj_len, pad=1.0).float() # O, 1024, 6
        data_dict['obj_locs']= self.pad_tensors(data_dict['obj_locs'],
                                                lens=self.max_obj_len, pad=0.0).float() # O, 3
        data_dict['obj_boxes']= self.pad_tensors(data_dict['obj_boxes'],
                                                 lens=self.max_obj_len, pad=0.0).float() # O, 3
        data_dict['obj_labels'] = self.pad_tensors(data_dict['obj_labels'],
                                                   lens=self.max_obj_len, pad=-100).long() # O
        # build sem mask, no mask
        data_dict['obj_sem_masks'] = (torch.arange(self.max_obj_len) < len(data_dict['obj_locs']))
        # build label for refer
        data_dict['tgt_object_label'] = data_dict['tgt_object_label'].long() # 1 or C
        data_dict['tgt_object_id'] = data_dict['tgt_object_id'].long() # 1 or O
        if len(data_dict['tgt_object_id']) > 1: # O, pad to max objet length
            data_dict['tgt_object_id'] = self.pad_tensors(data_dict['tgt_object_id'].long(),
                                                          lens=self.max_obj_len, pad=0).long() # O
        # build target
        if data_dict.get('tgt_object_id_iou25') is not None:
            data_dict['tgt_object_id_iou25'] = self.pad_tensors(data_dict['tgt_object_id_iou25'],
                                                                lens=self.max_obj_len, pad=0).long()
        if data_dict.get('tgt_object_id_iou50') is not None:
            data_dict['tgt_object_id_iou50'] = self.pad_tensors(data_dict['tgt_object_id_iou50'],
                                                                lens=self.max_obj_len, pad=0).long()
        # build input output for caption
        if self.split == 'train':
            masked_txt_ids, masked_lm_labels = random_caption_word(data_dict['txt_ids'],
                                                                   data_dict['txt_masks'],
                                                                   self.tokenizer, self.vocab,
                                                                   self.txt_mask_ratio)
            data_dict['txt_ids'] = masked_txt_ids
            data_dict['masked_lm_labels'] = masked_lm_labels
        else:
            data_dict['gt_ids'] = data_dict['txt_ids'].clone()
            sentence = ""
            for _ in range(self.max_seq_length - 2):
                sentence += '[MASK]'
            encoded_input = self.tokenizer(sentence, max_length=self.max_seq_length,
                        add_special_tokens=True, truncation=True,
                        padding='max_length', return_tensors="pt")
            data_dict['txt_ids'] = encoded_input['input_ids'].squeeze(0) # L
            data_dict['txt_masks'] = encoded_input['attention_mask'].squeeze(0) # L
        return data_dict

# @DATASETWRAPPER_REGISTRY.register()
# class VoxelDatasetWrapper(Dataset):
#     def __init__(self, cfg, dataset):
#         # TODO: add max_scene_points, voxel_scale, voxel_mode into config files
#         self.dataset = dataset
#         self.max_scene_points = 80000
#         self.voxel_scale = 50
#         self.voxel_mode = 4
#
#     def __len__(self):
#         return len(self.dataset)
#
#     def __getitem__(self, idx):
#         # Subsample points for scene point cloud
#         data_dict = self.dataset[idx]
#         scene_pcds = data_dict["scene_pcds"]
#         pcd_locs = scene_pcds[:, :3]
#         pcd_locs = pcd_locs * self.voxel_scale
#         min_locs, _ = pcd_locs.min(0)
#         pcd_locs = pcd_locs - min_locs
#         scene_pcds[:, :3] = pcd_locs.long()
#         indices = np.random.choice(
#             len(scene_pcds), size=self.max_scene_points, replace=len(scene_pcds) < self.max_scene_points
#         )
#         data_dict["scene_pcds"] = scene_pcds[indices]
#         data_dict["valid_pcd_length"] = len(scene_pcds)
#         return data_dict
#
#     def collate_fn(self, batch):
#         batch = default_collate(batch)
#         scene_pcds = batch["scene_pcds"]        # B, N, 6
#         valid_lengths = batch["valid_pcd_length"]
#         batch_size = scene_pcds.size(0)         # B
#         # Get valid feats and locs
#         scene_pcds = [x[:valid_lengths[i], :] for i, x in enumerate(scene_pcds)]
#         # Sparse tensor needed batch idx as an additional dimension over coordinates
#         pcd_locs = [x[:, :3] for x in scene_pcds]   # B, N, 3
#         batch_idx = [torch.ones(x.size(0), 1) * i for i, x in enumerate(pcd_locs)]  # B, N, 1
#         pcd_locs = torch.cat([torch.cat([x, y], dim=1) for x, y in zip(pcd_locs, batch_idx)], dim=0).long()
#         pcd_feats = torch.cat([x[:, 3:] for x in scene_pcds], dim=0)        # BN, 3
#
#         voxel_coords, v2p_map, p2v_map = sg_ops.voxelization_idx(
#             pcd_locs, batch_size, self.voxel_mode
#         )
#         voxel_feats = sg_ops.voxelization(pcd_feats.contiguous(), p2v_map, self.voxel_mode)
#         batch["voxel_coords"] = voxel_coords
#         batch["v2p_map"] = v2p_map
#         batch["p2v_map"] = p2v_map
#         batch["voxel_feats"] = voxel_feats
#         return batch


if __name__ == '__main__':
    pass
