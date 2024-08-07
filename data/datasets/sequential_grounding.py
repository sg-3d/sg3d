from copy import deepcopy
import json
import os
import random
import re
import jsonlines

from data.build import DATASET_REGISTRY
from data.datasets.sceneverse_base import SceneVerseBase

role_prompt = "You are an AI visual assistant situated in a 3D scene. "\
    "You can perceive (1) an ego-view image (accessible when necessary) and (2) the objects (including yourself) in the scene (always accessible). "\
        "You should properly respond to the USER's instruction according to the given visual information. "
#role_prompt = " " 
egoview_prompt = "Ego-view image:"
objects_prompt = "Objects (including you) in the scene:"
task_prompt = "USER: {instruction} ASSISTANT:"
    
def get_prompt(instruction):
    return {
            'prompt_before_obj': role_prompt,
            'prompt_middle_1': egoview_prompt,
            'prompt_middle_2': objects_prompt,
            'prompt_after_obj': task_prompt.format(instruction=instruction),
        }
    
class SequentialGrounding(SceneVerseBase):
    def __init__(self, cfg, dataset_name, split):
        # TODO: hack test split to be the same as val
        if split == 'test':
            split = 'val'
        super().__init__(cfg, dataset_name, split)
        self.sequential_grounding_base = cfg.data.get('sequential_grounding_base')
        dataset_cfg = cfg.data.get(self.__class__.__name__)
        self.init_dataset_params(dataset_cfg)
        if self.split == 'train':
            self.pc_type = 'gt'
        self.multi_step_context = cfg.data.get('multi_step_context', True)
        self.drop_data_percent = cfg.data.get('drop_data_percent', 0.0)
        print(f'Loading {self.__class__.__name__}')
        self.lang_data, self.scan_ids = self._load_lang()
        self.init_scan_data()

    def get_lang(self, index):
        item = self.lang_data[index]
        task_description = item['task_description']
        action_steps = item['action_steps']
        tgt_object_id = [int(action['target_id']) for action in action_steps]
        tgt_object_name = [action['label'] for action in action_steps]
        sentence = task_description
        for action in action_steps:
            sentence += ' ' + action['action']
        scan_id = item['scan_id']

        data_dict = get_prompt(task_description)
        data_dict['output_gt'] = ' '.join([action['action'] + ' <s>' for action in action_steps])
        data_dict['source'] = self.dataset_name
        data_dict['data_idx'] = item['item_id']

        return scan_id, tgt_object_id, tgt_object_name, sentence, data_dict

    def extract_filenames(self, directory, pattern):
        # List all files in the directory
        all_files = os.listdir(directory)
        # Filter files based on the regular expression
        matched_files = [filename for filename in all_files 
                        if os.path.isfile(os.path.join(directory, filename)) and re.match(pattern, filename)]
        return matched_files

    def _load_lang(self):
        split_scan_ids = self._load_split(self.cfg, self.split)
        lang_data = []
        scan_ids = set()
        # filter data to current dataset
        if self.split == 'train':
            anno_file_base = os.path.join(self.sequential_grounding_base, 'release_data', 'train.json')
        else:
            anno_file_base = os.path.join(self.sequential_grounding_base, 'release_data', 'test.json')
        anno_data_pre_filter = json.load(open(anno_file_base))
        anno_data = []
        for data in anno_data_pre_filter:
            dataset_name = data['scan_id'].split('_')[0]
            if dataset_name == self.dataset_name:
                anno_data.append(data)
        # debug mode
        if self.cfg.debug.flag == True and self.cfg.debug.debug_size != -1:
            anno_data = anno_data[:self.cfg.debug.debug_size]
        # read data
        for i, data in enumerate(anno_data):
            dataset_name = data['scan_id'].split('_')[0]
            scan_id = data['scan_id'][len(self.dataset_name) + 1:]
            scan_ids.add(scan_id)
            if self.split == 'train' and self.drop_data_percent > 0.0 and random.random() < self.drop_data_percent:
                continue 
            item = {'task_description': data['task_description'], 'action_steps': data['action_steps']}
            item['scan_id'] = scan_id
            item['item_id'] = f'f{scan_id}_{i}'
            if self.multi_step_context:
                lang_data.append(item)
            else:
                for j in range(len(item['action_steps'])):
                    new_item = deepcopy(item)
                    new_item['action_steps'] = [item['action_steps'][j]]
                    lang_data.append(new_item) 
                    
        return lang_data, scan_ids

@DATASET_REGISTRY.register()
class SequentialGroundingScanNet(SequentialGrounding):
    def __init__(self, cfg, split):
        super().__init__(cfg, 'ScanNet', split)

@DATASET_REGISTRY.register()
class SequentialGrounding3RScan(SequentialGrounding):
    def __init__(self, cfg, split):
        super().__init__(cfg, '3RScan', split)
        
@DATASET_REGISTRY.register()
class SequentialGroundingMultiScan(SequentialGrounding):
    def __init__(self, cfg, split):
        super().__init__(cfg, 'MultiScan', split)
        
@DATASET_REGISTRY.register()
class SequentialGroundingARKitScenes(SequentialGrounding):
    def __init__(self, cfg, split):
        super().__init__(cfg, 'ARKitScenes', split)

@DATASET_REGISTRY.register()
class SequentialGroundingHM3D(SequentialGrounding):
    def __init__(self, cfg, split):
        super().__init__(cfg, 'HM3D', split)

class SequentialGroundingSingleStep(SceneVerseBase):
    def __init__(self, cfg, dataset_name, split):
        # TODO: hack test split to be the same as val
        if split == 'test':
            split = 'val'
        super().__init__(cfg, dataset_name, split)
        self.sequential_grounding_base = cfg.data.get('sequential_grounding_base')
        dataset_cfg = cfg.data.get(self.__class__.__name__)
        self.init_dataset_params(dataset_cfg)
        if self.split == 'train':
            self.pc_type = 'gt'
        self.multi_step_context = cfg.data.get('multi_step_context', True)
        self.drop_data_percent = cfg.data.get('drop_data_percent', 0.0)
        print(f'Loading {self.__class__.__name__}')
        self.lang_data, self.scan_ids = self._load_lang()
        self.init_scan_data()

    def get_lang(self, index):
        item = self.lang_data[index]
        item_id = item['item_id']
        scan_id = item['scan_id']
        tgt_object_id = int(item['target_id'])
        tgt_object_name = item['instance_type']
        sentence = item['utterance']
        obj_key = f"{scan_id}|{tgt_object_id}|{tgt_object_name}" # used to group the captions for a single object

        data_dict = {
            "data_idx": item_id,
            "sentence": sentence,
            "obj_key": obj_key
        }

        return scan_id, tgt_object_id, tgt_object_name, sentence, data_dict
    
    def extract_filenames(self, directory, pattern):
        # List all files in the directory
        all_files = os.listdir(directory)
        # Filter files based on the regular expression
        matched_files = [filename for filename in all_files 
                        if os.path.isfile(os.path.join(directory, filename)) and re.match(pattern, filename)]
        return matched_files

    def _load_lang(self):
        split_scan_ids = self._load_split(self.cfg, self.split)
        lang_data = []
        scan_ids = set()
        # filter data to current dataset
        if self.split == 'train':
            anno_file_base = os.path.join(self.sequential_grounding_base, 'release_data', 'train.json')
        else:
            anno_file_base = os.path.join(self.sequential_grounding_base, 'release_data', 'test.json')
        anno_data_pre_filter = json.load(open(anno_file_base))
        anno_data = []
        for data in anno_data_pre_filter:
            dataset_name = data['scan_id'].split('_')[0]
            if dataset_name == self.dataset_name:
                anno_data.append(data)
        # debug mode
        if self.cfg.debug.flag == True and self.cfg.debug.debug_size != -1:
            anno_data = anno_data[:self.cfg.debug.debug_size]        
        # read data
        for i, data in enumerate(anno_data):
            dataset_name = data['scan_id'].split('_')[0]
            scan_id = data['scan_id'][len(self.dataset_name) + 1:]
            scan_ids.add(scan_id)
            if self.split == 'train' and self.drop_data_percent > 0.0 and random.random() < self.drop_data_percent:
                continue 
            item = {'task_description': data['task_description'], 'action_steps': data['action_steps']}
            item['scan_id'] = scan_id
            item['item_id'] = f'f{scan_id}_{i}'
            for j in range(len(item['action_steps'])):
                new_item = {}
                new_item['item_id'] = f'f{scan_id}_{i}'
                new_item['scan_id'] = scan_id
                new_item['target_id'] = int(item['action_steps'][j]['target_id'])
                new_item['instance_type'] = item['action_steps'][j]['label']
                utterance = item['task_description']
                if self.multi_step_context:
                    for k in range(j + 1):
                        utterance += ' ' + item['action_steps'][k]['action']
                else:
                    utterance += ' ' + item['action_steps'][j]['action']
                new_item['utterance'] = utterance
                lang_data.append(new_item)
        return lang_data, scan_ids

@DATASET_REGISTRY.register()
class SequentialGroundingSingleStepScanNet(SequentialGroundingSingleStep):
    def __init__(self, cfg, split):
        super().__init__(cfg, 'ScanNet', split)

@DATASET_REGISTRY.register()
class SequentialGroundingSingleStep3RScan(SequentialGroundingSingleStep):
    def __init__(self, cfg, split):
        super().__init__(cfg, '3RScan', split)
        
@DATASET_REGISTRY.register()
class SequentialGroundingSingleStepMultiScan(SequentialGroundingSingleStep):
    def __init__(self, cfg, split):
        super().__init__(cfg, 'MultiScan', split)
        
@DATASET_REGISTRY.register()
class SequentialGroundingSingleStepARKitScenes(SequentialGroundingSingleStep):
    def __init__(self, cfg, split):
        super().__init__(cfg, 'ARKitScenes', split)

@DATASET_REGISTRY.register()
class SequentialGroundingSingleStepHM3D(SequentialGroundingSingleStep):
    def __init__(self, cfg, split):
        super().__init__(cfg, 'HM3D', split)