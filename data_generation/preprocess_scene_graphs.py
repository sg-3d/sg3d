import os
import json
from tqdm import tqdm

datasets = ['3RScan', 'ScanNet', 'ARKitScenes', 'MultiScan', 'HM3D']

def find_object_by_id(objects, id):
    for obj in objects:
        if obj['id'] == id:
            return obj
    return None  # Return None if the object is not found

def add_relations(obj_key, rel, scene_graph_for_gpt, object_info):
    if 'target_id' in rel:
        if rel['target_id'] >= 0:
            target_obj = find_object_by_id(object_info, rel['target_id'])
            if target_obj:
                relation_text = f"{rel['relation']} {target_obj['label']}-{rel['target_id']}"
                scene_graph_for_gpt['object_info'][obj_key]['relations'].append(relation_text)
    elif 'target_id_list' in rel:
        for target_id in rel['target_id_list']:
            if target_id >= 0:
                target_obj = find_object_by_id(object_info, target_id)
                if target_obj:
                    relation_text = f"{rel['relation']} {target_obj['label']}-{target_id}"
                    scene_graph_for_gpt['object_info'][obj_key]['relations'].append(relation_text)

def transfer_for_gpt(scene_graph_w_obj_cap, strict=False):
    scene_graph_for_gpt = {
        'scene_id': scene_graph_w_obj_cap['scene_id'],
        'object_info': {}
    }
    for obj in scene_graph_w_obj_cap['object_info']:
        obj_key = f"{obj['label']}-{obj['id']}"
        scene_graph_for_gpt['object_info'][obj_key] = {'relations': []}

        # Exclude caption for objects with 'count' == 1 or labels 'wall' or 'floor'
        if obj['label'] not in ['wall', 'floor']:
            if (not strict) or ('count' in obj and obj['count'] > 1):
                if len(obj['caption']) <= 600: # Avoid too long captions, which are usually errors
                    scene_graph_for_gpt['object_info'][obj_key]['caption'] = obj['caption']

        # Adding relationships
        for rel in obj['relationships']:
            add_relations(obj_key, rel, scene_graph_for_gpt, scene_graph_w_obj_cap['object_info'])

    return scene_graph_for_gpt


for dataset in datasets:
    scene_graph_dir = os.path.join('/path/to/SceneVerse/scene/graphs', dataset) # TODO: change the path
    save_dir = os.path.join('scene_graphs_w_obj_cap', dataset)
    dataset_name = os.path.basename(os.path.normpath(save_dir))

    def find_caption(scan_id, obj_id, captions):
        for caption in captions:
            if scan_id == caption['scan_id'] and str(obj_id) == caption['target_id']:
                return caption['utterance']
        return ""

    # Load the captions JSON
    if dataset_name == 'HM3D':
        captions = []
    else:
        if dataset_name == 'ScanNet':
            captions_path = os.path.join('/path/to/SceneVerse', dataset, 'annotations/refer/ssg_obj_caption_gpt.json') # TODO: change the path
        else:
            captions_path = os.path.join('/path/to/SceneVerse', dataset, 'annotations/ssg_obj_caption_gpt.json') # TODO: change the path
        with open(captions_path, 'r') as file:
            captions = json.load(file)

    for dir_name in tqdm(os.listdir(scene_graph_dir)):
        try:
            scene_graph_path = os.path.join(scene_graph_dir, dir_name, 'objects.json')
            relationships_path = os.path.join(scene_graph_dir, dir_name, 'relationships.json')
            save_path = os.path.join(save_dir, dir_name, 'scene_graphs_w_obj_cap.json')

            # Load the scene graph JSON
            with open(scene_graph_path, 'r') as file:
                scene_graph = json.load(file)
            
            # Load the relationships JSON
            with open(relationships_path, 'r') as file:
                relationships_data = json.load(file)

            # Initialize the new structure
            new_structure = {
                "scene_id": dir_name,
                "object_info": [],
                "inst_to_label": {}
            }

            # Fill in the captions in the scene graph
            for scene_id, scene_data in scene_graph.items():
                new_structure['inst_to_label'] = scene_data['inst_to_label']
                for obj in scene_data['objects_info']:
                    obj['id'] = int(obj['id'])

                    obj['caption'] = find_caption(dir_name, obj['id'], captions)
                    obj['relationships'] = []
                    if 'mesh' in obj:
                        del obj['mesh']

                    # Add relationships to objects
                    for rel in relationships_data[scene_id]['relationships']:
                        target_id = rel[1] # id < 0: -2 for wall, -3 for floor
                        if obj['id'] == rel[0]:
                            obj['relationships'].append({'target_id': target_id, 'relation': rel[2]})

                    # Handle multi-object relationships
                    for rel in relationships_data[scene_id]['multi_objs_relationships']:
                        assert rel[1] in ['Aligned', 'in the middle of'], f"Unknown relationship: {rel[1]}"
                        if obj['id'] in rel[0]:
                            if rel[1] == "Aligned":
                                # For "Aligned", add all other object IDs to the target_id_list
                                target_id_list = [id for id in rel[0] if obj['id'] != id]
                                obj['relationships'].append({'target_id_list': target_id_list, 'relation': rel[1]})
                            elif rel[1] == "in the middle of" and obj['id'] == rel[0][0]:
                                # For "in the middle of", the first object is in the middle of the other two
                                target_id_list = [rel[0][1], rel[0][2]]
                                obj['relationships'].append({'target_id_list': target_id_list, 'relation': rel[1]})

                    new_structure["object_info"].append(obj)

            # save preprocessed scene graph
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            with open(save_path, 'w') as file:
                json.dump(new_structure, file, indent=4)

            # transfer for gpt and save
            scene_graph_for_gpt = transfer_for_gpt(new_structure)
            with open(save_path.replace('scene_graphs_w_obj_cap.json', 'scene_graphs_for_gpt.json'), 'w') as file:
                json.dump(scene_graph_for_gpt, file, indent=4)
            
        except Exception as e:
            print(f"{dataset_name} {dir_name}: {str(e)}\n")
            continue
