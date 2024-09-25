import os
import json
import csv
import pandas as pd


def get_representative_scans(dataset):
    representative_scans = []
    
    if dataset in ['MultiScan', 'ScanNet']:
        scene_graph_dir = 'scene_graphs_w_obj_cap/' + dataset
        for scan_id in os.listdir(scene_graph_dir):
            if scan_id.endswith('_00'):
                representative_scans.append(scan_id)
    
    elif dataset == '3RScan':
        with open(os.path.dirname(os.path.realpath(__file__)) + '/dataset_info/3RScan.json', 'r') as f:
            data = json.load(f)
        for entry in data:
            representative_scans.append(entry['reference'])
    
    elif dataset == 'ARKitScenes':
        df = pd.read_csv(os.path.dirname(os.path.realpath(__file__)) + '/dataset_info/metadata.csv')
        df = df.sort_values('visit_id')
        arkit_scans_dir = os.path.dirname(os.path.realpath(__file__)) + '/scene_graphs_w_obj_cap/ARKitScenes'
        existing_video_ids = set(os.listdir(arkit_scans_dir))

        for visit_id, group in df.groupby('visit_id'):
            for video_id in group['video_id']:
                if str(video_id) in existing_video_ids:
                    representative_scans.append(str(video_id))
                    break
    
    elif dataset == 'HM3D':
        hm3d_dir = os.path.dirname(os.path.realpath(__file__)) + '/scene_graphs_w_obj_cap/HM3D'
        for scan_id in os.listdir(hm3d_dir):
            file_path = f"{hm3d_dir}/{scan_id}/scene_graphs_w_obj_cap.json"
            with open(file_path, 'r') as f:
                data = json.load(f)
            if len(data.get("inst_to_label", {})) > 0:
                representative_scans.append(scan_id)
    
    else:
        raise ValueError(f'Unknown dataset: {dataset}')
    
    return representative_scans


def get_test_scans(dataset):
    if dataset == '3RScan':
        with open('dataset_info/3RScan.json') as f:
            data = json.load(f)
        return [item['reference'] for item in data if item['type'] == 'test']

    elif dataset == 'ScanNet':
        with open('dataset_info/scannetv2_val_sort.json') as f:
            return json.load(f)

    elif dataset == 'ARKitScenes':
        with open('dataset_info/metadata.csv', newline='') as f:
            reader = csv.DictReader(f)
            return [row['video_id'] for row in reader if row['fold'] == 'Validation']

    elif dataset == 'MultiScan':
        with open('dataset_info/MultiScan_val_split_non_overlap.txt') as f:
            return f.read().splitlines()

    elif dataset == 'HM3D':
        with open('dataset_info/HM3D_val_split_non_overlap.txt') as f:
            return f.read().splitlines()

    else:
        raise ValueError(f'Unknown dataset: {dataset}')


def filter_unmatch_ids(dataset, scan):
    scan_path = f'scene_graphs_w_obj_cap/{dataset}/{scan}/scene_graphs_w_obj_cap.json'
    with open(scan_path, 'r') as f:
        data = json.load(f)        
        for obj in data['object_info']:
            obj_id = str(obj['id'])
            obj_label = obj['label']
            if obj_id not in data['inst_to_label'] or data['inst_to_label'][obj_id] != obj_label:
                print(f'Unmatched object id and label in {dataset}: {scan}')
                return False    
    return True


def select_scans(representative_scans, test_scans, is_test=True):
    if is_test:
        return [scan for scan in representative_scans if scan in test_scans]
    else:
        return [scan for scan in representative_scans if scan not in test_scans]


def process_dataset(dataset):
    representative_scans = get_representative_scans(dataset)
    test_scans = get_test_scans(dataset)

    filtered_scans = [scan for scan in representative_scans if filter_unmatch_ids(dataset, scan)]

    test_representative_scans = select_scans(filtered_scans, test_scans, is_test=True)
    train_representative_scans = select_scans(filtered_scans, test_scans, is_test=False)

    with open(f'splits/{dataset}_test_scans.txt', 'w') as f:
        for scan in test_representative_scans:
            f.write(f'{scan}\n')

    with open(f'splits/{dataset}_train_scans.txt', 'w') as f:
        for scan in train_representative_scans:
            f.write(f'{scan}\n')


if __name__ == '__main__':
    os.makedirs('splits', exist_ok=True)
    
    for dataset in ['MultiScan', 'ScanNet', '3RScan', 'ARKitScenes', 'HM3D']:
        process_dataset(dataset)
        print(f'{dataset} train and test representative scans processed and saved.')
