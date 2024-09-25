import os
import json
import time
import datetime
import re
from tqdm import tqdm
from openai import AzureOpenAI
from concurrent.futures import ProcessPoolExecutor


# Constants and paths
SYSTEM_PROMPT_PATH = "system_prompt.txt"
SYSTEM_EXAMPLE_PATH = "examples.txt"
DATASETS = ["MultiScan", "ScanNet", "ARKitScenes", "3RScan", "HM3D"]
SPLIT = "test"
SCENE_GRAPH_DIR_BASE = "scene_graphs_w_obj_cap/"

# New API setup # TODO: change the API key and API base
REGION = "eastus2" # eastus2 or swedencentral
MODEL = "gpt-4-turbo-2024-04-09"
API_KEY = "your/api/key"
API_BASE = "your/api/base"
ENDPOINT = f"{API_BASE}/{REGION}"

# Initialize AzureOpenAI client
client = AzureOpenAI(
    api_key=API_KEY,
    api_version="2024-02-01",
    azure_endpoint=ENDPOINT,
)

def get_chat_response(messages, model=MODEL, temperature=1.0, max_tokens=256, n=1, frequency_penalty=0.0, presence_penalty=0.0, patience=1, sleep_time=0):
    while patience > 0:
        patience -= 1
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                n=n,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty
            )
            if n == 1:
                prediction = response.choices[0].message.content.strip()
                if prediction != "" and prediction != None:
                    return prediction
            else:
                prediction = [choice.message.content.strip() for choice in response.choices]
                if prediction[0] != "" and prediction[0] != None:
                    return prediction
        except Exception as e:
            print(e)
            if sleep_time > 0:
                time.sleep(sleep_time)
    return ""

def fetch_openai_response(dataset, scene_graph, timestamp):
    messages = load_prompt(SYSTEM_PROMPT_PATH, SYSTEM_EXAMPLE_PATH)
    scene_graph_content = json.dumps({"object_info": scene_graph["object_info"]})
    messages.append({"role": "user", "content": scene_graph_content})
    tasks = get_chat_response(messages, model=MODEL, max_tokens=1000, presence_penalty=0.0, patience=3, sleep_time=8)
    
    # Save tasks
    response_dir = f"responses/{timestamp}"
    os.makedirs(response_dir, exist_ok=True)
    with open(os.path.join(response_dir, f"{dataset}_{scene_graph['scene_id']}.txt"), 'w') as file:
        file.write(str(tasks))
    
    # Parse tasks and save refactored results
    objects_info = load_objects_info(scene_graph['scene_id'], dataset)
    task_list = tasks.split("===")
    task_list = [task.strip(" \n") for task in task_list]
    parsed_tasks = parse_task(task_list, objects_info)

    if parsed_tasks:
        with open(os.path.join(response_dir, f"{dataset}_{scene_graph['scene_id']}.json"), 'w') as file:
            json.dump(parsed_tasks, file, indent=4)

def load_system_prompt(system_prompt_path):
    with open(system_prompt_path, 'r') as file:
        data = file.read()
    return [{"role": "system", "content": data}]

def load_examples(system_example_path):
    with open(system_example_path, 'r') as file:
        lines = file.readlines()
    data = ""
    for line in lines:
        data += line
    return data

def load_prompt(system_prompt_path, system_example_path):
    system_prompts = load_system_prompt(system_prompt_path)
    system_examples = load_examples(system_example_path)
    system_prompts[0]['content'] = system_prompts[0]['content'].replace("<EXAMPLES>", system_examples)
    return system_prompts

def load_objects_info(scene_id, dataset):
    with open(os.path.join(SCENE_GRAPH_DIR_BASE, dataset, scene_id, "scene_graphs_w_obj_cap.json"), 'r') as file:
        data = json.load(file)
    return {str(obj["id"]): obj for obj in data["object_info"]}

def parse_task(task_list, objects_info):
    parsed_tasks = []
    for task in task_list:
        lines = task.split("\n")
        if not lines[0].startswith("Task:"):
            continue

        task_pattern = re.compile(r"Task:\s*(.+)")
        match = task_pattern.search(lines[0])
        task_description = match.group(1)

        if not (lines[1] == "Steps:" or lines[1] == "Steps: "):
            print(f"Error parsing steps: {lines[1]}")
            continue
        
        steps = []
        target_obj_missing = False

        for line in lines[2:]:
            try:
                i, j = line.index('['), line.index(']')
                step = line[:i].strip()
                target_object = line[i+1:j].strip()
                target_id = target_object.split("-")[-1]
                label = ' '.join(target_object.split("-")[:-1])
            except:
                target_obj_missing = True
                print(f"Error parsing line: {line}")
                break

            obj_info = objects_info.get(target_id, {})
            if not obj_info or label != obj_info.get("label"):
                target_obj_missing = True
                break

            steps.append({
                "action": step,
                "target_id": target_id,
                "label": label
            })
        
        if target_obj_missing:
            continue
        parsed_tasks.append({"task_description": task_description, "action_steps": steps})
    
    return parsed_tasks


def main():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    for dataset in DATASETS:
        with open(f'splits/{dataset}_{SPLIT}_scans.txt', 'r') as f:
            lines = f.readlines()
        all_scenes = [line.rstrip() for line in lines]

        scene_graph_dir = SCENE_GRAPH_DIR_BASE + dataset
        all_scene_graphs = []
        for scene in all_scenes:
            file_path = os.path.join(scene_graph_dir, scene, "scene_graphs_for_gpt.json")
            if file_path.endswith('.json'):
                with open(file_path, 'r') as f:
                    all_scene_graphs.append(json.load(f))

        with ProcessPoolExecutor(max_workers=4) as executor:
            for i, scene_graph in enumerate(tqdm(all_scene_graphs)):
                executor.submit(fetch_openai_response, dataset, scene_graph, timestamp)

        # merge data
        merged_data = []
        response_dir = f"responses/{timestamp}"
        for file in os.listdir(response_dir):
            if file.endswith('.json'):
                with open(os.path.join(response_dir, file), 'r') as f:
                    data = json.load(f)
                    for task in data:
                        task['scan_id'] = file.split('.')[0]
                    merged_data.extend(data)
        with open(os.path.join(response_dir, f'{SPLIT}.json'), 'w') as f:
            json.dump(merged_data, f, indent=4)

if __name__ == "__main__":
    main()
