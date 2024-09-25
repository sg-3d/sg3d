# Data Generation Guide

This directory contains scripts to generate tasks using GPT-4 with SceneVerse scene graphs. Follow the steps below to run the pipeline:

1. **Generate SceneVerse Scene Graphs**
   To generate SceneVerse scene graphs, refer to the [SceneVerse Scene Graph Generation Guide](https://github.com/scene-verse/SceneVerse/tree/main?tab=readme-ov-file#scene-graph-generation).

2. **Add Captions to Scene Graphs**
   Run the script `preprocess_scene_graphs.py` to add captions to SceneVerse scene graphs.  
   - **Output:** Refactored scene graphs will be saved in the `scene_graphs_w_obj_cap/` directory.

3. **Select Representative Scans & Split Datasets**
   Run `select_representative_scans.py` to select one representative scan per venue and split the dataset into training and testing sets.
   - **Output:** The split information will be saved in the `splits/` directory.

4. **Generate Task Data with GPT-4**
   Run `task_generation.py` to generate task data using GPT-4. The raw task data is saved as `.txt`, and the refactored data as `.json`. Additionally, the merged data from all scenes is saved as `{SPLITS}.json`, which matches the format of the distributed dataset and is ready for use with the model.
   - **Output:** Raw, refactored outputs, and the merged data will be saved in the `responses/` directory.
