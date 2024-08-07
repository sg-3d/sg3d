## Task-oriented Sequential Grounding in 3D Scenes

<p align="left">
    <a href=''>
      <img src='https://img.shields.io/badge/Paper-PDF-red?style=plastic&logo=adobeacrobatreader&logoColor=red' alt='Paper PDF'>
    </a>
    <a href=''>
      <img src='https://img.shields.io/badge/Paper-arXiv-green?style=plastic&logo=arXiv&logoColor=green' alt='Paper arXiv'>
    </a>
    <a href='https://sg-3d.github.io'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=plastic&logo=Google%20chrome&logoColor=blue' alt='Project Page'>
    </a>
    <a href=''>
      <img src='https://img.shields.io/badge/Demo-HuggingFace-yellow?style=plastic&logo=AirPlay%20Video&logoColor=yellow' alt='HuggigFace'>
    </a>
    <a href=''> <img src='https://img.shields.io/badge/Model-Checkpoints-orange?style=plastic&logo=Google%20Drive&logoColor=orange'  alt='Checkpoints(TODO)'>
    </a>
</p>


[Zhuofan Zhang](https://tongclass.ac.cn/author/zhuofan-zhang/), 
[Ziyu Zhu](https://zhuziyu-edward.github.io/), 
[Pengxiang Li](),
[Tengyu Liu](),
[Xiaojian Ma](https://jeasinema.github.io/), 
[Yixin Chen](https://yixchen.github.io/),
[Baoxiong Jia](https://buzz-beater.github.io),
[Siyuan Huang](https://siyuanhuang.com/),
[Qing Li](https://liqing-ustc.github.io/)📧

This repository is the official implementation of the paper "Task-oriented Sequential Grounding in 3D Sceness".

<div align=center>
<img src='https://sg-3d.github.io/assets/model-1.png' width=100%>
</div>

### News
- [ 2024.08 ] Release codes of model! 

### Abstract

Grounding natural language in physical 3D environments is essential for the advancement of embodied artificial intelligence. Current datasets and models for 3D visual grounding predominantly focus on identifying and localizing objects from static, object-centric descriptions. These approaches do not adequately address the dynamic and sequential nature of task-oriented grounding necessary for practical applications. In this work, we propose a new task: Task-oriented Sequential Grounding in 3D scenes, wherein an agent must follow detailed step-by-step instructions to complete daily activities by locating a sequence of target objects in indoor scenes. To facilitate this task, we introduce SG3D, a large-scale dataset containing 22,346 tasks with 112,236 steps across 4,895 real-world 3D scenes. The dataset is constructed using a combination of RGB-D scans from various 3D scene datasets and an automated task generation pipeline, followed by human verification for quality assurance. We adapted three state-of-the-art 3D visual grounding models to the sequential grounding task and evaluated their performance on SG3D. Our results reveal that while these models perform well on traditional benchmarks, they face significant challenges with task-oriented sequential grounding, underscoring the need for further research in this area.

### Install
1. Install conda package
```
conda env create --name envname 
pip3 install torch==2.0.0
pip3 install torchvision==0.15.1
pip3 install -r requirements.txt
```

2. install pointnet2
```
cd modules/third_party
# PointNet2
cd pointnet2
python setup.py install
cd ..
```

### Prepare data and checkpoint
1. download sceneverse data  from [scene_verse_base](https://github.com/scene-verse/sceneverse?tab=readme-ov-file) and change `data.scene_verse_base` to sceneverse data directory.
2. download segment level data from [scene_ver_aux](https://drive.google.com/drive/folders/1em0G5S4aH4lCIfnjLxyFO3DV3Rwuwnqh?usp=share_link) and change `data.scene_verse_aux` to download data directory.
3. download other data from [scene_verse_pred](https://drive.google.com/drive/folders/12BjbhXzV7lON4X0tx3e7DdvZHhI1Q1f2?usp=share_link) and change `data.scene_verse_pred` to download data directory.
4. download sequential grounding checkpoint and data from [sequential-grounding]() and change `data.sequential_grounding_base` to download directory, change `pretrained_weights_dir` to downloaded pointnet dir.
4. download Vicuna-7B form [Vicuna](https://huggingface.co/huangjy-pku/vicuna-7b/tree/main) and change `model.llm.cfg_path`
5. change TBD in config


### Run 3D-VisTA on SG3D benchmark
Training
```
python3 run.py --config-path configs/vista --config-name sequential-sceneverse.yaml
```
Testing
```
python3 run.py --config-path configs/vista --config-name sequential-sceneverse.yaml pretrain_ckpt_path=PATH
```
### Run PQ3D on SG3D benchmark
Training
```
python3 run.py --config-path configs/query3d --config-name sequential-sceneverse.yaml
```
Testing
```
python3 run.py --config-path configs/query3d --config-name sequential-sceneverse.yaml pretrain_ckpt_path=PATH
```
### Run 3D-LLM LEO on SG3D benchmark
Training
```
python3 run.py --config-path configs/sequential --config-name sequential-sceneverse.yaml
```
Testing
```
python3 run.py --config-path configs/sequential--config-name sequential-sceneverse.yaml pretrain_ckpt_path=PATH
```
For multi-gpu training usage.
```
python launch.py --mode ${launch_mode} \
    --qos=${qos} --partition=${partition} --gpu_per_node=4 --port=29512 --mem_per_gpu=80 \
    --config {config}  \
```

### Acknowledgement
We would like to thank the authors of [LEO](https://embodied-generalist.github.io), [Vil3dref](https://github.com/cshizhe/vil3dref), [Mask3d](https://github.com/JonasSchult/Mask3D), [Openscene](https://github.com/pengsongyou/openscene), [Xdecoder](https://github.com/microsoft/X-Decoder), and [3D-VisTA](https://github.com/3d-vista/3D-VisTA) for their open-source release.


### Citation:
```
@article{zhu2024unifying,
    title={Unifying 3D Vision-Language Understanding via Promptable Queries},
    author={Zhu, Ziyu and Zhang, Zhuofan and Ma, Xiaojian and Niu, Xuesong and Chen, Yixin and Jia, Baoxiong and Deng, Zhidong and Huang, Siyuan and Li, Qing},
    journal={arXiv preprint arXiv:2405.11442},
    year={2024}
}
```
