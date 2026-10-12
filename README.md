# SimGro: Evidence-linked Case-Prototype Learning for Preoperative Pathological Stratification of Lung Adenocarcinoma Spectrum Nodules

<div align="center">

**📖 Paper link will be added upon publication.**

Yuting Zheng<sup>1,2,3,&dagger;</sup> ·
Jiajie Wei<sup>4,&dagger;</sup> ·
Xian Lin<sup>5,&dagger;</sup> ·
Lin Teng<sup>6,&dagger;</sup> ·
Xiaofei Yue<sup>7,&dagger;</sup> ·
Huasheng Mao<sup>4</sup> ·
Qinyue Luo<sup>1,2,3</sup> ·
Yimeng He<sup>1,2,3</sup> ·
Yukun Cao<sup>1,2,3</sup> ·
Leqing Chen<sup>1,2,3</sup> ·
Shiqi Li<sup>1,2,3</sup> ·
Libo Shen<sup>8</sup> ·
Mengxue Gao<sup>1,2,3</sup> ·
Zhiyu Huang<sup>9</sup> ·
Zeng Xiong<sup>9</sup> ·
Meiyun Wang<sup>6</sup> ·
Haonan Gong<sup>10</sup> ·
Xiaocui Liu<sup>1,2,3</sup> ·
Jianbo Gao<sup>7</sup> ·
Jingrui Xu<sup>4</sup> ·
Xin Yang<sup>4</sup> ·
Jun Fan<sup>11,*</sup> ·
Yu Zhang<sup>12,*</sup> ·
Xiaoyu Han<sup>1,2,3,*</sup> ·
Heshui Shi<sup>1,2,3,*</sup> ·
Zengqiang Yan<sup>4,*</sup>

<br>

<sup>1</sup>Department of Radiology, Union Hospital, Tongji Medical College,
Huazhong University of Science and Technology, Wuhan 430022, China.<br>
<sup>2</sup>Hubei Provincial Clinical Research Center for Precision Radiology &
Interventional Medicine, Wuhan 430022, China.<br>
<sup>3</sup>Hubei Key Laboratory of Molecular Imaging, Wuhan 430022, China.<br>
<sup>4</sup>School of Electronic Information and Communications, Huazhong
University of Science and Technology, Wuhan 430074, China.<br>
<sup>5</sup>Department of Ophthalmology, Yong Loo Lin School of Medicine,
National University of Singapore, Singapore.<br>
<sup>6</sup>Henan Provincial People's Hospital & People's Hospital of Zhengzhou
University, Zhengzhou 450003, China.<br>
<sup>7</sup>Department of Radiology, The First Affiliated Hospital of Zhengzhou
University, Zhengzhou 450052, China.<br>
<sup>8</sup>Xiangyang Hospital of Traditional Chinese Medicine (Xiangyang
Institute of Traditional Chinese Medicine), Xiangyang 441001, China.<br>
<sup>9</sup>Department of Radiology, Xiangya Hospital, Central South University,
Changsha 410008, China.<br>
<sup>10</sup>Department of Medical Imaging, Nanchang Hongdu Hospital of
Traditional Chinese Medicine, Nanchang 330008, China.<br>
<sup>11</sup>Department of Pathology, Union Hospital, Tongji Medical College,
Huazhong University of Science and Technology, Wuhan 430022, China.<br>
<sup>12</sup>Shanghai Chest Hospital, Shanghai Jiao Tong University School of
Medicine, Shanghai, China.

<br>

<sup>*</sup>Corresponding authors:
[Jun Fan](mailto:fanjun0915@sina.com),
[Yu Zhang](mailto:zybz77@163.com),
[Xiaoyu Han](mailto:xiaoyuhan1123@163.com),
[Heshui Shi](mailto:heshuishi@hust.edu.cn), and
[Zengqiang Yan](mailto:z_yan@hust.edu.cn).<br>
<sup>&dagger;</sup>These authors contributed equally: Yuting Zheng, Jiajie Wei,
Xian Lin, Lin Teng, and Xiaofei Yue.

</div>

## 🚀 Overview

**Clinical problem.** Fine-grained preoperative pathological stratification of lung adenocarcinoma from computed tomography is limited by overlapping imaging appearances and pathology-defined categories.

**Our approach.** SimGro is a case-prototype learning framework that enables case-based reasoning for pathological stratification by comparing CT phenotypes with real case-anchored prototypes rather than relying on direct image-to-label classification, returning matched cases as inspectable evidence.

<p align="center">
  <img src="SimGro.png"
       alt="SimGro architecture and training strategy" width="100%" />
</p>

## ⚙️ Environment Setup

### 1. Create an environment

Create and activate a dedicated Python environment:

```bash
conda create -n simgro python=3.10 -y
conda activate simgro
```

### 2. Install PyTorch

Install a PyTorch build compatible with your CUDA environment by following the
[official PyTorch installation guide](https://pytorch.org/get-started/locally/).

### 3. Install the remaining dependencies

```bash
pip install numpy scipy scikit-learn tqdm tensorboard einops \
    dynamic-network-architectures torchvision opencv-python
```

The main dependencies are:

- PyTorch and TorchVision
- NumPy and SciPy
- scikit-learn
- TensorBoard
- einops
- dynamic-network-architectures
- OpenCV

## 📊 Data Preparation

Clinical imaging data are not distributed with this repository. By default,
`utils/config.py` expects the following structure under `./data/`:

```text
data/
├── imgs/
│   └── <scan_id>.npy
├── labels/
│   └── <scan_id>.npy
├── splits.json
└── pulmonary_data_dict.json
```

`splits.json` should contain a list of cross-validation folds. Each fold should
provide `train`, `val`, and `test` sample-ID lists:

```json
[
  {
    "train": ["anonymous_case_001"],
    "val": ["anonymous_case_002"],
    "test": ["anonymous_case_003"]
  }
]
```

`pulmonary_data_dict.json` maps each anonymous sample ID to its CT volume and
one-based pathological label:

```json
{
  "anonymous_case_001": {
    "current_ct_name": "anonymous_scan_001",
    "nodule_class": 1
  }
}
```

The data loader converts labels from `1–5` to zero-based labels `0–4`.

> Do not commit clinical data or identifiable metadata. Use only appropriately
> de-identified data for which ethical approval and data-use permission have been
> obtained.

## 📈 Training

Configure paths and hyperparameters in `utils/config.py`, then train one fold:

```bash
python train.py --fold 0
```

A representative command is:

```bash
python train.py \
    --fold 0 \
    --scheduler CosineAnnealing \
    --warm_up 30 \
    --updaterate 1 \
    --ce_weight 1 \
    --emd_weight 10 \
    --weight_sep 0.4 \
    --weight_cls 20 \
    --tau 0.1
```

Available learning-rate schedulers are:

- `None`
- `CosineAnnealing`
- `ReduceLROnPlateau`
- `Poly`

By default, model checkpoints are saved under `result/SimGro/`, and TensorBoard
logs are written under `tensorboard/`.

Launch TensorBoard with:

```bash
tensorboard --logdir tensorboard
```

## 📋 Evaluation

Evaluate a trained checkpoint on the corresponding test fold:

```bash
python test.py \
    --fold 0 \
    --checkpoint result/SimGro/<checkpoint>.pth
```

The evaluation script prints:

- macro accuracy;
- per-class accuracy;
- macro F1 score;
- accuracy, precision, recall, F1, and specificity;
- mean absolute error.

The streamlined evaluation pipeline does not generate prediction tables, plots,
or additional analysis files.

Only load checkpoints obtained from trusted sources. PyTorch checkpoint files may
contain serialized content that is unsafe when supplied by an untrusted party.

## 🔁 Reproducibility

- Keep the fold index consistent between training and evaluation.
- Keep the pathological class order consistent during data preparation.
- Record Python, PyTorch, CUDA, and GPU versions for each experiment.
- Use the same preprocessing and spatial crop size for all cohorts.
- The default input is a single-channel `48 × 48 × 48` CT volume.
- Checkpoints contain the model state and the prototype-to-case mapping.

## Citation

If you use this repository, please cite:

> **Evidence-linked case-prototype learning for preoperative pathological
> stratification of lung adenocarcinoma spectrum nodules.**

The final paper link and BibTeX metadata will be added upon publication.

## AI Contributor

This repository was developed with assistance from
[`@codex`](https://github.com/codex), OpenAI's coding agent.

<p>
  <a href="https://github.com/codex">
    <img src="https://github.com/codex.png" width="80" alt="Codex avatar" />
  </a>
</p>

## Acknowledgements

This codebase builds on the following open-source projects:

- [PyTorch](https://github.com/pytorch/pytorch)
- [dynamic-network-architectures](https://github.com/MIC-DKFZ/dynamic-network-architectures)
- [einops](https://github.com/arogozhnikov/einops)

We thank the developers and contributors of these projects.

## License

No software license has been selected for this repository yet. Add an appropriate
license before permitting reuse or redistribution, and verify the licenses and
provenance of all incorporated code.
