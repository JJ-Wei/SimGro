# SimGro: Evidence-linked Case-Prototype Learning for Preoperative Pathological Stratification of Lung Adenocarcinoma Spectrum Nodules

<div align="center">

**Paper link and complete author information will be added upon publication.**

</div>

## 🚀 Overview

**Clinical problem.** Preoperative pathological stratification of lung
adenocarcinoma spectrum nodules is challenging because adjacent pathological
subtypes may share highly similar radiological appearances while exhibiting
different levels of invasiveness.

**Our approach.** SimGro is an evidence-linked case-prototype learning framework
for five-class pulmonary nodule classification. It learns a compact bank of
class-specific prototypes in a normalized feature space and links each prototype
to a representative real case. This design provides both a classification result
and case-level reference evidence for interpreting the model decision.

**Learning strategy.** During training, SimGro:

- extracts normalized 3D image representations from cropped CT volumes;
- assigns samples to class-specific prototypes with balanced Sinkhorn assignment;
- optimizes cross-entropy, ordinal EMD, prototype separation, and prototype
  compactness objectives;
- periodically re-anchors each prototype to a representative real training case;
- stores the prototype-to-case mapping in the model checkpoint.

The current implementation supports five-fold data splits and five ordered
pathological classes.

## 🗂️ Repository Structure

```text
SimGro/
├── models/
│   ├── models.py
│   └── SimGro/
│       ├── img_network.py
│       ├── multimodal_network.py
│       └── pulmotarynetwork.py
├── utils/
│   ├── config.py
│   ├── dataset.py
│   ├── evaluation.py
│   └── loss.py
├── train.py
├── test.py
└── README.md
```

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
