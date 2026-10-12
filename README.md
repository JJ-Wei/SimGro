# SimGro

SimGro is a prototype-based 3D deep-learning framework for five-class pulmonary
nodule subtype classification. The model learns normalized image features and a
small bank of class-specific templates. During training, Sinkhorn assignments
associate samples with templates, and representative real samples periodically
re-anchor the template bank.

This repository contains the model, training pipeline, evaluation code, and data
loading utilities. Clinical imaging data and trained checkpoints are not included.

## Repository structure

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
└── test.py
```

## Requirements

The code requires Python and the following main packages:

```text
torch
torchvision
numpy
scipy
scikit-learn
tqdm
tensorboard
einops
dynamic-network-architectures
opencv-python
```

Install them in a dedicated environment. Select the PyTorch build appropriate for
your CUDA version from the official PyTorch installation instructions, then install
the remaining dependencies, for example:

```bash
pip install numpy scipy scikit-learn tqdm tensorboard einops \
    dynamic-network-architectures opencv-python
```

## Data layout

By default, `utils/config.py` expects data under `./data/`:

```text
data/
├── imgs/
│   └── <scan_id>.npy
├── labels/
│   └── <scan_id>.npy
├── splits.json
└── pulmonary_data_dict.json
```

`splits.json` should contain a list of cross-validation folds. Each fold is a
dictionary with `train`, `val`, and `test` sample-ID lists.

`pulmonary_data_dict.json` should map each sample ID to its metadata. The current
loader uses at least these fields:

```json
{
  "anonymous_sample_id": {
    "current_ct_name": "anonymous_scan_id",
    "nodule_class": 1
  }
}
```

Class labels in the metadata are expected to be one-based (`1` to `5`) and are
converted internally to zero-based labels.

> Do not commit clinical data or identifiable sample metadata. Use only data for
> which you have appropriate ethical approval and distribution rights.

## Configuration

Edit `utils/config.py` to set the data path, batch size, learning rate, number of
epochs, image size, device, and model dimensions. The default model expects
single-channel volumes cropped to `48 × 48 × 48` voxels.

The training script sets deterministic random seeds. `CUDA_VISIBLE_DEVICES` is
currently configured near the top of `train.py`; adjust it for your system if
needed.

## Training

Run a single cross-validation fold with:

```bash
python train.py --fold 0
```

Useful options include:

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

Supported scheduler names are `None`, `CosineAnnealing`, `ReduceLROnPlateau`, and
`Poly`.

By default, checkpoints are written below `result/SimGro/`, and TensorBoard logs
are written below `tensorboard/`.

Start TensorBoard with:

```bash
tensorboard --logdir tensorboard
```

## Evaluation

Evaluate a checkpoint with:

```bash
python test.py --fold 0 --checkpoint result/SimGro/<checkpoint>.pth
```

The test script prints the final macro accuracy, per-class accuracy, macro F1,
precision, recall, F1, specificity, and mean absolute error. It does not generate
plots or prediction files.

Only load checkpoints from trusted sources. PyTorch checkpoint files may contain
serialized data that is unsafe when obtained from an untrusted party.

## Reproducibility notes

- Keep the fold index consistent between training and evaluation.
- Record the package versions, CUDA version, and GPU model used for each run.
- Do not change the class order between metadata preparation, training, and testing.
- Checkpoints contain both the model state and the current template-to-sample map.

## License

No software license has been selected for this repository yet. Add an appropriate
license before permitting reuse or redistribution, and verify the licenses and
provenance of all incorporated code.
