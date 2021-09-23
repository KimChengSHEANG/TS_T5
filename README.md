# TS with T5

## Requirements

**Step1**. Install pytorch follow this link: https://pytorch.org/get-started/locally/

```bash
# Recent Pytorch installation script
# GPU version
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html

# It is too slow to train or evaluate on CPU, so please use GPU instead.
```

**Step2.** Install all requirements

```bash
pip install -r requirements.txt
```

## Train

```bash
python scripts/train.py
```

## Evaluate

Evaluate the lastest training.

```bash
python scripts/evaluate.py
```


# Pretrained model will be uploaded soon
