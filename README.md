# TS with T5



## Requirements

**Step1**. Install pytorch follow this link: https://pytorch.org/get-started/locally/

```bash
Examples:
# gpu version
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html

# cpu version
pip install torch==1.8.0+cpu torchvision==0.9.0+cpu torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
######
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

