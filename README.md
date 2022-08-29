# TS with T5



# Dependencies



* Tested with Python 3.7.5



## Install requirements

**Step1**. Install pytorch follow this link: https://pytorch.org/get-started/locally/

```bash
# Recent Pytorch installation script
# GPU version
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html

# Training on CPU is not recommended. It's too slow!
```

**Step2.** Install all requirements

```bash
pip install -r requirements.txt
```



# Usage

## Train

```bash
python scripts/train.py
```

## Evaluate

Evaluate the last trained model

```bash
python scripts/evaluate.py

```

# Generate

Edit the scripts/generate.py to pass input and output filepath.

```python
python scripts/generate.py
```



# Evaluate pretrained models



## 4 tokens model

```python
# evaluate TurkCorpus and ASSET datasets
python scripts/evaluate_4tokens_pretrained_model.py
```

### Default tokens and expected results

```
Used tokens: C: 0.95         L: 0.75         WR: 0.75        DTD: 0.75     

Expected results: 
TurkCorpus: 	SARI: 43.50      BLEU: 67.26     FKGL: 6.17
ASSET: 		SARI: 45.05      BLEU: 72.24     FKGL: 6.33
```

#### Generate using other input file

```python
 # edit the file to change input and output files. Default TurkCorpus complex is used as an example.
python scripts/generate_with_4tokens_pretrained_model.py
```



## 5 tokens model

```python
# evaluate TurkCorpus and ASSET datasets
python scripts/evaluate_5tokens_pretrained_model.py
```

### Default tokens and expected results

```
Used tokens: W: 1.05 	C: 0.95 	L: 0.75 	WR: 0.75 	DTD: 0.75

Expected results: 
TurkCorpus: 	SARI: 43.34 	 BLEU: 65.54 	 FKGL: 5.55 
ASSET: 		SARI: 45.26 	 BLEU: 72.19 	 FKGL: 5.92 
```

#### Generate using other input file

```python
 # edit the file to change input and output files. Default TurkCorpus complex is used as an example.
python scripts/generate_with_5tokens_pretrained_model.py
```





# Citation

If you make use of the code in this repository, please cite the following papers:

```
@inproceedings{sheang-saggion-2021-controllable,
    title = "Controllable Sentence Simplification with a Unified Text-to-Text Transfer Transformer",
    author = "Sheang, Kim Cheng  and Saggion, Horacio",
    booktitle = "Proceedings of the 14th International Conference on Natural Language Generation",
    month = aug,
    year = "2021",
    address = "Aberdeen, Scotland, UK",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.inlg-1.38",
    pages = "341--352"
}
```





# Credits

The preprocessing code was adopted from https://github.com/facebookresearch/access
