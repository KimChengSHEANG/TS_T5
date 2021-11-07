# TS with T5



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

4 tokens model

```python
# evaluate TurkCorpus and ASSET datasets
python scripts/evaluate_4tokens_pretrained_model.py

# generate using other input file
 # edit the file to change input and output files. Default TurkCorpus complex is used as an example.
python scripts/generate_with_4tokens_pretrained_model.py
```



5 tokens model

```python
# evaluate TurkCorpus and ASSET datasets
python scripts/evaluate_5tokens_pretrained_model.py

# generate using other input file
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

