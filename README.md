# Implementation of the article ["Unpaired Image Super-Resolution using Pseudo-Supervision"](https://openaccess.thecvf.com/content_CVPR_2020/papers/Maeda_Unpaired_Image_Super-Resolution_Using_Pseudo-Supervision_CVPR_2020_paper.pdf)

QUICK START:

For start training:
1. create your own config 
`implementation/configs/train_config.py` 
(like one of ours). 
2. run `python -m pip install -r implementation/requirements.txt`
or `python -m conda install -r implementation/requirements.txt`
3. run `cd implementation` and `export PYTHONPATH="."` 
4. run `export CUDA_VISIBLE_DEVICES=0` for selecting number of uor GPU
5. run `python3 configs/train_config.py`
