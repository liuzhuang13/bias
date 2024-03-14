# A Decade's Battle on Dataset Bias: Are We There Yet?



[A Decade's Battle on Dataset Bias: Are We There Yet?]() \
[Zhuang Liu](https://liuzhuang13.github.io/) and [Kaiming He](https://kaiminghe.github.io/)\
Meta AI Research, FAIR\
[[`arXiv`](https://arxiv.org/abs/2403.08632)] [[`code`](https://github.com/liuzhuang13/bias)]

--- 

<p align="center">
<img src="https://github.com/facebookresearch/dropout/assets/91447088/57222fea-8da0-48f1-b38e-7f690876361e" width=100% height=100% 
class="center">
</p>

These images are sampled from three modern datasets: YFCC, CC, and DataComp. *Can you specify which dataset each image is from?* While these datasets appear to be less biased, we discover that neural networks can easily accomplish this “dataset classification” task with surprisingly high accuracy on the held-out validation set.

<details>
  <summary>Answer (click)</summary>
  YFCC: 1, 4, 7, 10, 13; CC: 2, 5, 8, 11, 14; DataComp: 3, 6, 9, 12, 15.
</details>



## Code

We use the code from [ConvNeXt](https://github.com/facebookresearch/ConvNeXt). Please follow the instructions there for setup.

## Dataset Preparation

Download images from each dataset and organize them as follows:
```
/path/to/datasets_root/
  train/
    dataset1/
      ...
    dataset2/
      ...
    dataset3/
      ...
  val/
    dataset1/
      ...
    dataset2/
      ...
    dataset3/
      ...
```

## Training

We give example commands for single-machine and multi-node training below. 

### Multi-node
```
python run_with_submitit.py --nodes 4 --ngpus 8 \
--model convnext_tiny --opt_betas 0.9 0.95 \
--batch_size 128 --lr 1e-3 --update_freq 1 \
--weight_decay 0.3 --reprob 0 \
--data_set image_folder --nb_classes 3 \ 
--data_path /path/to/datasets_root/train \
--eval_data_path /path/to/datasets_root/val \
--job_dir /path/to/save_results
```

### Single-machine
```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model convnext_tiny --opt_betas 0.9 0.95 \
--batch_size 128 --lr 1e-3 --update_freq 1 \
--weight_decay 0.3 --reprob 0 \
--data_set image_folder --nb_classes 3 \ 
--data_path /path/to/datasets_root/train \
--eval_data_path /path/to/datasets_root/val \
--output_dir /path/to/save_results
```



## LICENSE
This project is released under the MIT license. Please see the LICENSE file for more information.

## Citation
```
@article{liu2024decade,
  title   = {A Decade's Battle on Dataset Bias: Are We There Yet?},
  author  = {Zhuang Liu and Kaiming He},
  year    = {2024},
  journal = {arXiv preprint arXiv:2403.08632},
}
```