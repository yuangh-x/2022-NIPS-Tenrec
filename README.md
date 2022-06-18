# Tenrec
Tenrec is a real-world, Large-scale Multipurpose Benchmark Dataset where data was collected from user feedback on feeds recommendation platform of Tencent, it is for research purpose only.

**Dataset in Tenrec:**

QK-video.csv: Users interactions with videos in QK.

QB-video.csv: Users interactions with videos in QB.

QK-article.csv: Users interactions with articles in QK.

QB-artilce.csv: Usesr interactions with articles in QB.

**Download the dataset:**

The Dataset can be downloaded from:

**Benchmark**
**CTR:**
AFM
```
python main.py --task_name=ctr --model_name=afm --dataset_path=data/ctr_1M.csv --train_batch_size=4096 --test_batch_size=4096 --epochs=20 --lr=0.00005
```
DeepFM
```
python main.py --task_name=ctr --model_name=deepfm --dataset_path=data/ctr_1M.csv --train_batch_size=4096 --test_batch_size=4096 --epochs=20 --lr=0.00005
```


**License:**

This dataset is licensed under a CC BY-NC 4.0 International License(https://creativecommons.org/licenses/by-nc/4.0/).
