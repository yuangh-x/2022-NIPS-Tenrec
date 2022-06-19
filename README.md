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

Please run the command as below to test the performance of each task

**CTR:**

AFM
```
python main.py --task_name=ctr --model_name=afm --dataset_path=data/ctr_data_1M.csv --train_batch_size=4096 --test_batch_size=4096 --epochs=20 --lr=0.00005
```
DeepFM
```
python main.py --task_name=ctr --model_name=deepfm --dataset_path=data/ctr_data_1M.csv --train_batch_size=4096 --test_batch_size=4096 --epochs=20 --lr=0.00005
```
xDeepFM
```
python main.py --task_name=ctr --model_name=xdeepfm --dataset_path=data/ctr_data_1M.csv --train_batch_size=4096 --test_batch_size=4096 --epochs=20 --lr=0.00005
```
NFM
```
python main.py --task_name=ctr --model_name=nfm --dataset_path=data/ctr_data_1M.csv --train_batch_size=4096 --test_batch_size=4096 --epochs=20 --lr=0.00005
```
Wide & Deep
```
python main.py --task_name=ctr --model_name=wdl --dataset_path=data/ctr_data_1M.csv --train_batch_size=4096 --test_batch_size=4096 --epochs=20 --lr=0.00005
```

**Session-based Recommendation**

NextItNet
```
python main.py --task_name=sequence --model_name=nextitnet --dataset_path=data/sbr_data_1M.csv --train_batch_size=32 --val_batch_size=32 --test_batch_size=32 --epochs=20 --lr=0.0001 --hidden_size=128 --block_num=16 --embedding_size=128 --is_pretrain==1
```
BERT4Rec
```
python main.py --task_name=sequence --model_name=bert4rec --dataset_path=data/sbr_data_1M.csv --train_batch_size=32 --val_batch_size=32 --test_batch_size=32 --epochs=20 --lr=0.0001 --hidden_size=128 --block_num=16 --embedding_size=128 --is_pretrain==1
```
SASRec
```
python main.py --task_name=sequence --model_name=sasrec --dataset_path=data/sbr_data_1M.csv --train_batch_size=32 --val_batch_size=32 --test_batch_size=32 --epochs=20 --lr=0.0001 --hidden_size=64 --block_num=8 --embedding_size=64 --is_pretrain==1
```
GRU4Rec
```
python main.py --task_name=sequence --model_name=gru4rec --dataset_path=data/sbr_data_1M.csv --train_batch_size=32 --val_batch_size=32 --test_batch_size=32 --epochs=20 --lr=0.0005 --hidden_size=64 --block_num=8 --embedding_size=64 --is_pretrain==1
```

**Multi-Task Learing**

ESMM
```
python main.py --task_name=mtl --model_name=esmm --dataset_path=data/ctr_data_1M.csv --train_batch_size=4096 --val_batch_size=4096 --test_batch_size=4096 --epochs=20 --lr=0.0001 --embedding_size=32
```

**License:**

This dataset is licensed under a CC BY-NC 4.0 International License(https://creativecommons.org/licenses/by-nc/4.0/).
