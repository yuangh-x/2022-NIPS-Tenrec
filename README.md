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
python main.py --task_name=ctr --seed=100 --model_name=afm --dataset_path='data/ctr_data_1M.csv' --train_batch_size=4096 --test_batch_size=4096 --epochs=20 --lr=0.00005
```
DeepFM
```
python main.py --task_name=ctr --seed=100 --model_name=deepfm --dataset_path='data/ctr_data_1M.csv' --train_batch_size=4096 --test_batch_size=4096 --epochs=20 --lr=0.00005
```
xDeepFM
```
python main.py --task_name=ctr --seed=100 --model_name=xdeepfm --dataset_path='data/ctr_data_1M.csv' --train_batch_size=4096 --test_batch_size=4096 --epochs=20 --lr=0.00005
```
NFM
```
python main.py --task_name=ctr --seed=100 --model_name=nfm --dataset_path='data/ctr_data_1M.csv' --train_batch_size=4096 --test_batch_size=4096 --epochs=20 --lr=0.00005
```
Wide & Deep
```
python main.py --task_name=ctr --seed=100 --model_name=wdl --dataset_path='data/ctr_data_1M.csv' --train_batch_size=4096 --test_batch_size=4096 --epochs=20 --lr=0.00005
```

**Session-based Recommendation**

NextItNet
```
python main.py --task_name=sequence --seed=100 --model_name=nextitnet --dataset_path='data/sbr_data_1M.csv' --train_batch_size=32 --val_batch_size=32 --test_batch_size=32 --epochs=20 --lr=0.0001 --hidden_size=128 --block_num=8 --embedding_size=128 --dilation=[1, 4] --kernel_size=3 --is_pretrain==1
```
BERT4Rec
```
python main.py --task_name=sequence --seed=100 --model_name=bert4rec --dataset_path='data/sbr_data_1M.csv' --train_batch_size=32 --val_batch_size=32 --test_batch_size=32 --epochs=20 --lr=0.0001 --hidden_size=128 --block_num=16 --embedding_size=128 --num_heads=4 --bert_mask_prob=0.3 --is_pretrain==1
```
SASRec
```
python main.py --task_name=sequence --seed=100 --model_name=sasrec --dataset_path='data/sbr_data_1M.csv' --train_batch_size=32 --val_batch_size=32 --test_batch_size=32 --epochs=20 --lr=0.0001 --hidden_size=64 --block_num=8 --embedding_size=64 --num_heads=4 --is_pretrain==1
```
GRU4Rec
```
python main.py --task_name=sequence --seed=100 --model_name=gru4rec --dataset_path='data/sbr_data_1M.csv' --train_batch_size=32 --val_batch_size=32 --test_batch_size=32 --epochs=20 --lr=0.0005 --hidden_size=64 --block_num=8 --embedding_size=64 --is_pretrain==1
```

**Multi-Task Learing**

Only click
```
python main.py --task_name=mtl --seed=100 --model_name=mmoe --dataset_path='data/ctr_data_1M.csv' --train_batch_size=4096 --val_batch_size=4096 --test_batch_size=4096 --epochs=20 --lr=0.0001 --embedding_size=32 ----mtl_task_num=1
```

Only like
```
python main.py --task_name=mtl --seed=100 --model_name=mmoe --dataset_path='data/ctr_data_1M.csv' --train_batch_size=4096 --val_batch_size=4096 --test_batch_size=4096 --epochs=20 --lr=0.0001 --embedding_size=32 ----mtl_task_num=0
```

ESMM
```
python main.py --task_name=mtl --seed=100 --model_name=esmm --dataset_path='data/ctr_data_1M.csv' --train_batch_size=4096 --val_batch_size=4096 --test_batch_size=4096 --epochs=20 --lr=0.0001 --embedding_size=32 ----mtl_task_num=2
```

MMOE
```
python main.py --task_name=mtl --seed=100 --model_name=esmm --dataset_path='data/ctr_data_1M.csv' --train_batch_size=4096 --val_batch_size=4096 --test_batch_size=4096 --epochs=20 --lr=0.0001 --embedding_size=32 ----mtl_task_num=2
```

**Transfer Learning**

Plese run the command of Session-based Recommendation Task

NextItNet with Pretrain
```
python main.py --task_name=transfer_learning --seed=100 --model_name=peterrec --dataset_path='data/sbr_data_1M.csv' --pretrain_path='checkpoint/sequence_nextitnet_seed100_is_pretrain_1_best_model_lr0.0001_wd0.0_block8_hd128_emb128.pth' --train_batch_size=32 --val_batch_size=32 --test_batch_size=32 --epochs=20 --lr=0.0005 --hidden_size=128 --block_num=16 --embedding_size=128 --dilation=[1, 4] --kernel_size=3 --is_pretrain==0
```

SASRec with Pretrain
```
python main.py --task_name=transfer_learning --seed=100 --model_name=sas4transfer --dataset_path='data/sbr_data_1M.csv' --pretrain_path='checkpoint/sequence_sasrec_seed100_is_pretrain_1_best_model_lr0.0001_wd0.0_block8_hd64_emb64.pth' --train_batch_size=32 --val_batch_size=32 --test_batch_size=32 --epochs=20 --lr=0.0001 --hidden_size=64 --block_num=8 --embedding_size=64 --num_heads=4 --is_pretrain==0
```

**User Profile Prediction**

DNN
```
python main.py --task_name=user_profile_represent --model_name=dnn4profile --dataset_path=data/sbr_data_1M.csv --train_batch_size=32 --val_batch_size=32 --test_batch_size=32 --epochs=20 --lr=0.0001 --hidden_size=128 --block_num=16 --embedding_size=128 --is_pretrain==2
```

BERT4Rec without Pretrain
```
python main.py --task_name=user_profile_represent --model_name=bert4profile --dataset_path=data/sbr_data_1M.csv --train_batch_size=32 --val_batch_size=32 --test_batch_size=32 --epochs=20 --lr=0.0001 --hidden_size=128 --block_num=16 --embedding_size=128 --num_heads=4 --is_pretrain==2
```

Peterrec without Pretrain
```
python main.py --task_name=user_profile_represent --model_name=peter4profile --dataset_path=data/sbr_data_1M.csv --train_batch_size=32 --val_batch_size=32 --test_batch_size=32 --epochs=20 --lr=0.00005 --hidden_size=128 --block_num=8 --embedding_size=128 --dilation=[1, 4] --kernel_size=3 --is_pretrain==2
```

BERT4Rec with Pretrain
```
python main.py --task_name=user_profile_represent --model_name=bert4profile --dataset_path=data/sbr_data_1M.csv --pretrain_path='checkpoint/sequence_bert4rec_seed100_is_pretrain_1_best_model_lr0.0001_wd0.0_block16_hd128_emb128.pth' --train_batch_size=32 --val_batch_size=32 --test_batch_size=32 --epochs=20 --lr=0.0001 --hidden_size=128 --block_num=16 --embedding_size=128 --num_heads=4 --is_pretrain==0
```

Peterrec with Pretrain
```
python main.py --task_name=user_profile_represent --model_name=peter4profile --dataset_path=data/sbr_data_1M.csv --pretrain_path='checkpoint/sequence_nextitnet_seed100_is_pretrain_1_best_model_lr0.0001_wd0.0_block8_hd128_emb128.pth' --train_batch_size=32 --val_batch_size=32 --test_batch_size=32 --epochs=20 --lr=0.00005 --hidden_size=128 --block_num=8 --embedding_size=128 --dilation=[1, 4] --kernel_size=3 --is_pretrain==0
```

**Cold-start Recommendation**

BERT4Rec without Pretrain
```
python main.py --task_name=cold_start --model_name=bert4coldstart --dataset_path=data/cold_start.csv --train_batch_size=32 --val_batch_size=32 --test_batch_size=32 --epochs=20 --lr=0.001 --hidden_size=128 --block_num=16 --embedding_size=128 --num_heads=4 --is_pretrain==2
```

Peterrec without Pretrain
```
python main.py --task_name=cold_start --model_name=peter4coldstart --dataset_path=data/cold_start.csv --train_batch_size=32 --val_batch_size=32 --test_batch_size=32 --epochs=20 --lr=0.005 --hidden_size=128 --block_num=8 --embedding_size=128 --dilation=[1, 4] --kernel_size=3 --is_pretrain==2
```

BERT4Rec with Pretrain
```
python main.py --task_name=cold_start --model_name=bert4coldstart --dataset_path=data/cold_start.csv --pretrain_path='checkpoint/sequence_bert4rec_seed100_is_pretrain_1_best_model_lr0.0001_wd0.0_block16_hd128_emb128.pth' --train_batch_size=32 --val_batch_size=32 --test_batch_size=32 --epochs=20 --lr=0.001 --hidden_size=128 --block_num=16 --embedding_size=128 --num_heads=4 --is_pretrain==0
```

Peterrec with Pretrain
```
python main.py --task_name=cold_start --model_name=peter4coldstart --dataset_path=data/cold_start.csv --pretrain_path='checkpoint/sequence_nextitnet_seed100_is_pretrain_1_best_model_lr0.0001_wd0.0_block8_hd128_emb128.pth' --train_batch_size=32 --val_batch_size=32 --test_batch_size=32 --epochs=20 --lr=0.005 --hidden_size=128 --block_num=8 --embedding_size=128 --dilation=[1, 4] --kernel_size=3 --is_pretrain==0
```

**Lifelong User Representation Learning**

Conure
```

```

**License:**

This dataset is licensed under a CC BY-NC 4.0 International License(https://creativecommons.org/licenses/by-nc/4.0/).
