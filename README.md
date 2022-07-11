# Tenrec: A Large-scale Multipurpose Benchmark Dataset for Recommender Systems
Tenrec is a large-scale multipurpose benchmark dataset for recommender systems where data was collected from two feeds (articles and videos) recommendation platforms of Tencent.

### Dataset in Tenrec:

QK-video.csv: User video action in QK.

QB-video.csv: User video action in QB.

QK-article.csv: User article action in QK.

QB-artilce.csv: User article action in QB.

//**Download the dataset:**

//The Dataset can be downloaded from:
//https://drive.google.com/file/d/1R1JhdT9CHzT3qBJODz09pVpHMzShcQ7a/view?usp=sharing

### Benchmark

We apply Tenrec on 10 recommendation tasks. Please run the commands as below to test the performance of each task.

#### CTR:

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

#### Session-based Recommendation

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

#### Multi-Task Learing

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

#### Transfer Learning

Plese run the command of Session-based Recommendation Task firstly.

NextItNet with Pretrain
```
python main.py --task_name=transfer_learning --seed=100 --model_name=peterrec --dataset_path='data/sbr_data_1M.csv' --pretrain_path='checkpoint/sequence_nextitnet_seed100_is_pretrain_1_best_model_lr0.0001_wd0.0_block8_hd128_emb128.pth' --train_batch_size=32 --val_batch_size=32 --test_batch_size=32 --epochs=20 --lr=0.0005 --hidden_size=128 --block_num=16 --embedding_size=128 --dilation=[1, 4] --kernel_size=3 --is_pretrain==0
```

SASRec with Pretrain
```
python main.py --task_name=transfer_learning --seed=100 --model_name=sas4transfer --dataset_path='data/sbr_data_1M.csv' --pretrain_path='checkpoint/sequence_sasrec_seed100_is_pretrain_1_best_model_lr0.0001_wd0.0_block8_hd64_emb64.pth' --train_batch_size=32 --val_batch_size=32 --test_batch_size=32 --epochs=20 --lr=0.0001 --hidden_size=64 --block_num=8 --embedding_size=64 --num_heads=4 --is_pretrain==0
```

#### User Profile Prediction

Plese run the command of Session-based Recommendation Task firstly.

DNN
```
python main.py --task_name=user_profile_represent --seed=100 --model_name=dnn4profile --dataset_path=data/sbr_data_1M.csv --train_batch_size=32 --val_batch_size=32 --test_batch_size=32 --epochs=20 --lr=0.0001 --hidden_size=128 --block_num=16 --embedding_size=128 --is_pretrain==2
```

BERT4Rec without Pretrain
```
python main.py --task_name=user_profile_represent --seed=100 --model_name=bert4profile --dataset_path=data/sbr_data_1M.csv --train_batch_size=32 --val_batch_size=32 --test_batch_size=32 --epochs=20 --lr=0.0001 --hidden_size=128 --block_num=16 --embedding_size=128 --num_heads=4 --is_pretrain==2
```

Peterrec without Pretrain
```
python main.py --task_name=user_profile_represent --seed=100 --model_name=peter4profile --dataset_path=data/sbr_data_1M.csv --train_batch_size=32 --val_batch_size=32 --test_batch_size=32 --epochs=20 --lr=0.00005 --hidden_size=128 --block_num=8 --embedding_size=128 --dilation=[1, 4] --kernel_size=3 --is_pretrain==2
```

BERT4Rec with Pretrain
```
python main.py --task_name=user_profile_represent --seed=100 --model_name=bert4profile --dataset_path=data/sbr_data_1M.csv --pretrain_path='checkpoint/sequence_bert4rec_seed100_is_pretrain_1_best_model_lr0.0001_wd0.0_block16_hd128_emb128.pth' --train_batch_size=32 --val_batch_size=32 --test_batch_size=32 --epochs=20 --lr=0.0001 --hidden_size=128 --block_num=16 --embedding_size=128 --num_heads=4 --is_pretrain==0
```

Peterrec with Pretrain
```
python main.py --task_name=user_profile_represent --model_name=peter4profile --dataset_path=data/sbr_data_1M.csv --pretrain_path='checkpoint/sequence_nextitnet_seed100_is_pretrain_1_best_model_lr0.0001_wd0.0_block8_hd128_emb128.pth' --train_batch_size=32 --val_batch_size=32 --test_batch_size=32 --epochs=20 --lr=0.00005 --hidden_size=128 --block_num=8 --embedding_size=128 --dilation=[1, 4] --kernel_size=3 --is_pretrain==0
```

#### Cold-start Recommendation

BERT4Rec without Pretrain
```
python main.py --task_name=cold_start --seed=10 --model_name=bert4coldstart --train_batch_size=32 --val_batch_size=32 --test_batch_size=32 --epochs=20 --lr=0.001 --hidden_size=128 --block_num=16 --embedding_size=128 --num_heads=4 --is_pretrain==2
```

Peterrec without Pretrain
```
python main.py --task_name=cold_start --seed=10 --model_name=peter4coldstart --train_batch_size=32 --val_batch_size=32 --test_batch_size=32 --epochs=20 --lr=0.005 --hidden_size=128 --block_num=8 --embedding_size=128 --dilation=[1, 4] --kernel_size=3 --is_pretrain==2
```

Please run the command of Session-based Recommendation Task firstly.

BERT4Rec with Pretrain
```
python main.py --task_name=cold_start --seed=10 --model_name=bert4coldstart --pretrain_path='checkpoint/sequence_bert4rec_seed100_is_pretrain_1_best_model_lr0.0001_wd0.0_block16_hd128_emb128.pth' --train_batch_size=32 --val_batch_size=32 --test_batch_size=32 --epochs=20 --lr=0.001 --hidden_size=128 --block_num=16 --embedding_size=128 --num_heads=4 --is_pretrain==0
```

Peterrec with Pretrain
```
python main.py --task_name=cold_start --seed=10 --model_name=peter4coldstart --pretrain_path='checkpoint/sequence_nextitnet_seed100_is_pretrain_1_best_model_lr0.0001_wd0.0_block8_hd128_emb128.pth' --train_batch_size=32 --val_batch_size=32 --test_batch_size=32 --epochs=20 --lr=0.005 --hidden_size=128 --block_num=8 --embedding_size=128 --dilation=[1, 4] --kernel_size=3 --is_pretrain==0
```

#### Lifelong User Representation Learning

SAS4Rec
```
python main.py --task_name=life_long --seed=100  --task_num=4 --model_name=sas4life --train_batch_size=32 --val_batch_size=32 --test_batch_size=32 --epochs=20 --re_epochs=20 --lr=0.0001 --hidden_size=64 --block_num=8 --embedding_size=64 --num_heads=4
```

Conure
```
python main.py --task_name=life_long --seed=100 --task_num=4 --model_name=conure --train_batch_size=32 --val_batch_size=32 --test_batch_size=32 --epochs=20 --re_epochs=20 --lr=0.0001 --hidden_size=128 --block_num=8 --embedding_size=128 --dilation=[1, 4] --kernel_size=3
```

#### Model Compression

SASRec
```
python main.py --task_name=model_compr --seed=100 --model_name=sas4cp --dataset_path='data/sbr_data_1M.csv' --train_batch_size=32 --val_batch_size=32 --test_batch_size=32 --epochs=20 --lr=0.0001 --hidden_size=64 --block_num=8 --embedding_size=64 --num_heads=4 --is_pretrain==1
```
Cprec
```
python main.py --task_name=model_compr --seed=100 --model_name=cprec --dataset_path='data/sbr_data_1M.csv' --train_batch_size=32 --val_batch_size=32 --test_batch_size=32 --epochs=20 --lr=0.0001 --hidden_size=128 --block_num=8 --embedding_size=128 --dilation=[1, 4] --kernel_size=3 --is_pretrain==1
```

#### Model Training Speedup

SASRec-shallow train
```
python main.py --task_name=model_acc --seed=100 --model_name=sas4acc --dataset_path='data/sbr_data_1M.csv' --train_batch_size=32 --val_batch_size=32 --test_batch_size=32 --epochs=20 --lr=0.0001 --hidden_size=64 --block_num=4 --embedding_size=64 --num_heads=4 --is_pretrain==1
```
SASRec-deep train
```
python main.py --task_name=model_acc --seed=100 --model_name=sas4acc --dataset_path='data/sbr_data_1M.csv' --pretrain_path='checkpoint/model_acc_sas4rec_seed100_is_pretrain_1_best_model_lr0.0001_wd0.0_block4_hd64_emb64.pth' --train_batch_size=32 --val_batch_size=32 --test_batch_size=32 --epochs=20 --lr=0.0001 --hidden_size=64 --block_num=4 --embedding_size=64 --num_heads=4 --is_pretrain==0 --add_num_times=2 
```
Stackrec-shallow train
```
python main.py --task_name=model_acc --seed=100 --model_name=stackrec --dataset_path='data/sbr_data_1M.csv' --pretrain_path='checkpoint/model_acc_stackrec_seed100_is_pretrain_1_best_model_lr0.0001_wd0.0_block4_hd128_emb128.pth' --train_batch_size=32 --val_batch_size=32 --test_batch_size=32 --epochs=20 --lr=0.0001 --hidden_size=128 --block_num=4 --embedding_size=128 --dilation=[1, 4] --kernel_size=3 --is_pretrain==1
```
Stackrec-deep train
```
python main.py --task_name=model_acc --seed=100 --model_name=stackrec --dataset_path='data/sbr_data_1M.csv' --train_batch_size=32 --val_batch_size=32 --test_batch_size=32 --epochs=20 --lr=0.0001 --hidden_size=128 --block_num=4 --embedding_size=128 --dilation=[1, 4] --kernel_size=3 --is_pretrain==0 --add_num_times=2 
```

#### Model Inference Speedup

SASRec
```
python main.py --task_name=inference_acc --seed=5 --model_name=sas4infacc --dataset_path='data/QB-video.csv' --train_batch_size=32 --val_batch_size=32 --test_batch_size=1 --epochs=20 --lr=0.0001 --hidden_size=64 --block_num=8 --embedding_size=64 --num_heads=4 --is_pretrain==1
```
Skiprec
```
python main.py --task_name=inference_acc --seed=5 --model_name=cprec --dataset_path='data/QB-video.csv' --train_batch_size=32 --val_batch_size=32 --test_batch_size=1 --epochs=20 --lr=0.0001 --hidden_size=128 --block_num=8 --embedding_size=128 --dilation=[1, 4] --kernel_size=3 --is_pretrain==1
```

#### Environments
Pytorch 1.7.0

Tensorflow 2.3.0

sklearn 0.24.2

python 3.6.8

**License:**

This dataset is licensed under a CC BY-NC 4.0 International License(https://creativecommons.org/licenses/by-nc/4.0/).
