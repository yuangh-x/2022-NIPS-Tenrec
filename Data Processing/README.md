# Data Processing

Please run the commands as below to generate datasets for each task.

ctr_data_1M.csv is used in CTR task (Section 3.1) and Multi-Task Learning  (Section 3.3).
```
python gen_ctr.py
```
sbr_data_1M.csv is used in Session-based Recommendation (Section 3.2), Transfer Learning (Section 3.4, used as pre-training dataset), User Profile Prediction (Section 3.5), Model Compression  (Section 3.8), Model Training Speedup (Section 3.9).
```
python gen_sbr.py
```
cold_data.csv/cold_data_1.csv/cold_data_0.7.csv/cold_data_0.3.csv are used for the  Cold-Start task (Section 3.6).
```
python gen_cold.py
```
task_0.csv/task_1.csv/task_2.csv/task_3.csv are used in Lifelong Learning (see Section 3.7). 
```
python gen_lifelong.py
```

Note that: 

Model Inference Speedup Task (Section 3.10):  the  data set (QB-video)  is in the original dataset : https://drive.google.com/file/d/1R1JhdT9CHzT3qBJODz09pVpHMzShcQ7a/view?usp=sharing

Transfer Learning Task (Section 3.4):  target dataset is also QB-video.
