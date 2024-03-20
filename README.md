# TSMMG
Code of "Instruction Multi-Constraint Molecular Generation Using a Teacher-Student Large Language Model"

# Environments
```shell
conda create -n tsmmg python=3.10.9
conda activate tsmmg
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2              # cuda 11.7
pip install transformers
pip install rdkit
pip install pandas
pip install accelerate
pip install weasyprint
pip install libsvm-official
```

# Data Preparation

Download data from：https://pan.baidu.com/s/1THcMLeusKGSgJpUVq83lsw?pwd=0az0 
Unzip the compressed file under **TSMMG/**

# Training

```python
accelerate launch train.py --batch_size=16 --gpu_num=8 --epochs=2
```

# Inference
```shell
python eval.py --eval_type=[TASK_NAME] --cuda=[CUDA_NO] --eval_model_path=[SAVED_MODEL_PATH]
```
**Example**
```shell
python eval.py --eval_type=drd2 --cuda=0 --eval_model_path='./model_save_675354_2'
python eval.py --eval_type=drd2_qed --cuda=0 --eval_model_path='./model_save_675354_2'
python eval.py --eval_type=drd2_qed_sa --cuda=0 --eval_model_path='./model_save_675354_2'
```