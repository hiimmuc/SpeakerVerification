# Voice Verification .

This repository contains the framework for training speaker verification model described in [2]  
with score normalization post-processing described in [3].

## Dependencies

```
pip install -r requirements.txt
```

## Data Preparation

1. Generate train, validate list
   (if ratio == -1, take 3 files for each speaker for validate)

```python
python dataprep.py --generate --split_ratio -1
```

In addition to the Python dependencies, `wget` and `ffmpeg` must be installed on the system.

## Pretrained models

Pretrained models and corresponding cohorts can be downloaded from [here](https://drive.google.com/drive/folders/15FYmgHGKlF_JSyPGKfJzBRhQpBY5JcBw?usp=sharing).

## Training

**Set cuda usage**

```
!export CUDA_VISIBLE_DEVICES=5
```

then add the device="cuda:5" to args
<br/>
**Single GPU**

```python
!CUDA_VISIBLE_DEVICES=0 python main.py --do_train --config yaml/configuration.yaml
```

**Data parallel**
```python
!CUDA_VISIBLE_DEVICES=6,7,0 python main.py --do_train --config yaml/configuration.yaml --data_parallel
```

**Distributed**

```python
!CUDA_VISIBLE_DEVICES=6,7,0 python main.py --do_train --config yaml/configuration.yaml --distributed --mixedprec --distributed_backend nccl --port 10001
```

Note: the best model is automaticly saved during the training process, if the initial_model is not provided, automaticly load from the best_state weight if possible.
add --augment to train with augment data

## Inference

1. prepare cohorts

```python
!!CUDA_VISIBLE_DEVICES=0 python main.py --do_infer --prepare --config yaml/configuration.yaml
```

2. Evaluate and tune thresholds

```python
!CUDA_VISIBLE_DEVICES=0 python main.py --do_infer --eval --config yaml/configuration.yaml
```

3. Run on Test set

```python
!CUDA_VISIBLE_DEVICES=0 python main.py --do_infer --test --config yaml/configuration.yaml
```

## Citation

[1] _In defence of metric learning for speaker recognition_

```
@inproceedings{chung2020in,
    title={In defence of metric learning for speaker recognition},
    author={Chung, Joon Son and Huh, Jaesung and Mun, Seongkyu and Lee, Minjae and Heo, Hee Soo and Choe, Soyeon and Ham, Chiheon and Jung, Sunghwan and Lee, Bong-Jin and Han, Icksang},
    booktitle={Interspeech},
    year={2020}
}
```

[2] _Clova baseline system for the VoxCeleb Speaker Recognition Challenge 2020_

```
@article{heo2020clova,
    title={Clova baseline system for the {VoxCeleb} Speaker Recognition Challenge 2020},
    author={Heo, Hee Soo and Lee, Bong-Jin and Huh, Jaesung and Chung, Joon Son},
    journal={arXiv preprint arXiv:2009.14153},
    year={2020}
}
```

[3] _Analysis of score normalization in multilingual speaker recognition_

```
@inproceedings{inproceedings,
    title = {Analysis of Score Normalization in Multilingual Speaker Recognition},
    author = {Matejka, Pavel and Novotny, Ondrej and Plchot, Oldřich and Burget, Lukas and Diez, Mireia and Černocký, Jan},
    booktitle = {Interspeech},
    year = {2017}
}
```
