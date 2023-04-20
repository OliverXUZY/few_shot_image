# few_shot_image

## Vision Tasks

### Running the code

**Datasets**
- [miniImageNet](https://mega.nz/#!rx0wGQyS!96sFlAr6yyv-9QQPCm5OBFbOm4XSD0t-HlmGaT5GaiE) (courtesy of [Spyros Gidaris](https://github.com/gidariss/FewShotWithoutForgetting))
- [tieredImageNet](https://drive.google.com/open?id=1nVGCTd9ttULRXFezh4xILQ9lUkg0WZCG) (courtesy of [Kwonjoon Lee](https://github.com/kjunelee/MetaOptNet))
- [domainNet](http://ai.bu.edu/M3SDA/)
- [ImageNet-800](http://image-net.org/challenges/LSVRC/2012/)

### result
#### 5-way accuracy (%) on *mini-ImageNet*
| Backbone  | Task(M) |shot images|query images|image per task(m)|Accuracy|
|------------|--------|-----------|------------|-----------------|-|
|CLIP-ViT_B32| 0      | None      | None       |None             |83.03 +- 0.24|
|            | 200    |1          | 7          |40               |88.53 +- 0.22|
|            |        |5          | 35         |200              |89.50 +- 0.20|
|            |        |25         | 175        |1000             |89.93 +- 0.20|
|            | 1000   |1          | 7          |40               |89.37 +- 0.20|
|            |        |5          | 35         |200              |90.81 +- 0.19|
|            |        |25         | 175        |1000             |90.97 +- 0.19|
|            | 5000   |1          | 7          |40               |89.95 +- 0.20|
|            |        |5          | 35         |200              |90.94 +- 0.19|
|            |        |25         | 175        |1000             |91.16 +- 0.18|


### M vs m trend
#### 15-way accuracy (%) on *mini-ImageNet*
<img width="944" alt="image" src="https://user-images.githubusercontent.com/43462304/230820902-64f19589-0fd6-4fa2-b6a9-e7de7c34e675.png">

| Backbone  | Task(M) |shot images|query images|image per task(m)|Accuracy|
|------------|--------|-----------|------------|-----------------|-|
|CLIP-ViT_B32| 0      | None      | None       |None             |68.41 +- 0.54|
|            | 200    |1          | 7          |120              |78.42 +- 0.15(78.62 +-0.15)|
|            | 200    |1          | 9          |150              |78.51 +- 0.15|
|            |        |2          | 18         |300              |80.26 +- 0.15|
|            |        |3          | 27         |450              |80.59 +- 0.14|
|            |        |4          | 36         |600              |80.67 +- 0.14|
|            | 400    |1          | 9          |150              |78.95 +- 0.15|
|            |        |2          | 18         |300              |80.85 +- 0.15|
|            |        |3          | 27         |450              |81.31 +- 0.14|
|            |        |4          | 36         |600              |81.51 +- 0.14|
|            | 600    |1          | 9          |150              |79.44 +- 0.15|
|            |        |2          | 18         |300              |80.87 +- 0.15|
|            |        |3          | 27         |450              |81.38 +- 0.14|
|            |        |4          | 36         |600              |81.65 +- 0.14|
|            | 800    |1          | 9          |150              |79.87 +- 0.15|
|            |        |2          | 18         |300              |81.06 +- 0.15|
|            |        |3          | 27         |450              |81.44 +- 0.14|
|            |        |4          | 36         |600              |81.65 +- 0.14|
|CLIP-RN50   | 0      | None      | None       |None             |61.31 +- 0.31|
|            | 200    |1          | 9          |150              |67.32 +- 0.16 |
|            |        |2          | 18         |300              |67.19 +- 0.16|
|            |        |3          | 27         |450              |67.12 +- 0.16 |
|            |        |4          | 36         |600              |-|
|            | 400    |1          | 9          |150              |67.25 +- 0.16|
|            |        |2          | 18         |300              |67.16 +- 0.16 |
|            |        |3          | 27         |450              |67.14 +- 0.16 |
|            |        |4          | 36         |600              |-|
|            | 600    |1          | 9          |150              |67.16 +- 0.16|
|            |        |2          | 18         |300              |67.38 +- 0.16|
|            |        |3          | 27         |450              |67.32 +- 0.16|
|            |        |4          | 36         |600              |-|
|            | 800    |1          | 9          |150              |67.47 +- 0.16|
|            |        |2          | 18         |300              |67.31 +- 0.16|
|            |        |3          | 27         |450              |-|
|            |        |4          | 36         |600              |-|
|DinoV2-ViT_b14   | 0      | None      | None       |None             |90.61 +- 0.19|
|            | 200    |1          | 9          |150              |-|

**5-shot in testing**
| Backbone   | Task(M) |shot images|query images|image per task(m)|Accuracy|
|------------|--------|-----------|------------|-----------------|-|
|CLIP-ViT_B32| 0      | None      | None       |None             |88.67 +- 0.14|
|            | 200    |1          | 9          |150              |93.22 +- 0.11|
|CLIP-RN50   | 0      | None      | None       |None             |82.03 +- 0.18|
|            | 200    |1          | 9          |150              |85.09 +- 0.17|
|DinoV2-ViT_b14   | 0      | None      | None       |None             |97.20 +- 0.06|
|            | 200    |1          | 9          |150              |-|

#### 15-way accuracy (%) on *tiered-ImageNet*
<img width="931" alt="image" src="https://user-images.githubusercontent.com/43462304/230820771-68a38cf2-c064-482c-9377-cc656e56ceb0.png">
<img width="935" alt="image" src="https://user-images.githubusercontent.com/43462304/230820826-4ac714d3-5993-463c-ae3e-6a190ede829f.png">

**1-shot in testing**
| Backbone   | Task(M) |shot images|query images|image per task(m)|Accuracy|
|------------|--------|-----------|------------|-----------------|-|
|CLIP-ViT_B32| 0      | None      | None       |None             |59.55 +- 0.21|
|            | 200    |1          | 9          |150              |68.57 +- 0.37|
|            |        |2          | 18         |300              |70.51 +- 0.37|
|            |        |3          | 27         |450              |71.14 +- 0.36|
|            |        |4          | 36         |600              |71.30 +- 0.36|
|            | 400    |1          | 9          |150              |70.19 +- 0.53|
|            |        |2          | 18         |300              |71.63 +- 0.36|
|            |        |3          | 27         |450              |72.33 +- 0.36|
|            |        |4          | 36         |600              |72.44 +- 0.36|
|            | 600    |1          | 9          |150              |70.54 +- 0.36|
|            |        |2          | 18         |300              |72.33 +- 0.36|
|            |        |3          | 27         |450              |72.64 +- 0.35|
|            |        |4          | 36         |600              |72.81 +- 0.36|
|            | 800    |1          | 9          |150              |71.44 +- 0.36 |
|            |        |2          | 18         |300              |72.28 +- 0.35|
|            |        |3          | 27         |450              |72.78 +- 0.36|
|            |        |4          | 36         |600              |72.93 +- 0.35|
|CLIP-RN50   | 0      | None      | None       |None             |51.76 +- 0.36|
|            | 200    |1          | 9          |150              |57.56 +- 0.36|
|            |        |2          | 18         |300              |57.64 +- 0.36|
|            |        |3          | 27         |450              |57.74 +- 0.36|
|            |        |4          | 36         |600              |58.90 +- 0.35|
|            | 400    |1          | 9          |150              |58.19 +- 0.36|
|            |        |2          | 18         |300              |58.33 +- 0.36|
|            |        |3          | 27         |450              |58.40 +- 0.36|
|            |        |4          | 36         |600              |59.24 +- 0.36|
|            | 600    |1          | 9          |150              |58.40 +- 0.50|
|            |        |2          | 18         |300              |58.63 +- 0.36|
|            |        |3          | 27         |450              |58.72 +- 0.36|
|            |        |4          | 36         |600              |59.63 +- 0.36|
|            | 800    |1          | 9          |150              |58.86 +- 0.35|
|            |        |2          | 18         |300              |58.95 +- 0.36|
|            |        |3          | 27         |450              |59.73 +- 0.36|
|            |        |4          | 36         |600              |59.91 +- 0.36|
|DinoV2-ViT_b14   | 0      | None      | None       |None             |82.33 +- 0.30|
|            | 200    |1          | 9          |150              |84.74 +- 0.30|
<!-- |clip-vit-B32            | 200    |1          | 7          |120              |69.25 +- 0.18| -->

**5-shot in testing**
| Backbone   | Task(M) |shot images|query images|image per task(m)|Accuracy|
|------------|--------|-----------|------------|-----------------|-|
|CLIP-ViT_B32| 0      | None      | None       |None             |79.51 +- 0.27|
|            | 200    |1          | 9          |150              |84.79 +- 0.22|
|CLIP-RN50   | 0      | None      | None       |None             |71.40 +- 0.30|
|            | 200    |1          | 9          |150              |75.80 +- 0.28|
|DinoV2-ViT_b14   | 0      | None      | None       |None             |92.90 +- 0.16|
|            | 200    |1          | 9          |150              |93.65 +- 0.16|

#### 15-way accuracy (%) on *domain-Net*
**1-shot in testing**
| Backbone   | Task(M) |shot images|query images|image per task(m)|Accuracy|
|------------|--------|-----------|------------|-----------------|-|
|CLIP-ViT_B32| 0      | None      | None       |None             |44.49 +- 0.37|
|            | 200    |1          | 9          |150              |-|
|CLIP-RN50   | 0      | None      | None       |None             |38.15 +- 0.39|
|            | 200    |1          | 9          |150              |-|
|DinoV2-ViT_b14| 0      | None      | None       |None             |61.65 +- 0.41|
|            | 200    |1          | 9          |150              |68.22 +- 0.40|

## Vary num shot
#### 15-way accuracy (%) on *mini-ImageNet*
| Backbone  | Task(M) |shot images|query images|image per task(m)|Accuracy|
|------------|--------|-----------|------------|-----------------|-|
|CLIP-ViT_B32| 0      | None      | None       |None             |68.41 +- 0.54|
|            | 200    |1          | 19         |300              |-|
|            |        |5          | 15         |300              |-|
|            |        |10         | 10         |300              |-|
|            |        |15         | 5          |300              |-|
|            | 400    |1          | 19         |300              |-|
|            |        |5          | 15         |300              |-|
|            |        |10         | 10         |300              |-|
|            |        |15         | 5          |300              |-|
|            | 400    |1          | 39         |600              |-|
|            |        |5          | 35         |600              |-|
|            |        |10         | 30         |600              |-|
|            |        |15         | 25         |600              |-|
|CLIP-RN50   | 0      | None      | None       |None             |-|
|            | 200    |1          | 19         |300              |-|
|            |        |5          | 15         |300              |-|
|            |        |10         | 10         |300              |-|
|            |        |15         | 5          |300              |-|
|            | 400    |1          | 19         |300              |-|
|            |        |5          | 15         |300              |-|
|            |        |10         | 10         |300              |-|
|            |        |15         | 5          |300              |-|
|            | 400    |1          | 39         |600              |-|
|            |        |5          | 35         |600              |-|
|            |        |10         | 30         |600              |-|
|            |        |15         | 25         |600              |-|

#### 15-way accuracy (%) on *tiered-ImageNet*
| Backbone  | Task(M) |shot images|query images|image per task(m)|Accuracy|
|------------|--------|-----------|------------|-----------------|-|
|CLIP-ViT_B32| 0      | None      | None       |None             |59.55 +- 0.21|
|            | 200    |1          | 19         |300              |69.10 +- 0.37|
|            |        |5          | 15         |300              |71.27 +- 0.36|
|            |        |10         | 10         |300              |71.10 +- 0.37|
|            |        |15         | 5          |300              |71.01 +- 0.36|
|            | 400    |1          | 19         |300              |70.84 +- 0.36|
|            |        |5          | 15         |300              |72.43 +- 0.36|
|            |        |10         | 10         |300              |72.42 +- 0.36|
|            |        |15         | 5          |300              |72.11 +- 0.35|
|            | 400    |1          | 39         |600              |70.89 +- 0.36|
|            |        |5          | 35         |600              |72.45 +- 0.36|
|            |        |10         | 30         |600              |72.42 +- 0.36|
|            |        |15         | 25         |600              |72.20 +- 0.36|
|CLIP-RN50   | 0      | None      | None       |None             |51.76 +- 0.36|
|            | 200    |1          | 19         |300              |59.38 +- 0.36|
|            |        |5          | 15         |300              |59.27 +- 0.36|
|            |        |10         | 10         |300              |58.86 +- 0.36|
|            |        |15         | 5          |300              |58.64 +- 0.36|
|            | 400    |1          | 19         |300              |60.47 +- 0.36|
|            |        |5          | 15         |300              |59.89 +- 0.36|
|            |        |10         | 10         |300              |59.77 +- 0.36|
|            |        |15         | 5          |300              |59.57 +- 0.36|
|            | 400    |1          | 39         |600              |60.44 +- 0.37|
|            |        |5          | 35         |600              |59.72 +- 0.36|
|            |        |10         | 30         |600              |59.54 +- 0.36|
|            |        |15         | 25         |600              |59.48 +- 0.36|
#### Examples
* python command
```
python finetune.py \
    --config=configs/clip/mini-imagenet/finetune_ViT.yaml \
    --n_batch_train 200 \
    --n_shot 1 \
    --sample_per_task 150
    
python test.py --config=configs/clip/mini-imagenet/test.yaml 
```
modify `path` in `test.yaml ` as saving model path or add `--path [path]` in command `python test.py`

* run one job
```
sbatch script.sh
```
Modify `script.sh` with different python command.

* run multiple jobs with varying one inputs
```
sbatch --array=1-3 jobArrayScript.sh
```
Modify `jobArrayScript.sh` and/or `input_file_list.txt`, `input_path_list.txt`.

* run multiple jobs with varying several inputs
```
bash multi_jobs.sh
```
Modify `script.sh`, `multi_jobs.sh`.
## Local data directory
```
datasets/
├── mini-imagenet
│   ├── miniImageNet_category_split_meta_train_limited100.pickle
│   ├── miniImageNet_category_split_meta_train_limited10.pickle
│   ├── miniImageNet_category_split_meta_train_limited20.pickle
│   ├── miniImageNet_category_split_meta_train_limited50.pickle
│   ├── miniImageNet_category_split_meta_train_limited_class16.pickle
│   ├── miniImageNet_category_split_meta_train_limited_class32.pickle
│   ├── miniImageNet_category_split_meta_train_limited_class8.pickle
│   ├── miniImageNet_category_split_test.pickle
│   ├── miniImageNet_category_split_train_phase_test.pickle
│   ├── miniImageNet_category_split_train_phase_train.pickle
│   ├── miniImageNet_category_split_train_phase_val.pickle
│   └── miniImageNet_category_split_val.pickle
```
