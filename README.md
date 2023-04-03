# few_shot_image

## Vision Tasks

### Running the code

**Datasets**
- [miniImageNet](https://mega.nz/#!rx0wGQyS!96sFlAr6yyv-9QQPCm5OBFbOm4XSD0t-HlmGaT5GaiE) (courtesy of [Spyros Gidaris](https://github.com/gidariss/FewShotWithoutForgetting))
- [tieredImageNet](https://drive.google.com/open?id=1nVGCTd9ttULRXFezh4xILQ9lUkg0WZCG) (courtesy of [Kwonjoon Lee](https://github.com/kjunelee/MetaOptNet))
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

#### 15-way accuracy (%) on *mini-ImageNet*
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
|            |        |4          | 36         |600              |-|


#### 15-way accuracy (%) on *tiered-ImageNet*
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
|            |        |2          | 18         |300              |57.85 +- 0.36|
|            |        |3          | 27         |450              |57.74 +- 0.36|
|            |        |4          | 36         |600              |57.54 +- 0.35|
|            | 400    |1          | 9          |150              |58.19 +- 0.36|
|            |        |2          | 18         |300              |58.53 +- 0.36|
|            |        |3          | 27         |450              |58.36 +- 0.36|
|            |        |4          | 36         |600              |57.98 +- 0.36|
|            | 600    |1          | 9          |150              |58.40 +- 0.50|
|            |        |2          | 18         |300              |58.63 +- 0.36|
|            |        |3          | 27         |450              |58.57 +- 0.36|
|            |        |4          | 36         |600              |58.11 +- 0.36|
|            | 800    |1          | 9          |150              |58.86 +- 0.35|
|            |        |2          | 18         |300              |58.95 +- 0.36|
|            |        |3          | 27         |450              |58.43 +- 0.36|
|            |        |4          | 36         |600              |58.24 +- 0.36|
<!-- |clip-vit-B32            | 200    |1          | 7          |120              |69.25 +- 0.18| -->


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
