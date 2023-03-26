# few_shot_image

## Vision Tasks

### Running the code

**Datasets**
- [miniImageNet](https://drive.google.com/file/d/1fJAK5WZTjerW7EWHHQAR9pRJVNg1T1Y7/view?usp=sharing) (courtesy of [Spyros Gidaris](https://github.com/gidariss/FewShotWithoutForgetting))
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


#### 10-way accuracy (%) on *tiered-ImageNet*
| Backbone   | Task(M) |shot images|query images|image per task(m)|Accuracy|
|------------|--------|-----------|------------|-----------------|-|
|CLIP-ViT_B32| 0      | None      | None       |None             |-|


#### Examples

* run one job
```
sbatch script.sh
```
Modify `script.sh` to finetune/test CLIP backbone.

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
