# few_shot_image

## Examples

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
