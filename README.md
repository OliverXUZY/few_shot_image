# few_shot_image
## Result
### Trend of M, m
* Verify the trend of different number of tasks and number of images per task. Each task contains 5 classes.
  * m = 50 =  5*(1+9): Each class contains 1-shot and 9-query.
  * m = 100 =  5*(2+18): Each class contains 2-shot and 18-query.
  * m = 200 =  5*(4+36): Each class contains 4-shot and 36-query.
* Due to limited sapce and time, I trained encoder one time. For test evaluation, I sampled 1500 tasks and show the accuracy confidence interval below.

<img width="945" alt="image" src="https://user-images.githubusercontent.com/43462304/215162793-514d2bce-0007-4826-a4fa-19ec2df0b3e3.png">

* For fixed number of tasks or samples per task, increasing samples or tasks will improve the accuracy. The total number of sample (M * m) will determine the overall performance. However:
* As pointed out by reviewer, we observed empirically m cannot be too small, 200 tasks with 100 images per task will have performance comparable to 800 tasks with 500 images per task.

| Backbone  | Task(M) |shot images|query images|image per task(m)|Accuracy|
|------------|--------|-----------|------------|-----------------|-|
|CLIP-ViT_B32| 0      | None      | None       |None             |83.03 +- 0.24|
|            | 800    |1          | 9          |50               |89.71 +- 0.19|
|            |        |2          | 18         |100              |90.27 +- 0.19|
|            |        |3          | 27         |150              ||
|            |        |4          | 36         |200              |90.80 +- 0.18|
|            | 600    |1          | 9          |50               ||
|            |        |2          | 18         |100              ||
|            |        |3          | 27         |150              ||
|            |        |4          | 36         |200              ||
|            | 400    |1          | 9          |50               |89.31 +- 0.19|
|            |        |2          | 18         |100              |90.11 +- 0.19|
|            |        |3          | 27         |150              ||
|            |        |4          | 36         |200              |90.70 +- 0.18|
|            | 200    |1          | 9          |50               |89.07 +- 0.20|
|            |        |2          | 18         |100              |89.95 +- 0.19|
|            |        |3          | 27         |150              ||
|            |        |4          | 36         |200              |90.09 +- 0.19|
## Examples

```
sbatch script.sh
```
select one of line 21/22/23 in `script.sh` to finetune/test CLIP backbone

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
