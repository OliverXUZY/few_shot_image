# Vision Language Model

## Datasets
Download datasets with class names: [here](https://lyy.mpi-inf.mpg.de/mtl/download/Lmzjm9tX.html)

## Result
* Evaluation: Nearest-centroids for few-shot task evaluation. In each task, we sample 10 classes, each class contains text features 10-query images.
* Text encoder: 8 templates adapted from CLIP, was reported to have better performance:
  ```
  'a photo of a {}',
  'itap of a {}.',
  'a bad photo of the {}.',
  'a origami {}.',
  'a photo of the large {}.',
  'a {} in a video game.',
  'art of the {}.',
  'a photo of the small {}.'
  ```

  The templates forward through clip text encoder become text features, act as _"shot images"_.

#### 10-way accuracy (%) on *mini-ImageNet*
| pre-train  | zero-shot NC     | Multi-task finetune + NC |
|------------|------------------|--------------------------|
|CLIP-ViT_B32| 94.43 +- 0.05    |  95.03 +- 0.05           |


#### 15-way accuracy (%) on *mini-ImageNet*
| pre-train  | zero-shot NC     | Multi-task finetune + NC |
|------------|------------------|--------------------------|
|CLIP-ViT_B32| 93.14 +- 0.04    |  93.49 +- 0.04           |

#### 15-way accuracy (%) on *tiered-ImageNet*
| pre-train  | zero-shot NC     | Multi-task finetune + NC |
|------------|------------------|--------------------------|
|CLIP-ViT_B32| 93.10 +- 0.26    |  93.69 +- 0.19           |

### Local data directory
Downloaded cached files [here](https://drive.google.com/file/d/1PSpCTF6U6bzOqWp0jF4XhhhybIpc3di8/view?usp=sharing)

inside `~/datasets/tiered-imagenet/tiered_imagenet`
```
./train
./cached_test_labels_vl-tiered-imagenet.npy
./tiered-imagenet_test_ViT-B32_text_representation.json
./val
./tiered-imagenet_train_ViT-B32_text_representation.json
./cached_train_labels_vl-tiered-imagenet.npy
./tiered-imagenet_val_ViT-B32_text_representation.json
./cached_val_labels_vl-tiered-imagenet.npy
./test

```
