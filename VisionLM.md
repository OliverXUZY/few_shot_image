# Vision Language Model

## Result
* Evaluation: Nearest-centroids for few-shot task evaluation. In each task, we sample 10 classes, each class contains text features 10-query images.
* Text encoder: 1 template `'a photo of a {class name}'`.
<!-- 8 templates adapted from CLIP, was reported to have better performance:
  ```
  ['a photo of a {}',
  'itap of a {}.',
  'a bad photo of the {}.',
  'a origami {}.',
  'a photo of the large {}.',
  'a {} in a video game.',
  'art of the {}.',
  'a photo of the small {}.']
  ```
 -->
  The templates forward through clip text encoder become text features, act as _"shot images"_.

#### 10-way accuracy (%) on *mini-ImageNet*
| pre-train  | zero-shot NC     | Multi-task finetune + NC |
|------------|------------------|--------------------------|
|CLIP-ViT_B32| 93.80 +- 0.06    | 94.20 +- 0.06            |
