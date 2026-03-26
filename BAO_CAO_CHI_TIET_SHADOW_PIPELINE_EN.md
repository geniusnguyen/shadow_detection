# Detailed Report on the Acoustic Shadow Detection Problem in Ultrasound Images

## 1. Problem Objective

The problem addressed in this repository is the detection of `posterior acoustic shadow` in ultrasound images. The system does not only predict a pixel-level mask, but also provides an image-level decision:

- Whether the image truly contains acoustic shadowing.
- Whether the predicted dark region is a `true posterior acoustic shadow` or merely an artifact that resembles shadowing.

The pipeline is divided into 4 stages:

1. `lesion_classifier`: pretrain an encoder on a 6-class ovarian tumor classification task.
2. `shadow_classifier`: fine-tune that encoder for `shadow segmentation + image-level shadow classification`.
3. `shadow_inference`: load the trained checkpoint and export inference results.
4. `shadow_postfilter`: apply `artifact-aware` post-processing to reduce false positives.

## 2. System Architecture Overview

The overall processing flow is:

```text
data_tumor/images
  -> lesion_classifier
  -> best_encoder.pt

data_shad_official/Image + Label
  -> shadow_classifier
  -> best.pt

best.pt + manifest.csv
  -> shadow_inference
  -> predictions.csv + input/pred_prob/pred_mask/overlay

best.pt + manifest.csv + rule_config + artifact_descriptions
  -> shadow_postfilter
  -> predictions_postfilter.csv + metrics_before_after.json
```

The core idea of the project is:

- Do not train the shadow model from scratch.
- First let the encoder learn ultrasound-domain features through lesion classification.
- Then transfer that encoder to the shadow task.
- Finally add a logic-based post-processing layer that uses shape, position, darkness/brightness, and mask continuity.

## 3. Module 1: `lesion_classifier`

### 3.1. Role

This is the `domain pretraining` stage. Its goal is not to detect shadow directly, but to help the backbone learn meaningful ovarian ultrasound features:

- texture;
- lesion morphology;
- anatomical layout;
- intensity and reflection patterns specific to this imaging domain.

The checkpoint produced at this stage is `best_encoder.pt`, which is then loaded into `shadow_classifier`.

### 3.2. Input

The data is read from:

```text
data_tumor/images/
  ubi/
  udac/
  unangdathuy/
  unangdathuydac/
  unangdonthuy/
  unangdonthuydac/
```

Each subfolder corresponds to one classification class.

### 3.3. Submodules

#### a. `src/data/ovatus_manifest.py`

Functions:

- Scan the whole classification dataset.
- Group images by `case`.
- Handle the naming rule `abc.jpg`, `abc_1.jpg`, `abc_2.jpg`.
- Split wide two-panel images into `left/right` when needed.
- Build the manifest and split `train/val` by `case` to avoid leakage.

Inputs:

- `data.root`
- `split_wide_if_no_views`
- `full_image_aspect_ratio_threshold`
- `extensions`

Outputs:

- `manifest.csv`
- `train_split.csv`
- `val_split.csv`
- `manifest_warnings.json`
- `class_to_idx.json`

Main logic:

- If `abc_1` and/or `abc_2` exist, those cropped views are preferred.
- If only `abc.jpg` exists and the image is wide, it is split into 2 views.
- If there is only one single image, it is kept as `single`.
- Each sample is assigned `sample_weight = 1 / num_views_in_case` so cases with many views do not dominate the loss and metrics.

#### b. `src/data/ovatus_cls_dataset.py`

Functions:

- Read images from the manifest.
- Crop by `crop_box` if the sample was split from a full image.
- Resize-pad to a square size.
- Apply augmentation and normalization.

Inputs:

- one manifest row;
- `image_size`;
- augmentation parameters.

Outputs per sample:

- `image`: tensor `[3, H, W]`
- `target`: classification label
- `sample_weight`
- `case_uid`
- `sample_id`

Augmentation:

- horizontal flip;
- random rotation;
- brightness/contrast jitter;
- gaussian blur.

#### c. `src/models/*.py`

The repository supports 3 backbone families:

- `ResNetClassifier`
- `EfficientNetClassifier`
- `SegFormerClassifier`

Input:

- image tensor `[B, 3, H, W]`

Output:

- `logits` of shape `[B, num_classes]`

Meaning:

- `encoder`: extracts features.
- `classifier`: linear head for the 6-class classification task.
- `encoder_state_dict()`: exports only the encoder weights so they can be reused for the shadow task.

#### d. `src/engine/classification_trainer.py`

Functions:

- Manage the train/eval loop.
- Compute class weights.
- Compute both view-level and case-level metrics.
- Save `latest.pt`, `best.pt`, and `best_encoder.pt`.

Loss:

- `CrossEntropyLoss` with:
  - class weighting;
  - label smoothing;
  - sample weighting based on the number of views in each case.

Metrics:

- `view_accuracy`
- `view_macro_f1`
- `view_balanced_accuracy`
- `case_accuracy`
- `case_macro_f1`
- `case_balanced_accuracy`

Here, `case-level` metrics are computed by averaging the logits of all views belonging to the same `case_uid`, then assigning the final case label.

### 3.4. Main Configuration Parameters

Three config files are currently provided:

- `lesion_classifier/configs/ovatus_resnet34.yaml`
- `lesion_classifier/configs/ovatus_efficientnet_b0.yaml`
- `lesion_classifier/configs/ovatus_segformer_b0.yaml`

Current default values:

- `image_size = 384`
- `val_fraction = 0.2`
- `random_seed = 42`
- optimizer: `AdamW`
- scheduler: `CosineAnnealingLR`
- `label_smoothing = 0.05`
- `AMP` enabled

### 3.5. Module Outputs

Each run produces:

- `manifest.csv`
- `train_split.csv`
- `val_split.csv`
- `class_to_idx.json`
- `manifest_warnings.json`
- `run_metadata.json`
- `history.json`
- `checkpoints/latest.pt`
- `checkpoints/best.pt`
- `checkpoints/best_encoder.pt`

### 3.6. Existing Empirical Results

After normalization, the pretraining dataset contains:

- `1689` view-level samples
- `1349` train samples
- `340` validation samples

Best results by `case_macro_f1`:

| Backbone | Best epoch | Case Macro-F1 | Case Accuracy | Case Balanced Accuracy |
|---|---:|---:|---:|---:|
| ResNet34 | 15 | 0.7012 | 0.7443 | 0.6963 |
| EfficientNet-B0 | 17 | 0.6863 | 0.7386 | 0.6790 |
| SegFormer-B0 | 24 | 0.6907 | 0.7614 | 0.7018 |

Observations:

- `ResNet34` achieves the highest `case_macro_f1`.
- `SegFormer-B0` achieves the highest `case_accuracy` and `case_balanced_accuracy`.
- This stage provides the encoder initialization for the downstream shadow task.

## 4. Module 2: `shadow_classifier`

### 4.1. Role

This is the core module of the project. It solves two tasks jointly:

- `segmentation`: predict a shadow mask at the pixel level.
- `classification`: predict whether the image contains shadow at the image level.

It uses the encoder pretrained in `lesion_classifier`.

### 4.2. Input

The data is read from:

```text
data_shad_official/Image
data_shad_official/Label
```

The polygon label used in the `Labelme JSON` files is `bc`.

### 4.3. Submodules

#### a. `src/data/shadow_manifest.py`

Functions:

- Read images and label JSON files.
- Extract polygons labeled `bc`.
- Build view-level samples.
- Split full-frame two-panel images into `left/right`.
- Generate `image_label` at the view level.
- Split by `study` instead of by file.

Inputs:

- `image_dir`
- `label_dir`
- `positive_label = bc`
- `split_wide_views`
- `full_image_aspect_ratio_threshold`

Outputs:

- a list of manifest rows;
- `warnings`;
- `stats`.

Image-level labeling rule:

- If the sample is a full image: `image_label = 1` if there is at least one polygon.
- If the sample is a `left/right` crop: `image_label = 1` if any polygon intersects that crop.

Important meaning:

- This is how the segmentation problem is converted into an image-level classification target while remaining consistent with polygon annotations.

#### b. `src/data/shadow_dataset.py`

Functions:

- Read the original image.
- Rasterize polygons into a binary mask.
- Crop by `crop_box` if needed.
- Resize-pad the image and mask in sync.
- Apply augmentation for training.
- Convert to tensors.

Input of each sample:

- one manifest row containing `source_path`, `polygons_json`, `crop_box`, `image_label`, `sample_weight`.

Outputs:

- `image`: tensor `[3, H, W]`
- `mask`: tensor `[1, H, W]`
- `image_label`: scalar float 0/1
- `sample_weight`
- `study_id`
- `sample_id`

Augmentation:

- horizontal flip;
- random rotation;
- brightness/contrast changes;
- gaussian blur.

#### c. `src/models/shadow_multitask.py`

Functions:

- Define 3 multitask architectures:
  - `ResNetShadowModel`
  - `EfficientNetShadowModel`
  - `SegFormerShadowModel`
- Each model uses one shared encoder and has two outputs:
  - `segmentation_logits`
  - `classification_logits`

Input:

- image tensor `[B, 3, H, W]`

Outputs:

- `segmentation_logits`: `[B, 1, H, W]`
- `classification_logits`: `[B]`

Architecture:

1. The backbone extracts multiscale features.
2. `classification head`:
   - global pooling;
   - dropout;
   - linear layer to 1 logit.
3. `segmentation head`:
   - a lightweight U-Net-like decoder;
   - upsampling + skip connections;
   - a `1x1` conv to produce a 1-channel mask.

Role of each backbone:

- `ResNet`: CNN encoder with clear skip features.
- `EfficientNet`: compact and parameter-efficient encoder.
- `SegFormer`: transformer backbone, better suited for layout and long-range context.

#### d. `load_pretrained_encoder(...)`

Functions:

- Load `best_encoder.pt` from the pretraining stage.
- Assign it to `model.encoder`.
- Report `missing_keys` and `unexpected_keys`.

For `shadow_resnet34_finetune`, the current preload information is:

- `missing_keys = []`
- `unexpected_keys = []`

This indicates that the encoder transfer from pretraining to fine-tuning is fully compatible.

#### e. `src/engine/shadow_trainer.py`

Functions:

- Run the multitask training loop.
- Compute the joint segmentation + classification loss.
- Evaluate segmentation and classification in the same loop.
- Select `best.pt` according to the monitored metric.

Loss formula:

1. Segmentation loss:

```text
seg_loss = BCEWithLogits(seg_logits, mask) + DiceLoss(seg_logits, mask)
```

2. Classification loss:

```text
cls_loss = BCEWithLogits(cls_logits, image_label)
```

3. Total loss:

```text
total_loss = sample_weight * (seg_loss_weight * seg_loss + cls_loss_weight * cls_loss)
```

Key points:

- `sample_weight_by_study = true`: studies with many views/frames do not dominate too much.
- `AMP`
- `grad_clip_norm`
- `CosineAnnealingLR`

### 4.4. Train Script Inputs and Outputs

Main script:

- `shadow_classifier/scripts/train_shadow_multitask.py`

Inputs:

- YAML config file;
- image data + labels;
- encoder checkpoint from the pretraining stage.

Outputs:

- `manifest.csv`
- `train_split.csv`
- `val_split.csv`
- `manifest_warnings.json`
- `manifest_stats.json`
- `run_metadata.json`
- `history.json`
- `checkpoints/latest.pt`
- `checkpoints/best.pt`
- `checkpoints/best_encoder.pt`

### 4.5. Main Configuration Parameters

Three config files:

- `shadow_classifier/configs/shadow_resnet34.yaml`
- `shadow_classifier/configs/shadow_efficientnet_b0.yaml`
- `shadow_classifier/configs/shadow_segformer_b0.yaml`

Shared settings:

- `image_size = 384`
- `val_fraction = 0.2`
- `random_seed = 42`
- `dropout = 0.2`
- `epochs = 40`
- optimizer: `AdamW`
- scheduler: `cosine`
- `seg_loss_weight = 1.0`
- `cls_loss_weight = 1.0`

Main differences:

- ResNet34:
  - `batch_size = 16`
  - `lr = 3e-4`
- EfficientNet-B0:
  - `batch_size = 12`
  - `lr = 2e-4`
- SegFormer-B0:
  - `batch_size = 8`
  - `lr = 2e-4`

### 4.6. Data After Manifest Construction

According to `shadow_classifier/outputs/shadow_resnet34_finetune/manifest_stats.json`:

- `331` original images
- `662` view-level samples after splitting into `left/right`
- `215` studies
- `546` positive samples
- `116` negative samples

The current validation set contains:

- `114` samples
- `43` studies

Observation:

- The problem is clearly imbalanced, because positive samples are much more frequent than negative samples.
- Therefore metrics such as `balanced_accuracy`, `macro_f1`, and `negative_recall` are very important; it is not enough to look only at `accuracy`.

### 4.7. Existing Fine-Tuning Results

The checkpoint is selected by `seg_dice`.

| Backbone | Best epoch | Seg Dice | Seg IoU | Cls Accuracy | Cls Macro-F1 | Cls Balanced Acc | Neg Recall | Pos Recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ResNet34 | 34 | 0.4554 | 0.3493 | 0.8333 | 0.6634 | 0.6474 | 0.3684 | 0.9263 |
| EfficientNet-B0 | 31 | 0.4661 | 0.3593 | 0.8158 | 0.6878 | 0.7000 | 0.5263 | 0.8737 |
| SegFormer-B0 | 35 | 0.4761 | 0.3685 | 0.7982 | 0.6580 | 0.6684 | 0.4737 | 0.8632 |

Observations:

- `SegFormer-B0` gives the best segmentation performance.
- `EfficientNet-B0` gives the best `classification macro_f1` and `balanced_accuracy`.
- `ResNet34` gives the highest `accuracy`, but its `negative_recall` is low, meaning it still tends to label negatives as positives.

## 5. Module 3: `shadow_inference`

### 5.1. Role

This module separates inference from training code. Its purpose is to:

- load `best.pt`;
- run on `train/val/all`;
- export results as images and CSV files for error analysis.

Main script:

- `shadow_inference/scripts/export_shadow_predictions.py`

### 5.2. Inputs

Input arguments:

- `--checkpoint`: path to `best.pt`
- `--manifest`: optional, if omitted the script auto-detects `manifest.csv`
- `--output-dir`
- `--split`: `all/train/val`
- `--max-samples`
- `--seg-threshold`
- `--cls-threshold`
- `--device`

Logical input of each sample:

- one row from the manifest;
- original image;
- ground-truth polygon;
- transform settings stored in the checkpoint config.

### 5.3. Processing

Processing flow:

1. Load the checkpoint and config.
2. Rebuild the model with the correct architecture.
3. Load the sample and rasterize the ground-truth mask.
4. Run the model.
5. Obtain:
   - `seg_prob = sigmoid(segmentation_logits)`
   - `cls_score = sigmoid(classification_logits)`
6. Threshold:
   - `pred_mask = seg_prob >= seg_threshold`
   - `pred_image_label = cls_score >= cls_threshold`
7. Save outputs.

### 5.4. Outputs

Run-level:

- `run_info.json`
- `predictions.csv`

Sample-level:

- `input.png`
- `gt_mask.png`
- `pred_prob.png`
- `pred_mask.png`
- `overlay.png`
- `summary.json`

Meaning of each file:

- `pred_prob.png`: pixel-level shadow probability map.
- `pred_mask.png`: thresholded mask.
- `overlay.png`: predicted mask overlaid on the original image.
- `summary.json`: summary of image-level prediction and mask area.

## 6. Module 4: `shadow_postfilter`

### 6.1. Role

This is a post-processing step that does not train a new model. Its goal is to:

- use `cls_score`, `seg_prob`, and `pred_mask`;
- extract geometric and intensity features;
- apply rules to distinguish:
  - `true_posterior_acoustic_shadow`
  - `edge_shadow`
  - `posterior_enhancement`
  - `reverberation_like_artifact`
  - `none`

The final purpose is to reduce false positives.

### 6.2. Submodules

#### a. `spf/io_utils.py`

Functions:

- load checkpoints;
- resolve the manifest path;
- build a model from a checkpoint;
- load samples for inference;
- denormalize images;
- save JSON and overlay outputs.

This is a utility layer so the post-filter script does not have to duplicate inference code.

#### b. `spf/feature_extractor.py`

Functions:

- extract features from `image_rgb`, `seg_prob`, and `pred_mask`.

Inputs:

- denormalized RGB image;
- segmentation probability map;
- thresholded mask.

Output:

- a `features` dictionary.

Main feature set:

- `mask_non_empty`
- `mask_area_pixels`
- `mask_area_ratio`
- `num_components`
- `largest_component_fraction`
- `bbox_fill_ratio`
- `depth_span_ratio`
- `width_span_ratio`
- `depth_to_width_ratio`
- `start_depth_ratio`
- `end_depth_ratio`
- `center_x_ratio`
- `left_border_fraction`
- `right_border_fraction`
- `edge_border_fraction`
- `mean_prob_in_mask`
- `high_prob_core_ratio`
- `grayscale_inside_mean`
- `grayscale_outside_mean`
- `darkness_delta`
- `avg_runs_per_occupied_row`
- `rows_with_multi_runs_ratio`
- `component_fragmentation`

Meaning:

- Geometry group: tells whether the mask is deep, thin, border-hugging, or coherent.
- Intensity group: tells whether the predicted region is darker or brighter than its surroundings.
- Probability group: tells whether the model is truly confident in the mask or only weakly activates it.

Note:

- `darkness_delta = grayscale_outside_mean - grayscale_inside_mean`
- A positive value means the region inside the mask is darker than the outside region.
- A negative value means the region inside the mask is brighter than the outside region, which suggests `posterior enhancement`.

#### c. `spf/rule_engine.py`

Functions:

- convert features into intermediate `signals`;
- compute a score for each artifact class;
- aggregate them into `final_shadow_confidence`.

Important signals:

- `mask_strength`
- `area_strength`
- `dark_shadow_signal`
- `enhancement_signal`
- `elongated_signal`
- `deep_signal`
- `thin_signal`
- `border_signal`
- `fragmented_signal`
- `weak_cls_signal`
- `weak_mask_signal`
- `small_mask_signal`
- `support_signal`
- `penalty_relief`

The rule engine then produces artifact scores for:

- `true_posterior_acoustic_shadow`
- `edge_shadow`
- `posterior_enhancement`
- `reverberation_like_artifact`
- `none`

And computes:

```text
final_shadow_confidence
```

General rule behavior:

- It favors true shadow when:
  - `cls_score` is not too low;
  - the mask is large enough;
  - the mask is coherent;
  - the region tends to extend along the depth axis;
  - the predicted area does not look too much like edge or reverberation artifacts.
- It suppresses false shadow when:
  - the mask is thin and border-hugging;
  - the posterior region is brighter rather than darker;
  - the mask is fragmented;
  - both `cls_score` and `mask_strength` are weak.

#### d. `spf/evaluation.py`

Functions:

- compute binary classification metrics from `y_true` and `y_pred`.

Three reporting levels:

- `raw_classifier`
- `gated_with_mask`
- `postfilter`

Meaning:

- `raw_classifier`: based only on `cls_score >= cls_threshold`.
- `gated_with_mask`: positive only if the classifier is positive and the mask is non-empty.
- `postfilter`: based on `final_shadow_confidence >= final_threshold`.

### 6.3. Default Rule Config

File:

- `shadow_postfilter/configs/default_rule_config.json`

Important thresholds:

- `mask_area_low = 0.002`
- `mask_area_good = 0.02`
- `dark_delta_good = 0.05`
- `enhancement_delta_good = 0.025`
- `enhancement_delta_high = 0.08`
- `depth_to_width_good = 1.8`
- `depth_span_good = 0.28`
- `width_span_thin = 0.08`
- `border_fraction_high = 0.45`
- `bbox_fill_low = 0.18`
- `avg_runs_high = 1.8`
- `fragmented_components_high = 4.0`
- `high_prob_good = 0.55`
- `weak_cls_low = 0.25`
- `cls_support_good = 0.75`
- `mask_strength_low = 0.2`
- `mask_strength_good = 0.65`

Important weights:

- `raw_cls = 0.62`
- `mask_strength = 0.24`
- `dark_shadow = 0.08`
- `elongated = 0.06`
- `deep = 0.06`
- `edge_penalty = 0.14`
- `enhancement_penalty = 0.12`
- `reverberation_penalty = 0.08`
- `none_penalty = 0.06`
- `true_shadow_bonus = 0.22`
- `support_bonus = 0.12`

Observation:

- The rule engine still places substantial emphasis on the original classifier.
- Mask features and darkness signals act as supporting evidence, while penalties help remove false positives.

### 6.4. Artifact Class Descriptions

File:

- `shadow_postfilter/artifacts/artifact_descriptions.json`

Five artifact classes:

1. `true_posterior_acoustic_shadow`
2. `edge_shadow`
3. `posterior_enhancement`
4. `reverberation_like_artifact`
5. `none`

Role of this file:

- standardize artifact interpretation;
- provide a conceptual basis for rule design;
- help the report reader understand why a region is down-weighted as shadow.

### 6.5. Inputs and Outputs

Main script:

- `shadow_postfilter/scripts/run_artifact_postfilter.py`

Inputs:

- `--checkpoint`
- `--manifest`
- `--output-dir`
- `--split`
- `--rules-config`
- `--artifact-descriptions`
- `--max-samples`
- `--seg-threshold`
- `--cls-threshold`
- `--final-threshold`
- `--device`
- `--save-visuals`
- `--visual-limit`

Outputs:

- `run_info.json`
- `rules_config_used.json`
- `artifact_descriptions_used.json`
- `predictions_postfilter.csv`
- `metrics_before_after.json`
- `visuals/`

`predictions_postfilter.csv` contains:

- sample information;
- `raw_cls_score`;
- `raw_cls_pred`;
- `gated_pred`;
- `final_shadow_confidence`;
- `postfilter_pred`;
- `predicted_artifact_type`;
- `artifact_score__*` fields;
- `signal__*` fields;
- `feature__*` fields;
- `decision_reasons`.

## 7. Evaluation Metrics

### 7.1. Classification Metrics

#### Accuracy

```text
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

This measures the overall proportion of correct predictions, but it can be misleading on imbalanced datasets.

#### Negative Recall

```text
Negative Recall = TN / (TN + FP)
```

This is very important in this problem because the goal of post-filtering is to reduce false positives.

#### Positive Recall

```text
Positive Recall = TP / (TP + FN)
```

This measures the ability to retain true positives.

#### Balanced Accuracy

```text
Balanced Accuracy = (Negative Recall + Positive Recall) / 2
```

This is more suitable than plain accuracy when the dataset is imbalanced.

#### Macro-F1

F1 is computed for each class and then averaged. This metric balances precision and recall for both the negative and positive classes.

### 7.2. Segmentation Metrics

#### Dice

```text
Dice = (2 * intersection + eps) / (pred_sum + target_sum + eps)
```

This measures overlap between the predicted mask and the ground-truth mask.

#### IoU

```text
IoU = (intersection + eps) / (union + eps)
```

This is stricter than Dice and is more sensitive to over- or under-segmentation.

### 7.3. Metrics in the Post-Filter Stage

The post-filter is evaluated only at the image level; it does not recompute segmentation metrics.

Three levels are compared:

- `raw_classifier`
- `gated_with_mask`
- `postfilter`

This helps identify:

- how much improvement comes from the original classifier;
- how much comes from requiring a non-empty mask;
- how much comes from the rule engine.

## 8. Existing Post-Filter Results

### 8.1. Overall Results by Backbone

| Run | Stage | Accuracy | Macro-F1 | Balanced Acc | Neg Recall | Pos Recall |
|---|---|---:|---:|---:|---:|---:|
| EfficientNet-B0 | Raw | 0.8158 | 0.6878 | 0.7000 | 0.5263 | 0.8737 |
| EfficientNet-B0 | Gated | 0.8246 | 0.7081 | 0.7263 | 0.5789 | 0.8737 |
| EfficientNet-B0 | Postfilter | 0.2456 | 0.2419 | 0.5263 | 0.9474 | 0.1053 |
| ResNet34 | Raw | 0.8333 | 0.6634 | 0.6474 | 0.3684 | 0.9263 |
| ResNet34 | Gated | 0.8509 | 0.7131 | 0.7000 | 0.4737 | 0.9263 |
| ResNet34 v1 | Postfilter | 0.2456 | 0.2398 | 0.5474 | 1.0000 | 0.0947 |
| ResNet34 v2 | Postfilter | 0.8509 | 0.7131 | 0.7000 | 0.4737 | 0.9263 |
| ResNet34 v3 | Postfilter | 0.8509 | 0.7258 | 0.7211 | 0.5263 | 0.9158 |
| SegFormer-B0 | Raw | 0.7982 | 0.6580 | 0.6684 | 0.4737 | 0.8632 |
| SegFormer-B0 | Gated | 0.7982 | 0.6704 | 0.6895 | 0.5263 | 0.8526 |
| SegFormer-B0 | Postfilter | 0.2982 | 0.2974 | 0.5789 | 1.0000 | 0.1579 |

Main observations:

- The `gated_with_mask` step already improves results clearly for all 3 backbones.
- `ResNet34 v3` is the only currently available post-filter version that performs better than `gated_with_mask`.
- `EfficientNet-B0` and `SegFormer-B0` are currently penalized too strongly by the post-filter.

### 8.2. Detailed Analysis of `resnet34_postfilter_val_v3`

This is currently the best run of the post-processing layer.

Run settings:

- `num_samples = 114`
- `split = val`
- `seg_threshold = 0.5`
- `cls_threshold = 0.5`
- `final_threshold = 0.63`

Results:

| Evaluation level | Accuracy | Macro-F1 | Balanced Acc | Neg Recall | Pos Recall |
|---|---:|---:|---:|---:|---:|
| Raw classifier | 0.8333 | 0.6634 | 0.6474 | 0.3684 | 0.9263 |
| Gated with mask | 0.8509 | 0.7131 | 0.7000 | 0.4737 | 0.9263 |
| Postfilter v3 | 0.8509 | 0.7258 | 0.7211 | 0.5263 | 0.9158 |

Compared with `gated_with_mask`, `v3`:

- increases `macro_f1` from `0.7131` to `0.7258`;
- increases `balanced_accuracy` from `0.7000` to `0.7211`;
- increases `negative_recall` from `0.4737` to `0.5263`;
- slightly decreases `positive_recall` from `0.9263` to `0.9158`.

This matches the objective of the task: reduce false positives while trying to preserve true positives.

### 8.3. Why `v3` Is Better Than `v2`

Inspection of the output files shows:

- `rules_config_used.json` is identical between `v2` and `v3`.
- The main difference is `final_threshold`:
  - `v2 = 0.50`
  - `v3 = 0.63`

That means:

- `v2` did not change the final decision relative to `gated_with_mask`;
- `v3` raises the final decision threshold, which removes some weaker positives.

In other words, the improvement in `v3` mainly comes from:

- the rule engine producing `final_shadow_confidence`;
- a better-tuned final threshold.

### 8.4. Artifact Distribution in `resnet34_postfilter_val_v3`

On the `114` validation samples:

- `87` are labeled as `true_posterior_acoustic_shadow`
- `11` are labeled as `posterior_enhancement`
- `12` are labeled as `none`
- `2` are labeled as `edge_shadow`
- `2` are labeled as `reverberation_like_artifact`

Meaning:

- Most samples are still considered true shadow.
- The most common artifact after `true_shadow` is `posterior_enhancement`.
- This suggests that the base model still tends to confuse brighter posterior regions with shadow in some cases.

### 8.5. Most Frequent Decision Reasons in `v3`

Top `decision_reasons`:

- `strong_model_support_guardrail`: 79 times
- `bright_posterior_region_penalty`: 53 times
- `elongated_along_depth_axis`: 46 times
- `fragmented_mask_penalty`: 12 times
- `weak_shadow_evidence_penalty`: 11 times
- `dark_region_supports_true_shadow`: 9 times

Meaning:

- The rule engine still respects the confidence of the original model, as reflected by `strong_model_support_guardrail`.
- Penalties related to `posterior enhancement` appear very frequently.
- The actual darkness feature is not yet the dominant cue in the current data, possibly because the predicted mask does not always fall on a region that is darker than its surroundings.

### 8.6. Cases Where Labels Changed After `post-filter v3`

Compared with `gated_with_mask`:

- `3` samples changed from positive to negative.
- `1` sample changed from negative to positive.

Among the 3 downgraded samples:

- 1 was a true negative that was correctly fixed;
- 2 were true positives that were incorrectly removed.

The sample promoted to positive was:

- `2400115495_002:left`

Meaning:

- `v3` improves negative recall, but it still pays the price of introducing a small number of new false negatives.
- The current trade-off is more reasonable than in previous post-filter versions.

## 9. Strengths and Limitations of the Pipeline

### 9.1. Strengths

1. The pipeline is clearly structured:
   - pretraining;
   - multitask fine-tuning;
   - inference;
   - post-filtering.

2. Splitting is done by `case/study` rather than by file:
   - this avoids leakage between views from the same patient/case.

3. The model is multitask:
   - segmentation helps localization;
   - classification helps image-level decision making.

4. The post-filter is explainable:
   - it provides an `artifact type`;
   - it provides `decision reasons`;
   - this is very useful for reporting and error analysis.

5. A sample weighting mechanism is included:
   - this reduces the dominance of studies/cases with many views.

### 9.2. Limitations

1. The shadow dataset is imbalanced:
   - there are far more positive samples than negative ones.

2. Segmentation performance is still moderate:
   - Dice is around `0.45 - 0.48`.

3. The current post-filter is not yet equally generalizable across backbones:
   - the rules currently fit `ResNet34 v3` better.

4. The intensity signal is not yet consistently strong:
   - in many cases, `darkness_delta` is not fully aligned with the expected shadow theory.

5. The rule engine is still manually tuned:
   - more error analysis is needed to refine thresholds for each backbone.

## 10. Suggested Improvements

1. Use a separate rule config for each backbone instead of a single shared rule set.

2. Tune `final_threshold` on the validation set with a clearer optimization goal:
   - prioritize `balanced_accuracy`;
   - or prioritize `negative_recall`.

3. Add more depth-aware intensity features:
   - compare darkness specifically in the posterior region behind the lesion rather than across the whole mask.

4. Try imbalance-aware loss functions for the image-level branch:
   - class-balanced BCE;
   - focal loss for the classification head.

5. Evaluate calibration:
   - verify whether `cls_score` and `final_shadow_confidence` can truly be interpreted as probabilities.

6. Build an FP/FN analysis table by artifact type:
   - edge shadow;
   - posterior enhancement;
   - reverberation.

## 11. Conclusion

The current pipeline is a multi-stage shadow detection system:

- stage 1 learns ultrasound-domain knowledge through lesion classification;
- stage 2 fine-tunes a multitask model for shadow segmentation and shadow classification;
- stage 3 exports predictions for qualitative analysis;
- stage 4 applies rule-based post-processing to reduce false positives.

Among the currently available results:

- `SegFormer-B0` is strongest for segmentation;
- `EfficientNet-B0` is more balanced in raw image-level classification;
- `ResNet34 + postfilter v3` is currently the best post-processing configuration.

If written as a short conclusion for a report, it can be summarized as:

> The system shows that combining domain-specific encoder pretraining, multitask learning, and artifact-aware post-filtering can improve the reliability of posterior acoustic shadow detection, especially in reducing false positives on the validation set.
