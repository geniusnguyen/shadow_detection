# Bao Cao Chi Tiet Bai Toan Phat Hien Bong Can Am Tren Anh Sieu Am

## 1. Muc tieu bai toan

Bai toan trong repo nay la phat hien `posterior acoustic shadow` tren anh sieu am. He thong khong chi du doan mask theo pixel ma con dua ra quyet dinh o muc anh:

- Anh nay co bong can am that su hay khong.
- Vung toi du doan la `true posterior acoustic shadow` hay chi la artifact gan giong shadow.

Pipeline duoc chia thanh 4 giai doan:

1. `lesion_classifier`: pretrain encoder tren bai toan phan loai 6 lop khoi u buong trung.
2. `shadow_classifier`: fine-tune encoder do cho bai toan `shadow segmentation + image-level shadow classification`.
3. `shadow_inference`: nap checkpoint da train va export ket qua suy luan.
4. `shadow_postfilter`: hau xu ly theo huong `artifact-aware` de giam false positive.

## 2. Tong quan kien truc he thong

Luong xu ly tong quat:

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

Y tuong trung tam cua bai toan:

- Khong train model shadow tu dau.
- Dau tien cho encoder hoc dac trung domain sieu am bang bai toan phan loai lesion.
- Sau do chuyen encoder sang bai toan shadow.
- Cuoi cung dung them mot tang logic hau xu ly dua tren hinh dang, vi tri, do toi/sang va do lien mach cua vung du doan.

## 3. Module 1: `lesion_classifier`

### 3.1. Vai tro

Day la giai doan `domain pretraining`. Muc dich khong phai phat hien shadow truc tiep, ma la lam cho backbone hoc duoc dac trung cua anh sieu am buong trung:

- texture;
- hinh thai ton thuong;
- bo cuc giai phau;
- dac trung intensity/phong xa co y nghia trong domain.

Checkpoint sinh ra o buoc nay la `best_encoder.pt`, sau do duoc nap vao `shadow_classifier`.

### 3.2. Dau vao

Du lieu duoc doc tu:

```text
data_tumor/images/
  ubi/
  udac/
  unangdathuy/
  unangdathuydac/
  unangdonthuy/
  unangdonthuydac/
```

Moi thu muc con la 1 lop classification.

### 3.3. Cac module con

#### a. `src/data/ovatus_manifest.py`

Chuc nang:

- Quet toan bo dataset classification.
- Gom anh theo `case`.
- Xu ly quy tac `abc.jpg`, `abc_1.jpg`, `abc_2.jpg`.
- Tach anh rong 2 panel thanh `left/right` neu can.
- Tao manifest va split `train/val` theo `case` de tranh leak.

Dau vao:

- `data.root`
- `split_wide_if_no_views`
- `full_image_aspect_ratio_threshold`
- `extensions`

Dau ra:

- `manifest.csv`
- `train_split.csv`
- `val_split.csv`
- `manifest_warnings.json`
- `class_to_idx.json`

Logic chinh:

- Neu ton tai `abc_1` va/hoac `abc_2`, uu tien dung cac crop nay.
- Neu chi co `abc.jpg` va anh rong, tach thanh 2 view.
- Neu chi co 1 anh don, giu `single`.
- Moi sample duoc gan `sample_weight = 1 / num_views_in_case` de case co nhieu view khong ap dao metric/loss.

#### b. `src/data/ovatus_cls_dataset.py`

Chuc nang:

- Doc anh tu manifest.
- Crop theo `crop_box` neu sample duoc tach tu full image.
- Resize-pad ve kich thuoc vuong.
- Augmentation va normalize.

Dau vao:

- 1 dong manifest.
- `image_size`
- tham so augmentation.

Dau ra cua moi sample:

- `image`: tensor `[3, H, W]`
- `target`: nhan lop classification
- `sample_weight`
- `case_uid`
- `sample_id`

Augmentation:

- horizontal flip;
- random rotation;
- brightness/contrast jitter;
- gaussian blur.

#### c. `src/models/*.py`

Repo ho tro 3 nhom backbone:

- `ResNetClassifier`
- `EfficientNetClassifier`
- `SegFormerClassifier`

Dau vao:

- tensor anh `[B, 3, H, W]`

Dau ra:

- `logits` kich thuoc `[B, num_classes]`

Y nghia:

- `encoder`: trich xuat dac trung.
- `classifier`: linear head cho bai toan 6 lop.
- `encoder_state_dict()`: xuat rieng trong so encoder de dung lai cho bai toan shadow.

#### d. `src/engine/classification_trainer.py`

Chuc nang:

- Quan ly train/eval loop.
- Tinh class weights.
- Tinh metric view-level va case-level.
- Luu `latest.pt`, `best.pt`, `best_encoder.pt`.

Loss:

- `CrossEntropyLoss` co:
  - class weighting;
  - label smoothing;
  - sample weighting theo so view trong case.

Metric:

- `view_accuracy`
- `view_macro_f1`
- `view_balanced_accuracy`
- `case_accuracy`
- `case_macro_f1`
- `case_balanced_accuracy`

Trong do `case-level` duoc tinh bang cach trung binh logits cua cac view cung `case_uid`, sau do moi suy ra nhan case.

### 3.4. Tham so cau hinh chinh

Ba file config:

- `lesion_classifier/configs/ovatus_resnet34.yaml`
- `lesion_classifier/configs/ovatus_efficientnet_b0.yaml`
- `lesion_classifier/configs/ovatus_segformer_b0.yaml`

Gia tri dang dung:

- `image_size = 384`
- `val_fraction = 0.2`
- `random_seed = 42`
- optimizer: `AdamW`
- scheduler: `CosineAnnealingLR`
- `label_smoothing = 0.05`
- co `AMP`

### 3.5. Dau ra cua module

Moi run se tao:

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

### 3.6. Ket qua thuc te hien co

Tap pretrain sau chuan hoa co:

- `1689` sample view-level
- `1349` train sample
- `340` val sample

Ket qua tot nhat theo `case_macro_f1`:

| Backbone | Best epoch | Case Macro-F1 | Case Accuracy | Case Balanced Accuracy |
|---|---:|---:|---:|---:|
| ResNet34 | 15 | 0.7012 | 0.7443 | 0.6963 |
| EfficientNet-B0 | 17 | 0.6863 | 0.7386 | 0.6790 |
| SegFormer-B0 | 24 | 0.6907 | 0.7614 | 0.7018 |

Nhan xet:

- `ResNet34` cho `case_macro_f1` cao nhat.
- `SegFormer-B0` cho `case_accuracy` va `case_balanced_accuracy` cao nhat.
- Buoc nay tao co so de nap sang bai toan shadow downstream.

## 4. Module 2: `shadow_classifier`

### 4.1. Vai tro

Day la module trung tam cua bai toan. No giai dong thoi 2 nhiem vu:

- `segmentation`: du doan mask shadow theo pixel.
- `classification`: du doan muc anh co shadow hay khong.

No su dung encoder da pretrain tu `lesion_classifier`.

### 4.2. Dau vao

Du lieu duoc doc tu:

```text
data_shad_official/Image
data_shad_official/Label
```

Nhan duong trong `Labelme JSON` duoc su dung la `bc`.

### 4.3. Cac module con

#### a. `src/data/shadow_manifest.py`

Chuc nang:

- Doc anh va file label JSON.
- Lay cac polygon co nhan `bc`.
- Tao sample view-level.
- Tach anh full-frame 2 panel thanh `left/right`.
- Tao `image_label` o muc view.
- Split theo `study` thay vi theo file.

Dau vao:

- `image_dir`
- `label_dir`
- `positive_label = bc`
- `split_wide_views`
- `full_image_aspect_ratio_threshold`

Dau ra:

- danh sach dong manifest;
- `warnings`;
- `stats`.

Quy tac gan nhan image-level:

- Neu sample la full image: `image_label = 1` neu co it nhat 1 polygon.
- Neu sample la crop `left/right`: `image_label = 1` neu co bat ky polygon nao giao voi crop do.

Y nghia quan trong:

- Day la cach dua bai toan segmentation ve bai toan classification muc anh ma van nhat quan voi annotation polygon.

#### b. `src/data/shadow_dataset.py`

Chuc nang:

- Doc anh goc.
- Rasterize polygon thanh binary mask.
- Crop theo `crop_box` neu can.
- Resize-pad anh va mask dong bo.
- Augment cho train.
- Chuyen sang tensor.

Dau vao cua moi sample:

- 1 dong manifest co `source_path`, `polygons_json`, `crop_box`, `image_label`, `sample_weight`.

Dau ra:

- `image`: tensor `[3, H, W]`
- `mask`: tensor `[1, H, W]`
- `image_label`: scalar float 0/1
- `sample_weight`
- `study_id`
- `sample_id`

Phan augmentation:

- horizontal flip;
- random rotation;
- thay doi brightness/contrast;
- gaussian blur.

#### c. `src/models/shadow_multitask.py`

Chuc nang:

- Dinh nghia 3 kien truc multitask:
  - `ResNetShadowModel`
  - `EfficientNetShadowModel`
  - `SegFormerShadowModel`
- Moi model dung 1 encoder chung va 2 dau ra:
  - `segmentation_logits`
  - `classification_logits`

Dau vao:

- tensor anh `[B, 3, H, W]`

Dau ra:

- `segmentation_logits`: `[B, 1, H, W]`
- `classification_logits`: `[B]`

Kien truc:

1. Backbone trich xuat dac trung da muc.
2. `classification head`:
   - global pooling;
   - dropout;
   - linear ra 1 logit.
3. `segmentation head`:
   - decoder kieu U-Net nho;
   - upsample + skip connection;
   - conv 1x1 ra 1 kenh mask.

Vai tro cua tung backbone:

- `ResNet`: encoder CNN co skip feature ro rang.
- `EfficientNet`: encoder gon, hieu qua tham so.
- `SegFormer`: transformer backbone, hop voi bo cuc va context xa hon.

#### d. `load_pretrained_encoder(...)`

Chuc nang:

- Nap `best_encoder.pt` tu buoc pretrain.
- Gan vao `model.encoder`.
- Bao cao `missing_keys` va `unexpected_keys`.

Trong `shadow_resnet34_finetune`, thong tin preload hien co:

- `missing_keys = []`
- `unexpected_keys = []`

Dieu nay cho thay encoder tuong thich tot giua buoc pretrain va buoc fine-tune.

#### e. `src/engine/shadow_trainer.py`

Chuc nang:

- Train loop multitask.
- Tinh tong loss segmentation + classification.
- Evaluate segmentation va classification trong cung 1 vong lap.
- Chon `best.pt` theo metric monitor.

Cong thuc loss:

1. Segmentation loss:

```text
seg_loss = BCEWithLogits(seg_logits, mask) + DiceLoss(seg_logits, mask)
```

2. Classification loss:

```text
cls_loss = BCEWithLogits(cls_logits, image_label)
```

3. Tong loss:

```text
total_loss = sample_weight * (seg_loss_weight * seg_loss + cls_loss_weight * cls_loss)
```

Noi bat:

- `sample_weight_by_study = true`: study co nhieu view/frame se khong chiem uu the qua lon.
- `AMP`
- `grad_clip_norm`
- `CosineAnnealingLR`

### 4.4. Dau vao va dau ra cua script train

Script chinh:

- `shadow_classifier/scripts/train_shadow_multitask.py`

Dau vao:

- file config YAML;
- du lieu anh + label;
- checkpoint encoder tu buoc pretrain.

Dau ra:

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

### 4.5. Thong so cau hinh chinh

Ba file config:

- `shadow_classifier/configs/shadow_resnet34.yaml`
- `shadow_classifier/configs/shadow_efficientnet_b0.yaml`
- `shadow_classifier/configs/shadow_segformer_b0.yaml`

Thong so chung:

- `image_size = 384`
- `val_fraction = 0.2`
- `random_seed = 42`
- `dropout = 0.2`
- `epochs = 40`
- optimizer: `AdamW`
- scheduler: `cosine`
- `seg_loss_weight = 1.0`
- `cls_loss_weight = 1.0`

Khac nhau chinh:

- ResNet34:
  - `batch_size = 16`
  - `lr = 3e-4`
- EfficientNet-B0:
  - `batch_size = 12`
  - `lr = 2e-4`
- SegFormer-B0:
  - `batch_size = 8`
  - `lr = 2e-4`

### 4.6. Du lieu sau khi tao manifest

Theo `shadow_classifier/outputs/shadow_resnet34_finetune/manifest_stats.json`:

- `331` anh goc
- `662` sample view-level sau khi tach `left/right`
- `215` study
- `546` positive sample
- `116` negative sample

Tap validation hien co:

- `114` sample
- `43` study

Nhan xet:

- Bai toan bi mat can bang ro rang, vi so sample duong tinh nhieu hon sample am tinh.
- Vi vay metric nhu `balanced_accuracy`, `macro_f1`, `negative_recall` rat quan trong, khong nen chi nhin `accuracy`.

### 4.7. Ket qua fine-tune hien co

Checkpoint duoc chon theo `seg_dice`.

| Backbone | Best epoch | Seg Dice | Seg IoU | Cls Accuracy | Cls Macro-F1 | Cls Balanced Acc | Neg Recall | Pos Recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ResNet34 | 34 | 0.4554 | 0.3493 | 0.8333 | 0.6634 | 0.6474 | 0.3684 | 0.9263 |
| EfficientNet-B0 | 31 | 0.4661 | 0.3593 | 0.8158 | 0.6878 | 0.7000 | 0.5263 | 0.8737 |
| SegFormer-B0 | 35 | 0.4761 | 0.3685 | 0.7982 | 0.6580 | 0.6684 | 0.4737 | 0.8632 |

Nhan xet:

- `SegFormer-B0` cho segmentation tot nhat.
- `EfficientNet-B0` cho `classification macro_f1` va `balanced_accuracy` tot nhat.
- `ResNet34` cho `accuracy` cao nhat nhung `negative_recall` thap, nghia la van co xu huong goi nham am tinh thanh duong tinh.

## 5. Module 3: `shadow_inference`

### 5.1. Vai tro

Module nay tach rieng suy luan khoi code train. Muc dich:

- nap `best.pt`;
- chay tren `train/val/all`;
- xuat ket qua thanh anh va CSV de phan tich loi.

Script chinh:

- `shadow_inference/scripts/export_shadow_predictions.py`

### 5.2. Dau vao

Tham so dau vao:

- `--checkpoint`: duong dan den `best.pt`
- `--manifest`: tuy chon, neu bo trong thi tu tim `manifest.csv`
- `--output-dir`
- `--split`: `all/train/val`
- `--max-samples`
- `--seg-threshold`
- `--cls-threshold`
- `--device`

Dau vao logic cua moi sample:

- 1 dong trong manifest;
- anh goc;
- polygon ground truth;
- thong so transform tu config da save trong checkpoint.

### 5.3. Xu ly

Luong xu ly:

1. Tai checkpoint va config.
2. Build lai model dung kien truc.
3. Load sample, rasterize mask ground truth.
4. Chay model.
5. Lay:
   - `seg_prob = sigmoid(segmentation_logits)`
   - `cls_score = sigmoid(classification_logits)`
6. Threshold:
   - `pred_mask = seg_prob >= seg_threshold`
   - `pred_image_label = cls_score >= cls_threshold`
7. Luu ket qua.

### 5.4. Dau ra

Cap run:

- `run_info.json`
- `predictions.csv`

Cap sample:

- `input.png`
- `gt_mask.png`
- `pred_prob.png`
- `pred_mask.png`
- `overlay.png`
- `summary.json`

Y nghia tung file:

- `pred_prob.png`: xac suat shadow theo pixel.
- `pred_mask.png`: mask sau threshold.
- `overlay.png`: do mask du doan len anh goc.
- `summary.json`: tom tat thong tin image-level va mask area.

## 6. Module 4: `shadow_postfilter`

### 6.1. Vai tro

Day la buoc hau xu ly khong train them model moi. Muc tieu:

- dung `cls_score`, `seg_prob`, `pred_mask`;
- trich dac trung hinh hoc va cuong do;
- ap rule de phan biet:
  - `true_posterior_acoustic_shadow`
  - `edge_shadow`
  - `posterior_enhancement`
  - `reverberation_like_artifact`
  - `none`

Muc dich cuoi cung la giam false positive.

### 6.2. Cac module con

#### a. `spf/io_utils.py`

Chuc nang:

- doc checkpoint;
- tim manifest;
- build model tu checkpoint;
- doc sample de inference;
- denormalize anh;
- save JSON/overlay.

Day la layer tien ich de script post-filter khong lap lai code inference.

#### b. `spf/feature_extractor.py`

Chuc nang:

- trich dac trung tu `image_rgb`, `seg_prob`, `pred_mask`.

Dau vao:

- anh RGB da denormalize;
- xac suat segmentation;
- mask sau threshold.

Dau ra:

- dictionary `features`.

Bo feature chinh:

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

Y nghia:

- Nhom hinh hoc: cho biet mask co dai theo truc sau, mong, bam bien, lien mach hay khong.
- Nhom intensity: cho biet vung du doan toi hon hay sang hon xung quanh.
- Nhom xac suat: cho biet model co that su tu tin vao mask hay chi la vung xac suat yeu.

Luu y:

- `darkness_delta = grayscale_outside_mean - grayscale_inside_mean`
- Gia tri duong nghia la vung trong mask toi hon ben ngoai.
- Gia tri am nghia la vung trong mask sang hon ben ngoai, de nghi nghieng ve `posterior enhancement`.

#### c. `spf/rule_engine.py`

Chuc nang:

- bien feature thanh cac `signal`;
- tinh diem cho tung artifact;
- tong hop thanh `final_shadow_confidence`.

Mot so signal quan trong:

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

Sau do rule engine tao cac artifact score:

- `true_posterior_acoustic_shadow`
- `edge_shadow`
- `posterior_enhancement`
- `reverberation_like_artifact`
- `none`

Va tinh:

```text
final_shadow_confidence
```

Quy tac tong quat:

- Thuong cho shadow that khi:
  - `cls_score` khong qua thap;
  - mask du lon;
  - mask lien mach;
  - xu huong keo dai theo truc sau;
  - vung du doan khong qua giong artifact canh/phan xa.
- Phat shadow gia khi:
  - mask mong, bam sat bien;
  - vung phia sau sang hon thay vi toi hon;
  - mask bi phan manh;
  - `cls_score` va `mask_strength` deu yeu.

#### d. `spf/evaluation.py`

Chuc nang:

- tinh metric binary classification tu `y_true`, `y_pred`.

Ba muc duoc bao cao:

- `raw_classifier`
- `gated_with_mask`
- `postfilter`

Y nghia:

- `raw_classifier`: chi dua vao `cls_score >= cls_threshold`.
- `gated_with_mask`: chi duong tinh neu classifier duong tinh va mask khong rong.
- `postfilter`: dua vao `final_shadow_confidence >= final_threshold`.

### 6.3. Rule config mac dinh

File:

- `shadow_postfilter/configs/default_rule_config.json`

Nguong quan trong:

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

Trong so quan trong:

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

Nhan xet:

- Rule engine van dat trong tam lon vao classifier goc.
- Dac trung mask va dark signal dong vai tro bo tro, sau do penalty giup cat false positive.

### 6.4. Mo ta artifact class

File:

- `shadow_postfilter/artifacts/artifact_descriptions.json`

5 nhom artifact:

1. `true_posterior_acoustic_shadow`
2. `edge_shadow`
3. `posterior_enhancement`
4. `reverberation_like_artifact`
5. `none`

Vai tro cua file nay:

- Chuan hoa cach dien giai artifact.
- Lam co so cho viec thiet ke rule.
- Giup phan doc bao cao de hieu tai sao mot vung bi giam diem shadow.

### 6.5. Dau vao va dau ra

Script chinh:

- `shadow_postfilter/scripts/run_artifact_postfilter.py`

Dau vao:

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

Dau ra:

- `run_info.json`
- `rules_config_used.json`
- `artifact_descriptions_used.json`
- `predictions_postfilter.csv`
- `metrics_before_after.json`
- `visuals/`

`predictions_postfilter.csv` gom:

- thong tin sample;
- `raw_cls_score`;
- `raw_cls_pred`;
- `gated_pred`;
- `final_shadow_confidence`;
- `postfilter_pred`;
- `predicted_artifact_type`;
- cac `artifact_score__*`;
- cac `signal__*`;
- cac `feature__*`;
- `decision_reasons`.

## 7. Cac tham so danh gia

### 7.1. Classification metric

#### Accuracy

```text
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

Dung de do ti le du doan dung tong the, nhung de bi danh lua khi dataset mat can bang.

#### Negative Recall

```text
Negative Recall = TN / (TN + FP)
```

Rat quan trong trong bai toan nay vi muc tieu hau xu ly la giam false positive.

#### Positive Recall

```text
Positive Recall = TP / (TP + FN)
```

Do kha nang giu lai duong tinh that.

#### Balanced Accuracy

```text
Balanced Accuracy = (Negative Recall + Positive Recall) / 2
```

Phu hop hon accuracy khi du lieu lech lop.

#### Macro-F1

Tinh F1 cho tung lop roi lay trung binh. Metric nay can bang giua precision va recall cua ca lop am va duong.

### 7.2. Segmentation metric

#### Dice

```text
Dice = (2 * intersection + eps) / (pred_sum + target_sum + eps)
```

Do do chong lap giua mask du doan va mask ground truth.

#### IoU

```text
IoU = (intersection + eps) / (union + eps)
```

Chat che hon Dice, nhay voi phan du thua/thieu trong mask.

### 7.3. Metric trong post-filter

Post-filter chi danh gia o muc image-level, khong tinh lai segmentation.

So sanh 3 muc:

- `raw_classifier`
- `gated_with_mask`
- `postfilter`

Dieu nay giup biet:

- cai thien den tu classifier goc;
- cai thien den tu viec yeu cau mask khong rong;
- cai thien den tu rule engine.

## 8. Ket qua post-filter hien co

### 8.1. Ket qua chung theo tung backbone

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

Nhan xet chinh:

- Buoc `gated_with_mask` da giup ro rang cho ca 3 backbone.
- `ResNet34 v3` la ban post-filter duy nhat hien co cho ket qua tot hon `gated_with_mask`.
- `EfficientNet-B0` va `SegFormer-B0` dang bi penalty qua manh trong post-filter hien tai.

### 8.2. Phan tich chi tiet `resnet34_postfilter_val_v3`

Day la run dang tot nhat cua tang hau xu ly.

Thong so run:

- `num_samples = 114`
- `split = val`
- `seg_threshold = 0.5`
- `cls_threshold = 0.5`
- `final_threshold = 0.63`

Ket qua:

| Muc danh gia | Accuracy | Macro-F1 | Balanced Acc | Neg Recall | Pos Recall |
|---|---:|---:|---:|---:|---:|
| Raw classifier | 0.8333 | 0.6634 | 0.6474 | 0.3684 | 0.9263 |
| Gated with mask | 0.8509 | 0.7131 | 0.7000 | 0.4737 | 0.9263 |
| Postfilter v3 | 0.8509 | 0.7258 | 0.7211 | 0.5263 | 0.9158 |

So voi `gated_with_mask`, `v3`:

- tang `macro_f1` tu `0.7131` len `0.7258`;
- tang `balanced_accuracy` tu `0.7000` len `0.7211`;
- tang `negative_recall` tu `0.4737` len `0.5263`;
- giam nhe `positive_recall` tu `0.9263` xuong `0.9158`.

Dieu nay phu hop voi muc tieu bai toan: giam false positive nhung van co gang giu lai duong tinh that.

### 8.3. Vi sao `v3` tot hon `v2`

Quan sat file output cho thay:

- `rules_config_used.json` cua `v2` va `v3` giong nhau.
- Khac biet chinh la `final_threshold`:
  - `v2 = 0.50`
  - `v3 = 0.63`

Nghia la:

- `v2` chua thay doi quyet dinh sau `gated_with_mask`;
- `v3` tang nguong quyet dinh cuoi, tu do cat bot mot so duong tinh yeu.

Noi cach khac, `v3` cai thien chu yeu nho:

- rule engine tao `final_shadow_confidence`;
- threshold cuoi duoc tune hop ly hon.

### 8.4. Phan bo artifact trong `resnet34_postfilter_val_v3`

Tren `114` sample validation:

- `87` duoc gan `true_posterior_acoustic_shadow`
- `11` duoc gan `posterior_enhancement`
- `12` duoc gan `none`
- `2` duoc gan `edge_shadow`
- `2` duoc gan `reverberation_like_artifact`

Y nghia:

- Da so sample van duoc xem la shadow that.
- Artifact pho bien nhat sau `true_shadow` la `posterior_enhancement`.
- Day la dau hieu cho thay mo hinh goc van co xu huong nham vung sang phia sau thanh shadow trong mot so truong hop.

### 8.5. Cac ly do quyet dinh xuat hien nhieu nhat trong `v3`

Top `decision_reasons`:

- `strong_model_support_guardrail`: 79 lan
- `bright_posterior_region_penalty`: 53 lan
- `elongated_along_depth_axis`: 46 lan
- `fragmented_mask_penalty`: 12 lan
- `weak_shadow_evidence_penalty`: 11 lan
- `dark_region_supports_true_shadow`: 9 lan

Y nghia:

- Rule engine van ton trong su tu tin cua model goc, the hien qua `strong_model_support_guardrail`.
- Penalty lien quan den `posterior enhancement` xuat hien rat nhieu.
- Dac trung `do toi that su` chua phai yeu to thong tri trong du lieu hien tai, co the vi mask du doan khong luon nam tren vung toi hon xung quanh.

### 8.6. Cac truong hop thay doi nhan sau post-filter `v3`

So voi `gated_with_mask`:

- `3` sample bi doi tu duong tinh sang am tinh.
- `1` sample bi doi tu am tinh sang duong tinh.

Trong 3 sample bi doi xuong:

- co 1 sample am tinh that duoc sua dung;
- 2 sample duong tinh that bi loai nham.

Sample duoc keo len duong tinh:

- `2400115495_002:left`

Y nghia:

- `v3` cai thien negative recall nhung van tra gia bang mot it false negative moi.
- Trade-off hien tai la hop ly hon cac ban post-filter cu.

## 9. Phan tich diem manh va han che cua pipeline

### 9.1. Diem manh

1. Pipeline co tinh he thong ro rang:
   - pretrain;
   - fine-tune multitask;
   - inference;
   - post-filter.

2. Split theo `case/study` thay vi theo file:
   - tranh leak giua cac view cung mot benh nhan/ca.

3. Mo hinh multitask:
   - segmentation giup dinh vi;
   - classification giup quyet dinh muc anh.

4. Post-filter giai thich duoc:
   - co `artifact type`;
   - co `decision reasons`;
   - de viet bao cao va phan tich loi.

5. Co co che sample weighting:
   - giam anh huong cua study/case co qua nhieu view.

### 9.2. Han che

1. Du lieu shadow bi lech lop:
   - positive rat nhieu so voi negative.

2. Segmentation hien tai chua cao:
   - Dice khoang `0.45 - 0.48`.

3. Post-filter hien tai chua tong quat cho moi backbone:
   - rule dang hop voi `ResNet34 v3` hon.

4. Signal intensity chua that su manh:
   - trong nhieu mau, `darkness_delta` khong dong nhat voi ly thuyet shadow.

5. Rule engine con duoc tune thu cong:
   - can them error analysis de dieu chinh threshold cho tung backbone.

## 10. De xuat huong cai tien

1. Tinh rieng rule config cho tung backbone thay vi dung 1 bo rule chung.

2. Tuning `final_threshold` tren tap validation theo muc tieu ro rang:
   - uu tien `balanced_accuracy`;
   - hoac uu tien `negative_recall`.

3. Bo sung feature intensity theo chieu sau:
   - so sanh do toi trong vung phia sau lesion thay vi toan mask chung.

4. Thu loss bat can bang cho image-level:
   - class-balanced BCE;
   - focal loss cho head classification.

5. Danh gia them calibration:
   - xem `cls_score` va `final_shadow_confidence` co duoc hieu dung la xac suat hay khong.

6. Xay dung bang loi FP/FN theo artifact:
   - edge shadow;
   - posterior enhancement;
   - reverberation.

## 11. Ket luan

Pipeline hien tai la mot he thong phat hien shadow nhieu tang:

- tang 1 hoc domain sieu am qua classification lesion;
- tang 2 fine-tune multitask cho shadow segmentation va shadow classification;
- tang 3 export de phan tich du doan;
- tang 4 hau xu ly rule-based de giam false positive.

Trong cac ket qua hien co:

- `SegFormer-B0` manh nhat o segmentation;
- `EfficientNet-B0` can bang hon o image classification goc;
- `ResNet34 + postfilter v3` la cau hinh hau xu ly hieu qua nhat hien tai.

Neu viet thanh ket luan ngan cho bao cao, co the tom tat:

> He thong da cho thay viec ket hop encoder pretraining theo domain, multitask learning va artifact-aware post-filter co kha nang nang cao do tin cay cua bai toan phat hien posterior acoustic shadow, dac biet o kha nang giam false positive tren tap validation.
