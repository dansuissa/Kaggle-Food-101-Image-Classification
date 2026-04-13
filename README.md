# Food Image Classification with ConvNeXt Ensembles

This repository contains my notebook for a multiclass food image classification project built around pretrained ConvNeXt models and out-of-fold ensembling. The goal was to classify food images into 10 categories using a pipeline that stays simple at the model-training level, but becomes much stronger through careful validation, test-time augmentation, and ensemble design.

The notebook ranked **1st in my school submission for this project**.

## Project overview

The work starts from a balanced 10-class image dataset and builds a full training and inference pipeline in a single notebook. Rather than training a vision model from scratch, I used transfer learning with pretrained ConvNeXt backbones and trained lightweight classification heads on top of frozen features. From there, I improved performance step by step through cross-validation, stronger test-time augmentation, probability blending, and stacking.

The notebook is designed to be read as a complete experiment. It includes data preparation, path cleaning, fold construction, model training, checkpointing, out-of-fold prediction generation, ensemble search, and final submission creation.

## What the notebook does

The first part of the notebook focuses on making the dataset reliable before training. It cleans and normalizes image paths, builds label mappings, checks that files exist, creates stratified 3-fold splits, and inspects the image size distribution. This ensures that the rest of the pipeline is reproducible and that model comparisons are fair.

The modeling stage uses two pretrained backbones from `timm`: **ConvNeXt Tiny** and **ConvNeXt Small**. In both cases, the backbone is frozen and only the final classifier head is trained on the project data. This gives a fast and stable transfer-learning setup while still adapting the model to the 10 food classes. Training is done with standard image augmentations, mixed precision, and fold-wise validation.

After training the base models, the notebook builds **out-of-fold predictions** across the full training set. These OOF predictions are central to the project because they provide an honest estimate of performance and make it possible to build ensembles without leakage.

The next stage applies **strong test-time augmentation** by averaging predictions over multiple deterministic crops and horizontal flips. On top of that, the notebook explores two ensemble strategies. The first is direct probability blending, including both linear and geometric combinations. The second is an OOF-safe **stacking** stage, where a logistic regression meta-model is trained on the base-model probability outputs to learn how to combine them.

The final part of the notebook produces competition-style submissions, including a quota-constrained assignment step that uses the balanced structure of the dataset to improve the final prediction set.

## Main ideas behind the approach

A central choice in this project was to rely on the quality of pretrained visual features instead of spending most of the effort on heavy fine-tuning. The strongest gains did not come from changing the backbone dramatically, but from improving robustness and combining complementary predictors well.

The notebook shows that even when two single models are already strong, their combination can still yield a significant improvement if they make different mistakes. This is why the ensemble stage became the most important part of the project.

## Results

Both ConvNeXt Tiny and ConvNeXt Small performed strongly as individual models under 3-fold cross-validation. Stronger test-time augmentation improved both of them further. The largest gain came from blending the Tiny and Small model families, which outperformed either single model alone. Stacking also helped, although the best simple blend remained extremely competitive. The final combined system produced the best overall results in the notebook.
