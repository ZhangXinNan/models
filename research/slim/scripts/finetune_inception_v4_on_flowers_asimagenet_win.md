# 1. build_imagenet_data_flower.py(参考build_imagenet_data.py)
```
python build_imagenet_data_flower.py `
    --train_directory D:/data_public/slim/flower_photos_train `
    --validation_directory  D:/data_public/slim/flower_photos_val `
    --output_directory D:/data_public/slim/flower_tfrecord `
    --labels_file D:/data_public/slim/flower_photos/labels.txt `
    --imagenet_metadata_file D:/data_public/slim/flower_photos/labels_name.txt `
    --num_threads 1
```

# 2. 新建imagenet_flowers.py (参考imagenet.py)

# 3. 新建 dataset_factory_auto.py (参考dataset_factory.py)
并在其中添加
```
from datasets import imagenet_flowers

datasets_map = {
    'cifar10': cifar10,
    'flowers': flowers,
    'imagenet': imagenet,
    'mnist': mnist,
    'auto': auto,
    'imagenet_flowers': imagenet_flowers,
}
```


# 4. 训练最后一层
不需要 把labels.txt放到$DATASET_DIR中
```
$DATASET_DIR='d:/data_public/slim/flower_tfrecord'
$TRAIN_DIR='d:/data_public/slim/flowers_models_v4_imagenet'
$PRETRAINED_CHECKPOINT_DIR='d:/data_public/googlenet'
# Fine-tune only the new layers for 1000 steps.
python train_image_classifier_auto.py `
  --train_dir=${TRAIN_DIR} `
  --dataset_name=imagenet_flowers `
  --dataset_split_name=train `
  --dataset_dir=${DATASET_DIR} `
  --model_name=inception_v4 `
  --checkpoint_path=${PRETRAINED_CHECKPOINT_DIR}/inception_v4.ckpt `
  --checkpoint_exclude_scopes=InceptionV4/Logits,InceptionV4/AuxLogits `
  --trainable_scopes=InceptionV4/Logits,InceptionV4/AuxLogits `
  --max_number_of_steps=1000 `
  --batch_size=32 `
  --learning_rate=0.01 `
  --learning_rate_decay_type=fixed `
  --save_interval_secs=60 `
  --save_summaries_secs=60 `
  --log_every_n_steps=100 `
  --optimizer=rmsprop `
  --weight_decay=0.00004
```

# 5 Run evaluation.
```
python eval_image_classifier_auto.py `
  --checkpoint_path=${TRAIN_DIR} `
  --eval_dir=${TRAIN_DIR} `
  --dataset_name=imagenet_flowers `
  --dataset_split_name=validation `
  --dataset_dir=${DATASET_DIR} `
  --model_name=inception_v4

# eval/Accuracy[0.81]
# eval/Recall_5[1]
# eval/Recall_5[1]eval/Accuracy[0.74]
# eval/Accuracy[0.763333321]eval/Recall_5[1]
```


# 6 Fine-tune all the new layers for 500 steps.
```
python train_image_classifier_auto.py `
  --train_dir=${TRAIN_DIR}/all `
  --dataset_name=imagenet_flowers `
  --dataset_split_name=train `
  --dataset_dir=${DATASET_DIR} `
  --model_name=inception_v4 `
  --checkpoint_path=${TRAIN_DIR} `
  --max_number_of_steps=500 `
  --batch_size=32 `
  --learning_rate=0.0001 `
  --learning_rate_decay_type=fixed `
  --save_interval_secs=60 `
  --save_summaries_secs=60 `
  --log_every_n_steps=10 `
  --optimizer=rmsprop `
  --weight_decay=0.00004
```

# 7 Run evaluation.
```
python eval_image_classifier_auto.py `
  --checkpoint_path=${TRAIN_DIR}/all `
  --eval_dir=${TRAIN_DIR}/all `
  --dataset_name=imagenet_flowers `
  --dataset_split_name=validation `
  --dataset_dir=${DATASET_DIR} `
  --model_name=inception_v4

# eval/Accuracy[0.883333325]
# eval/Recall_5[1]
```