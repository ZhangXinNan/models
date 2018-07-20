# DATASET_DIR=/Volumes/zhangxinExF/data_autohome/autoimg1
# TRAIN_DIR=/Volumes/zhangxinExF/data_autohome/autoimg1_tfrecord
PRETRAINED_CHECKPOINT_DIR=/Users/zhangxin/data_public/googlenet

DATASET_NAME=auto
DATASET_DIR=/Users/zhangxin/data_autohome/

# convert the dataset
python download_and_convert_data.py \
  --dataset_name=${DATASET_NAME} \
  --dataset_dir=${DATASET_DIR}

# Fine-tune only the new layers for 1000 steps.
TRAIN_DIR=/Users/zhangxin/data_autohome/car3_model
DATASET_DIR=/Users/zhangxin/data_autohome/car_tfrecord
python train_image_classifier.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=${DATASET_NAME} \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_v4 \
  --checkpoint_path=${PRETRAINED_CHECKPOINT_DIR}/inception_v4.ckpt \
  --checkpoint_exclude_scopes=InceptionV4/Logits,InceptionV4/AuxLogits \
  --trainable_scopes=InceptionV4/Logits,InceptionV4/AuxLogits \
  --max_number_of_steps=1000 \
  --batch_size=32 \
  --learning_rate=0.01 \
  --learning_rate_decay_type=fixed \
  --save_interval_secs=60 \
  --save_summaries_secs=60 \
  --log_every_n_steps=100 \
  --optimizer=rmsprop \
  --weight_decay=0.00004 \
  --clone_on_cpu=True

DATASET_NAME=auto
TRAIN_DIR=/Users/zhangxin/data_autohome/car3_model
DATASET_DIR=/Users/zhangxin/data_autohome/car_tfrecord
python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR} \
  --eval_dir=${TRAIN_DIR} \
  --dataset_name=${DATASET_NAME} \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_v4




python train_image_classifier.py \
  --train_dir=${TRAIN_DIR}/all \
  --dataset_name=flowers \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_v4 \
  --checkpoint_path=${TRAIN_DIR} \
  --max_number_of_steps=500 \
  --batch_size=32 \
  --learning_rate=0.0001 \
  --learning_rate_decay_type=fixed \
  --save_interval_secs=60 \
  --save_summaries_secs=60 \
  --log_every_n_steps=10 \
  --optimizer=rmsprop \
  --weight_decay=0.00004



python eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR}/all \
  --eval_dir=${TRAIN_DIR}/all \
  --dataset_name=flowers \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_v4


# car3
PB_FILE=/Users/zhangxin/data_autohome/car3_model_pb/car3_auto_v4.pb
python -u export_inference_graph_auto.py \
--model_name=inception_v4 \
--output_file=${PB_FILE} \
--dataset_name=${DATASET_NAME} \
--dataset_dir=${DATASET_DIR}

python -u /Users/zhangxin/github/tensorflow/tensorflow/python/tools/freeze_graph.py \
    --input_graph=${PB_FILE} \
    --input_checkpoint=/Users/zhangxin/data_autohome/car3_model/model.ckpt-1000 \
    --output_graph=car3_freeze.pb \
    --input_binary=True \
    --output_node_name=InceptionV4/Logits/Predictions



# 2836类模型导出
PB_FILE=/Users/zhangxin/data_autohome/ah_拍照识车/v4_3100_81.53_model_pb/v4.pb

python -u export_inference_graph_auto.py \
  --model_name=inception_v4 \
  --output_file=${PB_FILE} \
  --num_classes=2837

python -u /Users/zhangxin/github/tensorflow/tensorflow/python/tools/freeze_graph.py \
    --input_graph=${PB_FILE} \
    --input_checkpoint=/Users/zhangxin/data_autohome/ah_拍照识车/v4_3100_81.53_model/model.ckpt-447526 \
    --output_graph=/Users/zhangxin/data_autohome/ah_拍照识车/v4_3100_81.53_model_pb/v4_freeze.pb \
    --input_binary=True \
    --output_node_name=InceptionV4/Logits/Predictions
