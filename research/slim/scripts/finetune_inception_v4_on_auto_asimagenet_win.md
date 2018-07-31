# 1. build_imagenet_data_auto.py(参考build_imagenet_data.py)

```
$DATASET_DIR='D:/data_autohome/auto_tfrecord'
python build_imagenet_data_auto.py `
    --train_directory D:/data_autohome/autoimg1_det_yolo3_augm `
    --validation_directory D:/data_autohome/test_seriesdir_det `
    --output_directory $DATASET_DIR `
    --labels_file D:/data_autohome/autohome2836/labels.txt `
    --imagenet_metadata_file D:/data_autohome/autohome2836/metadata.txt `
    --num_threads 8
# 从训练数据中生成验证集，每类10张
$DATASET_DIR='D:/data_autohome/autoimg1_det_yolo3_augm10_tfrecord'
python build_imagenet_data_auto.py `
    --train_directory D:/data_autohome/autoimg1_det_yolo3_augm10 `
    --validation_directory D:/data_autohome/autoimg1_det_yolo3_augm10 `
    --output_directory $DATASET_DIR `
    --labels_file D:/data_autohome/autohome2836/labels.txt `
    --imagenet_metadata_file D:/data_autohome/autohome2836/metadata.txt `
    --num_threads 8
```

# 2. 新建imagenet_auto.py (参考imagenet.py)

# 3. 新建 dataset_factory_auto.py (参考dataset_factory.py)
并在其中添加
```
from datasets import imagenet_auto

datasets_map = {
    'cifar10': cifar10,
    'flowers': flowers,
    'imagenet': imagenet,
    'mnist': mnist,
    'auto': auto,
    'imagenet_flowers': imagenet_flowers,
    'imagenet_auto': imagenet_auto,
}
```


# 4. 训练最后一层
不需要 把labels.txt放到$DATASET_DIR中
```
# Fine-tune only the new layers for 1000 steps.

$DATASET_DIR='D:/data_autohome/auto_tfrecord'
$TRAIN_DIR='D:/data_autohome/auto_models_v4_imagenet'
$PRETRAINED_CHECKPOINT_DIR='d:/data_public/googlenet'

python train_image_classifier_auto.py `
  --train_dir=${TRAIN_DIR} `
  --dataset_name=imagenet_auto `
  --dataset_split_name=train `
  --dataset_dir=${DATASET_DIR} `
  --model_name=inception_v4 `
  --checkpoint_path=${PRETRAINED_CHECKPOINT_DIR}/inception_v4.ckpt `
  --checkpoint_exclude_scopes=InceptionV4/Logits,InceptionV4/AuxLogits `
  --trainable_scopes=InceptionV4/Logits,InceptionV4/AuxLogits `
  --max_number_of_steps=100000 `
  --batch_size=32 `
  --learning_rate=0.01 `
  --learning_rate_decay_type=fixed `
  --save_interval_secs=60 `
  --save_summaries_secs=60 `
  --log_every_n_steps=100 `
  --optimizer=rmsprop `
  --weight_decay=0.00004
```

## 使用周晖训练模型 进行FT
```
$DATASET_DIR='D:/data_autohome/auto_tfrecord'
$TRAIN_DIR='D:/data_autohome/auto_models_v4_ftzh'
$TRAIN_DIR_ZHOUHUI='D:/data_autohome/zhouhui/v4_3100_81.53'

python train_image_classifier_auto.py `
  --train_dir=${TRAIN_DIR} `
  --dataset_name=imagenet_auto `
  --dataset_split_name=train `
  --dataset_dir=${DATASET_DIR} `
  --model_name=inception_v4 `
  --checkpoint_path=${TRAIN_DIR_ZHOUHUI}/model.ckpt-447526 `
  --checkpoint_exclude_scopes=InceptionV4/Logits,InceptionV4/AuxLogits `
  --trainable_scopes=InceptionV4/Logits,InceptionV4/AuxLogits `
  --max_number_of_steps=500000 `
  --batch_size=32 `
  --learning_rate=0.001 `
  --learning_rate_decay_type=fixed `
  --save_interval_secs=600 `
  --save_summaries_secs=600 `
  --log_every_n_steps=1000 `
  --optimizer=rmsprop `
  --weight_decay=0.00004
```

tensorboard
```
tensorboard --logdir=./
```

### 出错
```
2018-07-30 07:37:04.348966: I T:\src\github\tensorflow\tensorflow\core\common_runtime\gpu\gpu_device.cc:1084] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 6399 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1070, pci bus id: 0000:01:00.0, compute capability: 6.1)
INFO:tensorflow:Error reported to Coordinator: <class 'ValueError'>, Can't load save_path when it is None.
Traceback (most recent call last):
  File "train_image_classifier_auto.py", line 583, in <module>
    tf.app.run()
  File "D:\Users\zhangxin\Anaconda3\envs\tensorflow_py3.5\lib\site-packages\tensorflow\python\platform\app.py", line 125, in run
    _sys.exit(main(argv))
  File "train_image_classifier_auto.py", line 579, in main
    session_config=session_config)
  File "D:\Users\zhangxin\Anaconda3\envs\tensorflow_py3.5\lib\site-packages\tensorflow\contrib\slim\python\slim\learning.py", line 748, in train
    master, start_standard_services=False, config=session_config) as sess:
  File "D:\Users\zhangxin\Anaconda3\envs\tensorflow_py3.5\lib\contextlib.py", line 59, in __enter__
    return next(self.gen)
  File "D:\Users\zhangxin\Anaconda3\envs\tensorflow_py3.5\lib\site-packages\tensorflow\python\training\supervisor.py", line 1005, in managed_session
    self.stop(close_summary_writer=close_summary_writer)
  File "D:\Users\zhangxin\Anaconda3\envs\tensorflow_py3.5\lib\site-packages\tensorflow\python\training\supervisor.py", line 833, in stop
    ignore_live_threads=ignore_live_threads)
  File "D:\Users\zhangxin\Anaconda3\envs\tensorflow_py3.5\lib\site-packages\tensorflow\python\training\coordinator.py", line 389, in join
    six.reraise(*self._exc_info_to_raise)
  File "D:\Users\zhangxin\Anaconda3\envs\tensorflow_py3.5\lib\site-packages\six.py", line 693, in reraise
    raise value
  File "D:\Users\zhangxin\Anaconda3\envs\tensorflow_py3.5\lib\site-packages\tensorflow\python\training\supervisor.py", line 994, in managed_session
    start_standard_services=start_standard_services)
  File "D:\Users\zhangxin\Anaconda3\envs\tensorflow_py3.5\lib\site-packages\tensorflow\python\training\supervisor.py", line 731, in prepare_or_wait_for_session
    init_feed_dict=self._init_feed_dict, init_fn=self._init_fn)
  File "D:\Users\zhangxin\Anaconda3\envs\tensorflow_py3.5\lib\site-packages\tensorflow\python\training\session_manager.py", line 289, in prepare_session
    init_fn(sess)
  File "D:\Users\zhangxin\Anaconda3\envs\tensorflow_py3.5\lib\site-packages\tensorflow\contrib\framework\python\ops\variables.py", line 697, in callback
    saver.restore(session, model_path)
  File "D:\Users\zhangxin\Anaconda3\envs\tensorflow_py3.5\lib\site-packages\tensorflow\python\training\saver.py", line 1745, in restore
    raise ValueError("Can't load save_path when it is None.")
ValueError: Can't load save_path when it is None.
```
解决方法 ：
```
 --checkpoint_path=${TRAIN_DIR_ZHOUHUI}
--》》
 --checkpoint_path=${TRAIN_DIR_ZHOUHUI}/model.ckpt-447526 
```

# 5 Run evaluation.

```
$DATASET_DIR='D:/data_autohome/auto_tfrecord'
$TRAIN_DIR='D:/data_autohome/auto_models_v4_imagenet'
python eval_image_classifier_auto.py `
  --checkpoint_path=${TRAIN_DIR} `
  --eval_dir=${TRAIN_DIR} `
  --dataset_name=imagenet_auto `
  --dataset_split_name=validation `
  --dataset_dir=${DATASET_DIR} `
  --model_name=inception_v4

# ft zhouhui
$DATASET_DIR='D:/data_autohome/auto_tfrecord'
$DATASET_DIR='D:/data_autohome/autoimg1_det_yolo3_augm10_tfrecord'
$TRAIN_DIR='D:/data_autohome/auto_models_v4_ftzh'

python eval_image_classifier_auto.py `
  --checkpoint_path=${TRAIN_DIR} `
  --eval_dir=${TRAIN_DIR} `
  --dataset_name=imagenet_auto `
  --dataset_split_name=validation `
  --dataset_dir=${DATASET_DIR} `
  --model_name=inception_v4
```


# 6 Fine-tune all the new layers for 500 steps.
```
$DATASET_DIR='D:/data_autohome/auto_tfrecord'
$TRAIN_DIR='D:/data_autohome/auto_models_v4_ftzh'
python train_image_classifier_auto.py `
  --train_dir=${TRAIN_DIR}/all `
  --dataset_name=imagenet_auto `
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
  --dataset_name=imagenet_auto `
  --dataset_split_name=validation `
  --dataset_dir=${DATASET_DIR} `
  --model_name=inception_v4

# 
# 
```