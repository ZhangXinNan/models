#encoding=utf8
import tensorflow as tf
import numpy as np
# from classify_image import NodeLookup
import os

# pb_file = '/Users/zhangxin/data_autohome/car3_model_pb/car3_freeze.pb'
pb_file = '/Users/zhangxin/data_autohome/ah_拍照识车/v4_3100_81.53_model_pb/v4_freeze.pb'

def init_graph(model_name=pb_file):
  with open(model_name, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')
 

# name = 'InceptionV3/Predictions/Reshape_1:0'
name = 'InceptionV4/Logits/Predictions:0'
def run_inference_on_image(file_name,node_lookup,sess):
  image_data = open(file_name, 'rb').read()
  softmax_tensor = sess.graph.get_tensor_by_name(name)
  predictions = sess.run(softmax_tensor,
                           {'input:0': image_data})
  print(predictions)
  predictions = np.squeeze(predictions)
  print predictions.shape
  return predictions
  # Creates node ID --> English string lookup.
#   node_lookup = node_lookup
#   top_k = predictions.argsort()[-12:][::-1]
#   top_names = []
#   #for node_id in top_k:
#   #  human_string = node_lookup.id_to_string(node_id)
#   #  top_names.append(human_string)
#   #  score = predictions[node_id]
#   #  print('id:[%d] name:[%s] (score = %.5f)' % (node_id, human_string, score))
#   #return predictions, top_k, top_names
#   print(file_name,node_lookup.id_to_string(top_k[0]))
#   return node_lookup.id_to_string(top_k[0])
 
#file_name = '/home/gcnan604/devdata/hdwei/TFExamples/plantSeedlings/train_convert/Common-Chickweed/0c25871d9.png'
# tempfile = '/Users/zhangxin/data_autohome/car_photos/1.8.871/710.1.6338.jpg'
tempfile = '/Users/zhangxin/pic/car.jpg'
 
# label_file, _ = os.path.splitext('my_freeze.pb')
# label_file = label_file + '.label'
# node_lookup = NodeLookup(label_file)
sess = tf.Session()
init_graph()
# with open('submission.scv','a+') as f:
#   for test_file in os.listdir(test_dir):
#     tempfile = os.path.join(test_dir,test_file)
node_lookup = {}
predictions = run_inference_on_image(tempfile,node_lookup,sess)
print tempfile, predictions
