import tensorflow as tf
from absl import app, flags, logging
from absl.flags import FLAGS
from core.yolov4 import YOLOv4, YOLOv3, YOLOv3_tiny, decode, anno
import core.utils as utils
from core.config import cfg

flags.DEFINE_string('weights', '', 'path to weights file')
flags.DEFINE_string('output', './output/dabbox3.h5', 'path to output')
flags.DEFINE_boolean('tiny', False, 'is yolov3-tiny or not')
flags.DEFINE_integer('input_size', 768, 'define input size of export model')
flags.DEFINE_string('model', 'anno', 'yolov3 or yolov4')

def save_tf():
  NUM_CLASS = len(utils.read_class_names(cfg.YOLO.CLASSES))
  input_layer = tf.keras.layers.Input([FLAGS.input_size, FLAGS.input_size, 3])
  if FLAGS.tiny:
    feature_maps = YOLOv3_tiny(input_layer, NUM_CLASS)
    bbox_tensors = []
    for i, fm in enumerate(feature_maps):
      bbox_tensor = decode(fm, NUM_CLASS, i)
      bbox_tensors.append(bbox_tensor)
    model = tf.keras.Model(input_layer, bbox_tensors)
    utils.load_weights_tiny(model, FLAGS.weights)
  else:
    if FLAGS.model == 'yolov3':
      feature_maps = YOLOv3(input_layer, NUM_CLASS)
      bbox_tensors = []
      for i, fm in enumerate(feature_maps):
        bbox_tensor = decode(fm, NUM_CLASS, i)
        bbox_tensors.append(bbox_tensor)
      model = tf.keras.Model(input_layer, bbox_tensors)
      utils.load_weights_v3(model, FLAGS.weights)
    elif FLAGS.model == 'yolov4':
      feature_maps = YOLOv4(input_layer, NUM_CLASS)
      bbox_tensors = []
      for i, fm in enumerate(feature_maps):
        bbox_tensor = decode(fm, NUM_CLASS, i)
        bbox_tensors.append(bbox_tensor)
      model = tf.keras.Model(input_layer, bbox_tensors)
      utils.load_weights(model, FLAGS.weights)
    elif FLAGS.model == 'anno':
      feature_maps = anno(input_layer, NUM_CLASS)
      bbox_tensors = []
      for i, fm in enumerate(feature_maps):
        bbox_tensor = decode(fm, NUM_CLASS, i)
        bbox_tensors.append(bbox_tensor)
      model = tf.keras.Model(input_layer, bbox_tensors)
      model.load_weights(FLAGS.weights)
    else:
      print("model option can be only 'yolov3' or 'yolov4'.")
      return

  model.summary()

  model.save(FLAGS.output)

def main(_argv):
  save_tf()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass

