from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import os
import shutil
import numpy as np
import tensorflow as tf
import core.utils as utils
from core.config import cfg
from core.yolov4 import YOLOv4, YOLOv3, YOLOv3_tiny, anno, decode

flags.DEFINE_string('weights', './output/checkpoints/ckpt-XXXX',
                    'path to weights file')
flags.DEFINE_string('framework', 'tf', 'select model type in (tf, tflite)'
                    'path to weights file')
flags.DEFINE_string('model', 'anno', 'yolov3 or yolov4')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 768, 'resize images to')
flags.DEFINE_string('annotation_path', "./data/dataset/syn_patent9_val_ref_fig.txt", 'annotation path')
flags.DEFINE_string('write_image_path', "./detection/", 'write image path')

# Fix for Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR
physical_devices = tf.config.experimental.list_physical_devices('GPU')
#assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
#tf.config.experimental.set_memory_growth(physical_devices[0], True)

def main(_argv):
    INPUT_SIZE = FLAGS.size
    if FLAGS.tiny:
        STRIDES = np.array(cfg.YOLO.STRIDES_TINY)
        ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS_TINY, FLAGS.tiny)
    else:
        STRIDES = np.array(cfg.YOLO.STRIDES)
        if FLAGS.model == 'yolov4':
            ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS, FLAGS.tiny)
        elif FLAGS.model =='anno':
            ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS)
            ANCHORS = ANCHORS[:2, :, :]
        else:
            ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS_V3, FLAGS.tiny)
    NUM_CLASS = len(utils.read_class_names(cfg.YOLO.CLASSES))
    CLASSES = utils.read_class_names(cfg.YOLO.CLASSES)
    predicted_dir_path = './mAP/predicted'
    ground_truth_dir_path = './mAP/ground-truth'
    if os.path.exists(predicted_dir_path): shutil.rmtree(predicted_dir_path)
    if os.path.exists(ground_truth_dir_path): shutil.rmtree(ground_truth_dir_path)
    if os.path.exists(cfg.TEST.DECTECTED_IMAGE_PATH): shutil.rmtree(cfg.TEST.DECTECTED_IMAGE_PATH)

    os.mkdir(predicted_dir_path)
    os.mkdir(ground_truth_dir_path)
    os.mkdir(cfg.TEST.DECTECTED_IMAGE_PATH)

    # Build Model
    if FLAGS.framework == 'tf':
        input_layer = tf.keras.layers.Input([INPUT_SIZE, INPUT_SIZE, 3])
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
                #utils.load_weights(model, FLAGS.weights)
                model.load_weights(FLAGS.weights)
            elif FLAGS.model == 'anno':
                feature_maps = anno(input_layer, NUM_CLASS)
                bbox_tensors = []
                for i, fm in enumerate(feature_maps):
                    bbox_tensor = decode(fm, NUM_CLASS, i)
                    bbox_tensors.append(bbox_tensor)
                model = tf.keras.Model(input_layer, bbox_tensors)
                model.load_weights(FLAGS.weights)

    else:
        # Load TFLite model and allocate tensors.
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        # Get input and output tensors.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)

    num_lines = sum(1 for line in open(FLAGS.annotation_path))
    with open(cfg.TEST.ANNOT_PATH, 'r') as annotation_file:
        for num, line in enumerate(annotation_file):
            annotation = line.strip().split()
            image_path = annotation[0]
            image_name = image_path.split('/')[-1]
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            bboxes_gt = []
            classes_gt = []
            for box in annotation[1:]:
                bboxes_gt.append(list(map(int, box.split(',')[:4])))
                classes_gt.append(list(map(int, box.split(',')[4:])))
            bboxes_gt = np.array(bboxes_gt)

            ground_truth_path = os.path.join(ground_truth_dir_path, str(num) + '.txt')

            print('=> ground truth of %s:' % image_name)
            num_bbox_gt = len(bboxes_gt)
            with open(ground_truth_path, 'w') as f:
                for i in range(num_bbox_gt):
                    scores_gt = np.zeros(len(CLASSES))
                    scores_gt[classes_gt[i]] = 1
                    class_name = utils.post_process_prediction(CLASSES, scores_gt)
                    xmin, ymin, xmax, ymax = list(map(str, bboxes_gt[i]))
                    bbox_mess = ' '.join([class_name, xmin, ymin, xmax, ymax]) + '\n'
                    f.write(bbox_mess)
                    print('\t' + str(bbox_mess).strip())
            print('=> predict result of %s:' % image_name)
            predict_result_path = os.path.join(predicted_dir_path, str(num) + '.txt')
            # Predict Process
            image_size = image.shape[:2]
            image_data = utils.image_preprocess(np.copy(image), [INPUT_SIZE, INPUT_SIZE])
            image_data = image_data[np.newaxis, ...].astype(np.float32)

            if FLAGS.framework == "tf":
                import time
                start = time.time()
                pred_bbox = model.predict(image_data)
                print('took ' + str(time.time() - start))
            else:
                interpreter.set_tensor(input_details[0]['index'], image_data)
                interpreter.invoke()
                pred_bbox = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            if FLAGS.model == 'yolov3':
                pred_bbox = utils.postprocess_bbbox(pred_bbox, ANCHORS, STRIDES)
            #elif FLAGS.model == 'yolov4':
            else:
                XYSCALE = cfg.YOLO.XYSCALE
                pred_bbox = utils.postprocess_bbbox(pred_bbox, ANCHORS, STRIDES, XYSCALE=XYSCALE)

            bboxes, class_probs = utils.postprocess_boxes(pred_bbox, image_size, INPUT_SIZE, cfg.TEST.SCORE_THRESHOLD)
            bboxes, class_probs = utils.nms(bboxes, class_probs, cfg.TEST.IOU_THRESHOLD, method='nms')
            #bboxes.sort(key=lambda x:x[-2], reverse=True)
            #bboxes = bboxes[:FLAGS.max_bboxes]

            if cfg.TEST.DECTECTED_IMAGE_PATH is not None:
                image = utils.draw_bbox(image, bboxes, class_probs)
                cv2.imwrite(cfg.TEST.DECTECTED_IMAGE_PATH + image_name, image)

            with open(predict_result_path, 'w') as f:
                for i in range(len(bboxes)):
                    bbox = bboxes[i]
                    coor = np.array(bbox[:4], dtype=np.int32)
                    score = bbox[4]
                    class_ind = int(bbox[5])
                    class_name = utils.post_process_prediction(CLASSES, class_probs[i])
                    score = '%.4f' % score
                    xmin, ymin, xmax, ymax = list(map(str, coor))
                    bbox_mess = ' '.join([class_name, score, xmin, ymin, xmax, ymax]) + '\n'
                    f.write(bbox_mess)
                    print('\t' + str(bbox_mess).strip())
            print(num, num_lines)

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass


