from absl import app, flags, logging
from absl.flags import FLAGS
import os
import shutil
import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision

from core.yolov4 import YOLOv4,YOLOv3, YOLOv3_tiny, anno, decode, compute_loss, decode_train
from core.dataset import Dataset
from core.config import cfg
import numpy as np
from core import utils
from core.utils import freeze_all, unfreeze_all

flags.DEFINE_string('model', 'anno', 'anno, yolov4, yolov3 or yolov3-tiny')
#flags.DEFINE_string('weights', './data/yolov4.weights', 'pretrained weights')
flags.DEFINE_string('weights', None, 'pretrained weights')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')

OUTPUT_DIR = 'output'
SUMMARY_STEPS = 10
CKPT_STEPS = 5000
USE_MIXED_PRECISION = True

def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    if USE_MIXED_PRECISION:
        mixed_precision.set_policy('mixed_float16')

    trainset = Dataset('train')
    testset = Dataset('test')
    logdir = OUTPUT_DIR
    ckpt_dir = os.path.join(OUTPUT_DIR, 'checkpoints')
    isfreeze = False
    steps_per_epoch = len(trainset)
    first_stage_epochs = cfg.TRAIN.FISRT_STAGE_EPOCHS
    second_stage_epochs = cfg.TRAIN.SECOND_STAGE_EPOCHS
    global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
    warmup_steps = cfg.TRAIN.WARMUP_STEPS
    total_steps = (first_stage_epochs + second_stage_epochs) * steps_per_epoch
    # train_steps = (first_stage_epochs + second_stage_epochs) * steps_per_period

    input_layer = tf.keras.layers.Input([cfg.TRAIN.INPUT_SIZE, cfg.TRAIN.INPUT_SIZE, 3])
    NUM_CLASS = len(utils.read_class_names(cfg.YOLO.CLASSES))
    STRIDES         = np.array(cfg.YOLO.STRIDES)
    IOU_LOSS_THRESH = cfg.YOLO.IOU_LOSS_THRESH
    XYSCALE = cfg.YOLO.XYSCALE
    ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS)

    if FLAGS.tiny:
        feature_maps = YOLOv3_tiny(input_layer, NUM_CLASS)
        bbox_tensors = []
        for i, fm in enumerate(feature_maps):
            bbox_tensor = decode_train(fm, NUM_CLASS, STRIDES, ANCHORS, i)
            bbox_tensors.append(fm)
            bbox_tensors.append(bbox_tensor)
        model = tf.keras.Model(input_layer, bbox_tensors)
    else:
        if FLAGS.model == 'yolov3':
            feature_maps = YOLOv3(input_layer, NUM_CLASS)
            bbox_tensors = []
            for i, fm in enumerate(feature_maps):
                bbox_tensor = decode_train(fm, NUM_CLASS, STRIDES, ANCHORS, i)
                bbox_tensors.append(fm)
                bbox_tensors.append(bbox_tensor)
            model = tf.keras.Model(input_layer, bbox_tensors)
        elif FLAGS.model == 'yolov4':
            feature_maps = YOLOv4(input_layer, NUM_CLASS)
            bbox_tensors = []
            for i, fm in enumerate(feature_maps):
                bbox_tensor = decode_train(fm, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
                bbox_tensors.append(fm)
                bbox_tensors.append(bbox_tensor)
            model = tf.keras.Model(input_layer, bbox_tensors)
        elif FLAGS.model == 'anno':
            feature_maps = anno(input_layer, NUM_CLASS)
            bbox_tensors = []
            for i, fm in enumerate(feature_maps):
                bbox_tensor = decode_train(fm, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
                bbox_tensors.append(fm)
                bbox_tensors.append(bbox_tensor)
            model = tf.keras.Model(input_layer, bbox_tensors)

    latest = tf.train.latest_checkpoint(ckpt_dir)
    if latest is not None:
        print('Loading checkpoint ' + latest)
        model.load_weights(latest)
        global_steps.assign(int(os.path.basename(latest).split('-')[-1]))
    elif FLAGS.weights == None:
        print("Training from scratch")
    else:
        if FLAGS.weights.split(".")[len(FLAGS.weights.split(".")) - 1] == "weights":
            if FLAGS.tiny:
                utils.load_weights_tiny(model, FLAGS.weights)
            else:
                if FLAGS.model == 'yolov3':
                    utils.load_weights_v3(model, FLAGS.weights)
                else:
                    utils.load_weights(model, FLAGS.weights)
        else:
            model.load_weights(FLAGS.weights)
        print('Restoring weights from: %s ... ' % FLAGS.weights)


    optimizer = tf.keras.optimizers.Adam(epsilon=1e-4)
    if USE_MIXED_PRECISION:
        optimizer = mixed_precision.LossScaleOptimizer(optimizer, loss_scale='dynamic')

    #if os.path.exists(logdir): shutil.rmtree(logdir)
    writer = tf.summary.create_file_writer(logdir)

    def train_step(image_data, target, write_summary):
        with tf.GradientTape() as tape:

            # update learning rate
            if global_steps < warmup_steps:
                lr = global_steps / warmup_steps * cfg.TRAIN.LR_INIT
            else:
                lr = cfg.TRAIN.LR_END + 0.5 * (cfg.TRAIN.LR_INIT - cfg.TRAIN.LR_END) * (
                    (1 + tf.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi))
                )
            optimizer.lr.assign(lr.numpy())

            pred_result = model(image_data, training=True)
            giou_loss = conf_loss = prob_loss = 0

            # optimizing process
            #for i in range(3):
            for i in range(len(cfg.YOLO.STRIDES)):
                conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]
                loss_items = compute_loss(pred, conv, target[i][0], target[i][1], STRIDES=STRIDES, NUM_CLASS=NUM_CLASS, IOU_LOSS_THRESH=IOU_LOSS_THRESH, i=i)
                giou_loss += loss_items[0]
                conf_loss += loss_items[1]
                prob_loss += loss_items[2]

            total_loss = giou_loss + conf_loss + prob_loss

            if USE_MIXED_PRECISION:
                scaled_loss = optimizer.get_scaled_loss(total_loss)
                scaled_grad = tape.gradient(scaled_loss, model.trainable_variables)
                gradients = optimizer.get_unscaled_gradients(scaled_grad)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            else:
                gradients = tape.gradient(total_loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            tf.print("=> STEP %4d   lr: %.6f   giou_loss: %4.2f   conf_loss: %4.2f   "
                     "prob_loss: %4.2f   total_loss: %4.2f" % (global_steps, optimizer.lr.numpy(),
                                                               giou_loss, conf_loss,
                                                               prob_loss, total_loss))

            global_steps.assign_add(1)

            # writing summary data
            if write_summary:
                with writer.as_default():
                    tf.summary.scalar("lr", optimizer.lr, step=global_steps)
                    tf.summary.scalar("loss/total_loss", total_loss, step=global_steps)
                    tf.summary.scalar("loss/giou_loss", giou_loss, step=global_steps)
                    tf.summary.scalar("loss/conf_loss", conf_loss, step=global_steps)
                    tf.summary.scalar("loss/prob_loss", prob_loss, step=global_steps)
                writer.flush()

    def test_step(image_data, target):
        with tf.GradientTape() as tape:
            pred_result = model(image_data, training=True)
            giou_loss = conf_loss = prob_loss = 0

            # optimizing process
            for i in range(3):
                conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]
                loss_items = compute_loss(pred, conv, target[i][0], target[i][1], STRIDES=STRIDES, NUM_CLASS=NUM_CLASS, IOU_LOSS_THRESH=IOU_LOSS_THRESH, i=i)
                giou_loss += loss_items[0]
                conf_loss += loss_items[1]
                prob_loss += loss_items[2]

            total_loss = giou_loss + conf_loss + prob_loss

            tf.print("=> TEST STEP %4d   giou_loss: %4.2f   conf_loss: %4.2f   "
                     "prob_loss: %4.2f   total_loss: %4.2f" % (global_steps, giou_loss, conf_loss,
                                                               prob_loss, total_loss))
            return total_loss

    for epoch in range(first_stage_epochs + second_stage_epochs):
        """
        if epoch < first_stage_epochs:
            if not isfreeze:
                isfreeze = True
                for name in ['conv2d_93', 'conv2d_101', 'conv2d_109']:
                    freeze = model.get_layer(name)
                    freeze_all(freeze)
        elif epoch >= first_stage_epochs:
            if isfreeze:
                isfreeze = False
                for name in ['conv2d_93', 'conv2d_101', 'conv2d_109']:
                    freeze = model.get_layer(name)
                    unfreeze_all(freeze)
        """
        for image_data, target in trainset:
            write_summary = global_steps % SUMMARY_STEPS == 0
            train_step(image_data, target, write_summary)

            if global_steps % CKPT_STEPS == 0:
                val_losses = []
                for image_data, target in testset:
                    val_losses.append(test_step(image_data, target))

                tf.print('AVG val loss: ' + str(np.mean(val_losses)))
                with writer.as_default():
                    tf.summary.scalar("val_loss/total_loss", np.mean(val_losses), step=global_steps)
                writer.flush()

                model.save_weights(os.path.join(ckpt_dir, "ckpt-" + str(int(global_steps))))

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass