from glob import glob
from src.model import *
from src.utils import *
import tensorflow as tf
from src.cityscapes import *
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler


batch_size = 12

train_dataset = get_dataset(
    './data/train_imgs/*',
    './data/train_labels/*',
    batch_size=batch_size
)

val_dataset = get_dataset(
    './data/val_imgs/*',
    './data/val_labels/*',
    batch_size=batch_size
)

loss = SparseCategoricalCrossentropy(from_logits=True)
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    model = FastSCNN(n_classes=19)
    for layer in model.layers:
        if isinstance(tf.keras.layers.Conv2D) or isinstance(tf.keras.layers.SeparableConv2D):
            layer.kernel_regularizer = tf.keras.regularizers.l2(4e-5)
    lr_scheduler = PolynomialDecay(0.045, 1000, 1e-4, 0.9)
    optimizer = tf.keras.optimizers.SGD(momentum=0.9, learning_rate=lr_scheduler)
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

tensorboard = TensorBoard(log_dir='logs', write_graph=True, update_freq='batch')
model_checkpoint = ModelCheckpoint(
    mode='min', filepath='fast_scnn_weights.h5',
    monitor='val_loss', save_best_only='True',
    save_weights_only='True', verbose=1
)

model.fit(
    train_dataset, validation_data=val_dataset, epochs=300,
    steps_per_epoch=len(glob('./data/train_imgs/*')) // batch_size,
    validation_steps=len(glob('./data/val_imgs/*')) // batch_size,
    callbacks=[tensorboard, model_checkpoint]
)