from model import build_Unet
from data import load_data, tf_dataset
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

if __name__ == '__main__':
    """ seeding """
    np.random.seed(42)
    tf.random.set_random_seed(42)

    """ load dataset """
    path = "E:\Dien_AI\Segment_Citydata\data"
    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = load_data(path)
    print(f"Dataset: Train: {len(x_train)} - Valid: {len(x_valid)} - Test: {len(x_test)}")

    """ Hyperparameters """
    shape = (2048,1024,3)
    num_classes = 30
    lr = 1e-4
    bacth_size = 8
    epochs = 10

    """ Model """
    model = build_Unet(shape, num_classes)
    model.compile(loss = "categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(lr),metrics=["accuracy"])
    model.summary()

    """ data """
    train_dataset = tf_dataset(x_train, y_train, batch=8)
    valid_dataset = tf_dataset(x_valid, y_valid, batch=8)

    train_steps = len(x_train)//bacth_size
    valid_steps = len(x_valid)//bacth_size

    callbacks = [
        ModelCheckpoint("model.h5",verbose=1,  save_best_only=True),
        ReduceLROnPlateau(monitor="val_loss",patience=1,factor=0.1, verbose=1,min_lr=1e-6),
        EarlyStopping(monitor="val_loss", patience=5,verbose=1)
    ]

    model.fit( train_dataset,
              steps_per_epoch=train_steps,
              validation_steps=valid_steps,
              validation_data=valid_dataset,
              epochs=epochs,
              callbacks=callbacks
              )