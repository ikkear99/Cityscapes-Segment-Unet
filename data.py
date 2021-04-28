import os
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd

H = 2048
W = 1024

def  data_process(x_path, y_path):
    images = []
    for i,(dirpath, dirnames, filenames) in enumerate(os.walk(x_path)):
        if dirpath is not x_path:
            for f in filenames:
                names = os.path.join(dirpath, f"{f}")
                images.append(names)

    masks = []
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(y_path)):
        if dirpath is not y_path:
            j = 0
            for f in filenames:
                if f.count("_gtFine_labelIds") == 1 :
                    names = os.path.join(dirpath,f"{f}")
                    masks.append(names)

    return images, masks

def load_data(path):
    x_train_path = os.path.join(path,"leftImg8bit/train")
    y_train_path = os.path.join(path,"gtFine/train")
    x_valid_path = os.path.join(path, "leftImg8bit/val")
    y_valid_path = os.path.join(path, "gtFine/val")
    x_test_path = os.path.join(path, "leftImg8bit/test")
    y_test_path = os.path.join(path, "gtFine/test")

    x_train, y_train = data_process(x_train_path,y_train_path)
    x_valid, y_valid = data_process(x_valid_path,y_valid_path)
    x_test, y_test = data_process(x_test_path,y_test_path)

    return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)

def read_image(x):
    x = cv2.imread(x,cv2.IMREAD_COLOR)
    x = x / 255.0
    x = x.astype(np.float32)
    return x

def read_mask(x):
    x = cv2.imread(x, cv2.IMREAD_GRAYSCALE)
    x = x -1
    x = x.astype(np.int32)
    return x

def tf_dataset(x,y, batch = 8):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.shuffle(buffer_size=5000)
    dataset = dataset.map(preprocess)
    dataset = dataset.batch(batch)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(2)
    return dataset

def preprocess(x, y):
    def f(x, y):
        x = x.decode()
        y = y.decode()

        image = read_image(x)
        mask = read_mask(y)

        return image, mask

    image, mask = tf.numpy_function(f, [x, y], [tf.float32, tf.int32])
    mask = tf.one_hot(mask, 3, dtype=tf.int32)
    image.set_shape([H, W, 3])
    mask.set_shape([H, W, 3])

    return image, mask

if __name__ == '__main__':
    path = "E:\Dien_AI\Segment_Citydata\data"
    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = load_data(path)
    print(f"Dataset: Train: {len(x_train)} - Valid: {len(x_valid)} - Test: {len(x_test)}")

    read_image(x_train[0])
    read_mask(y_train[0])
    dataset = tf_dataset(x_train, y_train, batch=8)
    for x, y in dataset:
        print(x.shape, y.shape)  ## (8, 2048, 1024, 3), (8, 2048, 1024, 3)

    print("End program")