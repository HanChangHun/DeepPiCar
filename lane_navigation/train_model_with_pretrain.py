import os
import pickle
import random
import datetime
from pathlib import Path

import cv2
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.client import device_lib

from sklearn.model_selection import train_test_split

import lane_navigation.image_augmentation as image_augmentation

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

gpu = tf.config.experimental.list_physical_devices("GPU")
try:
    tf.config.experimental.set_memory_growth(gpu[0], True)
except RuntimeError as e:
    print(e)


def my_imread(image_path):
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def img_preprocess(image):
    height, _, _ = image.shape
    image = image[
        int(height / 2) :, :, :
    ]  # remove top half of the image, as it is not relavant for lane following
    image = cv2.cvtColor(
        image, cv2.COLOR_RGB2YUV
    )  # Nvidia model said it is best to use YUV color space
    image = cv2.GaussianBlur(image, (3, 3), 0)
    image = cv2.resize(image, (200, 66))  # input image size (200,66) Nvidia model
    # image = image / 255 # normalizing, the processed image becomes black for some reason.  do we need this?
    image = image.astype(np.uint8)
    return image


def image_data_generator(image_paths, steering_angles, batch_size, is_training):
    while True:
        batch_images = []
        batch_steering_angles = []

        for i in range(batch_size):
            random_index = random.randint(0, len(image_paths) - 1)
            image_path = image_paths[random_index]
            image = my_imread(image_paths[random_index])
            steering_angle = steering_angles[random_index]
            if is_training:
                # training: augment image
                image, steering_angle = image_augmentation.random_augment(
                    image, steering_angle
                )

            image = img_preprocess(image)
            batch_images.append(image)
            batch_steering_angles.append(steering_angle)

        yield (np.asarray(batch_images), np.asarray(batch_steering_angles))


if __name__ == "__main__":
    image_paths = []
    lab_dir = Path("train_data_generation/data/drive_with_keypress/")
    data_dirs = []
    data_dirs.append(lab_dir / "28")
    data_dirs.append(lab_dir / "29")
    for data_dir in data_dirs:
        image_paths = list(data_dir.glob("*.png"))
    image_paths.sort()

    steering_angles = []
    for image_path in image_paths:
        steering_angles.append(int(float(image_path.stem[13:]) + 0.5))

    X_train, X_valid, y_train, y_valid = train_test_split(
        image_paths, steering_angles, test_size=0.1
    )

    batch_size = 4
    X_train_batch, y_train_batch = next(
        image_data_generator(X_train, y_train, batch_size, True)
    )
    X_valid_batch, y_valid_batch = next(
        image_data_generator(X_valid, y_valid, batch_size, False)
    )

    # model = nvidia_model.nvidia_model()
    model = keras.models.load_model("lane_navigation/model/lane_navigation_final.h5")

    model_output_dir = Path("lane_navigation/model/")
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(model_output_dir, "lane_navigation_w_pretrain_check.h5"),
        verbose=1,
        save_best_only=True,
    )

    history = model.fit_generator(
        image_data_generator(X_train, y_train, batch_size=100, is_training=True),
        steps_per_epoch=300,
        epochs=10,
        validation_data=image_data_generator(
            X_valid, y_valid, batch_size=100, is_training=False
        ),
        validation_steps=200,
        verbose=1,
        shuffle=1,
        callbacks=[checkpoint_callback],
    )

    model.save(os.path.join(model_output_dir, "lane_navigation_w_pretrain_final.h5"))

    date_str = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    history_path = os.path.join(model_output_dir, "history_w_pretrain.pickle")

    with open(history_path, "wb") as f:
        pickle.dump(history.history, f, pickle.HIGHEST_PROTOCOL)
