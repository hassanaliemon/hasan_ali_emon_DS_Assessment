import tensorflow as tf

import numpy as np
import os
import cv2

from dataloader import load_data

if __name__ == '__main__':
    import yaml
    conf_file = 'config.yaml'
    with open(conf_file) as f:
        conf_dict = yaml.load(f, Loader=yaml.SafeLoader)  # configuration dict
    # print(conf_dict)
    x_data, y_data = load_data(conf_dict['infer_dir'])
    model = tf.keras.models.load_model(conf_dict['efficientnet_model'], compile=False)
    cls_map = { 0: 'berry',  1:'bird',  2:'dog',  3:'flower' }

    for i, img_data in enumerate(x_data):
        img_data = np.expand_dims(img_data, axis=0)
        # print('shape', img_data.shape)
        output = model.predict(img_data)
        lbl = np.argmax(output)
        cls_name = cls_map[lbl]
        print(f'Prediction for ground truth {cls_map[y_data[i]]} is {cls_name}')