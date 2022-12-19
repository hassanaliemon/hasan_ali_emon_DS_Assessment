from tensorflow.keras.utils import normalize

import numpy as np
import os
import glob
import cv2

def load_data(data_dir):
    '''
    Loads dataset to feed during training or inference time

    Args:
        data_dir: Dir from where the dataset is to load

    Return:
        x, y: Images & Labels converted to numpy array
    '''
    img_list = []
    lbl_list = []
    cls_dict = { 'berry': 0,  'bird':1,  'dog': 2,  'flower': 3 }

    for cls_name in os.listdir(data_dir):
        clsfolder_dir = os.path.join(data_dir, cls_name)
        for img_name in os.listdir(clsfolder_dir):
            img_dir = os.path.join(clsfolder_dir, img_name)
            img = cv2.imread(img_dir)
            img = cv2.resize(img, (224, 224))
            # img = img/ 255.0
            img = normalize(img, axis=1)
            # print('converted img shape is',img.shape)
            lbl = cls_dict[cls_name]
            img_list.append(img)
            lbl_list.append(lbl)

    x, y= np.array(img_list), np.array(lbl_list, dtype=np.uint8)
    print('x, y shape', x.shape, y.shape)
    return x, y 


if __name__ == '__main__':
    import yaml
    conf_file = 'config.yaml'
    with open(conf_file) as f:
        conf_dict = yaml.load(f, Loader=yaml.SafeLoader)  # configuration dict
    print(conf_dict)
    x_train, y_train = load_data(conf_dict['train_dir'])
    x_test, y_test = load_data(conf_dict['test_dir'])
