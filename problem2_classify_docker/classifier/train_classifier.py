import tensorflow as tf
from tensorflow.keras.layers import Input, RandomFlip, RandomRotation, RandomContrast, Average, Dense, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.models import  Model, Sequential, load_model
from tensorflow.keras.applications import InceptionResNetV2, EfficientNetB0, ResNet50, ResNet101

import matplotlib.pyplot as plt
import os

from dataloader import load_data


def get_model(model_type):
    '''
    Loads the model
    Args: 
        model_type: model instance
    Returns:
        model: required model
    ''' 
    data_aug = Sequential( [ RandomFlip(), RandomRotation(factor=0.2), RandomContrast(factor=0.15) ] )
    
    model_type.trainable=True
    input = Input(shape=(224, 224, 3)) #input layer
    x = data_aug(input) # data augment layer
    # x = model_type(include_top=False, weights='imagenet')(x) # performance degrades
    output = model_type(include_top=True, weights=None, classes=4)(x)
    # x = GlobalAveragePooling2D()(x)
    # output = Dense(4)(x)

    # Instentiate model with config
    model = Model(input, output)
    # model compilation
    model.compile( optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'] )
    # show model summary
    model.summary()
    return model

def plot_loss_accuracy(history, save_name):
    ''' plots accuracy and loss

    Args:
        history: the history of a model
        save_name: specific model name
    '''  
    # plot accuracy  
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'{save_name} model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(os.path.join('logs', save_name+'_model_accuracy.png'))
    plt.clf() #clear buffer
    # plot loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'{save_name} model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(os.path.join('logs', save_name+'_model_loss.png'))
    plt.clf()

def train_model(x_train, y_train, x_test, y_test, tensorboard, conf_dict, model_type, model_path):
    '''
    Trains model with specific configuration
    Args:
        x_train, y_train: numpy array
        x_test, y_test: numpy array
        tensorboard: tensorboard callback
        conf_dic: configuration dictionary
        model_type[model instance]: Train model type
        model_path: to pretrain from or save model directory
    ''' 
    model = get_model(model_type)
    # checkpoint specification
    cp_callback = ModelCheckpoint(filepath= model_path, save_weights_only=False, save_best_only=True, verbose=1)
    # pretrain model loading
    if conf_dict['pretrain']:
        model.load_weights(model_path)
    
    # model training
    model_history = model.fit(x_train,y_train, epochs = conf_dict['epochs'], batch_size=conf_dict['batch_size'], 
                                validation_data=(x_test, y_test), shuffle=True, callbacks=[tensorboard, cp_callback])
    
    # saving model
    model.save(model_path)
    plot_loss_accuracy(model_history, save_name=os.path.basename(model_path)[:-3])


def train(conf_dict):
    '''
    Train the model using given dataset

    Args:
        conf_dict: Configuration Dictionary where configuration is specified
    '''  

    # load dataset
    x_train, y_train = load_data(conf_dict['train_dir'])
    x_test, y_test = load_data(conf_dict['test_dir'])
    print( f'Total train data {len(x_train)} and test data {len(x_test)}' )

    # model monitoring on tensorboard
    tensorboard = TensorBoard(log_dir=conf_dict['tensorboard_dir'])
    if conf_dict['train_on'] == 'efficientnet' or conf_dict['train_on'] == 'ensemble': #ensemble
        print('====================training Efficient net model============================')
        train_model(x_train, y_train, x_test, y_test, tensorboard, conf_dict, model_type=EfficientNetB0, \
                    model_path = conf_dict['efficientnet_model'])

    # if conf_dict['train_on'] == 'incep_resnet' or conf_dict['train_on'] == 'ensemble':
    #     print('========================training Inception Resnet model============================')
    #     train_model(x_train, y_train, x_test, y_test, tensorboard, conf_dict, model_type=InceptionResNetV2, \
    #                 model_path = conf_dict['incep_res_model'])
    if conf_dict['train_on'] == 'incep_resnet' or conf_dict['train_on'] == 'ensemble':
        print('========================training Inception Resnet model============================')
        train_model(x_train, y_train, x_test, y_test, tensorboard, conf_dict, model_type=ResNet101, \
                    model_path = conf_dict['incep_res_model'])
    # print('end of training')
    # exit(0)
    if conf_dict['train_on'] == 'ensemble':
        # load two models to ensemble
        incep_res = load_model(conf_dict['incep_res_model'], compile=False)
        efficientnet = load_model(conf_dict['efficientnet_model'], compile=False)
        # incep_res_model.get_layer(name='predictions').name='predictions_1'
        model_list = [incep_res, efficientnet]
        model_input = Input(shape=(224, 224, 3)) #input shape
        # input = Input(shape=(224, 224, 3))
        model_outputs = [model(model_input) for model in model_list] # store individual model's output
        ensemble_output = Average()(model_outputs) # average ensembling
        ensemble_model = Model(inputs=model_input, outputs=ensemble_output) # ensemble model 
        ensemble_model.save(conf_dict['ensemble_model']) # save ensembled model
    print('=====================training finished===============')


if __name__ == '__main__':
    import yaml
    conf_file = 'config.yaml'
    with open(conf_file) as f:
        conf_dict = yaml.load(f, Loader=yaml.SafeLoader)  # configuration dict
    print(conf_dict)
    train(conf_dict)