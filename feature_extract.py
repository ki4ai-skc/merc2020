"""
train code
"""
import os
import tensorflow as tf
import keras
from keras.layers import Dense, Lambda, AveragePooling1D
import numpy as np
from keras import backend as K
from keras.models import load_model, Model

os.environ["CUDA_VISIBLE_DEVICES"]='0'

def attention_pooling(model_input):
    """
    attention pooling module

    Args:
        model_input: sequential input

    Returns:
        attention_output: attention weight
    """

    # average pooling for lstm units
    model_input_mean = AveragePooling1D(pool_size=128, data_format='channels_first', padding='valid')(model_input)
    model_input_mean = Lambda(lambda x: K.squeeze(x, axis=2))(model_input_mean)

    # transposed input
    model_input_tran = Lambda(lambda x: K.permute_dimensions(x, [0, 2, 1]))(model_input)

    # calculate attention weight
    attention = Dense(50, activation='softmax', name='attention')(model_input_mean)

    # input * attention weight
    attention_output = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=(1, 2)))([attention, model_input_tran])

    return attention_output

def inter_model_load(model_path):
    speech_model = load_model(model_path, custom_objects={'attention_pooling': attention_pooling})  # Load best model
    inter_layer_model = Model(inputs=speech_model.input, outputs=speech_model.get_layer('dense_2').output)

    return inter_layer_model

def main(data_type):

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
    config = tf.ConfigProto(device_count={'GPU': 1, 'CPU': 30}, gpu_options=gpu_options)
    sess = tf.Session(config=config)
    keras.backend.set_session(sess)

    # Load mel-spectrogram numpy
    x = np.load('dataset/' + 'speech_' + data_type + '.npy')

    # Load model
    modelPath = 'model/speech_model_acc_0.3925.hdf5'
    model = inter_model_load(modelPath)
    model.summary()

    # Feature extraction
    feature = model.predict(x, verbose=1, batch_size=256)
    print(np.shape(feature))

    if not (os.path.isdir('features')):
        os.makedirs(os.path.join('features'))

    np.save('features/' + 'speech_BN_' + data_type + '.npy', feature)

    print("Finished")

    K.clear_session()

if __name__ == '__main__':
    print("train/val/test1/test2/test3:")
    data_type = input()
    main(data_type)
