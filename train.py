"""
train code
"""
import os
import tensorflow as tf
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Conv2D, Dropout, MaxPooling2D, Input, Flatten, Lambda, AveragePooling1D, Activation, TimeDistributed, LSTM, Bidirectional, BatchNormalization
import numpy as np
from keras import backend as K
from keras.models import Model


os.environ["CUDA_VISIBLE_DEVICES"]='0'


def attention_pooling(model_input):
    """
    attention pooling moddule

    Args:
        model_input: sequential input

    Returns:
        attention_output: attention weight
    """

    # average pooling for lstm units
    model_input_mean = AveragePooling1D(pool_size=64, data_format='channels_first', padding='valid')(model_input)
    model_input_mean = Lambda(lambda x: K.squeeze(x, axis=2))(model_input_mean)

    # transposed input
    model_input_tran = Lambda(lambda x: K.permute_dimensions(x, [0, 2, 1]))(model_input)

    # calculate attention weight
    attention = Dense(50, activation='softmax', name='attention')(model_input_mean)

    # input * attention weight
    attention_output = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=(1, 2)))([attention, model_input_tran])

    return attention_output


def speech_base_model():
    """
    speech baseline model

    Returns:
        model: speech baseline model
    """
    model_in = Input(shape=(400, 40, 1))

    # Layer1 : conv , batch norm, relu, and maxpool
    model_conv_1 = Conv2D(8, (5, 5), padding='same')(model_in)
    model_conv_1 = BatchNormalization()(model_conv_1)
    model_conv_1 = Activation(activation='relu')(model_conv_1)
    model_conv_1 = MaxPooling2D((2, 2), strides=(2, 2), padding='valid')(model_conv_1)

    # Layer2 : conv , batch norm, relu, and maxpool
    model_conv_2 = Conv2D(16, (5, 5), padding='same')(model_conv_1)
    model_conv_2 = BatchNormalization()(model_conv_2)
    model_conv_2 = Activation(activation='relu')(model_conv_2)
    model_conv_2 = MaxPooling2D((2, 2), strides=(2, 2), padding='valid')(model_conv_2)

    # Layer3 : conv , batch norm, relu, and maxpool
    model_conv_3 = Conv2D(32, (5, 5), padding='same')(model_conv_2)
    model_conv_3 = BatchNormalization()(model_conv_3)
    model_conv_3 = Activation(activation='relu')(model_conv_3)
    model_conv_3 = MaxPooling2D((2, 2), strides=(2, 2), padding='valid')(model_conv_3)

    # Flatten layer
    model_flat = TimeDistributed(Flatten())(model_conv_3)

    # bi-lstm and attention pooling
    model_lstm = Bidirectional(LSTM(32, return_sequences=True))(model_flat)
    model_att = attention_pooling(model_lstm)

    # dense layer
    model_dense_1 = Dense(64, activation='relu')(model_att)
    model_dense_1 = Dropout(0.5)(model_dense_1)
    model_dense_2 = Dense(7, name='output_layer')(model_dense_1)
    model_out = Activation(activation='softmax')(model_dense_2)

    model = Model(inputs=model_in, outputs=model_out)
    model.summary()

    return model


def main():
    # Load Training & Validation data
    _ROOT_PATH = "dataset/"
    x_train = np.load(_ROOT_PATH + "speech_train.npy")
    x_val = np.load(_ROOT_PATH + "speech_val.npy")
    y_train = np.load(_ROOT_PATH + "speech_train.npy")
    y_val = np.load(_ROOT_PATH + "speech_val.npy")

    # Convert labels to categorical one-hot encoding
    y_train = keras.utils.to_categorical(y_train, num_classes=7)
    y_val = keras.utils.to_categorical(y_val, num_classes=7)

    # Training Parameter setting
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    config = tf.ConfigProto(device_count={'GPU': 1, 'CPU': 30}, gpu_options=gpu_options)
    sess = tf.Session(config=config)
    keras.backend.set_session(sess)

    # Model build
    model = speech_base_model()

    # Model Check point
    model_path = 'model/' + 'speech_model_' + 'acc_{val_acc:.4f}.hdf5'
    checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', verbose=1, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_acc', min_delta=0.0005, patience=30, verbose=1, mode='auto')

    # Training
    adam = keras.optimizers.Adam(lr = 0.001, beta_1 = 0.5, beta_2 = 0.999, amsgrad=False)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=256, epochs=256, validation_data=(x_val, y_val), verbose=1, callbacks=[early_stopping, checkpoint])

    ### Evaluation
    score = model.evaluate(x_val, y_val, batch_size=256)

    print(score)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

if __name__ == '__main__':
    main()
