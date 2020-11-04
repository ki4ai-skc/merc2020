gpu_num = input('GPU Number:')


import librosa
import scipy
from pyvad import trim
from keras import backend as K
import tensorflow as tf
import os
import numpy as np
from keras.models import load_model, Model
from keras.layers import Dense, Conv2D, Dropout, MaxPooling2D, Input, Flatten, Lambda, AveragePooling1D, Activation, TimeDistributed, LSTM, Bidirectional, BatchNormalization

### Pre-processing
MAX_FRAME_LENGTH = 400    # max wave length (4 sec)
STRIDE = 0.01       # STRIDE (10ms)
WINDOW_SIZE = 0.025 # filter window size (25ms)
NUM_MELS = 40       # Mel filter number
PRE_EMPHASIS_COEFF = 0.97  # Pre-Emphasis filter coefficient

EMOTION_LIST = ['hap', 'ang', 'dis', 'fea', 'neu', 'sad', 'sur']

batch_size = 1

def init(gpu_num):
    ### GPU Setting
    os.environ["CUDA_VISIBLE_DEVICES"]=gpu_num

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

def preprocessing(wav, sampling_rate):
    """

    Args:
        wav: wave
        sr: sampling rate

    Returns:
        input_mels
    """
    # Resampling to 16kHz
    if sampling_rate != 16000:
        sampling_rate_re = 16000  # sampling rate of resampling
        wav = librosa.resample(wav, sampling_rate, sampling_rate_re)
        sampling_rate = sampling_rate_re

    # Denoising
    wav[np.argwhere(wav == 0)] = 1e-10
    wav_denoise = scipy.signal.wiener(wav, mysize=None, noise=None)

    # Pre Emphasis filter
    wav_emphasis = np.append(wav_denoise[0], wav_denoise[1:] - PRE_EMPHASIS_COEFF * wav_denoise[:-1])

    # Normalization (Peak)
    wav_max = np.abs(wav_emphasis).max() / 0.9
    wav_emphasis = wav_emphasis / wav_max  # normalize for VAD

    # Voice Activity Detection (VAD)
    vad_mode = 2  # VAD mode = 0 ~ 3
    wav_vad = trim(wav_emphasis, sampling_rate, vad_mode=vad_mode, thr=0.01)  ## trim
    if wav_vad is None:
        wav_vad = wav_emphasis

    # De normalization
    wav_vad = wav_vad * wav_max

    # Obtain the spectrogram
    sftf_vad = librosa.core.stft(y=wav_vad, hop_length=int(sampling_rate * STRIDE), n_fft=int(sampling_rate * WINDOW_SIZE))
    spec = np.abs(sftf_vad) ** 2

    # mel spectrogram
    mel_spec = librosa.feature.melspectrogram(S=spec, n_mels=NUM_MELS)

    # log scaled mel spectrogram
    log_weight = 1e+6
    log_mel_spec = np.log(1 + log_weight * mel_spec)

    frame_length = log_mel_spec.shape[1]

    # zero padding
    input_mels = np.zeros((NUM_MELS, MAX_FRAME_LENGTH), dtype=float)
    if frame_length < MAX_FRAME_LENGTH:
        input_mels[:, :frame_length] = log_mel_spec[:, :frame_length]
    else:
        input_mels[:, :MAX_FRAME_LENGTH] = log_mel_spec[:, :MAX_FRAME_LENGTH]

    return input_mels

def main():
    ### Parameter setting
    init(gpu_num)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
    config = tf.ConfigProto(device_count={'GPU': 1, 'CPU': 10}, gpu_options=gpu_options)
    sess = tf.Session(config=config)
    K.set_session(sess)

    # Load trained model
    model_path = './model/speech_model_acc_0.3925.hdf5'
    model_best = load_model(model_path, custom_objects={'attention_pooling':attention_pooling}) # Load best model
    model_best.summary()

    # load test samples
    example_path = './example.mp4'
    wav, sampling_rate = librosa.load(example_path)

    # preprocessing
    input_mels = preprocessing(wav, sampling_rate)

    x_test = np.zeros((1, input_mels.shape[1], input_mels.shape[0], 1))
    x_test[0, :, :, 0] = input_mels.T

    # prediction
    y_test_pred = model_best.predict(x_test, batch_size=batch_size, verbose=1)
    print(y_test_pred)

    # Feature extraction
    model_best_feature = Model(inputs=model_best.input, outputs=model_best.get_layer('dense_2').output)
    features = model_best_feature.predict(x_test, batch_size=batch_size, verbose=1)
    print(features)

    K.clear_session()

if __name__ == '__main__':
    main()

