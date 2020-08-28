"""
generate dataset_numpy
"""

import os
import numpy as np
import librosa
import scipy
from pyvad import trim

### Pre-processing
MAX_FRAME_LENGTH = 400    # max wave length (4 sec)
STRIDE = 0.01       # STRIDE (10ms)
WINDOW_SIZE = 0.025 # filter window size (25ms)
NUM_MELS = 40       # Mel filter number
PRE_EMPHASIS_COEFF = 0.97  # Pre-Emphasis filter coefficient

EMOTION_LIST = ['ang', 'dis', 'fea', 'hap', 'neu', 'sad', 'sur']

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

# Main Code
def main():
    """
    main code

    """
    # select data tye (train or val)
    data_type = 'train'
    # data_type = 'val'

    # data folder path
    load_path = './' + data_type

    # file list
    dir_files = os.listdir(load_path)
    file_list = []
    for files in dir_files:
        if '.mp4' in files:
            file_list.append(files)
    file_list.sort(key=lambda f: int(f.split("-")[0]))

    x_npy = np.zeros((len(file_list), MAX_FRAME_LENGTH, NUM_MELS, 1))
    y_npy = np.zeros((len(file_list),))
    for num_file, file_name in enumerate(file_list):
        file_path = load_path + '/' + file_name
        print('File path = ' + file_path)

        # Obtain file_id and Emotion label
        emotion = file_name.split("-")[6]
        emotion_num = EMOTION_LIST.index(emotion)  # Convert to emotion number

        # load wav
        wav, sampling_rate = librosa.load(file_path)

        # Preprocessing(Resampling, Normalization, Denoising, Pre-emphasis, VAD)
        input_mels = preprocessing(wav, sampling_rate)

        # save
        x_npy[num_file, :, :, 0] = input_mels.T
        y_npy[num_file] = emotion_num

    # Save numpy
    save_path = './dataset/'
    np.save(save_path + 'speech_' + data_type + '.npy', x_npy)
    np.save(save_path + 'label_' + data_type + '.npy', y_npy)

if __name__ == '__main__':
    main()
