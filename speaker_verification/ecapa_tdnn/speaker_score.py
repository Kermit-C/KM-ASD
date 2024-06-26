from __future__ import unicode_literals

from functools import reduce
from pydub import AudioSegment
import numpy as np

import os
import time
import random
import warnings
import argparse
import librosa
import threading
import scipy.io.wavfile as wf


def load_wav(vid_path, sr, mode='train'):
    wav, sr_ret = librosa.load(vid_path, sr=sr)
    assert sr_ret == sr
    if mode == 'train':
        extended_wav = np.append(wav, wav)
        if np.random.random() < 0.3:
            extended_wav = extended_wav[::-1]
        return extended_wav
    else:
        extended_wav = np.append(wav, wav[::-1])
        return extended_wav

def load_wav_from_mat(vid_mat, sr, mode='train'):
    wav, sr_ret = vid_mat, sr
    assert sr_ret == sr
    if mode == 'train':
        extended_wav = np.append(wav, wav)
        if np.random.random() < 0.3:
            extended_wav = extended_wav[::-1]
        return extended_wav
    else:
        extended_wav = np.append(wav, wav[::-1])
        return extended_wav

def load_wave(wave_file):
    buckets = build_buckets(10, 1, 0.01)
    data = get_fft_spectrum(wave_file, buckets)

    if data.shape[1] == 300:
        pass
    else:
        start = np.random.randint(0, data.shape[1] - 300)
        data = data[:, start:start+300]

    data = np.expand_dims(data, -1)
    return data


def lin_spectogram_from_wav(wav, hop_length, win_length, n_fft=1024):
    linear = librosa.stft(wav, n_fft=n_fft, win_length=win_length, hop_length=hop_length) # linear spectrogram
    return linear.T


def load_data(path, win_length=400, sr=16000, hop_length=160, n_fft=512, spec_len=250, mode='train'):
    wav = load_wav(path, sr=sr, mode=mode)
    linear_spect = lin_spectogram_from_wav(wav, hop_length, win_length, n_fft)
    mag, _ = librosa.magphase(linear_spect)  # magnitude
    mag_T = mag.T
    freq, time = mag_T.shape
    if mode == 'train':
        randtime = np.random.randint(0, time-spec_len)
        spec_mag = mag_T[:, randtime:randtime+spec_len]
    else:
        spec_mag = mag_T
    # preprocessing, subtract mean, divided by time-wise var
    mu = np.mean(spec_mag, 0, keepdims=True)
    std = np.std(spec_mag, 0, keepdims=True)
    res = (spec_mag - mu) / (std + 1e-5)
    res = np.array(res)
    res = np.expand_dims(res, -1)
    return res

def load_data_from_mat(mat, win_length=400, sr=16000, hop_length=160, n_fft=512, spec_len=250, mode='train'):
    wav = load_wav_from_mat(mat, sr=sr, mode=mode)
    linear_spect = lin_spectogram_from_wav(wav, hop_length, win_length, n_fft)
    mag, _ = librosa.magphase(linear_spect)  # magnitude
    mag_T = mag.T
    freq, time = mag_T.shape
    if mode == 'train':
        randtime = np.random.randint(0, time-spec_len)
        spec_mag = mag_T[:, randtime:randtime+spec_len]
    else:
        spec_mag = mag_T
    # preprocessing, subtract mean, divided by time-wise var
    mu = np.mean(spec_mag, 0, keepdims=True)
    std = np.std(spec_mag, 0, keepdims=True)
    res = (spec_mag - mu) / (std + 1e-5)
    res = np.array(res)
    res = np.expand_dims(res, -1)
    return res


class VoiceActivityDetector():
    """ Use signal energy to detect voice activity in wav file """
    
    def __init__(self, wave_input_filename):
        self._read_wav(wave_input_filename)._convert_to_mono()
        self.sample_window = 0.02 #20 ms
        self.sample_overlap = 0.01 #10ms
        self.speech_window = 0.5 #half a second
        self.speech_energy_threshold = 0.3 #60% of energy in voice band
        self.speech_start_band = 300
        self.speech_end_band = 3000
           
    def _read_wav(self, wave_file):
        self.rate, self.data = wf.read(wave_file)
        self.channels = len(self.data.shape)
        self.filename = wave_file
        return self
    
    def _convert_to_mono(self):
        if self.channels == 2 :
            self.data = np.mean(self.data, axis=1, dtype=self.data.dtype)
            self.channels = 1
        return self
    
    def _calculate_frequencies(self, audio_data):
        data_freq = np.fft.fftfreq(len(audio_data),1.0/self.rate)
        data_freq = data_freq[1:]
        return data_freq    
    
    def _calculate_amplitude(self, audio_data):
        data_ampl = np.abs(np.fft.fft(audio_data))
        data_ampl = data_ampl[1:]
        return data_ampl
        
    def _calculate_energy(self, data):
        data_amplitude = self._calculate_amplitude(data)
        data_energy = data_amplitude ** 2
        return data_energy
        
    def _znormalize_energy(self, data_energy):
        energy_mean = np.mean(data_energy)
        energy_std = np.std(data_energy)
        energy_znorm = (data_energy - energy_mean) / energy_std
        return energy_znorm
    
    def _connect_energy_with_frequencies(self, data_freq, data_energy):
        energy_freq = {}
        for (i, freq) in enumerate(data_freq):
            if abs(freq) not in energy_freq:
                energy_freq[abs(freq)] = data_energy[i] * 2
        return energy_freq
    
    def _calculate_normalized_energy(self, data):
        data_freq = self._calculate_frequencies(data)
        data_energy = self._calculate_energy(data)
        #data_energy = self._znormalize_energy(data_energy) #znorm brings worse results
        energy_freq = self._connect_energy_with_frequencies(data_freq, data_energy)
        return energy_freq
    
    def _sum_energy_in_band(self,energy_frequencies, start_band, end_band):
        sum_energy = 0
        for f in energy_frequencies.keys():
            if start_band<f<end_band:
                sum_energy += energy_frequencies[f]
        return sum_energy
    
    def _median_filter (self, x, k):
        assert k % 2 == 1, "Median filter length must be odd."
        assert x.ndim == 1, "Input must be one-dimensional."
        k2 = (k - 1) // 2
        y = np.zeros ((len (x), k), dtype=x.dtype)
        y[:,k2] = x
        for i in range (k2):
            j = k2 - i
            y[j:,i] = x[:-j]
            y[:j,i] = x[0]
            y[:-j,-(i+1)] = x[j:]
            y[-j:,-(i+1)] = x[-1]
        return np.median (y, axis=1)
        
    def _smooth_speech_detection(self, detected_windows):
        median_window=int(self.speech_window/self.sample_window)
        if median_window%2==0: median_window=median_window-1
        median_energy = self._median_filter(detected_windows[:,1], median_window)
        return median_energy
        
    def convert_windows_to_readible_labels(self, detected_windows):
        """ Takes as input array of window numbers and speech flags from speech
        detection and convert speech flags to time intervals of speech.
        Output is array of dictionaries with speech intervals.
        """
        speech_time = []
        is_speech = 0
        for window in detected_windows:
            if (window[1]==1.0 and is_speech==0): 
                is_speech = 1
                speech_label = {}
                speech_time_start = window[0] / self.rate
                speech_label['speech_begin'] = speech_time_start
                print(window[0], speech_time_start)
                #speech_time.append(speech_label)
            if (window[1]==0.0 and is_speech==1):
                is_speech = 0
                speech_time_end = window[0] / self.rate
                speech_label['speech_end'] = speech_time_end
                speech_time.append(speech_label)
                print(window[0], speech_time_end)
        return speech_time
      
    def plot_detected_speech_regions(self):
        """ Performs speech detection and plot original signal and speech regions.
        """
        data = self.data
        detected_windows = self.detect_speech()
        data_speech = np.zeros(len(data))
        it = np.nditer(detected_windows[:,0], flags=['f_index'])
        while not it.finished:
            data_speech[int(it[0])] = data[int(it[0])] * detected_windows[it.index,1]
            it.iternext()
        plt.figure()
        plt.plot(data_speech)
        plt.plot(data)
        plt.show()
        return self
       
    def detect_speech(self):
        """ Detects speech regions based on ratio between speech band energy
        and total energy.
        Output is array of window numbers and speech flags (1 - speech, 0 - nonspeech).
        """
        detected_windows = np.array([])
        sample_window = int(self.rate * self.sample_window)
        sample_overlap = int(self.rate * self.sample_overlap)
        data = self.data
        sample_start = 0
        start_band = self.speech_start_band
        end_band = self.speech_end_band
        while (sample_start < (len(data) - sample_window)):
            sample_end = sample_start + sample_window
            if sample_end>=len(data): sample_end = len(data)-1
            data_window = data[sample_start:sample_end]
            energy_freq = self._calculate_normalized_energy(data_window)
            sum_voice_energy = self._sum_energy_in_band(energy_freq, start_band, end_band)
            sum_full_energy = sum(energy_freq.values())
            speech_ratio = sum_voice_energy/sum_full_energy
            # Hipothesis is that when there is a speech sequence we have ratio of energies more than Threshold
            speech_ratio = speech_ratio>self.speech_energy_threshold
            detected_windows = np.append(detected_windows,[sample_start, speech_ratio])
            sample_start += sample_overlap
        detected_windows = detected_windows.reshape(int(len(detected_windows)/2),2)
        detected_windows[:,1] = self._smooth_speech_detection(detected_windows)
        return detected_windows
 


def convert_audio_file(audio_path, output_path=None):
    assert os.path.exists(audio_path)
    print(audio_path)
    soundfile = AudioSegment.from_file(audio_path, format=os.path.splitext(audio_path)[1])
    if output_path:
        assert os.path.exists(output_path)
        soundfile.export(output_path, format='wav')
        return
    output_path = os.path.join(os.path.dirname(audio_path), os.path.splitext(audio_path)[0]+'.wav')
    soundfile.export(output_path, format='wav')

def vad_whole_audio(audio_path, output_dir, segment_length=3):
    assert os.path.exists(audio_path)
    assert os.path.exists(output_dir)
    if audio_path[-3:] != 'wav':
        convert_audio_file(audio_path)
        audio_path = os.path.join(os.path.dirname(audio_path), os.path.splitext(audio_path)[0]+'.wav')

    v = VoiceActivityDetector(audio_path)
    voice = AudioSegment.from_wav(audio_path)
    raw_detection = v.detect_speech()
    speech_labels = v.convert_windows_to_readible_labels(raw_detection)
    counter = 0
    for idx, each in enumerate(speech_labels):
        if int(speech_labels[idx]['speech_end'] - speech_labels[idx]['speech_begin']) >= segment_length:
            sentence = voice[  speech_labels[idx]['speech_begin']*1000   :  speech_labels[idx]['speech_end']*1000  ]
            sentence.export(output_dir+'/{}'.format(str(counter)+'.wav'), format='wav')
            counter += 1

def vad_the_audio(audio_path, output_wav, segment_length=2):
    assert os.path.exists(audio_path)
    v = VoiceActivityDetector(audio_path)
    voice = AudioSegment.from_wav(audio_path)
    raw_detection = v.detect_speech()
    speech_labels = v.convert_windows_to_readible_labels(raw_detection)
    counter = 0
    for idx, each in enumerate(speech_labels):
        if int(speech_labels[idx]['speech_end'] - speech_labels[idx]['speech_begin']) >= segment_length:
            temp_s = voice[  speech_labels[idx]['speech_begin']*1000   :  speech_labels[idx]['speech_end']*1000  ]
            if counter == 0:
                sentence=temp_s
            else:
                sentence +=temp_s
            counter += 1

    if counter > 0:
        sentence.export(output_wav, format='wav')
    else:
        voice.export(output_wav, format='wav')
   
    return (counter)

def vad_the_audio2(audio_path, output_wav, segment_length=2):
    assert os.path.exists(audio_path)
    if audio_path[-3:] != 'wav':
        convert_audio_file(audio_path)
        audio_path = os.path.join(os.path.dirname(audio_path), os.path.splitext(audio_path)[0]+'.wav')

    v = VoiceActivityDetector(audio_path)
    voice = AudioSegment.from_wav(audio_path)
    raw_detection = v.detect_speech()
    speech_labels = v.convert_windows_to_readible_labels(raw_detection)
    counter = 0
    for idx, each in enumerate(speech_labels):
        if int(speech_labels[idx]['speech_end'] - speech_labels[idx]['speech_begin']) >= segment_length:
            temp_s = voice[  speech_labels[idx]['speech_begin']*1000   :  speech_labels[idx]['speech_end']*1000  ]
            if counter == 0:
                sentence=temp_s
            else:
                sentence +=temp_s
            counter += 1

    if counter > 0:
        sentence.export(output_wav, format='wav')
    else:
        voice.export(output_wav, format='wav')
