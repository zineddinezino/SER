import pandas as pd
import numpy as np

import os
import sys


import librosa
import librosa.display
import seaborn as sns
import matplotlib.pyplot as plt




########################### Sorting the dataset
import os
import shutil
from turtle import done
path = "/dataset"
# iterate over files in
# that directory

for root, dirs, files in os.walk(path):

    for filename in files:
        if 'W' == filename[5:6]:
            original = os.path.join(root, filename)
            folder_path = '/angry/'
            target = os.path.join(folder_path, filename)
            shutil.move(original,target)
        elif 'N' == filename[5:6]:
            original = os.path.join(root, filename)
            folder_path = '/neutral/'
            target = os.path.join(folder_path, filename)
            shutil.move(original,target)
            
        elif 'A' == filename[5:6]:
            original = os.path.join(root, filename)
            folder_path = '/fear/'
            target = os.path.join(folder_path, filename)
            shutil.move(original,target)
            
        elif 'F' == filename[5:6]:
            original = os.path.join(root, filename)
            folder_path = '/happy/'
            target = os.path.join(folder_path, filename)
            shutil.move(original,target)
            
        elif 'T' == filename[5:6]:
            original = os.path.join(root, filename)
            folder_path = '/sad/'
            target = os.path.join(folder_path, filename)
            shutil.move(original,target)
            
        elif 'E' == filename[5:6]:
            original = os.path.join(root, filename)
            folder_path = '/disgust/'
            target = os.path.join(folder_path, filename)
            shutil.move(original,target)
            
        elif 'L' == filename[5:6]:
            original = os.path.join(root, filename)
            folder_path = '/boredom/'
            target = os.path.join(folder_path, filename)
            shutil.move(original,target)



################ Samples with their path
import os
import shutil
from turtle import done
path = "/dataset/"
# iterate over files in
# that directory

file_emotion = []
file_path = []

for root, dirs, files in os.walk(path):
  for d in dirs:
    if d == 'angry':
      for root_d,dirs_d,files_d in os.walk(os.path.join(path,d)):
        for f in files_d:
          file_emotion.append('angry')
          file_path.append(root+d+'/'+f)
    elif d == 'boredom':
      for root_d,dirs_d,files_d in os.walk(os.path.join(path,d)):
        for f in files_d:
          file_emotion.append('boredom')
          file_path.append(root+d+'/'+f)
    elif d == 'disgust':
      for root_d,dirs_d,files_d in os.walk(os.path.join(path,d)):
        for f in files_d:
          file_emotion.append('disgust')
          file_path.append(root+d+'/'+f)
    elif d == 'fear':
      for root_d,dirs_d,files_d in os.walk(os.path.join(path,d)):
        for f in files_d:
          file_emotion.append('fear')
          file_path.append(root+d+'/'+f)
    elif d == 'happy':
      for root_d,dirs_d,files_d in os.walk(os.path.join(path,d)):
        for f in files_d:
          file_emotion.append('happy')
          file_path.append(root+d+'/'+f)
    elif d == 'neutral':
      for root_d,dirs_d,files_d in os.walk(os.path.join(path,d)):
        for f in files_d:
          file_emotion.append('neutral')
          file_path.append(root+d+'/'+f)
    elif d == 'sad':
      for root_d,dirs_d,files_d in os.walk(os.path.join(path,d)):
        for f in files_d:
          file_emotion.append('sad')
          file_path.append(root+d+'/'+f)
 
# dataframe for emotion of files
emotion_df = pd.DataFrame(file_emotion, columns=['emotion'])

# dataframe for path of files.
path_df = pd.DataFrame(file_path, columns=['sample_path'])
data_path = pd.concat([emotion_df, path_df], axis=1)

# data augmentation
def noise(data):
    noise_amp = 0.035*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data

def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate)

def shift(data):
    shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
    return np.roll(data, shift_range)

def pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)

# the feature are extracted and stacked horizonatally
def extract_features(data):
    # ZCR
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result=np.hstack((result, zcr)) 

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft)) 

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc)) 

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms)) 

    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel))
    
    return result

def get_features(path):
    # duration and offset are used to take care of the no audio in start and the ending of each audio files as seen above.
    data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)
    
    # feauture extraction without augmentation
    res1 = extract_features(data)
    result = np.array(res1)
    
    # data augmentation using noise
    noise_data = noise(data)
    res2 = extract_features(noise_data)
    result = np.vstack((result, res2)) # stacking vertically

    # data augmentation using stretching and pitching
    new_data = stretch(data)
    data_stretch_pitch = pitch(new_data, sample_rate)
    res3 = extract_features(data_stretch_pitch)
    result = np.vstack((result, res3)) # stacking vertically
    
    return result

# performing the feauture extraction and the data augmentation
X, Y = [], []
for path, emotion in zip(data_path.sample_path, data_path.emotion):
    feature = get_features(path)
    for ele in feature:
        X.append(ele)
        Y.append(emotion)

# saving the features 
Features = pd.DataFrame(X)
Features['labels'] = Y
Features.to_csv('/features.csv', index=False)
Features.head()