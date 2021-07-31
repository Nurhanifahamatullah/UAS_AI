# In []
import os
import pandas as pd
import librosa
import numpy as np
import IPython.display as ipd
from glob import glob
import matplotlib.pyplot as plt
import librosa.display
# %matplotlib inline 
# In[]
labels = pd.read_csv(
   "train.csv",
    names=['ID', 'Class']
)
# In[]
classes = labels[1:].Class.unique()
class_nums = pd.factorize(classes)
cdict = { k:v for (k,v) in zip(class_nums[1], class_nums[0])}
inverted_cdict = dict([[v,k] for k,v in cdict.items()])
train_files = glob('{}/data/Train/*.wav'.format(os.path.abspath('.')))
cdict

TARGET_LEN = 88200
HZ_SLICE = 14
# In[]
def data_to_spec(data):
    D = librosa.stft(data)
    return librosa.power_to_db(np.abs(D)**2, ref=np.median)
# In[]
def data_to_mfcc(data):
    return librosa.feature.mfcc(data) 
# In[]
def file_to_spec(filename):
    data, rate = librosa.load(filename)
    if data.shape[0] != TARGET_LEN: # handle short sounds
        data = extend_short_sounds(data)
    return data_to_spec(data)
# In[]
def file_to_mfcc(filename):
    data, rate = librosa.load(filename)
    if data.shape[0] != TARGET_LEN: # handle short sounds
        data = extend_short_sounds(data)
    return data_to_mfcc(data)
# In[]
def file_to_label(filename):
    fname = os.path.basename(filename)
    cid = fname.replace('.wav', '')
    classname = labels.loc[labels['ID'] == cid]
    return classname.values.tolist()[0][1] 
# In[]
def show_file(filename):
    plt.title(file_to_label(filename))
    librosa.display.specshow(file_to_spec(filename), x_axis='time', y_axis='linear');
    plt.colorbar(); 
# In[]    
def extend_short_sounds(data):
    stretched = librosa.effects.time_stretch(data, rate=data.shape[0]/TARGET_LEN)
    return stretched[:TARGET_LEN:]
# In[]
def drop_frequency_bands(data):
    return data[::HZ_SLICE,]
# In[]
def show_progress(progress):
    if progress % 40:
        print('.', end="")
    else:
        print('.')

f = train_files[0]
ss = file_to_spec(f)
mm = file_to_mfcc(f)
print(ss.shape)
mm.shape
# In[]
def wav_to_stft(files, labels):
    info = []
    data_arr = []
    progress = 0
    for filename in files:
        try:
            d = file_to_mfcc(filename)
            #d = drop_frequency_bands(d)
            data_arr.append(d)
            if isinstance(labels, pd.DataFrame): 
                # labeled case
                info.append( file_to_label(filename) )
            else:                                
                # unlabled case
                info.append(filename)
            progress+=1
            show_progress(progress)
            #print(d.shape)
        except:
            # As mentioned above, this notebook does not include the wav files
            # so if running with incomplete data, just skip via exception.
            pass 
    return data_arr, info
# In[]
# uncomment to calculate and save

train_data, train_labels = wav_to_stft(train_files, labels)
np.save('train_spec_mfcc', train_data)
np.save('train_label_mfcc', train_labels)
print('done')

# Load the saved results (comment in and out as needed)
train_spec_load = np.load('./train_spec_mfcc.npy')
train_labels_load = np.load('./train_label_mfcc.npy')
train_spec_load.shape
# In[]
def mean_mfcc_features(x):
    return x.mean(axis=2)

t = train_spec_load


train_spec = mean_mfcc_features(t)
train_spec.shape

plt.plot(t[0], 'b')
plt.figure()
plt.plot(train_spec[0], 'r')

ipd.Audio('data/Train/2140.wav')

# In[]
# load audio files with librosa
wav, sr = librosa.load('data/Train/2140.wav')
"""## Extracting MFCCs"""
# In[]
mfccs = librosa.feature.mfcc(y=wav, n_mfcc=13, sr=sr)

mfccs.shape
plt.figure(figsize=(25, 10))
librosa.display.specshow(mfccs, 
                         x_axis="time", 
                         sr=sr)
plt.colorbar(format="%+2.f")
plt.show()
