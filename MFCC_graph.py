import numpy as np
import librosa.display
import matplotlib.pyplot as plt

dir_path  = "C:\\Users\\user\\PycharmProjects\\Module2\\trainData\\"
file_name = input("Input speaker name: ")

y, sr = librosa.load(dir_path+file_name+".wav")
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
librosa.display.specshow(librosa.power_to_db(S, ref=np.max), sr=sr, y_axis='mel', x_axis = 'time')
plt.title('Mel Spectogram')
plt.show()

mfccs=librosa.feature.mfcc(y=y,sr=sr, n_mfcc=40)
librosa.display.specshow(mfccs, x_axis='time')
plt.title('MFCC')
plt.tight_layout()
plt.show()
