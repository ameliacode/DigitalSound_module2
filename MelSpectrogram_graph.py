import numpy as np
import librosa.display
import matplotlib.pyplot as plt

def mel_spectogram(dir_path, file_name):
    y, sr = librosa.load(dir_path+file_name+".wav")
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max), sr=sr, y_axis='mel', x_axis = 'time')
    plt.title('Mel Spectogram')
    plt.show()
