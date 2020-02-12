import matplotlib.pyplot as plt
import librosa.display

#MFCC
def mfcc_graph(dir_path, file_name):
    y, sr = librosa.load(dir_path+file_name+".wav")
    mfccs=librosa.feature.mfcc(y=y,sr=sr, n_mfcc=40)
    librosa.display.specshow(mfccs, x_axis='time')
    plt.title('MFCC')
    plt.tight_layout()
    plt.show()
