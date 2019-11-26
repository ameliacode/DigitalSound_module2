import _pickle as cPickle
import numpy as np
from scipy.io.wavfile import read
from sklearn.mixture import GaussianMixture
from speakerfeatures import extract_features
from inputDataFiles import inputFile
import warnings
warnings.filterwarnings("ignore")

#path to training data
source   = "C:\\Users\\user\\PycharmProjects\\Module2\\trainData\\"

#path where training speakers will be saved
dest = "C:\\Users\\user\\PycharmProjects\\Module2\\"
train_file = "development_set_enroll.txt"

inputFile(train_file, source, 156)
file_paths = open(train_file,'r')

count = 1 #주어진 모델 데이터는 하나 밖에 없으므로
# Extracting features for each speaker (5 files per speakers)
features = np.asarray(())
for path in file_paths:
    path = path.strip()
    print(path)

    # read the audio
    sr,audio = read(source + path)

    # extract 40 dimensional MFCC & delta MFCC features
    vector = extract_features(audio,sr)

    if features.size == 0:
        features = vector
    else:
        features = np.vstack((features, vector))
    # when features of 1 files of speaker are concatenated, then do model training

    if count == 13:
        gmm = GaussianMixture(n_components = 16, covariance_type='diag',n_init = 3)
        gmm.fit(features)

        # dumping the trained gaussian model
        picklefile = path.split("9.wav")[0]+".gmm"
        cPickle.dump(gmm,open(dest + picklefile,'wb'))
        print('+ modeling completed for speaker:',picklefile," with data point = ",features.shape)
        features = np.asarray(())
        count = 0
    count = count + 1


