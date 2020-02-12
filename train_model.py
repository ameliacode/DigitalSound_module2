import os
import _pickle as cPickle
import numpy as np
from scipy.io.wavfile import read
from sklearn.mixture import GaussianMixture
from speakerfeatures import extract_features
from inputDataFiles import inputFile
import warnings
warnings.filterwarnings("ignore")

#path to training data
file_name = input("Input training data folder name: ")
train_num = int("Input the number of training data set")
source = os.path.join(os.getcwd(), file_name)

#path where training speakers will be saved
dest = os.getcwd()
train_file = "development_set_enroll.txt"

inputFile(train_file, source, train_num)
file_paths = open(train_file,'r')

count = 1
# Extracting features for each speaker (13 files per speakers)
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
        gmm = GaussianMixture(n_components = 16, covariance_type='diag',n_init = 3)     #n_components = the number of mixture components, n_init = initialization
        gmm.fit(features) #추출된 특징들 토대로 EM 알고리즘을 통해 모델 파라미터 측정

        # dumping the trained gaussian model
        picklefile = path.split(".wav")[0]+".gmm"
        cPickle.dump(gmm,open(dest + picklefile,'wb'))
        print('+ modeling completed for speaker:',picklefile," with data point = ",features.shape)
        features = np.asarray(())
        count = 0
    count = count + 1


