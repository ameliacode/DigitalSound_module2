import os
import _pickle as cPickle
import numpy as np
from scipy.io.wavfile import read
from speakerfeatures import extract_features
import warnings
warnings.filterwarnings("ignore")
import time

#샘플 파일의 총 갯수를 입력합니다
#total_num = int(input("Input total number of test samples: "))

#테스트할 샘플 파일 목록 작성 함수
def inputFile(test_file, path_dir, num ):
    f= open(test_file, "w")
    file_list = os.listdir(path_dir)
    file_list_wav = [file for file in file_list if file.endswith(".wav")]
    for i in range(0,num):
        f.write(file_list_wav[i]+"\n")
    f.close()

#path to training data
source   = "C:\\Users\\user\\PycharmProjects\\Module2\\sampleTest\\"
modelpath = "C:\\Users\\user\\PycharmProjects\\Module2\\"
test_file = "development_set_test.txt"

inputFile(test_file, source, 144)              #테스트 파일 목록화

print("All sample test files listed")

file_paths = open(test_file,'r')

gmm_files = [os.path.join(modelpath,fname) for fname in os.listdir(modelpath) if fname.endswith('.gmm')]

#Load the Gaussian gender Models
models = [cPickle.load(open(fname,'rb')) for fname in gmm_files]
#models = []
speakers = [fname.split("\\")[-1].split(".gmm")[0] for fname in gmm_files]

# Read the test directory and get the list of test audio files
for path in file_paths:

    path = path.strip()
    print(path)
    sr,audio = read(source + path)
    vector   = extract_features(audio,sr)

    log_likelihood = np.zeros(len(models))

    for i in range(len(models)):
        gmm = models[i]  #checking with each model one by one
        scores = np.array(gmm.score(vector))
        log_likelihood[i] = scores.sum()

    winner = np.argmax(log_likelihood)
    print("\tdetected as - ", speakers[winner])
    time.sleep(1.0)

