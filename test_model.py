import os
import _pickle as cPickle
import numpy as np
from scipy.io.wavfile import read
from speakerfeatures import extract_features
from inputDataFiles  import inputFile
import warnings
warnings.filterwarnings("ignore")
import time

#샘플 파일의 총 갯수를 입력합니다 + 테스트하게 될 폴더명을 입력합니다
total_num = int(input("Input total number of test samples: "))
folder_name = input("Input Folder name: ")

#path to training data
source = os.path.join(os.getcwd(), folder_name)
modelpath = os.getcwd()
test_file = "development_set_test.txt"

inputFile(test_file, source, total_num)              #테스트 파일 목록화

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

    log_likelihood = np.zeros(len(models))  #모델의 갯수만큼 0으로 초기화

    for i in range(len(models)):
        gmm = models[i]  #checking with each model one by one
        scores = np.array(gmm.score(vector)) #추출된 벡터의 per-sample average log-likehood를 array화
        log_likelihood[i] = scores.sum() #각 스코어의 합을 log_likelihood에 저장

    winner = np.argmax(log_likelihood) #유사값이 가장 큰 색인 = winner
    print("\tdetected as - ", speakers[winner])
    time.sleep(1.0)

