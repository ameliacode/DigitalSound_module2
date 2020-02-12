import os

#테스트할 샘플 파일 목록 작성 함수
def inputFile(test_file, path_dir, num ):
    f= open(test_file, "w")     #해당 파일 쓰기
    file_list = os.listdir(path_dir)    #os 라이브러리 listdir: 해당 디렉터리 파일의 리스트
    file_list_wav = [file for file in file_list if file.endswith(".wav")] #wav로 끝나는 파일들을 리스트 저장
    for i in range(0,num):
        f.write(file_list_wav[i]+"\n")  #파일 리스트명 저장
    f.close()
    print("All sample test files listed")

