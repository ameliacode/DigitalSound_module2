import os

#테스트할 샘플 파일 목록 작성 함수
def inputFile(test_file, path_dir, num ):
    f= open(test_file, "w")
    file_list = os.listdir(path_dir)
    file_list_wav = [file for file in file_list if file.endswith(".wav")]
    for i in range(0,num):
        f.write(file_list_wav[i]+"\n")
    f.close()
    print("All sample test files listed")

