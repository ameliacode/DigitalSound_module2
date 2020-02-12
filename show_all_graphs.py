from MelSpectrogram_graph import mel_spectogram
from MFCC_graph import mfcc_graph
from GMMcluster_graph import gmm_cluster_graph
from GMM_graph import gmm_graph

dir_path = input("Input train data abs directory: ")
file_name = input("Input speaker name: ")

#Mel Spectogram
mel_spectogram(dir_path, file_name)

#MFCC
mfcc_graph(dir_path, file_name)

#GMM clustered
gmm_cluster_graph(dir_path, file_name)

#GMM graph
gmm_graph(dir_path, file_name)
