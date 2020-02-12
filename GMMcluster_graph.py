import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from speakerfeatures import extract_features
from scipy.io.wavfile import read

def gmm_cluster_graph(dir_path, file_name):
    sr, audio = read(dir_path+file_name+".wav")
    vector = extract_features(audio, sr)
    features = vector
    model= GaussianMixture(n_components=16, covariance_type='diag',n_init=3, init_params='random', random_state=0, tol=1e-9, max_iter=200)
    model.fit(features)
    labels=model.predict(features)
    plt.scatter(features[:,0], features[:,1], s=10, linewidth=1, cmap=plt.cm.get_cmap('tab20',16), c=labels)
    plt.title("Gaussian Mixture Model")
    plt.show()

