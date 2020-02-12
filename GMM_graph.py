import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from speakerfeatures import extract_features
from scipy.io.wavfile import read

def gmm_graph(dir_path, file_name):
    sr, audio = read(dir_path+file_name+".wav")
    vector=extract_features(audio, sr)
    features=vector
    model= GaussianMixture(n_components=16, covariance_type='full',n_init=3, init_params='random', random_state=0, tol=1e-9, max_iter=200)
    features=np.array(features).reshape(-1,1)
    model.fit(features)

    gmm_x=np.array(np.linspace(np.min(features), np.max(features), len(features))).reshape(-1,1)
    gmm_y=np.exp(model.score_samples(gmm_x))

    fig, ax = plt.subplots()
    ax.hist(features, bins=50, normed=True, alpha = 0.5, color="#0070FF")
    ax.plot(gmm_x, gmm_y, color="black", lw=1, label="Gaussian Mixture 1D Plot")

    ax.set_ylabel("Probability density")
    ax.set_xlabel("Arbitrary units")

    plt.show()

