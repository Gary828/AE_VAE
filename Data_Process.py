import scipy.io as scio
import torch

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
############################################ Load the datasets  ########################################################
class Load_Data():
    def __init__(self, dataset):
        self.dataset = dataset

    def CPU(self):
        # path1 = os.path.abspath('.')
        path = './Datasets/{}.mat'.format(self.dataset)
        data = scio.loadmat(path)

        Labels = data['Y']
        if Labels.shape[0] == 1:
            Labels = np.reshape(Labels, (Labels.shape[1], 1))
        features = data['X']

        if features.shape[1] == Labels.shape[0]:
           features = features.T  # change the data into n×d

        return features, Labels

    def GPU(self):
        path = './Dataset/{}.mat'.format(self.dataset)
        data = scio.loadmat(path)

        labels = data['Y']
        Labels = np.array(labels).flatten()

        if labels.shape[0] == 1:
            labels = np.reshape(labels, (labels.shape[1], 1))

        features = data['X']

        if features.shape[1] == labels.shape[0]:
            features = features.T

        return features, Labels

    def Graph(self):
        path = './Datasets/Graph_Datasets/{}/'.format(self.dataset)

        Features = sp.load_npz(path + 'Features.npz')
        Features = Features.toarray()

        Labels = sp.load_npz(path + 'Labels.npz')
        Labels = Labels.toarray()
        Labels = Labels.reshape(Labels.shape[1], 1)

        Adjacency = sp.load_npz(path + 'Adjacency.npz')
        Adjacency = Adjacency.toarray()

        return Features, Labels, Adjacency

######################################### Data preprocess ##############################################################
class Normalized():
    def __init__(self, X):
        self.X = X

    def Normal(self):
        # X: n×d，normalize each dimension of X
        return (self.X - np.mean(self.X, axis = 0)) / (np.std(self.X, axis = 0, ddof=1) + 1e-4)

    def Length(self):
        # X:n*d, Make each row of X equal to each other.
        meth = np.sum(self.X, axis = 1)
        meth = meth.reshape(meth.shape[0], 1)
        return self.X / (meth+ 1e-6)

    def MinMax(self):
        # Normalize matrix by Min and Max
        # X: n*d, apply to each columns
        min = np.min(self.X, axis=0)
        max = np.max(self.X, axis=0)
        return (self.X - min) / (max - min + 1e-6)

def L2_distance_2(A, B):
    A = A.T
    B = B.T
    AA = torch.sum(A*A, dim=0, keepdims=True)
    BB = torch.sum(B*B, dim=0, keepdims=True)
    AB = (A.T).mm(B)
    D = ((AA.T).repeat(1, BB.shape[1])) + (BB.repeat(AA.shape[1], 1)) - 2 * AB
    D = torch.abs(D)
    return D
######################################################### mySVM ########################################################
def mySVM(Latent_representation, Labels, scale=0.3):
    X_train, X_test, Y_train, Y_test = train_test_split(Latent_representation, Labels, test_size=scale, random_state=0)
    clf = svm.SVC(probability=True)
    clf.fit(X_train, Y_train)

    Pred_Y = clf.predict(X_test)
    score = f1_score(Pred_Y, Y_test, average='weighted')
    return score

############################################# plot the t-SNE ###########################################################
def plot_embeddings(embeddings, Features, Labels):

    # norm = Normalized(embeddings)
    # embeddings = norm.MinMax()

    emb_list = []
    for k in range(Features.shape[0]):
        emb_list.append(embeddings[k])
    emb_list = np.array(emb_list)

    model = TSNE(n_components=2, init="pca")
    # model = TSNE(n_components=2)
    node_pos = model.fit_transform(emb_list)

    color_idx = {}
    for i in range(Features.shape[0]):
        color_idx.setdefault(Labels[i][0], [])
        color_idx[Labels[i][0]].append(i)

    for c, idx in color_idx.items():
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c, s = 5) # c=node_colors)
    plt.axis('off')
    # plt.legend()
    plt.gca.legend_ = None
    plt.show()

def get_weight_initial(d1, d2):
    bound = torch.sqrt(torch.Tensor([6.0 / (d1 + d2)]))
    nor_W = -bound + 2*bound*torch.rand(d1, d2)
    return torch.Tensor(nor_W)