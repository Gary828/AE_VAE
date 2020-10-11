import sys
sys.path.append('D:\OneDrive - mail.nwpu.edu.cn\Optimal\Public\Python\Pre_Process')
from Model_AE import *
from Metrics import *
import scipy.io as scio
from Data_Process import *
from sklearn.cluster import KMeans
import numpy as np
import torch
import time
import warnings
warnings.filterwarnings('ignore')
path_result = "./Latent_representation/"
######################################################### Setting #################################################
dataset = 'Yale'
Classification = True
Clustering = False
t_SNE = True

#######################################Load dataset  ##############################################################
load_data = Load_Data(dataset)
Features, Labels = load_data.CPU()

###  Normalization
norm = Normalized(Features)
Features = norm.Normal()

Features = torch.Tensor(Features)
################################################## parameters ######################################################
Epoch_Num = 200
learning_rate = 1e-3

hidden_layer_1 = 512
hidden_layer_2 = 128
hidden_layer_3 = 512

batch_n = Features.shape[0]
input_dim = Features.shape[1]
output_dim = input_dim

################################################# Result Initialization ################################################
ACC_VAE_total = []
NMI_VAE_total = []
PUR_VAE_total = []

ACC_VAE_total_STD = []
NMI_VAE_total_STD = []
PUR_VAE_total_STD = []

F1_score = []
################################################  Loss_Function ########################################################
def Loss_Function(Features_Reconstruction, Features, H_2_mean, H_2_std):

    re_loss = torch.nn.MSELoss(size_average=False)
    Reconstruction_Loss = re_loss(Features_Reconstruction, Features)
    KLD_element = 1 + 2 * H_2_std - H_2_mean.pow(2) - H_2_std.exp() ** 2
    KL_Divergence = torch.sum(KLD_element).mul_(-0.5)
    return Reconstruction_Loss, KL_Divergence

###############################################  Model ###############################################################
model_VAE = myVAE(input_dim, hidden_layer_1, hidden_layer_2, hidden_layer_3, output_dim)
optimzer = torch.optim.Adam(model_VAE.parameters(), lr=learning_rate)

for epoch in range(Epoch_Num):
    Latent_Representation, Features_Reconstruction, H_2_mean, H_2_std = model_VAE(Features)
    Reconstruction_Loss, KL_Divergence = Loss_Function(Features_Reconstruction, Features, H_2_mean, H_2_std)
    loss = Reconstruction_Loss + KL_Divergence

    optimzer.zero_grad()
    loss.backward()
    optimzer.step()

    Latent_Representation = Latent_Representation.cpu().detach().numpy()
    ##################################################### Results  ####################################################
    if Classification and (epoch + 1) % 5 == 0:
        print("Epoch:{},Loss:{:.4f}".format(epoch + 1, loss.item()))
        score = mySVM(Latent_Representation, Labels, scale=0.3)
        print("Epoch[{}/{}], scale = {}".format(epoch + 1, Epoch_Num, score))
        np.save(path_result + "{}.npy".format(epoch + 1), Latent_Representation)
        F1_score.append(score)

    elif Clustering and (epoch + 1) % 5 == 0:
        print("Epoch[{}/{}], Reconstruction_Loss: {:.4f}, KL_Divergence: {:.4f}"
              .format(epoch + 1, epoch_n,  Reconstruction_Loss.item(), KL_Divergence))


        ACC_H2 = []
        NMI_H2 = []
        PUR_H2 = []
        for i in range(5):
            kmeans = KMeans(n_clusters=max(np.int_(Labels).flatten()))
            Y_pred_OK = kmeans.fit_predict(Latent_Representation.detach().numpy())
            Y_pred_OK = np.array(Y_pred_OK)
            Labels = np.array(Labels)
            Labels = Labels.flatten()
            AM = clustering_metrics(Y_pred_OK, Labels)
            ACC, NMI, PUR = AM.evaluationClusterModelFromLabel(print_msg=False)
            ACC_H2.append(ACC)
            NMI_H2.append(NMI)
            PUR_H2.append(PUR)
        print('ACC_H2=', 100 * np.mean(ACC_H2), '\n', 'NMI_H2=', 100 * np.mean(NMI_H2), '\n', 'PUR_H2=',
              100 * np.mean(PUR_H2))
        ACC_VAE_total.append(100 * np.mean(ACC_H2))
        NMI_VAE_total.append(100 * np.mean(NMI_H2))
        PUR_VAE_total.append(100 * np.mean(PUR_H2))

        ACC_VAE_total_STD.append(100 * np.std(ACC_H2))
        NMI_VAE_total_STD.append(100 * np.std(NMI_H2))
        PUR_VAE_total_STD.append(100 * np.std(PUR_H2))


##################################################  Result ##################################################
if Clustering:
    Index_MAX = np.argmax(ACC_VAE_total)

    ACC_VAE_max = np.float(ACC_VAE_total[Index_MAX])
    NMI_VAE_max = np.float(NMI_VAE_total[Index_MAX])
    PUR_VAE_max = np.float(PUR_VAE_total[Index_MAX])

    ACC_STD = np.float(ACC_VAE_total_STD[Index_MAX])
    NMI_STD = np.float(NMI_VAE_total_STD[Index_MAX])
    PUR_STD = np.float(PUR_VAE_total_STD[Index_MAX])

    print('ACC_VAE_max={:.2f} +- {:.2f}'.format(ACC_VAE_max, ACC_STD))
    print('NMI_VAE_max={:.2f} +- {:.2f}'.format(NMI_VAE_max, NMI_STD))
    print('PUR_VAE_max={:.2f} +- {:.2f}'.format(PUR_VAE_max, PUR_STD))
elif Classification:
    Index_MAX = np.argmax(F1_score)
    print("VAE: F1-score_max is {:.2f}".format(100 * np.max(F1_score)))
    ########################################################## t- SNE #################################################
if t_SNE:
    print("dataset is {}".format(dataset))
    print("Index_Max = {}".format(Index_MAX))
    Latent_Representation_max = np.load(path_result + "{}.npy".format((Index_MAX + 1) * 5))
    Features = np.array(Features)
    plot_embeddings(Latent_Representation_max, Features, Labels)