from Model_AE import *
from Metrics import *
from Data_Process import *
import scipy.io as scio
from sklearn.cluster import KMeans
import numpy as np
import torch
import time
#from tensorboardX import SummaryWriter
import warnings
warnings.filterwarnings('ignore')
path_result = "./Latent_representation/"

######################################################### Setting #################################################
dataset = 'Yale'
Classification = True
Clustering = False
t_SNE = False

######################################  Load dataset  ###############################################################
load_data = Load_Data(dataset)
Features, Labels = load_data.CPU()

###  Normalization
norm = Normalized(Features)
Features = norm.MinMax()

Features = torch.Tensor(Features)
########################################### hyper-parameters##########################################
Epoch_Num = 200
learning_rate = 1e-3

Input_Dim = Features.shape[1]
hidden_layer_1 = 512
hidden_layer_2 = 128
hidden_layer_3 = hidden_layer_1
Output_Dim = Input_Dim

############################################ Results  Initialization ######################################
ACC_AE_total = []
NMI_AE_total = []
PUR_AE_total = []

ACC_AE_total_STD = []
NMI_AE_total_STD = []
PUR_AE_total_STD = []

F1_score = []
########################################################  Loss function ##############################################
loss_fn = torch.nn.MSELoss(size_average=False)
model_AE = myAE(Input_Dim, hidden_layer_1, hidden_layer_2, hidden_layer_3, Output_Dim)
optimzer = torch.optim.Adam(model_AE.parameters(), lr=learning_rate)

############################################################ Training ###################################################
for epoch in range(Epoch_Num):

    Latent_Representation, Features_Reconstrction = model_AE(Features)
    loss = loss_fn(Features, Features_Reconstrction)

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

            print("Epoch:{},Loss:{:.4f}".format(epoch+1, loss.item()))
            Latent_Representation = Latent_Representation.cpu().detach().numpy()

            ACC_H2 = []
            NMI_H2 = []
            PUR_H2 = []
            for i in range(10):
                kmeans = KMeans(n_clusters=max(np.int_(Labels).flatten()))
                Y_pred_OK = kmeans.fit_predict(Latent_Representation)
                Y_pred_OK = np.array(Y_pred_OK)
                Labels = np.array(Labels).flatten()
                AM = clustering_metrics(Y_pred_OK, Labels)
                ACC, NMI, PUR = AM.evaluationClusterModelFromLabel(print_msg=False)
                ACC_H2.append(ACC)
                NMI_H2.append(NMI)
                PUR_H2.append(PUR)

            print(f'ACC_H2=', 100 * np.mean(ACC_H2), '\n', 'NMI_H2=', 100 * np.mean(NMI_H2), '\n', 'PUR_H2=',
                  100 * np.mean(PUR_H2))

            ACC_AE_total.append(100 * np.mean(ACC_H2))
            NMI_AE_total.append(100 * np.mean(NMI_H2))
            PUR_AE_total.append(100 * np.mean(PUR_H2))

            ACC_AE_total_STD.append(100 * np.std(ACC_H2))
            NMI_AE_total_STD.append(100 * np.std(NMI_H2))
            PUR_AE_total_STD.append(100 * np.std(PUR_H2))
            np.save(path_result + "{}.npy".format(epoch + 1), Latent_Representation)
##################################################  Result #############################################################
if Clustering:

    Index_MAX = np.argmax(ACC_AE_total)

    ACC_AE_max = np.float(ACC_AE_total[Index_MAX])
    NMI_AE_max = np.float(NMI_AE_total[Index_MAX])
    PUR_AE_max = np.float(PUR_AE_total[Index_MAX])

    ACC_STD = np.float(ACC_AE_total_STD[Index_MAX])
    NMI_STD = np.float(NMI_AE_total_STD[Index_MAX])
    PUR_STD = np.float(PUR_AE_total_STD[Index_MAX])

    print('ACC_AE_max={:.2f} +- {:.2f}'.format(ACC_AE_max, ACC_STD))
    print('NMI_AE_max={:.2f} +- {:.2f}'.format(NMI_AE_max, NMI_STD))
    print('PUR_AE_max={:.2f} +- {:.2f}'.format(PUR_AE_max, PUR_STD))

elif Classification:
    Index_MAX = np.argmax(F1_score)
    print("AE: F1-score_max is {:.2f}".format(100*np.max(F1_score)))

########################################################### t- SNE #################################################
if t_SNE:
    print("dataset is {}".format(dataset))
    print("Index_Max = {}".format(Index_MAX))
    Latent_Representation_max = np.load(path_result + "{}.npy".format((Index_MAX+1) * 5))
    Features = np.array(Features)
    plot_embeddings(Latent_Representation_max, Features, Labels)



