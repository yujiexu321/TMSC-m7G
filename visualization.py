import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def ROC_curv(roc_data1,roc_data2,roc_data3,roc_data4):
    roc_data1_x, roc_data1_y, roc_auc1 = roc_data1
    roc_data2_x, roc_data2_y, roc_auc2 = roc_data2
    roc_data3_x, roc_data3_y, roc_auc3 = roc_data3
    roc_data4_x, roc_data4_y, roc_auc4 = roc_data4
    plt.plot(roc_data1_x,roc_data1_y,'b', label='TE+Enc:AUC = %0.4f' % roc_auc1)
    plt.plot(roc_data2_x, roc_data2_y, 'y', label='TE+PE+Enc:AUC = %0.4f' % roc_auc2)
    plt.plot(roc_data3_x, roc_data3_y, 'g', label='MSSE+PE+Enc:AUC = %0.4f' % roc_auc3)
    plt.plot(roc_data4_x, roc_data4_y, 'r', label='TMSC-m7G:AUC = %0.4f' % roc_auc4)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'p--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

# roc_data_E_Enc = np.load("TE+Enc_roc.npy",allow_pickle=True)
# roc_data_E_PE_Enc = np.load("TE+PE+Enc_roc.npy",allow_pickle=True)
# roc_data_MSSE_PE_Enc = np.load("MSSE+PE+Enc_roc.npy",allow_pickle=True)
# roc_data_TMSC_m7G = np.load("TMSC-m7G_roc.npy",allow_pickle=True)

# ROC_curv(roc_data_E_Enc, roc_data_E_PE_Enc, roc_data_MSSE_PE_Enc, roc_data_TMSC_m7G)
#
def PRC_curv(prc_data1,prc_data2,prc_data3,prc_data4):
    prc_data1_x, prc_data1_y, prc_auc1 = prc_data1
    prc_data2_x, prc_data2_y, prc_auc2 = prc_data2
    prc_data3_x, prc_data3_y, prc_auc3 = prc_data3
    prc_data4_x, prc_data4_y, prc_auc4 = prc_data4
    plt.plot(prc_data1_x,prc_data1_y,'b', label='TE+Enc:AP = %0.4f' % prc_auc1)
    plt.plot(prc_data2_x, prc_data2_y, 'y', label='TE+PE+Enc:AP = %0.4f' % prc_auc2)
    plt.plot(prc_data3_x, prc_data3_y, 'g', label='MSSE+PE+Enc:AP = %0.4f' % prc_auc3)
    plt.plot(prc_data4_x, prc_data4_y, 'r', label='TMSC-m7G:AP = %0.4f' % prc_auc4)
    plt.legend(loc='lower left')
    plt.plot([0, 1], [0, 1], 'p--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.show()

# prc_data_E_Enc = np.load("TE+Enc_prc.npy",allow_pickle=True)
# prc_data_E_PE_Enc = np.load("TE+PE+Enc_prc.npy",allow_pickle=True)
# prc_data_MSSE_PE_Enc = np.load("MSSE+PE+Enc_prc.npy",allow_pickle=True)
# prc_data_TMSC_m7G = np.load("TMSC-m7G_prc.npy",allow_pickle=True)

# PRC_curv(prc_data_E_Enc, prc_data_E_PE_Enc, prc_data_MSSE_PE_Enc, prc_data_TMSC_m7G)



import umap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def rep_plt(X,y,epoch):
    title = 'Learned Feature UMAP Visualisation, Epoch[{}]'.format(epoch)
    reducer = umap.UMAP(min_dist=0.1, n_components=2)   # umap
    # reducer = PCA(n_components=2)     # pca
    # reducer = TSNE(n_components=2, perplexity=12)    # tsne

    X_embedded = reducer.fit_transform(X)
    palette = sns.color_palette("bright", 2)
    sns.scatterplot(X_embedded[:, 0], X_embedded[:, 1],hue=y, legend='full', palette=palette)
    plt.title(title)
    plt.show()

