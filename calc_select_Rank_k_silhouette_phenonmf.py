

import os
import time
import pandas as pd
import numpy as np

from classRun_HexaNMF_CCLE_mask_multi import *


if __name__ == '__main__':

    start = time.time()

    """
    Read input data files: X1, X2, X3
    """
    X1ori = pd.read_csv("/Users/yutongdai/Desktop/project/MIMIC/processed data/sampling_dig_9540.csv",header=0, index_col=0, na_values='NaN')
    X1 = X1ori[X1ori.notnull().any(axis=1)]  
    # X1 = X1ori.dropna()    
    # X1 = X1ori[X1ori.notnull().all(axis=1)]    
    maskX1 = 1 - X1.isnull()
    X1 = X1.fillna(0)

    X2ori = pd.read_csv('/Users/yutongdai/Desktop/project/MIMIC/processed data/sampling_labeve100_9540.csv',
        header=0, index_col=0, na_values='NaN')
    X2 = X2ori[X1ori.notnull().any(axis=1)]  
    # X2 = X2ori[X1ori.notnull().all(axis=1)]    
    maskX2 = 1 - X2.isnull()
    X2 = X2.fillna(0)

    # X3ori = pd.read_csv('C://Users/taiho/Desktop/desktop/Academic/NMF_Python/11_HexaNMF_CCLE/input_CCLE_linage_binmat_5data_modified.csv',
    X3ori = pd.read_csv('/Users/yutongdai/Desktop/project/MIMIC/processed data/sampling_drug300_9540.csv',
                        header=0, index_col=0, na_values='NaN')
    # X3ori = pd.read_csv('/Users/yutongdai/Desktop/paper/hexNMF/hex data_DAI/X2_MUT_cbio_normalized_binary_data_Complete.edition.csv',header=0, index_col=0, na_values='NaN')
    # X3 = X3ori[X1ori.notnull().any(axis=1)]
    
    X3 = X3ori.loc[:, X3ori.notnull().any(axis=0)]
    X3 = X3.loc[X1ori.notnull().any(axis=1), :]  

    # X3 = X3.ix[X1ori.notnull().all(axis=1),:] 
    maskX3 = 1 - X3.isnull()
    X3 = X3.fillna(0)

    print("input over")

    """
    Set parameters
    """
    nloop = 5
    maxiter = 3000

    #c_store = pd.DataFrame(columns=['RANK', 'CCC'])
    c_store = pd.DataFrame(columns=['RANK', 'CCC', 'silhouette_avg'])
    for K in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]:
        print("rank k :", K)

        """
        Set output directory
        """
        savedir = '/Users/yutongdai/Desktop/project/MIMIC/kselect_test_cell_k%d_n%d_iter%d' % (K, nloop, maxiter)
        if not os.path.exists(savedir):
            os.mkdir(savedir)

        print("set dir over")
        """
        Run Joint NMF
        Calculate consensus matrix
        """
        print("running start")
        print(nloop)
        nmf = Run_HexaNMF(X1, X2, X3, maskX1, maskX2, maskX3, K, maxiter, nloop)
        #report = nmf.runNMF_singlerun(K, nloop)

        cmatrix, sdata, bdata, report = nmf.runNMF_multirun(K, nloop)

        """
        Save output files: W, H1, H2
        """


        #sdata.W.to_csv(savedir + '/jnmf_dataW_CCLE_k%d.csv' % K)
        #sdata.H1.to_csv(savedir + '/jnmf_dataH1_CCLE_k%d.csv' % K)
        #sdata.H2.to_csv(savedir + '/jnmf_dataH2_CCLE_k%d.csv' % K)
        #sdata.H3.to_csv(savedir + '/jnmf_dataH3_CCLE_k%d.csv' % K)
        #sdata.H4.to_csv(savedir + '/jnmf_dataH4_CCLE_k%d.csv' % K)
        #sdata.H5.to_csv(savedir + '/jnmf_dataH5_CCLE_k%d.csv' % K)
        #sdata.H6.to_csv(savedir + '/jnmf_dataH6_CCLE_k%d.csv' % K)
        #bdata.W.to_csv(savedir + '/jnmf_bestW_CCLE_k%d.csv' % K)
        #bdata.H1.to_csv(savedir + '/jnmf_bestH1_CCLE_k%d.csv' % K)
        #bdata.H2.to_csv(savedir + '/jnmf_bestH2_CCLE_k%d.csv' % K)
        #bdata.H3.to_csv(savedir + '/jnmf_bestH3_CCLE_k%d.csv' % K)
        #bdata.H4.to_csv(savedir + '/jnmf_bestH4_CCLE_k%d.csv' % K)
        #bdata.H5.to_csv(savedir + '/jnmf_bestH5_CCLE_k%d.csv' % K)
        #bdata.H6.to_csv(savedir + '/jnmf_bestH6_CCLE_k%d.csv' % K)


        report.to_csv(savedir + '/jnmf_distance_report_CCLE_k%d_n%d_iter%d.csv' % (K, nloop, maxiter))
        """
        Plot consensus matrix
        """
        reorderM_hierarchicalcmW, reorderM_kmeanscmW, c, silhouette_avg = cmatrix.perform_clustering_analysis(cmatrix.cmW, K)
        #reorderedcmW, c, silhouette_avg = cmatrix.reorderConsensusMatrix(cmatrix.cmW)
        new_row = {"RANK": K, "CCC": c, "silhouette_avg": silhouette_avg}
        new_row = pd.DataFrame([new_row])
        c_store = pd.concat([c_store, new_row], ignore_index=True)

        imshow(reorderM_hierarchicalcmW, cmap='Blues', interpolation="nearest")
        plt.colorbar()
        savefig(savedir + '/jnmf_CCLE_ConsensusMatrixW_k%d.clustered.png' % K)
        plt.close()

        imshow(reorderM_kmeanscmW, cmap='Blues', interpolation="nearest")
        plt.colorbar()
        savefig(savedir + '/jnmf_CCLE_Kmeans_ConsensusMatrixW_k%d.clustered.png' % K)
        plt.close()
        """
        imshow(cmatrix.cmPanH, cmap='Blues', interpolation="nearest")
        plt.colorbar()
        savefig(savedir + '/jnmf_CCLE_ConsensusMatrixPanH_k%d.png' % K)
        plt.close()
        """

        """
        Save output files: cmW, cmPanH
        """

        #cmatrix.cmW.to_csv(savedir + '/jnmf_CCLE_ConsensusMatrixW_k%d.csv' % K)
        #cmatrix.cmPanH.to_csv(savedir + '/jnmf_CCLE_ConsensusMatrixPanH_k%d.csv' % K)
        reorderM_hierarchicalcmW.to_csv(savedir + '/jnmf_CCLE_ConsensusMatrixW_k%d.clustered.csv' % K)
        reorderM_kmeanscmW.to_csv(savedir + '/jnmf_CCLE_Kmeans_ConsensusMatrixW_k%d.clustered.csv' % K)
        c_store.to_csv(savedir + '/CCC_k%d.csv' % K)
        elapsed_time = time.time() - start
        print("time cost is :")
        print(elapsed_time)