

import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from matplotlib.pyplot import savefig, imshow, set_cmap
from classHexaNMF_mask import *
#from classHexaConsensusMatrix import *
# test sil
from classHexaConsensusMatrix_silhouette import *
from main.classSaveData_of_PhenoNMF import *
from multiprocessing import Pool
from multiprocessing import Process

class Run_HexaNMF(object):
    '''
    classdocs
    '''

    def __init__(self, X1, X2, X3, maskX1, maskX2, maskX3, K, maxiter, nloop):
        '''
        Constructor
        '''
        self.X1 = X1
        self.X2 = X2
        self.X3 = X3

        self.maskX1 = maskX1
        self.maskX2 = maskX2
        self.maskX3 = maskX3

        self.K = K
        self.maxiter = maxiter
        self.nloop = nloop
    #this　function is designed for runing once    
    def runNMF_singlerun(self, iloop):
        
        """
        Run Joint NMF
        """
        jnmf = HexaNMF_mask(self.X1, self.X2, self.X3, self.maskX1, self.maskX2, self.maskX3, self.K, self.maxiter)
        jnmf.check_nonnegativity()
        jnmf.check_samplesize()
        jnmf.initialize_W_H()
        #jnmf.update_euclidean_multiplicative()
        print("start updating")
        jnmf.wrapper_calc_euclidean_multiplicative_update()
        print("updating over")
        print("prepare for output")
        jnmf.print_distance_report_of_HW_to_X(iloop)
        jnmf.set_PanH()
        return jnmf
    #this　function is designed for runing multiple times            
    def runNMF_multirun(self, K, nloop):
        self.K = K
        self.nloop = nloop
        
        """
        Run Joint NMF
        Calculate consensus matrix
        """
        #FOR silhouette add K
        cmatrix = HexaConsensusMatrix(self.X1, self.X2, self.X3, self.K)
        sdata = SaveHexaDataWH(self.X1, self.X2, self.X3, self.K)
        bestdata = SaveBestHexaDataWH(self.X1, self.X2, self.X3, self.K)
        report = pd.DataFrame()

        #p = Pool(self.nloop)
        #jnmfs = p.map(MulHelper(self, 'runNMF_singlerun'), range(self.nloop))

        for i in range(self.nloop):
            #jnmf = jnmfs[i]
            jnmf = self.runNMF_singlerun(i)
            report = pd.concat([report, jnmf.get_distance_report_of_HW_to_X(i)], ignore_index=True)
            #report.append(jnmf.get_distance_report_of_HW_to_X(i))
            connW = cmatrix.calcConnectivityW(jnmf.W)
            connH1 = cmatrix.calcConnectivityH(jnmf.H1)
            connH2 = cmatrix.calcConnectivityH(jnmf.H2)
            connH3 = cmatrix.calcConnectivityH(jnmf.H3)
            connPanH = cmatrix.calcConnectivityH(jnmf.PanH)
            cmatrix.addConnectivityMatrixtoConsensusMatrix(connW, connH1, connH2, connH3, connPanH)
            sdata.save_W_H(jnmf.W, jnmf.H1, jnmf.H2, jnmf.H3, K, i)
            if jnmf.eucl_dist < bestdata.bestscore:
                bestdata.bestscore = jnmf.eucl_dist
                bestdata.save_W_H(jnmf.W, jnmf.H1, jnmf.H2, jnmf.H3, K, i)
            del connW, connH1, connH2, connH3, connPanH    
        cmatrix.finalizeConsensusMatrix(2)

        
        """
        Save output files: W, H1, H2, and H3
        """
        return cmatrix, sdata, bestdata, report

class MulHelper(object):
    def __init__(self, cls, mtd_name):
        self.cls = cls
        self.mtd_name = mtd_name
    def __call__(self, *args, **kwargs):
        return getattr(self.cls, self.mtd_name)(*args, **kwargs)


