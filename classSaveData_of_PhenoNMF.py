'''

This section is designed to store the output matrices
'''
import numpy as np
import pandas as pd

class SaveHexaDataWH(object):
    '''
    Joint NMF
    '''

    def __init__(self, X1, X2, X3, rank):
        '''
        Constructor
        '''
        self.rank = rank
        self.W = pd.DataFrame()
        self.H1 = pd.DataFrame(columns = X1.columns)
        self.H2 = pd.DataFrame(columns = X2.columns)
        self.H3 = pd.DataFrame(columns = X3.columns)

    # Saving the W and H matrices 
    def save_W_H(self, W, H1, H2, H3, rank, trial):
        self.W = pd.concat([self.W, self.make_M(W, rank, trial)], ignore_index=True)
        #self.W = self.W.append(self.make_M(W, rank, trial))
        self.W = pd.concat([self.H1, self.make_M(H1, rank, trial)], ignore_index=True)
        #self.H1 = self.H1.append(self.make_M(H1, rank, trial))
        self.W = pd.concat([self.H2, self.make_M(H2, rank, trial)], ignore_index=True)
        #self.H2 = self.H2.append(self.make_M(H2, rank, trial))
        self.W = pd.concat([self.H3, self.make_M(H3, rank, trial)], ignore_index=True)
        #self.H3 = self.H3.append(self.make_M(H3, rank, trial))

            
    def make_M(self, M, rank, trial):
        M['rank'] = rank
        M['trial'] = trial
        return M

class SaveBestHexaDataWH(object):
    '''
    Save Best Matrix from Joint NMF results
    '''

    def __init__(self, X1, X2, X3, rank):
        '''
        Constructor
        '''
        self.rank = rank
        self.W = pd.DataFrame()
        self.H1 = pd.DataFrame(columns = X1.columns)
        self.H2 = pd.DataFrame(columns = X2.columns)
        self.H3 = pd.DataFrame(columns = X3.columns)
        self.bestscore = float("inf")
                
    def save_W_H(self, W, H1, H2, H3, rank, trial):
        self.W = self.make_M(W, rank, trial)
        self.H1 = self.make_M(H1, rank, trial)
        self.H2 = self.make_M(H2, rank, trial)
        self.H3 = self.make_M(H3, rank, trial)
    
    def make_M(self, M, rank, trial):
        M['rank'] = rank
        M['trial'] = trial
        return M
    
#    def run(self):
        