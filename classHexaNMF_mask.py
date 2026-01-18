'''

This part is the updated formula part of the algorithm, and the specific formula is consistent as derived in the main text

'''
import numpy as np
import pandas as pd

class HexaNMF_mask(object):
    '''
    Joint NMF
    '''


    def __init__(self, X1, X2, X3, maskX1, maskX2, maskX3, rank, maxiter):
        '''
        Constructor
        '''
        self.X1 = X1
        self.X2 = X2
        self.X3 = X3
        self.maskX1 = maskX1
        self.maskX2 = maskX2
        self.maskX3 = maskX3
        self.rank = rank
        self.maxiter = maxiter
        self.report = pd.DataFrame(columns = ["K", "iter", "difference of WH - WH_pre", "Euclidean distance of X - WH", "MASKED Euclidean distance of X - WH", "Error: avg' |X - WH|/X"])
    
    #this　function is designed for Ensuring non-negativity    
    def check_nonnegativity(self):
        if(self.X1.min().min() < 0 or self.X2.min().min() < 0 or self.X3.min().min() < 0 ):
            raise Exception('non negativity')
    #this　function is designed for checking the input size of input matrix
    def check_samplesize(self):
        if(self.X1.shape[0] != self.X2.shape[0] or self.X1.shape[0] != self.X3.shape[0]):
            print(self.X1.shape[0])
            print(self.X2.shape[0])
            print(self.X3.shape[0])
            raise Exception('sample size')

    #this　function is designed for randomly initializing the matrix       
    def initialize_W_H(self):
        self.W = pd.DataFrame(np.random.rand(self.X1.shape[0], self.rank), index = self.X1.index, columns = map(str, range(1, self.rank+1)))
        self.H1 = pd.DataFrame(np.random.rand(self.rank, self.X1.shape[1]), index = map(str, range(1, self.rank+1)), columns = self.X1.columns)
        self.H2 = pd.DataFrame(np.random.rand(self.rank, self.X2.shape[1]), index = map(str, range(1, self.rank+1)), columns = self.X2.columns)
        self.H3 = pd.DataFrame(np.random.rand(self.rank, self.X3.shape[1]), index = map(str, range(1, self.rank+1)), columns = self.X3.columns)
        self.X1r_pre = np.dot(self.W, self.H1)
        self.X2r_pre = np.dot(self.W, self.H2)
        self.X3r_pre = np.dot(self.W, self.H3)
        self.eps = np.finfo(self.W.to_numpy().dtype).eps

    #this　function is designed for implementation of the update formula (as demonstrated in the main text)
    '''
    def calc_euclidean_multiplicative_update(self):
        self.H1 = np.multiply(self.H1, np.divide(np.dot(self.W.T, np.multiply(self.maskX1, self.X1)), np.dot(self.W.T, np.multiply(self.maskX1, np.dot(self.W, self.H1)+self.eps))))
        self.H2 = np.multiply(self.H2, np.divide(np.dot(self.W.T, np.multiply(self.maskX2, self.X2)), np.dot(self.W.T, np.multiply(self.maskX2, np.dot(self.W, self.H2)+self.eps))))
        self.H3 = np.multiply(self.H3, np.divide(np.dot(self.W.T, np.multiply(self.maskX3, self.X3)), np.dot(self.W.T, np.multiply(self.maskX3, np.dot(self.W, self.H3)+self.eps))))
        maskPanX = np.c_[self.maskX1, self.maskX2, self.maskX3]
        PanX = np.c_[self.X1, self.X2, self.X3]
        PanH = np.c_[self.H1, self.H2, self.H3]
        self.W = np.multiply(self.W, np.divide(np.dot(np.multiply(maskPanX, PanX), np.transpose(PanH)), (np.dot(np.multiply(maskPanX, np.dot(self.W, PanH)), np.transpose(PanH))+self.eps)))

    '''
    def calc_euclidean_multiplicative_update(self, lambda_H1=1, lambda_H2=0.2, lambda_H3=0.3):
        # Update H1, H2, H3 with L1 regularization
        self.H1 = np.multiply(self.H1, np.divide(np.dot(self.W.T, np.multiply(self.maskX1, self.X1)), np.dot(self.W.T,np.multiply(self.maskX1,np.dot(self.W,self.H1) + self.eps)) + lambda_H1 * np.sign(self.H1)))
        self.H2 = np.multiply(self.H2, np.divide(np.dot(self.W.T, np.multiply(self.maskX2, self.X2)), np.dot(self.W.T, np.multiply(self.maskX2,np.dot(self.W,self.H2) + self.eps)) + lambda_H2 * np.sign(self.H2)))
        self.H3 = np.multiply(self.H3, np.divide(np.dot(self.W.T, np.multiply(self.maskX3, self.X3)), np.dot(self.W.T,np.multiply(self.maskX3,np.dot(self.W,self.H3) + self.eps)) + lambda_H3 * np.sign(self.H3)))

        # Combined mask and data for PanX
        maskPanX = np.c_[self.maskX1, self.maskX2, self.maskX3]
        PanX = np.c_[self.X1, self.X2, self.X3]
        PanH = np.c_[self.H1, self.H2, self.H3]

        # Update
        self.W = np.multiply(self.W, np.divide(np.dot(np.multiply(maskPanX, PanX), np.transpose(PanH)), (np.dot(np.multiply(maskPanX, np.dot(self.W, PanH)), np.transpose(PanH)) + self.eps)))

    #This function is designed for updating the W and H matrices according to a set iteration count
    def wrapper_calc_euclidean_multiplicative_update(self):
        for run in range(self.maxiter):
            self.calc_euclidean_multiplicative_update()
            self.calc_distance_of_HW_to_X()
            #self.print_distance_report_of_HW_to_X(run)
            self.get_distance_report_of_HW_to_X(run)
        return self.report
    #This function is designed for calculateing the euclidean distance between two matrices
    def calc_distance_of_HW_to_X(self):
        self.X1r = np.dot(self.W, self.H1)
        self.X2r = np.dot(self.W, self.H2)
        self.X3r = np.dot(self.W, self.H3)
        self.diff = np.sum(np.abs(self.X1r_pre-self.X1r)) + np.sum(np.abs(self.X2r_pre-self.X2r)) + np.sum(np.abs(self.X3r_pre-self.X3r))
        self.X1r_pre = self.X1r
        self.X2r_pre = self.X2r
        self.X3r_pre = self.X3r
        self.eucl_dist1 = self.calc_euclidean_dist(self.X1, self.X1r)
        self.eucl_dist2 = self.calc_euclidean_dist(self.X2, self.X2r)
        self.eucl_dist3 = self.calc_euclidean_dist(self.X3, self.X3r)
        self.eucl_dist = self.eucl_dist1 + self.eucl_dist2 + self.eucl_dist3
        self.masked_eucl_dist1 = self.calc_masked_euclidean_dist(self.X1, self.X1r, self.maskX1)
        self.masked_eucl_dist2 = self.calc_masked_euclidean_dist(self.X2, self.X2r, self.maskX2)
        self.masked_eucl_dist3 = self.calc_masked_euclidean_dist(self.X3, self.X3r, self.maskX3)
        self.masked_eucl_dist = self.masked_eucl_dist1 + self.masked_eucl_dist2 + self.masked_eucl_dist3
        self.error1 = np.mean(np.mean(np.abs(self.X1-self.X1r)))/np.mean(np.mean(self.X1))
        self.error2 = np.mean(np.mean(np.abs(self.X2-self.X2r)))/np.mean(np.mean(self.X2))
        self.error3 = np.mean(np.mean(np.abs(self.X3-self.X3r)))/np.mean(np.mean(self.X3))

        self.error = self.error1 + self.error2 + self.error3
        
    #This function is designed for　printing the euclidean distance    
    def print_distance_report_of_HW_to_X(self, text):

        #print("[%s] diff = %f, eucl_dist = %f, MASKED_eucl_dist = %f, error = %f" % (text, self.diff, self.eucl_dist, self.masked_eucl_dist, self.error))
        print(text)
        print("Difference sum:", self.diff)
        print("eucl_dist:", self.eucl_dist)
        print("masked_eucl_dist:", self.masked_eucl_dist)
        print("error:", self.error)

    #This function is designed for　adding the euclidean distance into report 
    def get_distance_report_of_HW_to_X(self, text):
        newreport = pd.DataFrame([[self.rank, text, self.diff, self.eucl_dist, self.masked_eucl_dist, self.error]], columns = ["K", "iter", "difference of WH - WH_pre", "Euclidean distance of X - WH", "MASKED Euclidean distance of X - WH", "Error: avg' |X - WH|/X"])
        #self.report = self.report.append(newreport)
        self.report = pd.concat([self.report, newreport], ignore_index=True)
        return self.report
        
    def calc_euclidean_dist(self, X, Y):
        dist = np.sum(np.sum(np.power(X-Y, 2)))
        return dist

    def calc_masked_euclidean_dist(self, X, Y, maskX):
        dist = np.sum(np.sum(np.power(np.multiply(maskX, X-Y), 2)))
        return dist
    
    def set_PanH(self):
        self.PanH = pd.concat([self.H1, self.H2, self.H3], axis=1)
        columnsPanH = ["X1_" + str(x) for x in self.H1.columns] + ["X2_" + str(x) for x in self.H2.columns] + ["X3_" + str(x) for x in self.H3.columns]
        self.PanH.columns = columnsPanH  
            
#    def run(self):
        