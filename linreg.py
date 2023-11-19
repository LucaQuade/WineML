import numpy as np

class LinReg:
#Simple Linear Regression class

    def fit(self, X, Y):
    
        #calculates weights theta that minimize the loss function

        N = X.shape[0]
        X = np.c_[X, np.ones(N)]
        
        #Solve gradient of loss function = 0 to calculate weights theta
        theta = np.linalg.solve(X.T @ X, X.T @ Y)
        return theta


    def predict(self, X, theta):
    
        #returns array of predicted values for test cases X with weights theta
        
        N = X.shape[0]
        X = np.c_[X, np.ones(N)]
        
        Y_pred = X @ theta
        return Y_pred


    def mse(self, Y_pred, Y_target):
    
        #calculate mean squared error of prediction
        
        se = ((Y_pred - Y_target) ** 2).sum()
        return se / len(Y_pred)
