import numpy as np
from linreg import LinReg

def load_data(path, num_train):

    data = np.loadtxt(path, delimiter=';', skiprows=1)
    
    X_train, Y_train = data[:num_train, :-1], data[:num_train, -1]
    X_test, Y_test = data[num_train:, :-1], data[num_train:, -1]
        
    return X_train, Y_train, X_test, Y_test
    
def main():
    
    num_train = 4000
    path = r'winequality\winequality-white.csv'
    
    X_train, Y_train, X_test, Y_test = load_data(path, num_train)
    
    linreg = LinReg()
    
    weights = linreg.fit(X_train, Y_train)
    prediction = linreg.predict(X_test, weights)
    error = linreg.mse(prediction, Y_test)
    
    print("Mean squared error of prediction is " + str(error))
    
    
if __name__ == "__main__":
    main()