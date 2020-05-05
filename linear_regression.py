import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn import linear_model
from sklearn.metrics import mean_squared_error


def fit_LinRegr(X_train, y_train):
    
    #create a "w" vector with d + 1
    w = np.zeros((1,np.size(X_train, 1) + 1))
    
    # additional column for input matrix with value of 1
    my_ones = np.ones((np.size(X_train, 0),1))
    
    #add the additional column to input matrix
    X_train = np.hstack((X_train,my_ones))
    
    #transpose input matrix for "least squares solution" 
    x_transpose =  np.transpose(X_train)
    
    #part of "least squares solution" 
    w1 = np.linalg.inv(np.dot(x_transpose , X_train)) 
    
    #part of "least squares solution" 
    w2 = np.dot (x_transpose,y_train )
    
    #part of "least squares solution"  
    w = np.dot(w1,w2)
    
    # return the "least squares solution" aka "w"
    return w

def mse(X_train,y_train,w):
     
    my_ones = np.ones((np.size(X_train, 0),1))
    
    X_train = np.hstack((X_train,my_ones))
      
    result = 0
    # index i to access data point
    i = 1
    # get number of rows
    num_rows = np.size(X_train, 0)
    # access data point on position i
    while i  <= num_rows:
        
         # access data point i
         data_point = X_train[i-1:i,:]
         
         # convert to vector
         data_point.flatten()
         
         # access the real y value for data point i
         y_value = y_train[i-1:i]
         
         # convert to vector
         y_value = y_value.flatten()
         
         # take the value of y as number
         y_value = y_value[0]
         
         #get the value of y for a data point based on w coefficent
         value_of_line = pred(data_point,w)
         
         # sum the values
         result = result + ((value_of_line - y_value)**2) 
         
         #increase the index i to access other data point
         i = i + 1
         
       
    #divide the sum by num of rows to get the error    
    result = result / num_rows
    #return error
    return result  
def pred(X_train,w):
    
    #convert X_train  to a vector
    X_train = X_train.flatten()
    
    #convert "w" to a vector
    w = w.flatten()
    
    #calculate dot product of vector X_train and "w"
    dot_product = np.dot(w,X_train) 
    
    #return dot product result
    return dot_product
def test_SciKit(X_train, X_test, Y_train, Y_test):
    
   # create linear_model.LinearRegression object
   reg = linear_model.LinearRegression()
   
   #train the model 
   reg.fit(X_train,Y_train)
   
   #get predictions using testing sets
   predictions = reg.predict(X_test)
   
   #return the mean_squared_error
   return mean_squared_error(Y_test,predictions)
def testFn_Part2():
    X_train, y_train = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X_train,y_train,test_size=0.2)
    
    #Testing Part 2a
    w=fit_LinRegr(X_train, y_train)
    e=mse(X_test,y_test,w)
    
    #Testing Part 2b
    scikit=test_SciKit(X_train, X_test, y_train, y_test)
    
    print("Mean squared error from Part 2a is ", e)
    print("Mean squared error from Part 2b is ", scikit)

testFn_Part2()












