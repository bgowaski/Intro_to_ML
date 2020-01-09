# Importing the Tensorflow library 
import tensorflow as tf 
  
# Importing the NumPy library 
import numpy as np 
  
# Importing the matplotlib.pylot function 
import matplotlib.pyplot as plt 
  
# A vector of size 15 with values from -5 to 5 
a = np.linspace(-5, 5, 15) 
  
# Applying the softplus function and 
# storing the result in 'b' 
b = tf.nn.softplus(a, name ='softplus') 
  
# Initiating a Tensorflow session 
with tf.Session() as sess: 
    print('Input:', a) 
    print('Output:', sess.run(b)) 
    plt.plot(a, sess.run(b), color = 'red', marker = "o")  
    plt.title("tensorflow.nn.softplus")  
    plt.xlabel("X")  
    plt.ylabel("Y")  
  
    plt.show() 