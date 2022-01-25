import numpy as np  
x = np.array([[1, 3, 5], [11, 35, 56]])  

z = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
 
y = np.ravel(x, order='F')  
w = np.ravel(z, order='F')  
#print(x)
print(z)
print(w)
#print(w)
   