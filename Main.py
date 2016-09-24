X = np.array([  [0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1] ])
    
# output dataset            
y = np.array([[0,0,1,1]]).T

a = ANN(X ,y,5,3,1)

print ("prueba de foward")        
at = a.foward([0,0,1]);

print("prediccion")
print (at)


a.entrenar()