X = np.array([  [0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1] ])
    
# output dataset            
y = np.array([[0,0,1,1]]).T

a = ANN(X ,y,5,3,1)

a.entrenar()

print ("prueba de [0,0,1]")        
at1 = a.clasificar([0,0,1])

print("Se espera 0 ,prediccion:")
print (at1)

print ("prueba de [0,1,1]")        
at2 = a.clasificar([0,1,1]);

print("Se espera 0 ,prediccion:")
print (at2)

print ("prueba de [1,0,1]")        
at3 = a.clasificar([1,0,1]);

print("Se espera 1 ,prediccion:")
print (at3)

print ("prueba de [1,1,1]")        
at4 = a.clasificar([1,1,1]);

print("Se espera 1 ,prediccion:")
print (at4)