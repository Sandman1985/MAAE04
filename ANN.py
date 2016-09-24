#import numpy as np



class ANN:
     
    datos   = None      #Conjunto de datos de entrenamiento
    resultados  = None  #Conjunto de las salidas asociadas a los datos de entrenmiento
    sizeoculta = None   #Cantidad de neuronas de la capa oculta
    entradas = None     #Cantidad de valores de entrada
    salidas = None      #Cantidad de valores de salida
    ni = None           #Parametro de aprendizaje
    
    pesos_oculta = None
    pesos_salida = None
    
    iteraciones = 0     #controla la cantidad de iteraciones
    
    #FUNCIONES
    # funcion sigmoidal 
    def sig(self,x):
        return 1/(1+np.exp(-x))

    # derivada funcion sigmoidal
    def devSig(self,x):
        return x*(1-x)
    
    # propagacion hacia adelante
    def foward(self,dato):
        l1 = self.sig(np.dot(self.pesos_oculta,dato))
        return self.sig(np.dot(self.pesos_salida,l1))
    
    #calcula el delta en la salida
    def delta_salida(self,prediccion,resultado):
        return self.devSig(prediccion)* (prediccion - resultado)

    #calcula el delta en la capa oculta
    def delta_oculta(self,prediccion,ds):
        return self.devSig(prediccion)* self.pesos_salida * ds
       # return self.devSig(prediccion)* np.dot(self.pesos_salida,ds)
        #complica porque es una sola salida y toma self.pesos_salida como una matriz indefinida en lugar de un escalar
    
    #actualiza los valore de w
    def actualizar_pesos(self,ds,do,dato):
        l1 = self.sig(np.dot(self.pesos_oculta,dato))
        print(dato)
        print(do)
        self.pesos_oculta +=  dato * do
        self.pesos_salida +=  np.dot(self.ni , l1.dot(ds))
        

        
    # controla el fin de la ejecucion -- por ahora por cantidad d eiteraciones
    def condicion():
        self.iteraciones+= 1
        if self.iteraciones > 10.000:
            return true
        else:
            return false
        
    
    def __init__(self,datos,resultados,sizeoculta,entradas=1,salidas=1,ni=0.1):
        self.datos   = datos
        self.resultados  = resultados
        self.sizeoculta = sizeoculta
        self.entradas = entradas
        self.salidas = salidas
        self.ni = ni

        # Los pesos se inicializan con valores randomicos chicos y media cero

        #pesos entre entrada y capa oculta pesos_oculta[i][j] es el peso desde la entrada j a la neurona oculta i
        self.pesos_oculta =  2*np.random.random((self.sizeoculta,self.entradas)) -1
        #pesos entre  capa oculta y salida pesos_salida[i][j] es el peso desde  la neurona oculta j a la salida i
        self.pesos_salida =  2*np.random.random((self.salidas,self.sizeoculta)) -1
        print ("inicializacion capa oculta")
        print(self.pesos_oculta)
        print ("inicializacion capa salida")
        print(self.pesos_salida)
            
        
    def entrenar(self):
        while True:            
            for dato, resultado in zip(self.datos, self.resultados):
                prediccion = self.foward(dato)                
                ds = self.delta_salida(prediccion,resultado)
                do = self.delta_oculta(prediccion,ds)
                ##############################################
                self.actualizar_pesos(ds,do,dato)
            if self.condition():
                break
                

  
		