import numpy as np



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
    max_It = None
    lista_error = []
    ##################################################################################################################
    def __init__(self,datos,resultados,sizeoculta,entradas=1,salidas=1,ni=0.1,max_It=10000):
        self.datos   = np.array(datos)
        self.resultados  = np.array(resultados)
        self.sizeoculta = sizeoculta
        self.entradas = entradas
        self.salidas = salidas
        self.ni = ni
        self.max_It = max_It
        
        # Los pesos se inicializan con valores randomicos chicos         
        #pesos entre entrada y capa oculta pesos_oculta[i][j] es el peso desde la entrada j a la neurona oculta i
        self.pesos_oculta =  np.random.random((self.entradas,self.sizeoculta)) 
        #pesos entre  capa oculta y salida pesos_salida[i][j] es el peso desde  la neurona oculta j a la salida i
        self.pesos_salida =  np.random.random((self.sizeoculta,self.salidas)) 

            
        
    def entrenar(self):
        while True:            
            for dato, resultado in zip(self.datos, self.resultados):
                prediccion = self.foward(dato)  
                error = list((self.sig(resultado) -prediccion))  
                self.lista_error.extend(error[0])
                deltaSalida = self.delta_salida(prediccion,error)
                deltaOculta = self.delta_oculta(prediccion,deltaSalida)
                self.actualizar_pesos(deltaSalida,deltaOculta,dato)
                if ( self.iteraciones == 100 or self.iteraciones == 1000 or self.iteraciones == 10000 or self.iteraciones == 10000):
                    self.grafica_actual()
                if self.condition(True):
                    break
            if self.condition(False):
                break
        self.graficar_error()       
    
    def clasificar(self,dato):
        return self.foward(dato)
    
     ##########################################################################################################################   
   
    #FUNCIONES
    # funcion sigmoidal 
    def sig(self,x):
        return 1/(1+np.exp(-x))

    # derivada funcion sigmoidal
    def devSig(self,x):
        return x*(1-x)
    
    # propagacion hacia adelante
    def foward(self,dato):
        intermedio = self.sig(np.dot(dato,self.pesos_oculta))
        return self.sig(np.dot(intermedio,self.pesos_salida))

    
    #calcula el delta en la salida
    def delta_salida(self,prediccion,error):
        return error * self.devSig(prediccion)

    #calcula el delta en la capa oculta
    def delta_oculta(self,dato,ds):
        return self.devSig(self.sig(np.dot(dato,self.pesos_oculta))) * ds.dot(self.pesos_salida.T)
       # return self.devSig(prediccion)* np.dot(self.pesos_salida,ds)
        #complica porque es una sola salida y toma self.pesos_salida como una matriz indefinida en lugar de un escalar
    
    #actualiza los valore de w
    def actualizar_pesos(self,deltaSalida,deltaOculta,dato):     
        # actualiza pesos de capa oculta       

        #self.pesos_oculta +=   dato.T.dot(deltaOculta) * self.ni
        self.pesos_oculta +=   dato * deltaOculta * self.ni
        
        # entrada de la capa de salida 
        intermedio = self.sig(np.dot(dato,self.pesos_oculta))
        # actualiza pesos capa de salida
        self.pesos_salida +=  intermedio.T.dot(deltaSalida) *  self.ni         

        
    # controla el fin de la ejecucion -- por ahora por cantidad d eiteraciones
    def condition(self,incrementar):
        self.iteraciones+= 1 if incrementar else 0
        if self.iteraciones > self.max_It:
            return True
        else:
            return False
        
#################################################################################################################### 
    def graficar_error(self):
        base = [x for x in range(0,self.max_It+1) ]
        b = np.array(base)
        e = np.array(self.lista_error)
        plt.plot(b, e, 'bo', b, e, 'k')
        plt.show()
    
    def estimado(self,dato):
        intermedio = self.sig(np.dot(dato,self.pesos_oculta))
        return np.dot(intermedio,self.pesos_salida)
    
    def grafica_actual(self):
        
        e = [self.estimado(x)[0] for x in self.datos]
    
        print(self.datos)
        print("-----------")
        print(e)
        print("-----------")
   
        plt.plot(self.datos, e, 'bo', self.datos, e, 'k')
        plt.plot(self.datos, self.datos**2, 'bo', self.datos, self.datos**2, 'k')
        plt.show() 
