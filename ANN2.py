 

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
        z =[]
        for x,y in zip(prediccion,resultado):
            z.extend((x-y) * self.devSig(y))        
        return  z

    #calcula el delta en la capa oculta
    def delta_oculta(self,prediccion,ds):
        return self.devSig(prediccion)* self.pesos_salida * ds
        z =[]
        for x,y in zip(ds,prediccion):
            z.extend((x-y) * self.devSig(y))        
        return  z
    
    
       # return self.devSig(prediccion)* np.dot(self.pesos_salida,ds)
        #complica porque es una sola salida y toma self.pesos_salida como una matriz indefinida en lugar de un escalar
    
    #actualiza los valore de w
    def actualizar_pesos(self,ds,do,dato):
        d = np.array([dato]) 
        # actualiza pesos de capa oculta
        
        # self.pesos_oculta += (do.T.dot(d)* self.ni)
        
        #version para una entrada
        self.pesos_oculta +=  do.T *(d)* self.ni
        
        # entrada de la capa de salida 
        l1 = self.sig(np.dot(self.pesos_oculta,dato))
        # actualiza pesos capa de salida
        self.pesos_salida +=  (ds.dot(l1[0]) *  self.ni)   
        

        
    # controla el fin de la ejecucion -- por ahora por cantidad d eiteraciones
    def condition(self):
        self.iteraciones+= 1
        if self.iteraciones > self.max_It:
            return True
        else:
            return False
        
    
    def __init__(self,datos,resultados,sizeoculta,entradas=1,salidas=1,ni=0.1,max_It=10000):
        self.datos   = datos
        self.resultados  = resultados
        self.sizeoculta = sizeoculta
        self.entradas = entradas
        self.salidas = salidas
        self.ni = ni
        self.max_It = max_It
        
        np.random.seed(1)
        # Los pesos se inicializan con valores randomicos chicos y media cero

        #pesos entre entrada y capa oculta pesos_oculta[i][j] es el peso desde la entrada j a la neurona oculta i
        self.pesos_oculta =  2*np.random.random((self.sizeoculta,self.entradas)) -1
        #pesos entre  capa oculta y salida pesos_salida[i][j] es el peso desde  la neurona oculta j a la salida i
        self.pesos_salida =  2*np.random.random((self.salidas,self.sizeoculta)) -1

            
        
    def entrenar(self):
        while True:
            prediccion =[]
            for dato, resultado in zip(self.datos, self.resultados):
                prediccion.extend( self.foward(dato) ) 
                
            error = (self.resultados - np.array(prediccion) )
            self.lista_error.extend(error[0])
            ds = self.delta_salida(np.array(prediccion),self.resultados)
            do = self.delta_oculta(prediccion,ds)
                ##############################################
            self.actualizar_pesos(ds,do,dato)
            if self.condition():
                break
          
        self.graficar_error()
    
    def clasificar(self,dato):
        return self.foward(dato)
        

        
    def graficar_error(self):
        base = [x for x in range(0,self.max_It) ]
        print(len(base))
        print(len(self.lista_error))
        b = np.array(base)
        e = np.array(self.lista_error)
        plt.plot(b, e, 'bo', b, e, 'k')
        plt.show()