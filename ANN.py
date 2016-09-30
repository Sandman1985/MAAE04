import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})


class ANN:
     
    datos   = None      #Conjunto de datos de entrenamiento
    
    resultados  = None  #Conjunto de las salidas asociadas a los datos de entrenmiento
    sizeoculta = None   #Cantidad de neuronas de la capa oculta
    entradas = None     #Cantidad de valores de entrada
    salidas = None      #Cantidad de valores de salida
    ni = None           #Parametro de aprendizaje
    k = None            #Parametro de aprendizaje
    alfa = None         #Parametro de momento
    pesos_oculta = None
    pesos_salida = None
    
    tang = 0
    iteraciones = 0     #controla la cantidad de iteraciones
    max_It = None
    lista_error = []
    
    #memoria entre iteraciones, para aplicar momento
    variacion_salida = 0
    variacion_oculta= 0
    ##################################################################################################################
    def __init__(self,datos,datosEx, resultados,sizeoculta,entradas=1,salidas=1,ni=0.1,k = 1, alfa = 0.1,tang = 0, max_It=100000):
        self.datos   = datos
        self.datosEx = datosEx
        
        
        self.resultados  = resultados
        self.sizeoculta = sizeoculta
        self.entradas = entradas
        self.salidas = salidas
        self.ni = ni
        self.k = k
        self.alfa = alfa
        self.tang = tang
        self.max_It = max_It
        
        np.random.seed(5)
        # Los pesos se inicializan con valores randomicos chicos         
        #pesos entre entrada y capa oculta pesos_oculta[i][j] es el peso desde la entrada j a la neurona oculta i
        self.pesos_oculta =  2*np.random.random((self.entradas+1,self.sizeoculta)) -1
        #pesos entre  capa oculta y salida pesos_salida[i][j] es el peso desde  la neurona oculta j a la salida i
        self.pesos_salida =  2*np.random.random((self.sizeoculta,self.salidas)) -1

            
        
    def entrenar(self):
        while True:            
            # resultados predictos por la ANN
            prediccion = self.foward()
            # diferencia con respecto a las salidas esperadas
            error = self.resultados - prediccion       
            # almaceno promedio del error para graficar    
            self.lista_error.extend(self.error_avg(error))            
            deltaSalida = self.delta_salida(prediccion,error)
            deltaOculta = self.delta_oculta(deltaSalida)
            self.actualizar_pesos(deltaSalida,deltaOculta)
            
            if ( self.iteraciones == 100 or self.iteraciones == 1000 or self.iteraciones == 10000 or self.iteraciones == 100000 ):
                self.grafica_actual()
            if self.condition(True):
                break
        self.graficar_error()       
    
    def clasificar(self,dato):
        return self.foward(dato)
    
     ##########################################################################################################################   
   
    #FUNCIONES
    # funcion sigmoidal 
    def sig(self,x):
        if self.tang == 0:
            return 1/(1+np.exp(-self.k*x))
        else:
            return np.tanh(x)   
        

    # derivada funcion sigmoidal
    def devSig(self,x):
        if self.tang == 0:
            return x*(1-x)
        else:
            return 1 - np.power(x, 2)
        
    
    # propagacion hacia adelante
    def foward(self):
        intermedio = self.sig(np.dot(self.datosEx,self.pesos_oculta))
        return self.sig(np.dot(intermedio,self.pesos_salida))
    
    #error promedio
    def error_avg(self,error):
        e1 = np.abs(error)
        e2= sum (e1,0.0)
        return e2 / float(len(error))
    
    #calcula el delta en la salida
    def delta_salida(self,prediccion,error):
        return error * self.devSig(prediccion)

    #calcula el delta en la capa oculta
    def delta_oculta(self,deltaSalida):        
        return  deltaSalida.dot(self.pesos_salida.T) * self.devSig(self.sig(np.dot(self.datosEx,self.pesos_oculta)))

    
    #actualiza los valore de w
    def actualizar_pesos(self,deltaSalida,deltaOculta):     
        # actualiza pesos de capa oculta  

        
        self.variacion_oculta =  ( self.datosEx.T.dot(deltaOculta) * self.ni + (self.alfa * self.variacion_oculta ) )
        self.pesos_oculta +=  self.variacion_oculta

        # entrada de la capa de salida 
        intermedio = self.sig(np.dot(self.datosEx,self.pesos_oculta))
        # actualiza pesos capa de salida
        
        self.variacion_salida =  (intermedio.T.dot(deltaSalida) *  self.ni   + (self.alfa * self.variacion_salida )  ) 
        self.pesos_salida +=  self.variacion_salida
        
        
      

        
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

        
        plt.plot(b, e, 'bo',)
        plt.show()
    
    
    def grafica_actual(self):
        e = self.foward()
        if self.entradas ==1:             
            plt.plot(self.datos, e, 'bo', self.datos, e, 'k')
            plt.plot(self.datos, self.resultados, 'bo', self.datos, self.resultados, 'k')
        else:
            por_ejes = zip(*self.datos)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            ax.scatter(por_ejes[0], por_ejes[1], self.resultados ,  c='r', marker='o')
            ax.scatter(por_ejes[0], por_ejes[1], e,  c='b', marker='^')

            ax.set_xlabel('X')
            ax.set_ylabel('Y ')
            ax.set_zlabel('H(x,y)')
        plt.show()