from ANN import ANN 
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import numpy as np

def f (x):
    return [x[0]**2]

def g (x):
    return [math.sin(1.5*math.pi *x[0])]

def h(entrada):    
    if ((entrada[0]**2 +entrada[1]**2) <0.5):
        return [1]
    return [0]

puntos =np.array( sorted([ [(n*(0.05))*x for x in range(1,2) ] for n in range(-20,20)]))
puntosEx =np.array( ([[ 1, (n*(0.05))*x] for x in range(1,2)  for n in range(-20,20)]))
doble= np.array ([ [x, y ]for x in (0.1,0.3,0.5,0.7,0.9) for y in (0.2,0.4,0.6,0.8)])
dobleEx= np.array ([ [1,x, y ]for x in (0.1,0.3,0.5,0.7,0.9) for y in (0.2,0.4,0.6,0.8)])


sol_f = np.array(list(map(f,puntos)))
sol_g= np.array(list(map(g,puntos)))
sol_h= np.array(list(map(h,doble)))


#Configuracion f
sizeoculta = 5
entradas=1
salidas=1
ni=0.1
k = 7
alfa = 0.1
tang = 0
max_It=10000

#a = ANN(puntos,puntosEx ,sol_f ,sizeoculta,entradas,salidas,ni,k,alfa,tang,max_It)
#a.entrenar()

#Configuracion g
sizeoculta = 3
entradas=1
salidas=1
ni=0.05
k = 1
alfa = 0.1
tang = 1
max_It=100000
#b = ANN(puntos,puntosEx ,sol_g ,sizeoculta,entradas,salidas,ni,k,alfa,tang,max_It)
#b.entrenar()

#Configuracion h
sizeoculta = 3
entradas=2
salidas=1
ni=0.1
k = 1
alfa = 0.1
tang = 0
max_It=100000
c = ANN(doble,dobleEx ,sol_h, sizeoculta,entradas,salidas,ni,k,alfa,tang,max_It)
c.entrenar()