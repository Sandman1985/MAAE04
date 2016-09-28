import ANN
import math
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})


def f (x):
    return [x[0]**2]

def g (x):
    return [math.sin(1.5*math.pi *x[0])]

def h(entrada):
    if ((entrada[0][0]**2 +entrada[1][0]**2) <0.5):
        return [1]
    return [0]

p = np.array([ [1] for n in range(-20,20)])

puntos =np.array( sorted ([ [(1*(n*(1/20)))*x for x in range(1,2) ] for n in range(-20,20)]))
puntosEx =np.array( ([[ 1, (1*(n*(1/20)))*x] for x in range(1,2)  for n in range(-20,20)]))


doble= list(zip(puntos,puntos))


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
sizeoculta = 5
entradas=1
salidas=1
ni=0.1
k = 1
alfa = 0.1
tang = 1
max_It=100000
b = ANN(puntos,puntosEx ,sol_g ,sizeoculta,entradas,salidas,ni,k,alfa,tang,max_It)
b.entrenar()

#c = ANN(doble,puntosEx ,sol_g ,5,2,1,0.1,0.1,7,0,10000)
#c.entrenar()

