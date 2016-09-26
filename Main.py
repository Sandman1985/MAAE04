import ANN
import math
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})


def f (x):
    return x**2

def g (x):
    return math.sin(1.5*math.pi *x)

def h(x,y):
    if ((x**2 +y**2) <0.5):
        return 1
    return 0

puntos = sorted ([1*(n*(1/20)) if n != 0 else 0 for n in range(-20,20)])
doble= list(zip(puntos,puntos))

sol_f = [f(x) for x in puntos]

sol_g= [g(x) for x in puntos]

sol_h= [h(x,y) for (x,y) in doble]


a = ANN(puntos ,sol_f ,10,1,1,0.1,10000)
a.entrenar()

#b = ANN(puntos ,sol_g ,10,1,1,0.1,10000)
#b.entrenar()

#c = ANN(doble ,sol_g ,10,2,1,0.1,10000)
#c.entrenar()
