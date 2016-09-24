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

sol_f = [f(x) for x in puntos]

sol_g= [g(x) for x in puntos]

plt.plot(puntos, sol_f, 'bo', puntos, sol_f, 'k')
plt.show()


a = ANN(puntos ,sol_f,5,1,1,0.1,1000)

a.entrenar()