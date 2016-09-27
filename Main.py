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

puntos =np.array( sorted ([ [(1*(n*(1/20)))*x for x in range(1,2) ] for n in range(-20,20)]))

doble= list(zip(puntos,puntos))


sol_f = np.array(list(map(f,puntos)))


sol_g= list(map(g,puntos))

sol_h= list(map(h,doble))



a = ANN(puntos ,sol_f ,10,1,1,0.1,7,0,100000)
a.entrenar()

#b = ANN(puntos ,sol_g ,10,1,1,0.1,7,0,10000)
#b.entrenar()

#c = ANN(doble ,sol_g ,10,2,1,0.1,7,0,10000)
#c.entrenar()

