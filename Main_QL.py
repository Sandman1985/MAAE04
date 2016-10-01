from QL import QL
import numpy as np

# Se define la matriz R
r = np.array([[[0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0]],
              [[0,0,0,0], [0,-20,0,0], [0,0,80,0], [0,0,80,0], [0,0,0,-20], [0,0,0,0]],
              [[0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0]],
              [[0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0]],
              [[0,0,0,0], [0,0,0,0], [20,0,0,0], [20,0,0,0], [0,0,0,0], [0,0,0,0]],
              [[0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0]]])

# Se definen los GOALS
goals = [(2,2), (2,3), (3,2), (3,3)]

# Configuracion parametrica
parms = {
    "DF = 0,8":{
        "episodios":[30],
        "df":0.8
    },
    # "DF = 0,4":{
    #     "episodios":[5,10,30],
    #     "df":0.4
    # }
}

for caso, param in parms.items():
    print "Caso: " + caso + "\n"
    df = param['df']
    for ep in param['episodios']:
        print "Episodios: " + str(ep)
        learner = QL(r, goals, df = df, episodios = ep)
        learner.ejecutar()
