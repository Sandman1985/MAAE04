import numpy as np
from random import randrange, uniform, choice

class QL:

    #FUNCIONES
    def __init__(self,R, goals, df = 0.8, episodios = 100 , epsiolon = 0.2):
        '''
        Inicializa la matriz estado-recompenza
        '''
        self.R = R
        self.tam_tablero = R.size
        self.golas = goals
        self.Q = R #FIXME

        self.df = df
        self.episodios = episodios
        self.epsilon = epsiolon

    def actQ(self, historico):
        '''
        Actualiza la matriz Q
        '''
        for estado, accion in reversed(historico):
            est_fila, est_col = estado
            r_sa = r[est_fila][est_col][accion]
            nuevo_estado = self.mover(estado, accion)
            movimientos = self.movimientos_validos(nuevo_estado)
            q_actualizado =r_sa + self.df * max([self.Q[nuevo_estado[0]][nuevo_estado[1]][a] for a in movimientos])
            self.Q[est_fila][est_col][accion] = q_actualizado

    def mover(estado, accion):
        '''
        Devuelve el estado resultante de moverse en la
        direccion "accion" estando en el estado "estado"
        '''
        fila, columna = estado
        if accion == 0:
            return fila - 1, columna
        elif accion == 1:
            return fila, columna + 1
        elif accion == 2:
            return fila + 1, columna
        elif accion == 3:
            return fila, columna - 1


    def movimientos_validos(self,estado):
        '''
        Despliega una lista de movimientos calidos para cada estado,
        Los movimientos se devuelven en el siguiente formato
            0 = arriba
            1 = derecha
            2 = abajo
            3 = izquierda
        '''
        movimientos = []
        if estado[0] > 0:
            movimientos.append(0)
        if estado[0] > self.tam_tablero:
            movimientos.append(1)
        if estado[1] > 0:
            movimientos.append(2)
        if estado[1] < self.tam_tablero :
            movimientos.append(3)
        return movimientos

    def execute(self):
        '''
        Ejecuta el algoritmo
        '''
        for self.episodios in range(self.episodios):
            est_actual = randrange(0,6), randrange(0,6) # mal
            historico = []
            while not est_actual in goals:
                # Recorre la partida con los valores sin modificar, conservando
                # en cada paso el estado y el la recompensa
                movimientos = self.movimientos_validos(est_actual)
                if uniform(0, 1) < self.epsilon:
                    # Accion aleatoria para exploracion
                    accion = choice(movimientos)
                else:
                    # Accion elegida a partir del mejor Q
                    # (aleatoriamente entre las opciones de igual valor)
                    accion = self.mejor_mov(est_actual,movimientos)
                historico.append(est_actual, accion)
                est_actual = self.mover(est_actual, accion)
            self.actQ(historico)
        print self.Q

    def mejor_mov(self, estado, movimientos):
        '''
        Devuelve la accion que optimice Q en el estado "estado"
        (En caso de haber varias devuelve una aleatoria)
        '''
        maximo = max(movimientos, key = lambda mov: self.Q[estado[0]][estado[1]][mov])
        return choice([mov for mov in movimientos if self.Q[estado[0]][estado[1]][mov] == maximo])


r = np.array([[[0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0]],
              [[0,0,0,0], [0,-20,0,0], [0,0,80,0], [0,0,80,0], [0,0,0,-20], [0,0,0,0]],
              [[0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0]],
              [[0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0]],
              [[0,0,0,0], [0,0,0,0], [20,0,0,0], [20,0,0,0], [0,0,0,0], [0,0,0,0]],
              [[0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0]]])

goals = [(3,3), (3,4), (4,3), (3,4)]

learner = QL.new(r,goals)
