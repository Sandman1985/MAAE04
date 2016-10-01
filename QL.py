import numpy as np
from random import randrange, uniform, choice

class QL:

    #FUNCIONES
    def __init__(self,R, goals, df = 0.99, episodios = 10 , epsiolon = 0.2):
        '''
        Inicializa la matriz estado-recompenza
        '''
        self.R = R
        self.cant_filas = len(R)
        self.cant_col = len(R[0])
        self.goals = goals
        self.Q = np.zeros_like(R)

        self.df = df
        self.episodios = episodios
        self.epsilon = epsiolon

    def actQ(self, historico):
        '''
        Actualiza la matriz Q
        '''
        for estado, accion in reversed(historico):
            est_fila, est_col = estado
            r_sa = self.R[est_fila][est_col][accion]
            nuevo_estado = self.mover(estado, accion)
            movimientos = self.movimientos_validos(nuevo_estado)
            q_actualizado =r_sa + self.df * max([self.Q[nuevo_estado[0]][nuevo_estado[1]][a] for a in movimientos])
            self.Q[est_fila][est_col][accion] = q_actualizado

    def mover(self,estado, accion):
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
        if estado[0] < self.cant_filas - 1:
            movimientos.append(2)
        if estado[1] < self.cant_col - 1:
            movimientos.append(1)
        if estado[1] > 0:
            movimientos.append(3)
        return movimientos

    def execute(self):
        '''
        Ejecuta el algoritmo
        '''
        for self.episodios in range(self.episodios):
            est_actual = self.inicializador()
            historico = []
            while not est_actual in self.goals:
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
                historico.append((est_actual, accion))
                est_actual = self.mover(est_actual, accion)
            self.actQ(historico)
        self.imprimir_Q()

    def inicializador(self):
        '''
        Elige aleatoriamente un estado inicial para cada episodio
        (No se admiten estados finales como iniciales)
        '''
        estado = randrange(0,self.cant_filas), randrange(0,self.cant_col)
        while estado in self.goals:
            estado = randrange(0,self.cant_filas), randrange(0,self.cant_col)
        return estado

    def mejor_mov(self, estado, movimientos):
        '''
        Devuelve la accion que optimice Q en el estado "estado"
        (En caso de haber varias devuelve una aleatoria)
        '''
        maximo = max([self.Q[estado[0]][estado[1]][mov] for mov in movimientos])
        return choice([mov for mov in movimientos if self.Q[estado[0]][estado[1]][mov] == maximo])
        print estado

    def imprimir_Q(self):
        '''
        Imprime una representacion de los valores de Q en pantalla
        '''
        linea = "\n" + "-"*61 + "\n"
        matriz = linea
        for row in self.Q:
            secciones = zip(*row)
            arriba = "|"
            for u in secciones[0]:
                arriba += "   %03d   |" % u
            medio = "|"
            for m in zip(secciones[3], secciones[1]):
                medio += "%03d   %03d|" % m

            abajo = "|"
            for d in secciones[2]:
                abajo += "   %03d   |" % d

            matriz += arriba + "\n" + medio + "\n" + abajo + linea
        print matriz
