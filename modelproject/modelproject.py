from scipy import optimize
from types import SimpleNamespace

class modelprojectclass:
    
    def __init__(self, T, G, M, P, a, b, c, d, e, f):

        par = self.par = SimpleNamespace()

        # Assign values to the parameters
        par.T = T
        par.G = G
        par.M = M
        par.P = P
        par.a = a
        par.b = b
        par.c = c
        par.d = d
        par.e = e
        par.f = f
