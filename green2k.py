import numpy as np


def make_Gkomega(Gijs, ks):
    omegamesh = Gijs.shape[0]
    L = Gijs.shape[1]
    Gkomegas = []
    for iomega in range(omegamesh):
        Gks = []
        for k in ks:
            Gk = 0+0j
            for ix in range(L):
                Gk += np.exp(-1.0j*k*ix)*Gijs[iomega][ix]
            Gks.append(Gk)
        Gkomegas.append(Gks)
    Gkomegas = np.array(Gkomegas)
    Gkomegas = Gkomegas/np.sqrt(L)
    return Gkomegas
