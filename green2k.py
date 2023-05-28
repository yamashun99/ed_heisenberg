import numpy as np
import matplotlib.pyplot as plt


def make_Gkomega(Gijs, ks):
    omegamesh = Gijs.shape[0]
    L = Gijs.shape[1]
    Gkomegas = []
    for iomega in range(omegamesh):
        Gks = []
        for k in ks:
            Gk = 0+0j
            for ix in range(L):
                Gk += np.exp(1.0j*k*ix)*Gijs[iomega][ix]
            Gks.append(Gk)
        Gkomegas.append(Gks)
    Gkomegas = np.array(Gkomegas)
    return Gkomegas


if __name__ == '__main__':
    Gijs = np.load('Gijs.npy')
    L = Gijs.shape[1]
    xs = [i for i in range(L)]
    ks = [np.pi*(2*ik/L) for ik in range(L)]
    Gkomegas = make_Gkomega(Gijs, ks)
    #
    # omegamax = 5
    # omegamesh = 40
    # omegas = [iomega / omegamesh * omegamax for iomega in range(omegamesh)]
    #

    fig = plt.figure()
    ax = fig.add_subplot()
    mappable = ax.pcolor(xs, omegas, Gijs.real)
    fig.colorbar(mappable)
    fig.savefig('Gij_real.pdf')

    fig = plt.figure()
    ax = fig.add_subplot()
    mappable = ax.pcolor(xs, omegas, Gijs.imag)
    fig.colorbar(mappable)
    fig.savefig('Gij_imag.pdf')

    fig = plt.figure()
    ax = fig.add_subplot()
    mappable = ax.pcolor(ks, omegas, Gks.real)
    fig.colorbar(mappable)
    fig.savefig('Gkomega_real.pdf')

    fig = plt.figure()
    ax = fig.add_subplot()
    mappable = ax.pcolor(ks, omegas, Gks.imag)
    fig.colorbar(mappable)
    fig.savefig('Gkomega_imag.pdf')
