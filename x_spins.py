import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from two_spins import make_spin_ops

import matplotlib

matplotlib.use('pgf')

plt.rcParams.update({
    "font.size": 15,
    'figure.subplot.bottom': 0.12,
    #'figure.subplot.top': 1.0,
    #'figure.subplot.left': 0.12,
    #'figure.subplot.right': 1.0,
})
np.set_printoptions(precision=3)
aaa


def make_matrix(ss):
    res = np.array([1])
    for s in reversed(ss):
        res = np.kron(res, s)
    return res


def dynamical_structure_factor(hamil, eigval, eigvec, S0, S1, omega):
    G = 0 + 0j
    E0 = eigval[0]
    for j, Ej in enumerate(eigval):
        #G += eigvec[:, 0].conj() @ S1 @ np.array(
        #    [eigvec[:, j]]).T * eigvec[:, j].conj() @ S0 @ np.array(
        #        [eigvec[:, 0]]).T / (E0 - Ej + omega + 0.0001 + 0.0001j)
        G += eigvec[:, 0].conj() @ S1 @ np.array(
            [eigvec[:, j]]).T * eigvec[:, j].conj() @ S0 @ np.array(
                [eigvec[:, 0]]).T / (E0 - Ej + omega + 0.0001 + 0.0001j)
    return G[0]


def plot_dynamical_structure_factor(hamil, L):
    sp_ops = make_spin_ops()
    sz = sp_ops['Sz']
    sp = sp_ops['S+']
    sm = sp_ops['S-']
    s0 = sp_ops['I']
    eigval, eigvec = np.linalg.eigh(hamil)
    S0 = make_matrix([sz] + [s0] * (L - 1))
    Gs = []
    omegamesh = 10
    omegas = [5 * iomega / omegamesh for iomega in range(omegamesh)]
    for iomega in range(omegamesh):
        omega = omegas[iomega]
        Gs += [
            dynamical_structure_factor(hamil, eigval, eigvec,
                                       S0.transpose().conj(), S0, omega).imag
        ]
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(omegas, Gs)
    fig.savefig('G.pdf')

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(eigval)
    fig.savefig('eigval.pdf')


def Gij(hamil, L, i, j, omega):
    sp_ops = make_spin_ops()
    sz = sp_ops['Sz']
    sp = sp_ops['S+']
    sm = sp_ops['S-']
    s0 = sp_ops['I']
    eigval, eigvec = np.linalg.eigh(hamil)
    print("E0 = ", eigval[0])
    szi = [s0] * i + [sz] + [s0] * (L - i - 1)
    szj = [s0] * j + [sz] + [s0] * (L - j - 1)
    Si = make_matrix(szi)
    Sj = make_matrix(szj)
    print(Si)
    #G = dynamical_structure_factor(hamil, eigval, eigvec,
    #                               Sj.transpose().conj(), Si, omega)
    G = dynamical_structure_factor(hamil, eigval, eigvec, Sj, Si, omega)
    return G


def plot_dynamical_structure_factor_k(hamil, L):
    sp_ops = make_spin_ops()
    sz = sp_ops['Sz']
    sp = sp_ops['S+']
    sm = sp_ops['S-']
    s0 = sp_ops['I']
    eigval, eigvec = np.linalg.eigh(hamil)
    print("E0 = ", eigval[0])
    sz0 = [sz] + [s0] * (L - 1)
    omegamesh = 5
    Gkw = []
    Gw = []
    omegas = [5 * iomega / omegamesh for iomega in range(omegamesh)]
    ks = [2.0 * np.pi / (L**2) * (ik) for ik in range(L + 1)]
    for iomega in range(omegamesh):
        print(iomega)
        omega = omegas[iomega]
        Gs_iomega = []
        for ik in range(L + 1):
            k = ks[ik]
            Sk = make_matrix(sz0) * (0.0 + 0.0j)
            for ix in range(L):
                Sk += np.exp(1.0j * k * ix) * make_matrix(sz0[ix:] + sz0[:ix])
            Sk /= np.sqrt(L)
            G = dynamical_structure_factor(hamil, eigval, eigvec,
                                           Sk.transpose().conj(), Sk, omega)
            #G = dynamical_structure_factor(hamil, eigval, eigvec, Sk, Sk,
            #                               omega)
            Gs_iomega.append(G)
        Gw.append(np.sum(np.array(Gs_iomega)))
        Gkw.append(Gs_iomega)

    Gkw = np.array(Gkw)
    fig = plt.figure()
    ax = fig.add_subplot()
    mappable = ax.pcolor(ks, omegas, Gkw.imag)
    fig.colorbar(mappable)
    fig.savefig('Gkw.pdf')

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for ik, k in enumerate(ks):
        komega = [k for i in range(len(omegas))]
        ax.plot(komega, omegas, -Gkw.imag[:, ik])
    fig.savefig('Gkw_3d.pdf')

    Gw = np.array(Gw)
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(omegas, Gw.imag)
    fig.savefig('Gw.pdf')


def main():
    sp_ops = make_spin_ops()
    print(sp_ops)
    sz = sp_ops['Sz']
    sp = sp_ops['S+']
    sm = sp_ops['S-']
    s0 = sp_ops['I']
    L = 6
    hamil = np.zeros((2**L, 2**L))
    for i in range(L):
        szsz = [sz, sz] + [s0] * (L - 2)
        spsm = [sp, sm] + [s0] * (L - 2)
        smsp = [sm, sp] + [s0] * (L - 2)
        hamil += make_matrix(szsz[i:] + szsz[:i]) + (make_matrix(
            spsm[i:] + spsm[:i]) + make_matrix(smsp[i:] + smsp[:i])) / 2.0
    Gijs = []
    omegamax = 5
    omegamesh = 20
    for iomega in range(omegamesh):
        Gijs_axis0 = []
        for j in range(L):
            omega = iomega / omegamesh * omegamax
            Gijs_axis0.append(Gij(hamil, L, 0, j, omega))
        Gijs.append(Gijs_axis0)
    Gijs = np.array(Gijs)
    np.save('Gijs.npy', Gijs)
    print(Gijs)


main()
