import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def make_spin_ops():
    ops = {}
    ops['Sz'] = np.diag((0.5, -0.5))
    sp = np.zeros((2, 2))
    sp[0, 1] = 1
    ops['S+'] = sp
    sm = sp.transpose()
    ops['S-'] = sm
    ops['Sx'] = (sp + sm) / 2.0
    ops['Sy'] = (sp - sm) / 2.0j
    ops['I'] = np.identity(2)
    return ops


def make_matrix(ss):
    res = np.array([1])
    for s in reversed(ss):
        res = np.kron(res, s)
    return res


def make_dynamical_structure_factor(hamil, eigval, eigvec, S0, S1, omega):
    G = 0 + 0j
    E0 = eigval[0]
    for j, Ej in enumerate(eigval):
        G += (eigvec[:, 0].conj() @ S0 @ eigvec[:, j]) * (eigvec[:,
                                                                 j].conj() @ S1 @ eigvec[:, 0]) / (E0 - Ej + omega + 0.1 + 0.1j)
    return G


def make_Gij(hamil, L, i, j, omega):
    sp_ops = make_spin_ops()
    sz = sp_ops['Sz']
    sp = sp_ops['S+']
    sm = sp_ops['S-']
    s0 = sp_ops['I']
    eigval, eigvec = np.linalg.eigh(hamil)
    szi = [s0] * i + [sz] + [s0] * (L - i - 1)
    szj = [s0] * j + [sz] + [s0] * (L - j - 1)
    Si = make_matrix(szi)
    Sj = make_matrix(szj)
    G = make_dynamical_structure_factor(hamil, eigval, eigvec, Sj, Si, omega)
    return G


if __name__ == '__main__':
    sp_ops = make_spin_ops()
    sz = sp_ops['Sz']
    sp = sp_ops['S+']
    sm = sp_ops['S-']
    s0 = sp_ops['I']
    L = 8
    hamil = np.zeros((2**L, 2**L))
    for i in range(L):
        szsz = [sz, sz] + [s0] * (L - 2)
        spsm = [sp, sm] + [s0] * (L - 2)
        smsp = [sm, sp] + [s0] * (L - 2)
        hamil += make_matrix(szsz[i:] + szsz[:i]) + (make_matrix(
            spsm[i:] + spsm[:i]) + make_matrix(smsp[i:] + smsp[:i])) / 2.0
    Gijs = []
    omegamax = 5
    omegamesh = 40
    for iomega in range(omegamesh):
        print(iomega)
        Gijs_axis0 = []
        for j in range(L):
            omega = iomega / omegamesh * omegamax
            Gij = make_Gij(hamil, L, 0, j, omega)
            Gijs_axis0.append(Gij)
        Gijs.append(Gijs_axis0)
    Gijs = np.array(Gijs)
    np.save('Gijs.npy', Gijs)
