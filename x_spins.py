import numpy as np


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


def make_dynamical_structure_factor(eigval, eigvec, S0, S1, omega):
    G = 0 + 0j
    E0 = eigval[0]
    for j, Ej in enumerate(eigval):
        G += (eigvec[:, 0].conj() @ S0 @ eigvec[:, j]) * (eigvec[:,
                                                                 j].conj() @ S1 @ eigvec[:, 0]) / (E0 - Ej + omega + 0.1j)
    return G


def make_Gij(eigval, eigvec, L, i, j, omega):
    sp_ops = make_spin_ops()
    sz = sp_ops['Sz']
    sp = sp_ops['S+']
    sm = sp_ops['S-']
    s0 = sp_ops['I']
    szi = [s0] * i + [sz] + [s0] * (L - i - 1)
    szj = [s0] * j + [sz] + [s0] * (L - j - 1)
    Si = make_matrix(szi)
    Sj = make_matrix(szj)
    G = make_dynamical_structure_factor(eigval, eigvec, Sj, Si, omega)
    return G
