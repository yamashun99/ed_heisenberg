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


def make_matrix(s1, s2):
    return np.kron(s1, s2)


if __name__ == '__main__':
    sp_ops = make_spin_ops()
    sz = sp_ops['Sz']
    sp = sp_ops['S+']
    sm = sp_ops['S-']
    s0 = sp_ops['I']
    hamil = make_matrix(sz,
                        sz) + (make_matrix(sp, sm) + make_matrix(sm, sp)) / 2.0
    print("H=", hamil)
    eigval, eigvec = np.linalg.eigh(hamil)
    print("Eigenvalues=", eigval)
    print("Eigenvectors=", eigvec)
    print(eigvec[:, 0])
    print(hamil @ eigvec)
