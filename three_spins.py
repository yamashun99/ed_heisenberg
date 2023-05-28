import numpy as np
from two_spins import make_spin_ops


def make_matrix(s1, s2, s3):
    return np.kron(s1, np.kron(s2, s3))


def main():
    sp_ops = make_spin_ops()
    sz = sp_ops['Sz']
    sp = sp_ops['S+']
    sm = sp_ops['S-']
    s0 = sp_ops['I']

    hamil = make_matrix(sz, sz, s0) + (make_matrix(sp, sm, s0) + make_matrix(
        sm, sp, s0)) / 2.0 + make_matrix(s0, sz, sz) + (make_matrix(
            s0, sp, sm) + make_matrix(s0, sm, sp)) / 2.0 + make_matrix(
                sz, s0,
                sz) + (make_matrix(sp, s0, sm) + make_matrix(sm, s0, sp)) / 2.0

    print("H=", hamil)
    eigval, eigvec = np.linalg.eigh(hamil)
    print("Eigenvalues=", eigval)
    print("Eigenvectors=", eigvec)
    print(eigvec[:, 0])
    print(hamil @ eigvec)


main()
