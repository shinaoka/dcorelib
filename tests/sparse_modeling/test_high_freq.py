import numpy as np

import sparse_ir
from dcorelib.sparse_gf import high_freq
import pytest


@pytest.mark.parametrize("statistics", ["F", "B"])
def test_high_freq_moments(statistics):
    """
    G(iv) = A/(iv - e) =  G_1/iv + G_2/(iv)^2 + O(1/(iv)^3),
    where
        A = [[1, 0], [0, 1]],
        G1 = A,
        G2 = e * A.
    """
    lambda_ = 1e+3
    beta = 1e+2
    eps = 1e-7
    basis = sparse_ir.FiniteTempBasis(
        statistics, beta, lambda_/beta, eps=eps,
        kernel=sparse_ir.LogisticKernel(lambda_)
    )
    nf = 2

    smpl = sparse_ir.MatsubaraSampling(basis)
    A = np.identity(2)
    e = 0.1
    assert lambda_/beta > e
    high_freq_mom = [A, e*A]
    num_moments = len(high_freq_mom)

    iv = 1J * np.pi * smpl.sampling_points/beta
    giv = np.einsum('w,ij->wij', 1/(iv - e), A)
    gl = smpl.fit(giv, axis=0)
    high_freq_mom_reconst = high_freq.high_freq_moment(gl, basis, num_moments=2, axis=0)

    high_freq_mom_reconst2 = []
    for m in range(num_moments):
        ev = high_freq.evalulator_high_freq_moment(basis, m+1)
        high_freq_mom_reconst2.append(np.einsum('l,lij->ij', ev, gl))

    for m in range(num_moments):
        print(m, high_freq_mom_reconst[m])
        assert (np.abs(high_freq_mom_reconst[m] - high_freq_mom[m]).max()) < (100**m) * 100 * eps
        assert (np.abs(high_freq_mom_reconst2[m] - high_freq_mom[m]).max()) < (100**m) * 100 * eps
