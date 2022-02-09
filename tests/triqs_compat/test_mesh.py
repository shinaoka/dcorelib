import pytest
import numpy as np
from dcorelib.triqs_compat.gf.meshes import *

def test_meshimfreq():
    beta = 10
    mesh = MeshImFreq(beta, statistic="F", n_points=100)
    assert mesh.positive_only() == False