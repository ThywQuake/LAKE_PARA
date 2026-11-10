"""The Fortran constants that come from default-real literals are float32-rounded."""

import numpy as np

from lakematrix.constants import C


def test_default_real_literals_are_float32_rounded():
    assert C.Kelvin0 == float(np.float32(273.15))
    assert C.g == float(np.float32(9.814))
    assert C.aMagw == float(np.float32(7.6326))
    assert C.bMagw == float(np.float32(241.9))
    assert C.emissivityofwater == float(np.float32(0.98))
    assert C.albedoofwater_lw == float(np.float32(0.02))
    # literals with a d exponent stay double
    assert C.sigma == 5.67e-8
    assert C.pref == 1.0e5
    assert C.T_phase_threshold == 1.0e-5
