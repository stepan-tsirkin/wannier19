"""Test symmetry objects"""

from copy import deepcopy
import numpy as np
import pytest

import wannierberri.symmetry as sym

from common_systems import symmetries_GaAs, symmetries_Fe


@pytest.fixture
def check_symgroup_equal():

    def _inner(g1, g2):
        assert g1.size == g2.size, "Symmetry group size is different"
        for s in g1.symmetries:
            if s not in g2.symmetries:
                assert False, f"Symmetry {s} in group 1 is not in group 2"

    return _inner


def test_symmetry_group():
    assert sym.Group([sym.Inversion, sym.TimeReversal]).size == 4
    assert sym.Group([sym.C3z, sym.C6z]).size == 6
    assert sym.Group([sym.Inversion, sym.C4z, sym.TimeReversal * sym.C2x]).size == 16


def test_symmetry_group_failure():
    # sym.Group should fail for this generator
    with pytest.raises(RuntimeError):
        c = np.cos(0.1)
        s = np.sin(0.1)
        sym.Group([sym.Symmetry(np.array([[1, 0, 0], [0, c, s], [0, -s, c]]))])


def test_symmetry_spglib_GaAs(system_GaAs_W90, check_symgroup_equal):
    system_explicit = deepcopy(system_GaAs_W90)
    system_explicit.set_symmetry(symmetries_GaAs)

    system_spglib = deepcopy(system_GaAs_W90)
    positions = [[0., 0., 0.], [0.25, 0.25, 0.25]]
    labels = ["Ga", "As"]
    system_spglib.set_structure(positions, labels)
    system_spglib.set_symmetry_from_structure()

    check_symgroup_equal(system_explicit.symgroup, system_spglib.symgroup)


def test_symmetry_spglib_Fe(system_Fe_W90, check_symgroup_equal):
    system_explicit = deepcopy(system_Fe_W90)

    # Magnetic symmetries involving time-reversal is not implemented in spglib.
    # So, we exclude symmetries involving time reversal from the generators.
    symmetries_Fe_except_TR = [sym for sym in symmetries_Fe if not sym.TR]
    system_explicit.set_symmetry(symmetries_Fe_except_TR)

    system_spglib = deepcopy(system_Fe_W90)
    positions = [[0., 0., 0.]]
    labels = ["Fe"]
    magnetic_moments = [[0., 0., 1.]]
    system_spglib.set_structure(positions, labels, magnetic_moments)
    system_spglib.set_symmetry_from_structure()

    check_symgroup_equal(system_explicit.symgroup, system_spglib.symgroup)

    # Raise error if magnetic_moments is set to a number, not a 3d vector
    with pytest.raises(Exception):
        system_spglib.set_structure(positions, labels, [1.])
