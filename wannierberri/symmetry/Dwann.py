
import numpy as np
from .orbitals import num_orbitals
from ..__utility import UniqueListMod1


class Dwann:

    """
    A class to generate the Wannier transformation matrices D_wann 
    same as those in the .dmn file of pw2wannier90.x output.

    Parameters
    ----------
    spacegroup : irrep.SpaceGroup
        A space group object. (note, the order of operations should coincide with dmn file,
        so it should be initialized with reading the .sym file)
    positions : list(np.ndarray(shape=(3,), dtype=float))
        List of positions of Wannier centers in reduced coordinates.
        may be initialized with only one position. Then the other positions in the orbit will be generated
        by applying the symmetry operations of the spacegroup.
        if the file will be used with an already generated .amn file, 
        the positions should be the same as the .amn file, including the order.
    projection : str
        The projection type. Default is "_" which means "s". 
        (not implemented yet, but should be implemented in the future)
    spinor : bool
        Whether the Wannier functions are spinors.


    Attributes
    ----------
    orbit : list(np.ndarray(shape=(3,), dtype=float))
        List of positions of Wannier centers in reduced coordinates.
    spacegroup : irrep.SpaceGroup
        A space group object.
    num_points : int
        Number of points in the orbit.
    nsym : int
        Number of symmetry operations in the spacegroup.
    atommap : np.ndarray(shape=(num_points, nsym), dtype=int)
        A matrix that maps the orbit points to each other by the symmetry operations of the spacegroup.

    Notes
    -----
    * the spin ordering is always assumed "iterlaced", i.e. like in QE (or new VASP). If you are using an old VASP version,
    you should change the spin_ordering to "block" in the of w90data.amn object.
    """

    def __init__(self, spacegroup, positions, orbital="_",
                orbital_rotator=None,
                 spinor=False):

        self.nsym = spacegroup.size
        if spinor:
            self.spinor = True
            self.nspinor = 2
        else:
            self.spinor = False
            self.nspinor = 1

        if orbital != "_":
            assert orbital_rotator is not None
            self.rot_orb = [orbital_rotator(orbital, i)
                       for i in range(self.nsym)]
            self.num_orbitals_scal = num_orbitals(orbital)
        else:
            self.rot_orb = [np.eye(1) for _ in range(self.nsym)]
            self.num_orbitals_scal = 1

        if self.spinor:
            for isym, symop in enumerate(spacegroup.symmetries):
                S = symop.spinor_rotation
                if symop.time_reversal:
                    S = np.array([[0, 1], [-1, 0]]) @ S.conj()
                self.rot_orb[isym] = np.kron(self.rot_orb[isym], S)
        self.rot_orb = np.array(self.rot_orb)

        self.num_orbitals = self.num_orbitals_scal * self.nspinor

        self.orbit = orbit_from_positions(spacegroup, positions)
        self.spacegroup = spacegroup

        self.num_points = len(self.orbit)
        self.num_wann_scal = self.num_points * self.num_orbitals_scal
        self.num_wann = self.num_wann_scal * self.nspinor
        self.atommap = -np.ones((self.num_points, self.nsym), dtype=int)
        self.T = np.zeros((self.num_points, self.nsym, 3), dtype=float)

        for ip, p in enumerate(self.orbit):
            for isym, symop in enumerate(spacegroup.symmetries):
                p2 = symop.transform_r(p)
                # print (f"ip={ip}, p={p}, isym={isym}, p2={p2}")
                ip2 = self.orbit.index(p2)
                self.atommap[ip, isym] = ip2
                p2 = symop.transform_r(p)
                p2a = self.orbit[ip2]
                self.T[ip, isym] = p2a - p2
        T_round = np.round(self.T)
        assert np.allclose(self.T, T_round, atol=1e-7), f"T=\n{self.T}, \nT_round=\n{T_round}, \n max_diff={np.max(np.abs(self.T - T_round))}"
        self.T = T_round.astype(int)

        assert np.all(self.atommap >= 0), f"atommap={self.atommap}"




    def get_on_points(self, kptirr, kpt, isym):
        """
        Get the Wannier transformation matrix D_wann between a given 
        irreducible k-point and a given k-point.

        Parameters
        ----------
        kptirr : np.ndarray(shape=(3,), dtype=float)
            The irreducible k-point in reduced coordinates.
        kpt : np.ndarray(shape=(3,), dtype=float)
            The k-point in reduced coordinates.
        isym : int
            The index of the symmetry operation in the spacegroup.

        Returns
        -------
        Dwann : np.ndarray(shape=(num_points, num_points), dtype=complex)
            The Wannier transformation matrix D_wann.

        """
        symop = self.spacegroup.symmetries[isym]
        kptirr1 = symop.transform_k(kptirr)
        g = kpt - kptirr1
        assert np.all(abs(g - np.round(g)) < 1e-7), f"isym={isym}, g={g}, k1={kptirr}, k1p={kptirr1}, k2={kpt}"
        Dwann = np.zeros((self.num_wann, self.num_wann), dtype=complex)
        for ip, _ in enumerate(self.orbit):
            jp = self.atommap[ip, isym]
            Dwann[jp * self.num_orbitals:(jp + 1) * self.num_orbitals,
                  ip * self.num_orbitals:(ip + 1) * self.num_orbitals
                  ] = np.exp(2j * np.pi * (np.dot(kptirr1, self.T[ip, isym]))) * self.rot_orb[isym]
        # Here we assume that all the orbitals are real, so we don't need to take the complex conjugate
        return Dwann

    def get_on_points_all(self, kpoints, ikptirr, ikptirr2kpt):
        """
        generate the Wannier transformation matrices D_wann for all k-points
        on a grid.

        Parameters
        ----------
        kpoints : np.ndarray(shape=(nkpt,3), dtype=float)
            The k-points in reduced coordinates.
        ikptirr : np.ndarray(shape=(NKirr,), dtype=int)
            The indices of the irreducible k-points in the grid.
        ikptirr2kpt : np.ndarray(shape=(NKirr,nsym), dtype=int)
            The indices of the k-points in the grid that are related 
            to the irreducible k-points by the symmetry operations of the spacegroup.

        Returns
        -------
        Dwann : np.ndarray(shape=(NKirr,nsym, num_points,num_points), dtype=complex)
            The Wannier transformation matrices D_wann.
        """
        Dwann = np.zeros((len(ikptirr), self.nsym, self.num_wann, self.num_wann), dtype=complex)
        for ikirr, ik in enumerate(ikptirr):
            for isym in range(self.nsym):
                Dwann[ikirr, isym] = self.get_on_points(kpoints[ik], kpoints[ikptirr2kpt[ikirr, isym]], isym)
        return Dwann


def orbit_from_positions(spacegroup, positions):
    """
    Generate the orbit of Wannier centers from a list of positions
    and the symmetry operations of the spacegroup.

    Parameters
    ----------
    spacegroup : irrep.SpaceGroup
        A space group object.
    positions : list(np.ndarray(shape=(3,), dtype=float))
        List of positions of Wannier centers in reduced coordinates.

    Returns
    -------
    orbit : list(np.ndarray(shape=(3,), dtype=float))
        List of positions of Wannier centers in reduced coordinates.
    """
    orbit = UniqueListMod1()
    positions = np.array(positions)
    assert positions.ndim in [1, 2]
    if positions.ndim == 1:
        positions = positions[None, :]
    assert positions.shape[-1] == 3
    for p in positions:
        orbit.append(p)
    for symop in spacegroup.symmetries:
        for p in positions:
            orbit.append((symop.transform_r(p)) % 1)
    return orbit
