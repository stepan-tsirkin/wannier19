import copy
import warnings
import numpy as np
import os
from functools import cached_property
from collections import defaultdict
import glob
import multiprocessing
from .system import System, pauli_xyz
from ..__utility import alpha_A, beta_A, clear_cached, one2three
from ..symmetry.point_symmetry import PointSymmetry, PointGroup, TimeReversal
from .ws_dist import ws_dist_map



class System_R(System):
    """
        The base class for describing a system. Does not have its own constructor,
        please use the child classes, e.g  :class:`System_w90` or :class:`System_tb`


        Parameters
        -----------
        berry : bool
            set ``True`` to enable evaluation of external term in  Berry connection or Berry curvature and their
            derivatives.
        spin : bool
            set ``True`` if quantities derived from spin  will be used.
        morb : bool
            set ``True`` to enable calculation of external terms in orbital moment and its derivatives.
            Requires the ``.uHu`` file.
        OSD : bool
            set ``True`` to enable calculation of external terms in orbital contribution to Optical Spatial dispersion
            Requires the `uIu`` and ``.uHu`` files.
        periodic : [bool,bool,bool]
            set ``True`` for periodic directions and ``False`` for confined (e.g. slab direction for 2D systems). If less then 3 values provided, the rest are treated as ``False`` .
        SHCryoo : bool
            set ``True`` if quantities derived from Ryoo's spin-current elements will be used. (RPS 2019)
        SHCqiao : bool
            set ``True`` if quantities derived from Qiao's approximated spin-current elements will be used. (QZYZ 2018).
        use_ws : bool
            minimal distance replica selection method :ref:`sec-replica`.  equivalent of ``use_ws_distance`` in Wannier90.
            (Note: for :class:`System_tb` the method is not employed in the constructor. use `do_ws_dist()` if needed)
        _getFF : bool
            generate the FF_R matrix based on the uIu file. May be used for only testing so far. Default : ``{_getFF}``
        use_wcc_phase: bool
            using wannier centers in Fourier transform. Corresponding to Convention I (True), II (False) in Ref."Tight-binding formalism in the context of the PythTB package". Default: ``{use_wcc_phase}``
        npar : int
            number of nodes used for parallelization in the `__init__` method. Default: `multiprocessing.cpu_count()`

        Notes
        -----
        + The system is described by its real lattice, symmetry group, and Hamiltonian and other real-space matrices.
        + The lattice is given by the lattice vectors in the Cartesian coordinates.
        + The system can be either periodic or confined in some directions.

        Attributes
        ----------
        needed_R_matrices : set
            the set of matrices that are needed for the current calculation. The matrices are set in the constructor.
        use_ws : bool
            minimal distance replica selection method :ref:`sec-replica`.  equivalent of ``use_ws_distance`` in Wannier90.
            (Note: for :class:`System_tb` the method is not employed in the constructor. use `do_ws_dist()` if needed)
        npar : int
            number of nodes used for parallelization in the `__init__` method. Default: `multiprocessing.cpu_count()`
        use_wcc_phase: bool
            using wannier centers in Fourier transform. Corresponding to Convention I (True), II (False) in Ref."Tight-binding formalism in the context of the PythTB package". Default: ``{use_wcc_phase}``    
        _XX_R : dict(str:array)
            dictionary of real-space matrices. The keys are the names of the matrices, the values are the matrices themselves.
        wannier_centers_cart : array(float)
            the positions of the Wannier centers in the Cartesian coordinates.
        wannier_centers_reduced : array(float)
            the positions of the Wannier centers in the reduced coordinates.
        iRvec : array(int)
            the array of the R-vectors in the reduced coordinates.
        num_wann : int
            the number of Wannier functions.
        real_lattice : array(float, shape=(3,3))
            the lattice vectors of the model.
        nRvec : int
            the number of R-vectors.
        iR0 : int
            the index of the R-vector [0,0,0] in the iRvec array.
        NKFFT_recommended : int
            the recommended size of the FFT grid to be used in the interpolation.
        """

    def __init__(self,
                 berry=False,
                 morb=False,
                 spin=False,
                 SHCryoo=False,
                 SHCqiao=False,
                 OSD=False,
                 use_ws=True,
                 use_wcc_phase=True,
                 npar=None,
                 _getFF=False,
                 **parameters):

        super().__init__(**parameters)
        self.use_ws = use_ws
        self.needed_R_matrices = {'Ham'}
        self.npar = multiprocessing.cpu_count() if npar is None else npar
        self.use_wcc_phase = use_wcc_phase
        if not self.use_wcc_phase:
            warnings.warn("use_wcc_phase=False is not recommended")

        if morb:
            self.needed_R_matrices.update(['AA', 'BB', 'CC'])
        if berry:
            self.needed_R_matrices.add('AA')
        if spin:
            self.needed_R_matrices.add('SS')
        if _getFF:
            self.needed_R_matrices.add('FF')
        if SHCryoo:
            self.needed_R_matrices.update(['AA', 'SS', 'SA', 'SHA', 'SH'])
        if SHCqiao:
            self.needed_R_matrices.update(['AA', 'SS', 'SR', 'SH', 'SHR'])
        if OSD:
            self.needed_R_matrices.update(['AA', 'BB', 'CC', 'GG', 'OO'])

        if self.force_internal_terms_only:
            self.needed_R_matrices = self.needed_R_matrices.intersection(['Ham', 'SS'])

        self._XX_R = dict()

    def set_wannier_centers(self, wannier_centers_cart=None, wannier_centers_reduced=None):
        """
            set self.wannier_centers_cart. Only one of parameters should be provided.
            If both are None: self.wannier_centers_cart is set to zero.
        """
        lcart = (wannier_centers_cart is not None)
        lred = (wannier_centers_reduced is not None)
        if lred:
            _wannier_centers_cart = wannier_centers_reduced.dot(self.real_lattice)
            if lcart:
                assert abs(wannier_centers_cart - _wannier_centers_cart).max() < 1e-8
            else:
                self.wannier_centers_cart = _wannier_centers_cart
        else:
            if lcart:
                self.wannier_centers_cart = wannier_centers_cart
            else:
                self.wannier_centers_cart = np.zeros((self.num_wann, 3))
        self.clear_cached_wcc()

    def need_R_any(self, keys):
        """returns True is any of the listed matrices is needed in to be set

        keys : str or list of str
            'AA', 'BB', 'CC', etc
        """
        if not isinstance(keys, (list, tuple)):
            keys = [keys]
        for k in keys:
            if k in self.needed_R_matrices:
                return True

    def get_R_mat(self, key):
        try:
            return self._XX_R[key]
        except KeyError:
            raise ValueError(f"The real-space matrix elements '{key}' are not set in the system," +
                             " but are required for the current calculation. please check parameters of the System() initializer")

    def has_R_mat(self, key):
        return key in self._XX_R

    def has_R_mat_any(self, keys):
        for k in keys:
            if self.has_R_mat(k):
                return True

    def set_R_mat(self, key, value, diag=False, R=None, reset=False, add=False, Hermitian=False):
        """
        Set real-space matrix specified by `key`. Either diagonal, specific R or full matrix.  Useful for model calculations

        Parameters
        ----------
        key : str
            'SS', 'AA' , etc
        value : array
            * `array(num_wann,...)` if `diag=True` . Sets the diagonal part ( if `R` not set, `R=[0,0,0]`)
            * `array(num_wann,num_wann,..)`  matrix for `R` (`R` should be set )
            * `array(num_wann,num_wann,nRvec,...)` full spin matrix for all R

            `...` denotes the vector/tensor cartesian dimensions of the matrix element
        diag : bool
            set only the diagonal for a specific R-vector (if specified), or fpr R=[0,0,0]
        R : list(int)
            list of 3 integer values specifying R. if
        reset : bool
            allows to reset matrix if it is already set
        add : bool
            add matrix to the already existing
        Hermitian : bool
            force the value to be Hermitian (only if all vectors are set at once)
        """
        assert value.shape[0] == self.num_wann
        if diag:
            if R is None:
                R = [0, 0, 0]
            XX = np.zeros((self.num_wann, self.num_wann) + value.shape[1:], dtype=value.dtype)
            XX[self.range_wann, self.range_wann] = value
            self.set_R_mat(key, XX, R=R, reset=reset, add=add)
        elif R is not None:
            XX = np.zeros((self.num_wann, self.num_wann, self.nRvec) + value.shape[2:], dtype=value.dtype)
            XX[:, :, self.iR(R)] = value
            self.set_R_mat(key, XX, reset=reset, add=add)
        else:
            if Hermitian:
                value = 0.5 * (value + self.conj_XX_R(value))
            if key in self._XX_R:
                if reset:
                    self._XX_R[key] = value
                elif add:
                    self._XX_R[key] += value
                else:
                    raise RuntimeError(f"setting {key} for the second time without explicit permission. smth is wrong")
            else:
                self._XX_R[key] = value

    def spin_block2interlace(self, backward=False):
        """
        Convert the spin ordering from block (like in the amn file old versions of VASP) to interlace (like in the amn file of QE and new versions of VASP)
        """
        nw2 = self.num_wann // 2
        mapping = np.zeros(self.num_wann, dtype=int)

        if backward:
            mapping[:nw2] = np.arange(nw2) * 2
            mapping[nw2:] = np.arange(nw2) * 2 + 1
        else:
            mapping[::2] = np.arange(nw2)
            mapping[1::2] = np.arange(nw2) + nw2

        for key, val in self._XX_R.items():
            self._XX_R[key] = val[:, mapping][mapping, :]
        self.wannier_centers_cart = self.wannier_centers_cart[mapping]
        self.clear_cached_R()
        self.clear_cached_wcc()


    def spin_interlace2block(self, backward=False):
        """
        Convert the spin ordering from interlace (like in the amn file of QE and new versions of VASP) to block (like in the amn file old versions of VASP)
        """
        self.spin_block2interlace(backward=not backward)


    @property
    def Ham_R(self):
        return self.get_R_mat('Ham')

    def symmetrize2(self, symmetrizer):
        if not self.use_wcc_phase:
            raise NotImplementedError("Symmetrization is implemented only for convention I")

        from ..symmetry.sym_wann_2 import SymWann
        symmetrize_wann = SymWann(
            symmetrizer=symmetrizer,
            iRvec=self.iRvec,
            wannier_centers_cart=self.wannier_centers_cart,
            use_wcc_phase=self.use_wcc_phase,
            silent=self.silent,
        )

        self.check_AA_diag_zero(msg="before symmetrization", set_zero=True)
        logfile = self.logfile

        logfile.write(f"Wannier Centers cart (raw):\n {self.wannier_centers_cart}\n")
        logfile.write(f"Wannier Centers red: (raw):\n {self.wannier_centers_reduced}\n")

        self._XX_R, self.iRvec, self.wannier_centers_cart = symmetrize_wann.symmetrize(XX_R=self._XX_R)
        self.set_symmetry(spacegroup=symmetrizer.spacegroup)
        self.clear_cached_R()
        self.clear_cached_wcc()


    def symmetrize(self, proj, positions, atom_name, soc=False, magmom=None, spin_ordering='interlace', store_symm_wann=False,
                   method="new",
                   rotations=None, translations=None):
        """
        Symmetrize Wannier matrices in real space: Ham_R, AA_R, BB_R, SS_R,... , as well as Wannier centers


        Parameters
        ----------
        positions: array
            Positions of each atom.
        atom_name: list
            Name of each atom.
        proj: list
            Should be the same with projections card in relative Wannier90.win.

            eg: ``['Te: s','Te:p']``

            If there is hybrid orbital, grouping the other orbitals.

            eg: ``['Fe':sp3d2;t2g]`` Plese don't use ``['Fe':sp3d2;dxz,dyz,dxy]``

                ``['X':sp;p2]`` Plese don't use ``['X':sp;pz,py]``

            Note: If in `wannier90.win` file one sets several projections in one line like ``['Fe':d;sp3]``
            the actual order (as written to the `wannier90.nnkp` file) may be different. It is ordered by the orbital number l,
            and the hybrids are assigned negative numbers (e.g. for sp3 l=-3, see
            `Wannier90 user guide <https://raw.githubusercontent.com/wannier-developers/wannier90/v3.1.0/doc/compiled_docs/user_guide.pdf>`__
            chapter 3). So, the actual order will be ``['Fe':sp3;d]``. To  avoid confusion, it is recommended to put the different groups of projectons
            as separate lines of the `wannier90.win` file. See also `here <https://github.com/wannier-developers/wannier90/issues/463>`__
        soc: bool
            Spin orbital coupling.
        magmom: 2D array
            Magnetic momens of each atoms.
        spin_ordering : str, "block" or "interlace"
            The ordering of the wannier functions in the spinor case.
            "block" means that first all the orbitals of the first spin are written, then the second spin. (like in the amn file old versions of VASP)
            "interlace" means that the orbitals of the two spins are interlaced. (like in the amn file of QE and new versions of VASP)
        store_symm_wann: bool
            Store the (temporary) SymWann object in the `sym_wann` attribute of the System object.
            Can be useful for evaluating symmetry eigenvalues of wavefunctions, etc.
        rotations: array-like (shape=(N,3,3))
            Rotations of the symmetry operations. (optional)
        translations: array-like (shape=(N,3))
            Translations of the symmetry operations. (optional)

        Notes
        -----
            Works only with phase convention I (`use_wcc_phase=True`)

            rotations and translations should be either given together or not given at all. Make sense to preserve consistensy in the order
            of the symmetry operations, when store_symm_wann is set to True.
        """
        if method == "new":
            assert spin_ordering == "interlace", "Symmetrization method 'new' is implemented only for spin_ordering='interlace'"
            from irrep.spacegroup import SpaceGroup
            from ..symmetry.sawf import SymmetrizerSAWF
            from ..wannierise.projections import Projection

            index = {key: i for i, key in enumerate(set(atom_name))}
            atom_num = np.array([index[key] for key in atom_name])

            spacegroup = SpaceGroup(cell=(self.real_lattice, positions, atom_num),
                                    magmom=magmom, include_TR=True,
                                    spinor=soc)
            spacegroup.show()

            assert len(atom_name) == len(positions), "atom_name and positions should have the same length"

            proj_list = []
            for proj_str in proj:
                atom, orbital = [l.strip() for l in proj_str.split(':')]
                pos = np.array([positions[i] for i, name in enumerate(atom_name) if name == atom])
                for suborbital in orbital.split(';'):
                    suborbital = suborbital.strip()
                    proj = Projection(position_num=pos, orbital=suborbital, spacegroup=spacegroup)
                    # print (f"adding projection {proj} ({pos} {suborbital})")
                    proj_list.append(proj)
            symmetrizer = SymmetrizerSAWF().set_spacegroup(spacegroup).set_D_wann_from_projections(projections_obj=proj_list)
            self.symmetrize2(symmetrizer)
            return symmetrizer
        else:
            assert method == "old", f"unknown symmetrization method {method}. expected 'old' or 'new'"


        if not self.use_wcc_phase:
            raise NotImplementedError("Symmetrization is implemented only for convention I")


        from ..symmetry.sym_wann import SymWann
        symmetrize_wann = SymWann(
            num_wann=self.num_wann,
            lattice=self.real_lattice,
            positions=positions,
            atom_name=atom_name,
            projections=proj,
            iRvec=self.iRvec,
            XX_R=self._XX_R,
            soc=soc,
            wannier_centers_cart=self.wannier_centers_cart,
            magmom=magmom,
            use_wcc_phase=self.use_wcc_phase,
            spin_ordering=spin_ordering,
            rotations=rotations,
            translations=translations,
            silent=self.silent,
        )

        self.check_AA_diag_zero(msg="before symmetrization", set_zero=True)
        logfile = self.logfile

        logfile.write(f"Wannier Centers cart (raw):\n {self.wannier_centers_cart}\n")
        logfile.write(f"Wannier Centers red: (raw):\n {self.wannier_centers_reduced}\n")
        self._XX_R, self.iRvec, self.wannier_centers_cart = symmetrize_wann.symmetrize()

        logfile.write(f"Wannier Centers cart (symmetrized):\n {self.wannier_centers_cart}\n")
        logfile.write(f"Wannier Centers red: (symmetrized):\n {self.wannier_centers_reduced}\n")
        self.clear_cached_R()
        self.clear_cached_wcc()
        self.check_AA_diag_zero(msg="after symmetrization", set_zero=True)
        self.symmetrize_info = dict(proj=proj, positions=positions, atom_name=atom_name, soc=soc, magmom=magmom,
                                    spin_ordering=spin_ordering)

        if store_symm_wann:
            del symmetrize_wann.matrix_dict_list
            del symmetrize_wann.matrix_list
            self.sym_wann = symmetrize_wann


    def check_AA_diag_zero(self, msg="", set_zero=True):
        if self.has_R_mat('AA') and self.use_wcc_phase:
            A_diag = self.get_R_mat('AA')[:, :, self.iR0].diagonal()
            A_diag_max = abs(A_diag).max()
            if A_diag_max > 1e-5:
                warnings.warn(
                    f"the maximal value of diagonal position matrix elements {msg} is {A_diag_max}."
                    f"This may signal a problem\n {A_diag}")
                if set_zero:
                    warnings.warn("setting AA diagonal to zero")
            if set_zero:
                self.get_R_mat('AA')[self.range_wann, self.range_wann, self.iR0, :] = 0

    def check_periodic(self):
        exclude = np.zeros(self.nRvec, dtype=bool)
        for i, per in enumerate(self.periodic):
            if not per:
                sel = (self.iRvec[:, i] != 0)
                if np.any(sel):
                    warnings.warn(f"you declared your system as non-periodic along direction {i},"
                                  f"but there are {sum(sel)} of total {self.nRvec} R-vectors with R[{i}]!=0."
                                  "They will be excluded, please make sure you know what you are doing")
                    exclude[sel] = True
        if np.any(exclude):
            notexclude = np.logical_not(exclude)
            self.iRvec = self.iRvec[notexclude]
            for X in ['Ham', 'AA', 'BB', 'CC', 'SS', 'FF']:
                if X in self._XX_R:
                    self.set_R_mat(X, self.get_X_mat(X)[:, :, notexclude], reset=True)

    def set_spin(self, spins, axis=(0, 0, 1), **kwargs):
        """
        Set spins along axis in  SS(R=0).  Useful for model calculations.
        Note : The spin matrix is purely diagonal, so that <up | sigma_x | down> = 0
        For more cversatility use :func:`~wannierberri.system.System.set_R_mat`
        :func:`~wannierberri.system.System.set_spin_pairs`, :func:`~wannierberri.system.System.set_spin_from_projections`

        Parameters
        ----------
        spin : one on the following
            1D `array(num_wann)` of `+1` or `-1` spins are along `axis`
        axis : array(3)
            spin quantization axis (if spin is a 1D array)
        **kwargs :
            optional arguments 'R', 'reset', 'add' see :func:`~wannierberri.system.System.set_R_mat`
        """
        spins = np.array(spins)
        if max(abs(spins) - 1) > 1e-3:
            warnings.warn("some of your spins are not +1 or -1, are you sure you want it like this?")
        axis = np.array(axis) / np.linalg.norm(axis)
        value = np.array([s * axis for s in spins], dtype=complex)
        self.set_R_mat(key='SS', value=value, diag=True, **kwargs)

    def set_spin_pairs(self, pairs):
        """set SS_R, assuming that each Wannier function is an eigenstate of Sz,

        Parameters
        ----------
        pairs : list of tuple
            list of pairs of indices of bands `[(up1,down1), (up2,down2), ..]`

        Notes
        -----
        * For abinitio calculations this is a rough approximation, that may be used on own risk.
        * See also :func:`~wannierberri.system.System.set_spin_from_projections`

        """
        assert all(len(p) == 2 for p in pairs)
        all_states = np.array(sum((list(p) for p in pairs), []))
        assert np.all(all_states >= 0) and (np.all(all_states < self.num_wann)), (
            f"indices of states should be 0<=i<num_wann-{self.num_wann}, found {pairs}")
        assert len(set(all_states)) == len(all_states), "some states appear more then once in pairs"
        if len(pairs) < self.num_wann / 2:
            warnings.warn(f"number of spin pairs {len(pairs)} is less then num_wann/2 = {self.num_wann / 2}."
                          "For other states spin properties will be set to zero. are yoiu sure ?")
        SS_R0 = np.zeros((self.num_wann, self.num_wann, 3), dtype=complex)
        for i, j in pairs:
            dist = np.linalg.norm(self.wannier_centers_cart[i] - self.wannier_centers_cart[j])
            if dist > 1e-3:
                warnings.warn(f"setting spin pair for Wannier function {i} and {j}, distance between them {dist}")
            SS_R0[i, i] = pauli_xyz[0, 0]
            SS_R0[i, j] = pauli_xyz[0, 1]
            SS_R0[j, i] = pauli_xyz[1, 0]
            SS_R0[j, j] = pauli_xyz[1, 1]
            self.set_R_mat(key='SS', value=SS_R0, diag=False, R=[0, 0, 0], reset=True)

    def set_spin_from_projections(self, spin_ordering="interlace"):
        """set SS_R, assuming that each Wannier function is an eigenstate of Sz,
         according to the ordering of the ab-initio code

        Parameters
        ----------
        spin_ordering : str, "block" or "interlace"
            The ordering of the wannier functions in the spinor case.
             * "block" means that first all the orbitals of the first spin are written, then the second spin. (like in the amn file old versions of VASP)
             * "interlace" means that the orbitals of the two spins are interlaced. (like in the amn file of QE and new versions of VASP)

        Notes
        -------
        * This is a rough approximation, that may be used on own risk
        * The pure-spin character may be broken by maximal localization. Recommended to use `num_iter=0` in Wannier90
        * if your DFT code has a different spin ordering, use   :func:`~wannierberri.system.System.set_spin_pairs`
        """
        assert self.num_wann % 2 == 0, f"odd number of Wannier functions {self.num_wann} cannot be grouped into spin pairs"
        nw2 = self.num_wann // 2
        if spin_ordering.lower() == 'block':
            pairs = [(i, i + nw2) for i in range(nw2)]
        elif spin_ordering.lower() == 'interlace':
            pairs = [(2 * i, 2 * i + 1) for i in range(nw2)]
        else:
            raise ValueError(f"unknown spin ordering {spin_ordering}. expected 'block' or 'interlace'")
        self.set_spin_pairs(pairs)

    def set_spacegroup(self, spacegroup):
        """
        Set the space group of the :class:`System`, which will be used for symmetrization
        R-space and k-space 
        Also sets the pointgroup

        Parameters
        ----------
        spacegroup : :class:`irrep.spacegroup.SpaceGroup`
            The space group of the system. The point group will be evaluated by the space group.
        """
        self.spacegroup = spacegroup
        self.set_pointgroup(spacegroup=spacegroup)

    def do_at_end_of_init(self):
        self.set_pointgroup()
        self.check_periodic()
        logfile = self.logfile
        logfile.write(f"Real-space lattice:\n {self.real_lattice}\n")
        logfile.write(f"Number of wannier functions: {self.num_wann}\n")
        logfile.write(f"Number of R points: {self.nRvec}\n")
        logfile.write(f"Recommended size of FFT grid {self.NKFFT_recommended}\n")

    def do_ws_dist(self, mp_grid, wannier_centers_cart=None):
        """
        Perform the minimal-distance replica selection method
        As a side effect - it sets the variable _NKFFT_recommended to mp_grid and self.use_ws=True

        Parameters:
        -----------
        wannier_centers_cart : array(float)
            Wannier centers used (if None -- use those already stored in the system)
        mp_grid : [nk1,nk2,nk3] or int
            size of Monkhorst-Pack frid used in ab initio calculation.
        """
        logfile = self.logfile
        try:
            mp_grid = one2three(mp_grid)
            assert mp_grid is not None
        except AssertionError:
            raise ValueError(f"mp_greid should be one integer, of three integers. found {mp_grid}")
        self._NKFFT_recommended = mp_grid
        self.use_ws = True
        if wannier_centers_cart is None:
            wannier_centers_cart = self.wannier_centers_cart
        ws_map = ws_dist_map(
            self.iRvec, wannier_centers_cart, mp_grid, self.real_lattice, npar=self.npar)
        for key, val in self._XX_R.items():
            logfile.write(f"using ws_dist for {key}\n")
            self.set_R_mat(key, ws_map(val), reset=True)
        self.iRvec = np.array(ws_map._iRvec_ordered, dtype=int)
        self.clear_cached_R()

    def to_tb_file(self, tb_file=None):
        """
        Write the system in the format of the wannier90_tb.dat file
        Note : it is always written in phase convention II
        """
        logfile = self.logfile
        if tb_file is None:
            tb_file = self.seedname + "_fromchk_tb.dat"
        f = open(tb_file, "w")
        f.write("written by wannier-berri form the chk file\n")
        logfile.write(f"writing TB file {tb_file}\n")
        np.savetxt(f, self.real_lattice)
        f.write(f"{self.num_wann}\n")
        f.write(f"{self.nRvec}\n")
        Ndegen = np.ones(self.nRvec, dtype=int)
        for i in range(0, self.nRvec, 15):
            a = Ndegen[i:min(i + 15, self.nRvec)]
            f.write("  ".join(f"{x:2d}" for x in a) + "\n")
        for iR in range(self.nRvec):
            f.write("\n  {0:3d}  {1:3d}  {2:3d}\n".format(*tuple(self.iRvec[iR])))
            _ham = self.Ham_R[:, :, iR] * Ndegen[iR]
            f.write(
                "".join(
                    f"{m + 1:3d} {n + 1:3d} {_ham[m, n].real:15.8e} {_ham[m, n].imag:15.8e}\n"
                    for n in self.range_wann for m in self.range_wann)
            )
        if self.has_R_mat('AA'):
            AA = np.copy(self.get_R_mat('AA'))
            if self.use_wcc_phase:
                AA[self.range_wann, self.range_wann, self.iR0] += self.wannier_centers_cart
            for iR in range(self.nRvec):
                f.write("\n  {0:3d}  {1:3d}  {2:3d}\n".format(*tuple(self.iRvec[iR])))
                _aa = AA[:, :, iR] * Ndegen[iR]
                f.write(
                    "".join(
                        f"{m + 1:3d} {n + 1:3d} " + " ".join(f"{a.real:15.8e} {a.imag:15.8e}" for a in _aa[m, n]) + "\n"
                        for n in self.range_wann for m in self.range_wann
                    )
                )
        f.close()

    def _FFT_compatible(self, FFT, iRvec):
        """check if FFT is enough to fit all R-vectors"""
        return np.unique(iRvec % FFT, axis=0).shape[0] == iRvec.shape[0]

    @property
    def NKFFT_recommended(self):
        """finds a minimal FFT grid on which different R-vectors do not overlap"""
        if hasattr(self, '_NKFFT_recommended'):
            return self._NKFFT_recommended
        NKFFTrec = np.ones(3, dtype=int)
        for i in range(3):
            R = self.iRvec[:, i]
            if len(R[R > 0]) > 0:
                NKFFTrec[i] += R.max()
            if len(R[R < 0]) > 0:
                NKFFTrec[i] -= R.min()
        assert self._FFT_compatible(NKFFTrec, self.iRvec)
        return NKFFTrec

    @cached_property
    def cRvec(self):
        return self.iRvec.dot(self.real_lattice)

    @cached_property
    def cRvec_p_wcc(self):
        """
        With self.use_wcc_phase=True it is R+tj-ti. With self.use_wcc_phase=False it is R. [i,j,iRvec,a] (Cartesian)
        """
        if self.use_wcc_phase:
            return self.cRvec[None, None, :, :] + self.diff_wcc_cart[:, :, None, :]
        else:
            return self.cRvec[None, None, :, :]

    def clear_cached_R(self):
        clear_cached(self, ['cRvec', 'cRvec_p_wcc', 'reverseR', 'index_R'])

    @cached_property
    def diff_wcc_cart(self):
        """
        With self.use_wcc_phase=True it is tj-ti. With self.use_wcc_phase=False it is 0. [i,j,a] (Cartesian)
        """
        wannier_centers = self.wannier_centers_cart
        return wannier_centers[None, :, :] - wannier_centers[:, None, :]

    @cached_property
    def diff_wcc_red(self):
        """
        With self.use_wcc_phase=True it is tj-ti. With self.use_wcc_phase=False it is 0. [i,j,a] (Reduced)
        """
        wannier_centers = self.wannier_centers_reduced
        return wannier_centers[None, :, :] - wannier_centers[:, None, :]

    @property
    def wannier_centers_cart_wcc_phase(self):
        """returns zero array if use_wcc_phase = False"""
        if self.use_wcc_phase:
            return self.wannier_centers_cart
        else:
            return np.zeros((self.num_wann, 3), dtype=float)

    def clear_cached_wcc(self):
        clear_cached(self, ['diff_wcc_cart', 'cRvec_p_wcc', 'diff_wcc_red', "wannier_centers_reduced"])

    @cached_property
    def wannier_centers_reduced(self):
        return self.wannier_centers_cart.dot(np.linalg.inv(self.real_lattice))

    def convention_II_to_I(self):
        R_new = {}
        if self.wannier_centers_cart is None:
            raise ValueError("use_wcc_phase = True, but the wannier centers could not be determined")
        if self.has_R_mat('AA'):
            AA_R_new = np.copy(self.get_R_mat('AA'))
            AA_R_new[self.range_wann, self.range_wann, self.iR0, :] -= self.wannier_centers_cart
            R_new['AA'] = AA_R_new
        if self.has_R_mat('BB'):
            BB_R_new = self.get_R_mat('BB').copy() - self.get_R_mat('Ham')[:, :, :,
                                                     None] * self.wannier_centers_cart[None, :, None, :]
            R_new['BB'] = BB_R_new
        if self.has_R_mat('CC'):
            norm = np.linalg.norm(self.get_R_mat('CC') - self.conj_XX_R(key='CC'))
            assert norm < 1e-10, f"CC_R is not Hermitian, norm={norm}"
            assert self.has_R_mat('BB'), "if you use CC_R and use_wcc_phase=True, you need also BB_R"
            T = self.wannier_centers_cart[:, None, None, :, None] * self.get_R_mat('BB')[:, :, :, None, :]
            CC_R_new = self.get_R_mat('CC').copy() + 1.j * sum(
                s * (
                    -T[:, :, :, a, b] -  # -t_i^a * B_{ij}^b(R)
                    self.conj_XX_R(T[:, :, :, b, a]) +  # - B_{ji}^a(-R)^*  * t_j^b
                    self.wannier_centers_cart[:, None, None, a] * self.Ham_R[:, :, :, None] *
                    self.wannier_centers_cart[None, :, None, b]  # + t_i^a*H_ij(R)t_j^b
                ) for (s, a, b) in [(+1, alpha_A, beta_A), (-1, beta_A, alpha_A)])
            norm = np.linalg.norm(CC_R_new - self.conj_XX_R(CC_R_new))
            assert norm < 1e-10, f"CC_R after applying wcc_phase is not Hermitian, norm={norm}"
            R_new['CC'] = CC_R_new
        if self.has_R_mat('SA'):
            SA_R_new = self.get_R_mat('SA').copy() - self.get_R_mat('SS')[:, :, :,
                    None, :] * self.wannier_centers_cart[None, :, None, :, None]
            R_new['SA'] = SA_R_new
        if self.has_R_mat('SHA'):
            SHA_R_new = self.get_R_mat('SHA').copy() - self.get_R_mat('SH')[:, :, :,
                    None, :] * self.wannier_centers_cart[None, :, None, :, None]
            R_new['SHA'] = SHA_R_new

        unknown = set(self._XX_R.keys()) - set(['Ham', 'AA', 'BB', 'CC', 'SS', 'SH', 'SA', 'SHA'])
        if len(unknown) > 0:
            raise NotImplementedError(f"Conversion of conventions for {list(unknown)} is not implemented")

        for X in ['AA', 'BB', 'CC', 'SA', 'SHA']:
            if self.has_R_mat(X):
                self.set_R_mat(X, R_new[X], reset=True)

    @property
    def iR0(self):
        return self.iRvec.tolist().index([0, 0, 0])

    @cached_property
    def index_R(self):
        return {tuple(R): i for i, R in enumerate(self.iRvec)}

    def iR(self, R):
        R = np.array(np.round(R), dtype=int).tolist()
        return self.iRvec.tolist().index(R)

    @cached_property
    def reverseR(self):
        """indices of R vectors that has -R in irvec, and the indices of the corresponding -R vectors."""
        mapping = np.all(self.iRvec[:, None, :] + self.iRvec[None, :, :] == 0, axis=2)
        # check if some R-vectors do not have partners
        notfound = np.where(np.logical_not(mapping.any(axis=1)))[0]
        if len(notfound) > 0 and not self.ignore_mR_not_found:
            for ir in notfound:
                warnings.warn(f"R[{ir}] = {self.iRvec[ir]} does not have a -R partner")
        # check if some R-vectors have more then 1 partner
        morefound = np.where(np.sum(mapping, axis=1) > 1)[0]
        if len(morefound > 0):
            raise RuntimeError(
                f"R vectors number {morefound} have more then one negative partner : "
                f"\n{self.iRvec[morefound]} \n{np.sum(mapping, axis=1)}")
        lst_R, lst_mR = [], []
        for ir1 in range(self.nRvec):
            ir2 = np.where(mapping[ir1])[0]
            if len(ir2) == 1:
                lst_R.append(ir1)
                lst_mR.append(ir2[0])
        lst_R = np.array(lst_R)
        lst_mR = np.array(lst_mR)
        # Check whether the result is correct
        assert np.all(self.iRvec[lst_R] + self.iRvec[lst_mR] == 0)
        return lst_R, lst_mR

    def conj_XX_R(self, val: np.ndarray = None, key: str = None, ignore_mR_not_found=False):
        """ reverses the R-vector and takes the hermitian conjugate """
        assert (key is not None) != (val is not None)
        if key is not None:
            XX_R = self.get_R_mat(key)
        else:
            XX_R = val
        XX_R_new = np.zeros(XX_R.shape, dtype=complex)
        self.ignore_mR_not_found = ignore_mR_not_found
        lst_R, lst_mR = self.reverseR
        XX_R_new[:, :, lst_R] = XX_R[:, :, lst_mR]
        return XX_R_new.swapaxes(0, 1).conj()

    @property
    def nRvec(self):
        return self.iRvec.shape[0]

    def check_hermitian(self, key):
        if key in self._XX_R.keys():
            _X = self.get_R_mat(key).copy()
            assert (np.max(abs(_X - self.conj_XX_R(key=key))) < 1e-8), f"{key} should obey X(-R) = X(R)^+"
        else:
            self.logfile.write(f"{key} is missing, nothing to check\n")

    def set_structure(self, positions, atom_labels, magnetic_moments=None):
        """
        Set atomic structure of the system.

        Parameters
        ----------
        positions : (num_atom, 3) array_like of float
            Atomic positions in fractional coordinates.
        atom_labels: (num_atom,) list
            labels (integer, string, etc.) to distinguish species.
        magnetic_moments: (num_atom, 3) array_like of float (optional)
            Magnetic moment vector of each atom.
        """
        if len(positions) != len(atom_labels):
            raise ValueError("length of positions and atom_labels must be the same")
        if magnetic_moments is not None:
            if len(magnetic_moments) != len(positions):
                raise ValueError("length of positions and magnetic_moments must be the same")
            if not all([len(x) == 3 for x in magnetic_moments]):
                raise ValueError("magnetic_moments must be a list of 3d vector")
        self.positions = positions
        self.atom_labels = atom_labels
        self.magnetic_moments = magnetic_moments

    def get_spglib_cell(self):
        """Returns the atomic structure as a cell tuple in spglib format"""
        try:
            # assign integer to self.atom_labels
            atom_labels_unique = list(set(self.atom_labels))
            atom_numbers = [atom_labels_unique.index(label) for label in self.atom_labels]
            if self.magnetic_moments is None:
                return self.real_lattice, self.positions, atom_numbers
            else:
                return self.real_lattice, self.positions, atom_numbers, self.magnetic_moments
        except AttributeError:
            raise AttributeError("set_structure must be called before get_spglib_cell")

    def set_symmetry_from_structure(self):
        """
        Set the symmetry group of the :class:`System`. Requires spglib to be installed.
        :meth:`System.set_structure` must be called in advance.

        For magnetic systems, symmetries involving time reversal are not detected because
        spglib does not support time reversal symmetry for noncollinear systems.
        """
        import spglib

        spglib_symmetry = spglib.get_symmetry(self.get_spglib_cell())
        symmetry_gen = []
        for isym, W in enumerate(spglib_symmetry["rotations"]):
            # spglib gives real-space rotations in reduced coordinates. Here,
            # 1) convert to Cartesian coordinates, and
            # 2) take transpose to go to reciprocal space.
            W = spglib_symmetry["rotations"][isym]
            Wcart = self.real_lattice.T @ W @ np.linalg.inv(self.real_lattice).T
            R = Wcart.T
            try:
                TR = spglib_symmetry['time_reversals'][isym]
                tr_found = True
            except KeyError:
                TR = False
                tr_found = False
            symmetry_gen.append(PointSymmetry(R, TR=TR))

        if self.magnetic_moments is None:
            symmetry_gen.append(TimeReversal)
        elif not tr_found:
            warnings.warn(
                "you specified magnetic moments but spglib did not detect symmetries involving time-reversal. "
                f"proobably it is because you have an old spglib version {spglib.__version__}."
                "We suggest upgrading to spglib>=2.0.2")
        else:
            if not all([len(x) for x in self.magnetic_moments]):
                raise ValueError("magnetic_moments must be a list of 3d vector")
            warnings.warn("spglib does not find symmetries including time reversal operation. "
                          "To include such symmetries, use set_symmetry.")

        self.pointgroup = PointGroup(symmetry_gen, recip_lattice=self.recip_lattice, real_lattice=self.real_lattice)

    def get_sparse(self, min_values={'Ham': 1e-3}):
        min_values = copy.copy(min_values)
        ret_dic = dict(
            real_lattice=self.real_lattice,
            wannier_centers_reduced=self.wannier_centers_reduced,
            matrices={},
            use_wcc_phase=self.use_wcc_phase
        )
        if hasattr(self, 'symmetrize_info'):
            ret_dic['symmetrize_info'] = self.symmetrize_info

        def array_to_dict(A, minval):
            A_tmp = abs(A.reshape(A.shape[:3] + (-1,))).max(axis=-1)
            wh = np.argwhere(A_tmp >= minval)
            dic = defaultdict(lambda: dict())
            for w in wh:
                iR = tuple(self.iRvec[w[2]])
                dic[iR][(w[0], w[1])] = A[tuple(w)]
            return dict(dic)

        for k, v in min_values.items():
            ret_dic['matrices'][k] = array_to_dict(self.get_R_mat(k), v)
        return ret_dic

    @cached_property
    def essential_properties(self):
        return ['num_wann', 'real_lattice', 'iRvec', 'periodic',
                'use_wcc_phase', 'is_phonon', 'wannier_centers_cart', 'pointgroup']

    @cached_property
    def optional_properties(self):
        return ["positions", "magnetic_moments", "atom_labels"]

    def _R_mat_npz_filename(self, key):
        return "_XX_R_" + key + ".npz"

    def save_npz(self, path, extra_properties=(), exclude_properties=(), R_matrices=None, overwrite=True):
        """
        Save system to a directory of npz files
        Parameters
        ----------
        path : str
            path to saved files. If does not exist - will be created (unless overwrite=False)
        extra_properties : list of str
            names of properties which are not essential for reconstruction, but will also be saved
        exclude_properties : list of str
            dp not save certain properties - duse on your own risk
        R_matrices : list of str
            list of the R matrices, e.g. ```['Ham','AA',...]``` to be saved. if None: all R-matrices will be saved
        overwrite : bool
            if the directory already exiists, it will be overwritten
        """
        logfile = self.logfile

        properties = [x for x in self.essential_properties + list(extra_properties) if x not in exclude_properties]
        if R_matrices is None:
            R_matrices = list(self._XX_R.keys())

        try:
            os.makedirs(path, exist_ok=overwrite)
        except FileExistsError:
            raise FileExistsError(f"Directorry {path} already exists. To overwrite it set overwrite=True")

        for key in properties:
            logfile.write(f"saving {key}\n")
            fullpath = os.path.join(path, key + ".npz")
            a = getattr(self, key)
            if key in ['pointgroup']:
                np.savez(fullpath, **a.as_dict())
            else:
                np.savez(fullpath, a)
            logfile.write(" - Ok!\n")
        for key in self.optional_properties:
            if key not in properties:
                fullpath = os.path.join(path, key + ".npz")
                if hasattr(self, key):
                    a = getattr(self, key)
                    np.savez(fullpath, a)
        for key in R_matrices:
            logfile.write(f"saving {key}")
            np.savez_compressed(os.path.join(path, self._R_mat_npz_filename(key)), self.get_R_mat(key))
            logfile.write(" - Ok!\n")

    def load_npz(self, path, load_all_XX_R=False, exclude_properties=()):
        """
        Save system to a directory of npz files
        Parameters
        ----------
        path : str
            path to saved files. If does not exist - will be created (unless overwrite=False)
        load_all_XX_R : list of str
            load all matrices which were saved
        exclude_properties : list of str
            dp not save certain properties - duse on your own risk
        """
        logfile = self.logfile
        all_files = glob.glob(os.path.join(path, "*.npz"))
        all_names = [os.path.splitext(os.path.split(x)[-1])[0] for x in all_files]
        properties = [x for x in all_names if not x.startswith('_XX_R_') and x not in exclude_properties]
        for key in properties:
            logfile.write(f"loading {key}")
            a = np.load(os.path.join(path, key + ".npz"), allow_pickle=False)

            # pointgroup was previouslly named symgroup. This is for backward compatibility
            if key == 'symgroup':
                key_loc = 'pointgroup'
            else:
                key_loc = key

            if key_loc == 'pointgroup':
                val = PointGroup(dictionary=a)
            else:
                val = a['arr_0']
            setattr(self, key_loc, val)
            logfile.write(" - Ok!\n")
        if load_all_XX_R:
            R_files = glob.glob(os.path.join(path, "_XX_R_*.npz"))
            R_matrices = [os.path.splitext(os.path.split(x)[-1])[0][6:] for x in R_files]
            self.needed_R_matrices.update(R_matrices)
        for key in self.needed_R_matrices:
            logfile.write(f"loading R_matrix {key}")
            a = np.load(os.path.join(path, self._R_mat_npz_filename(key)), allow_pickle=False)['arr_0']
            self.set_R_mat(key, a)
            logfile.write(" - Ok!\n")
