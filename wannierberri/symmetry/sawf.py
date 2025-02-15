from functools import cached_property, lru_cache
from time import time
import warnings
from irrep.bandstructure import BandStructure
from irrep.spacegroup import SpaceGroupBare
import numpy as np
from ..__utility import get_inverse_block, rotate_block_matrix, orthogonalize
from ..wannierise.projections import ProjectionsSet

from ..w90files import DMN
from .Dwann import Dwann
from .orbitals import OrbitalRotator


class SymmetrizerSAWF(DMN):
    """
    An extended class for DMN object, which cannot be written to a w90 file, because it 
    contains more information than can be stored in the wannier90.dmn file
    Now the wannierisation in wannier-berri is NOT compatible with the format of the wannier90.dmn file,
    Thus the name "dmn" is kept for historical reasons, but the class is not compatible with the wannier90.dmn file
    """

    def __init__(self, NK=1):
        self.npz_tags = ['D_wann_block_indices', '_NB',
                    'kpt2kptirr', 'kptirr', 'kptirr2kpt', 'kpt2kptirr_sym',
                   '_NK', 'num_wann', 'comment', 'NKirr', 'Nsym', 'time_reversals',]
        self.npz_tags_optional = ["eig_irr", "kpoints_all"]
        self.default_tags = {}
        self._NB = 0
        self.num_wann = 0
        self.D_wann_block_indices = np.zeros((0, 2), dtype=int)
        self.NKirr = 0
        self.Nsym = 0
        self.kpoints_all = np.zeros((0, 3), dtype=float)
        self.kpt2kptirr = np.zeros(0, dtype=int)
        self.kptirr = np.zeros(0, dtype=int)
        self.kptirr2kpt = np.zeros((0, 0), dtype=int)


    def from_irrep(self, bandstructure: BandStructure,
                 grid=None, degen_thresh=1e-2, store_eig=True):
        """
        Initialize the object from the BandStructure object

        Parameters
        ----------
        bandstructure : irrep.bandstructure.BandStructure
            the object containing the band structure
        grid : tuple(int), optional
            the grid of kpoints (3 integers), if None, the grid is determined from the kpoints
            may be used to reduce the grid (by an integer factor) for the symmetry analysis
        degen_thresh : float, optional
            the threshold for the degeneracy of the bands. Only transformations between bands
             with energy difference smaller than this value are considered
        """
        data = bandstructure.get_dmn(grid=grid, degen_thresh=degen_thresh, unitary=True)
        self.grid = data["grid"]
        self.kpoints_all = data["kpoints"]
        self.kpt2kptirr = data["kpt2kptirr"]
        self.kptirr = data["kptirr"]
        self.kptirr2kpt = data["kptirr2kpt"]
        self.d_band_blocks = data["d_band_blocks"]
        self.d_band_block_indices = data["d_band_block_indices"]

        self.comment = "Generated by wannierberri with irrep"
        self.D_wann = []
        self.spacegroup = bandstructure.spacegroup
        self.Nsym = bandstructure.spacegroup.size
        self.time_reversals = np.array([symop.time_reversal for symop in self.spacegroup.symmetries])
        self.NKirr = len(self.kptirr)
        self._NK = len(self.kpoints_all)
        self._NB = bandstructure.num_bands
        self.clear_inverse()
        if store_eig:
            self.set_eig([bandstructure.kpoints[ik].Energy_raw for ik in self.kptirr])
        return self

    @cached_property
    def orbital_rotator(self):
        return OrbitalRotator([symop.rotation_cart for symop in self.spacegroup.symmetries])


    def set_D_wann_from_projections(self,
                                    projections=None,
                                    projections_obj=None,
                                    kpoints=None,
                                    ):
        """
        Parameters
        ----------
        projections : list( (np.array(float, shape=(3,)), str) )
            the list of projections. Each projection is a tuple of the position and the orbital name. e.g [(np.array([0, 0, 0]), "s"), (np.array([0, 0.5, 0.5]), "p")]
        projections_obj : ProjectionsSet or list(Projection)
            alternative way to provide the projections. Will be appended to the projections list
        kpoints : np.array(float, shape=(npoints,3,))
            the kpoints in fractional coordinates (neede only if the kpoints are not stored in the object yet) 
        """
        if projections is None:
            projections = []
        if not hasattr(self, "kpoints_all") or self.kpoints_all is None:
            if kpoints is None:
                warnings.warn("kpoints are not provided, neither stored in the object. Assuming Gamma point only")
                kpoints = np.array([[0, 0, 0]])
            self.kpoints_all = kpoints
            self._NK = len(kpoints)

        if projections_obj is not None:
            if isinstance(projections_obj, ProjectionsSet):
                projections_obj = projections_obj.projections
            for proj in projections_obj:
                orbitals = proj.orbitals
                print(f"orbitals = {orbitals}")
                if len(orbitals) > 1:
                    warnings.warn(f"projection {proj} has more than one orbital. it will be split into separate blocks, please order them in the win file consistently")
                for orb in orbitals:
                    projections.append((proj.positions, orb))

        D_wann_list = []
        self.T_list = []
        self.atommap_list = []
        self.rot_orb_list = []
        for positions, proj in projections:
            print(f"calculating Wannier functions for {proj} at {positions}")
            _Dwann = Dwann(spacegroup=self.spacegroup, positions=positions, orbital=proj, orbital_rotator=self.orbital_rotator, spinor=self.spacegroup.spinor)
            _dwann = _Dwann.get_on_points_all(kpoints=self.kpoints_all, ikptirr=self.kptirr, ikptirr2kpt=self.kptirr2kpt)
            D_wann_list.append(_dwann)
            self.T_list.append(_Dwann.T)
            self.atommap_list.append(_Dwann.atommap)
            self.rot_orb_list.append(_Dwann.rot_orb)
        print(f"len(D_wann_list) = {len(D_wann_list)}")
        self.set_D_wann(D_wann_list)
        return self

    @cached_property
    def rot_orb_dagger_list(self):
        return [rot_orb.swapaxes(1, 2).conj()
            for rot_orb in self.rot_orb_list]


    def symmetrize_smth(self, wannier_property):
        ncart = (wannier_property.ndim - 1)
        if ncart == 0:
            wcc_red_in = wannier_property
        elif ncart == 1:
            wcc_red_in = wannier_property @ self.spacegroup.lattice_inv
        else:
            raise ValueError("The input should be either a vector or a matrix")
        WCC_red_out = np.zeros((self.num_wann,) + (3,) * ncart, dtype=float)
        for isym, symop in enumerate(self.spacegroup.symmetries):
            for block, (ws, _) in enumerate(self.D_wann_block_indices):
                norb = self.rot_orb_list[block][0].shape[0]
                T = self.T_list[block][:, isym]
                num_points = T.shape[0]
                atom_map = self.atommap_list[block][:, isym]
                for atom_a in range(num_points):
                    start_a = ws + atom_a * norb
                    atom_b = atom_map[atom_a]
                    start_b = ws + atom_b * norb
                    XX_L = wcc_red_in[start_a:start_a + norb]
                    if ncart > 0:
                        XX_L = symop.transform_r(XX_L) + T[atom_a]
                    # XX_L = symop.transform_r(wcc_red_in[start_a:start_a + norb]) + T[atom_a]
                    # NOTE : I do not fully understand why the transpose are needed here but it works TODO  : check
                    transformed = np.einsum("ij,j...,ji->i...", self.rot_orb_dagger_list[block][isym].T, XX_L, self.rot_orb_list[block][isym].T).real
                    WCC_red_out[start_b:start_b + norb] += transformed
        if ncart > 0:
            WCC_red_out = WCC_red_out @ self.spacegroup.lattice
        return WCC_red_out / self.spacegroup.size

    def set_eig(self, eig):
        eig = np.array(eig, dtype=float)
        assert eig.ndim == 2
        assert eig.shape[1] == self.NB
        if eig.shape[0] == self.NK:
            self.eig_irr = eig[self.kptirr]
        elif eig.shape[0] == self.NKirr:
            self.eig_irr = eig
        else:
            raise ValueError(f"The shape of eig should be either ({self.NK}, {self.NB}) or ({self.NKirr}, {self.NB}), not {eig.shape}")

    def symmetrize_WCC(self, wannier_centers_cart):
        return self.symmetrize_smth(wannier_centers_cart)

    def symmetrize_spreads(self, wannier_spreads):
        return self.symmetrize_smth(wannier_spreads)

    def set_spacegroup(self, spacegroup):
        self.spacegroup = spacegroup
        self.time_reversals = np.array([symop.time_reversal for symop in self.spacegroup.symmetries])
        self.Nsym = spacegroup.size
        return self

    def as_dict(self):
        dic = super().as_dict()
        for k, val in self.spacegroup.as_dict().items():
            dic["spacegroup_" + k] = val
        for attrname in ["T", "atommap", "rot_orb"]:
            if hasattr(self, attrname + "_list"):
                for i, t in enumerate(self.__getattribute__(attrname + "_list")):
                    dic[f'{attrname}_{i}'] = t
        return dic


    def from_dict(self, dic):
        t0 = time()
        super().from_dict(dic)
        t1 = time()
        prefix = "spacegroup_"
        l = len(prefix)
        dic_spacegroup = {k[l:]: v for k, v in dic.items() if k.startswith(prefix)}
        if len(dic_spacegroup) > 0:
            self.spacegroup = SpaceGroupBare(**dic_spacegroup)
        t2 = time()
        for prefix in ["T", "atommap", "rot_orb"]:
            keys = sorted([k for k in dic.keys() if k.startswith(prefix)])
            lst = [dic[k] for k in keys]
            self.__setattr__(prefix + "_list", lst)
        t3 = time()
        print(f"time for read_npz dmn {t3 - t0}\n super {t1 - t0} \n spacegroup {t2 - t1}\n  T {t3 - t2} ")
        return self


    @lru_cache
    def ndegen(self, ikirr):
        return len(set(self.kptirr2kpt[ikirr]))


    def U_to_full_BZ(self, U, include_k=None):
        """
        Expands the U matrix from the irreducible to the full BZ

        Parameters
        ----------
        U : list of NKirr np.ndarray(dtype=complex, shape = (nBfree,nWfree,))
            The input matrix to be expanded
        all_k : bool
            If True, the U matrices are expanded at all reducible kpoints (self.include_k is ignored)
            if False, the U matrices are expanded only at the irreducible kpoints and their neighbours,
            for the rest of the kpoints, the U matrices are set to None

        Returns
        -------
        U : list of NK np.ndarray(dtype=complex, shape = (nBfree,nWfree,))
            The expanded matrix. if all_k is False, the U matrices at the kpoints not included in self.include_k are set to None
        """
        all_k = include_k is None
        Ufull = [None for _ in range(self.NK)]
        for ikirr in range(self.NKirr):
            for isym in range(self.Nsym):
                iRk = self.kptirr2kpt[ikirr, isym]
                if Ufull[iRk] is None and (all_k or include_k[iRk]):
                    Ufull[iRk] = self.rotate_U(U[ikirr], ikirr, isym, forward=True)
        return Ufull

    def get_symmetrizer_Uirr(self, ikirr):
        return Symmetrizer_Uirr(self, ikirr)

    def get_symmetrizer_Zirr(self, ikirr, free=None):
        if free is None:
            free = np.ones(self.NB, dtype=bool)
        return Symmetrizer_Zirr(self, ikirr, free=free)


class Symmetrizer_Uirr(SymmetrizerSAWF):

    def __init__(self, dmn, ikirr):
        self.ikirr = ikirr
        self.isym_little = dmn.isym_little[ikirr]
        self.nsym_little = len(self.isym_little)
        self.ikpt = dmn.kptirr[ikirr]
        self.d_indices = dmn.d_band_block_indices[ikirr]
        self.D_indices = dmn.D_wann_block_indices
        self.d_band_blocks = dmn.d_band_blocks[ikirr]
        self.D_wann_blocks_inverse = dmn.D_wann_blocks_inverse[ikirr]
        self.nb = dmn.NB
        self.num_wann = dmn.num_wann
        self.time_reversals = dmn.time_reversals



    def rotate_U(self, U, isym):
        # forward = not forward
        Uloc = U.copy()
        if self.time_reversals[isym]:
            Uloc = Uloc.conj()
        Uloc = rotate_block_matrix(Uloc,
                                   lblocks=self.d_band_blocks[isym],
                                   lindices=self.d_indices,
                                   rblocks=self.D_wann_blocks_inverse[isym],
                                   rindices=self.D_indices)
        return Uloc



    def __call__(self, U):
        Usym = sum(self.rotate_U(U, isym) for isym in self.isym_little) / self.nsym_little
        return orthogonalize(Usym)


class Symmetrizer_Zirr(SymmetrizerSAWF):

    def __init__(self, dmn, ikirr, free):
        self.ikirr = ikirr
        self.isym_little = dmn.isym_little[ikirr]
        self.nsym_little = len(self.isym_little)
        self.ikpt = dmn.kptirr[ikirr]
        self.nb = dmn.NB
        self.num_wann = dmn.num_wann
        self.time_reversals = dmn.time_reversals

        if free is not None:
            (
                d_band_block_indices_free,
                d_band_blocks_free
            ) = dmn.select_window(dmn.d_band_blocks[ikirr], dmn.d_band_block_indices[ikirr], free)
            d_band_blocks_free_inverse = get_inverse_block(d_band_blocks_free)

            self.lblocks = d_band_blocks_free_inverse
            self.rblocks = d_band_blocks_free
            self.indices = d_band_block_indices_free
        else:
            self.lblocks = dmn.d_band_blocks_inverse,
            self.rblocks = dmn.d_band_blocks
            self.indices = dmn.d_band_block_indices


    def __call__(self, Z):
        # return Z # temporary for testing
        if Z.shape[0] == 0:
            return Z
        else:
            Z_rotated = [self.rotate_Z(Z, isym) for isym in self.isym_little]
            Z[:] = sum(Z_rotated) / self.nsym_little
            return Z

    def rotate_Z(self, Z, isym):
        """
        Rotates the zmat matrix at the irreducible kpoint
        Z = d_band^+ @ Z @ d_band
        """
        Zloc = Z.copy()
        # if self.time_reversals[isym]:
        #     Zloc = Zloc.conj()
        Zloc = rotate_block_matrix(Zloc, lblocks=self.lblocks[isym],
                                 lindices=self.indices,
                                 rblocks=self.rblocks[isym],
                                 rindices=self.indices,
                                )
        if self.time_reversals[isym]:
            Zloc = Zloc.conj()

        return Zloc


class VoidSymmetrizer(SymmetrizerSAWF):

    """
    A fake symmetrizer that does nothing
    Just to be able to use the same with and without site-symmetry
    """

    def __init__(self, *args, NK=1, **kwargs):
        self.NKirr = NK
        self._NK = NK
        self.kptirr = np.arange(NK)
        self.kptirr2kpt = self.kptirr[:, None]
        self.kpt2kptirr = np.arange(NK)
        self.Nsym = 1

    def symmetrize_U_kirr(self, U, ikirr):
        return np.copy(U)

    def symmetrize_Z(self, Z):
        return np.copy(Z)

    def symmetrize_Zk(self, Z, ikirr):
        return np.copy(Z)

    def U_to_full_BZ(self, U, include_k=None):
        return np.copy(U)

    def __call__(self, X):
        return np.copy(X)

    def get_symmetrizer_Uirr(self, ikirr):
        return VoidSymmetrizer()

    def get_symmetrizer_Zirr(self, ikirr, free=None):
        return VoidSymmetrizer()

    def symmetrize_smth(self, wannier_property):
        return wannier_property
