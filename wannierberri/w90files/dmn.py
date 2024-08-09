from functools import cached_property, lru_cache
import warnings
from numba import njit  
from irrep.bandstructure import BandStructure
from irrep.gvectors import symm_matrix
from irrep.utility import get_block_indices
import numpy as np
from copy import deepcopy

from ..__utility import UniqueListMod1
from ..wannierise.projections import ProjectionsSet

from .utility import grid_from_kpoints, writeints, readints
from .w90file import W90_file
from .amn import AMN
from .Dwann import Dwann


class DMN(W90_file):
    """
    Class to read and store the wannier90.dmn file
    the symmetry transformation of the Wannier functions and ab initio bands

    Parameters
    ----------
    seedname : str
        the prefix of the file (including relative/absolute path, but not including the extensions, like `.dmn`)
        if None, the object is initialized with void values (zeroes)
    num_wann : int
        the number of Wannier functions (in the case of void initialization)
    num_bands : int
        the number of ab initio bands (in the case of void initialization)
    nkpt : int
        the number of kpoints (in the case of void initialization)

    Attributes
    ----------
    comment : str
        the comment at the beginning of the file
    NB : int
        the number of ab initio bands
    Nsym : int
        the number of symmetries
    NKirr : int
        the number of irreducible kpoints
    NK : int
        the number of kpoints
    num_wann : int
        the number of Wannier functions
    kptirr : numpy.ndarray(int, shape=(NKirr,))
        the list of irreducible kpoints
    kpt2kptirr : numpy.ndarray(int, shape=(NK,))
        the mapping from kpoints to irreducible kpoints (each number denotes the index of the irreducible kpoint in kptirr)
    kptirr2kpt : numpy.ndarray(int, shape=(NKirr, Nsym))
        the mapping from irreducible kpoints to all kpoints 
    kpt2kptirr_sym : numpy.ndarray(int, shape=(NK,))    
        the symmetry that brings the irreducible kpoint from self.kpt2kptirr into the reducible kpoint in question
    d_band_blocks : list(list(numpy.ndarray(complex, shape=(NB, NB))))
        the ab initio band transformation matrices in the block form (between almost degenerate bands)
    d_band_block_indices : list(np.array(int, shape=(Nblocks, 2)))
        the indices of the blocks in the ab initio band transformation matrices
    D_wann_blocks : list(list(numpy.ndarray(complex, shape=(num_wann, num_wann))))
        the Wannier function transformation matrices in the block form. Within the same irreducible representation of WFs
    D_wann_block_indices : np.array(int, shape=(Nblocks, 2))
        the indices of the blocks in the Wannier function transformation matrices
    """

    def __init__(self, seedname="wannier90", num_wann=None, num_bands=None, nkpt=None,
                 empty=False,
                 **kwargs):
        self.npz_tags = [ 'D_wann_block_indices', '_NB',
                    'kpt2kptirr', 'kptirr', 'kptirr2kpt', 'kpt2kptirr_sym',
                   '_NK', 'num_wann', 'comment', 'NKirr', 'Nsym',]
        
        if empty:
            self._NB = 0
            self.num_wann =0 
            return
        if seedname is None:
            self.set_identiy(num_wann, num_bands, nkpt)
            return

        super().__init__(seedname, "dmn", **kwargs)

    def to_npz(self, f_npz):
        dic = {k: self.__getattribute__(k) for k in self.npz_tags}
        for ik in range(self.NKirr):
            dic[f'd_band_block_indices_{ik}'] = self.d_band_block_indices[ik]
            for isym in range(self.Nsym):
                for i in range(len(self.d_band_block_indices[ik])):
                    dic[f'd_band_blocks_{ik}_{isym}_{i}'] = self.d_band_blocks[ik][isym][i]
                for i in range(len(self.D_wann_block_indices)):
                    dic[f'D_wann_blocks_{ik}_{isym}_{i}'] = self.D_wann_blocks[ik][isym][i]
        print (f"saving to {f_npz} : ")
        for k in dic:
            try:
                print (f"{k} : {dic[k].shape}")
            except:
                print (f"{k} : {dic[k]}")	
        np.savez_compressed(f_npz, **dic)

    def from_npz(self, f_npz):
        dic = np.load(f_npz)
        for k in self.npz_tags:
            self.__setattr__(k, dic[k])
        self.d_band_block_indices = [[] for _ in range(self.NKirr)]
        self.d_band_blocks = [[[] for s in range(self.Nsym)] for ik in range(self.NKirr)]
        self.D_wann_blocks = [[[] for s in range(self.Nsym)] for ik in range(self.NKirr)]
        for ik in range(self.NKirr):
            self.d_band_block_indices[ik] = dic[f'd_band_block_indices_{ik}']
            for isym in range(self.Nsym):
                for i in range(len(self.d_band_block_indices[ik])):
                    self.d_band_blocks[ik][isym].append(np.ascontiguousarray(dic[f'd_band_blocks_{ik}_{isym}_{i}']))
                for i in range(len(self.D_wann_block_indices)): 
                    self.D_wann_blocks[ik][isym].append(np.ascontiguousarray(dic[f'D_wann_blocks_{ik}_{isym}_{i}']))




    @property
    def NK(self):
        return self._NK

    @property
    def NB(self):
        if hasattr(self, '_NB'):
            return self._NB
        else:
            return 0

    @cached_property
    def isym_little(self):
        return [np.where(self.kptirr2kpt[ik] == self.kptirr[ik])[0] for ik in range(self.NKirr)]

    @cached_property
    def kpt2kptirr_sym(self):
        return np.array([np.where(self.kptirr2kpt[self.kpt2kptirr[ik], :] == ik)[0][0] for ik in range(self.NK)])

    def from_w90_file(self, seedname="wannier90", num_wann=0, eigenvalues=None):
        """
        eigenvalues np.array(shape=(NK,NB))
        The eigenvalues used to determine the degeneracy of the bandsm and the corresponding blocks
        of matrices which are non-zero
        """
        fl = open(seedname + ".dmn", "r")
        self.comment = fl.readline().strip()
        self._NB, self.Nsym, self.NKirr, self._NK = readints(fl, 4)
        self.kpt2kptirr = readints(fl, self.NK) - 1
        self.kptirr = readints(fl, self.NKirr) - 1
        self.kptirr2kpt = np.array([readints(fl, self.Nsym) for _ in range(self.NKirr)]) - 1
        assert np.all(self.kptirr2kpt.flatten() >= 0), "kptirr2kpt has negative values"
        assert np.all(self.kptirr2kpt.flatten() < self.NK), "kptirr2kpt has values larger than NK"
        assert (set(self.kptirr2kpt.flatten()) == set(range(self.NK))), "kptirr2kpt does not cover all kpoints"
        # print(self.kptirr2kpt.shape)
        # find an symmetry that brings the irreducible kpoint from self.kpt2kptirr into the reducible kpoint in question


        # read the rest of lines and convert to conplex array
        data = [l.strip("() \n").split(",") for l in fl.readlines()]
        data = np.array([x for x in data if len(x) == 2], dtype=float)
        data = data[:, 0] + 1j * data[:, 1]
        print("number of numbers in the dmn file :", data.shape)
        print("of those > 1e-8 :", np.sum(np.abs(data) > 1e-8))
        print("of those > 1e-5 :", np.sum(np.abs(data) > 1e-5))
        
        num_wann = np.sqrt(data.shape[0] // self.Nsym // self.NKirr - self.NB**2)
        assert abs(num_wann - int(num_wann)) < 1e-8, f"num_wann is not an integer : {num_wann}"
        self.num_wann = int(num_wann)
        assert data.shape[0] == (self.num_wann**2 + self.NB**2) * self.Nsym * self.NKirr, \
            f"wrong number of elements in dmn file found {data.shape[0]} expected {(self.num_wann**2 + self.NB**2) * self.Nsym * self.NKirr}"
        n1 = self.num_wann**2 * self.Nsym * self.NKirr
        # in fortran the order of indices is reversed. therefor transpose
        D_wann = data[:n1].reshape(self.NKirr, self.Nsym, self.num_wann, self.num_wann
                                          ).transpose(0, 1, 3, 2)
        d_band = data[n1:].reshape(self.NKirr, self.Nsym, self.NB, self.NB).transpose(0, 1, 3, 2)

        # arranging d_band in the block form
        if eigenvalues is not None:
            print ("DMN: eigenvalues are used to determine the block structure")
            self.d_band_block_indices = [get_block_indices(eigenvalues[ik], thresh=1e-2, cyclic=False) for ik in self.kptirr]
        else:
            print ("DMN: eigenvalues are NOT provided, the bands are considered as one block")
            self.d_band_block_indices = [ [(0, self.NB)] for _ in range(self.NKirr)]
        self.d_band_block_indices = [np.array(self.d_band_block_indices[ik]) for ik in range(self.NKirr)]
        # np.ascontinousarray is used to speedup with Numba
        self.d_band_blocks = [[[np.ascontiguousarray(d_band[ik, isym, start:end, start:end]) 
                                for start, end in self.d_band_block_indices[ik] ] 
                                for isym in range(self.Nsym)] for ik in range(self.NKirr)]

        # arranging D_wann in the block form
        self.wann_block_indices=[]
        #determine the block indices from the D_wann, excluding areas with only zeros
        start = 0
        thresh = 1e-5
        while start < self.num_wann:
            for end in range(start+1, self.num_wann):
                if np.all( abs(D_wann[:,:,start:end,end:])<thresh) and np.all( abs(D_wann[:,:,end:,start:end])<thresh):
                    self.wann_block_indices.append((start, end))
                    start = end
                    break
            else:
                self.wann_block_indices.append((start, self.num_wann))
                break
        # arange blocks
        self.D_wann_block_indices = np.array(self.wann_block_indices)
        # np.ascontinousarray is used to speedup with Numba
        self.D_wann_blocks = [[[np.ascontiguousarray(D_wann[ik, isym, start:end, start:end]) for start, end in self.D_wann_block_indices]
                                for isym in range(self.Nsym)] for ik in range(self.NKirr)]
        
    @lru_cache
    def d_band_diagonal(self,ikirr,isym):
        if ikirr is None:
            return np.array([self.d_band_diagonal(ikirr, isym) for ikirr in range(self.NKirr)])
        if isym is None:
            return np.array([self.d_band_diagonal(ikirr, isym) for isym in range(self.Nsym)])
        
        result = np.zeros(self.NB, dtype=complex)
        for (start, end),block in zip(self.d_band_block_indices[ikirr], self.d_band_blocks[ikirr][isym]):
            result[start:end] = block.diagonal()
        return result

    def d_band_full_matrix(self, ikirr=None, isym=None):
        """
        Returns the full matrix of the ab initio bands transformation matrix
        """
        if ikirr is None:
            return np.array([self.d_band_full_matrix(ikirr, isym) for ikirr in range(self.NKirr)])
        if isym is None:
            return np.array([self.d_band_full_matrix(ikirr, isym) for isym in range(self.Nsym)])
        
        result = np.zeros((self.NB, self.NB), dtype=complex)
        for (start, end),block in zip(self.d_band_block_indices[ikirr], self.d_band_blocks[ikirr][isym]):
            result[start:end, start:end] = block
        return result
    
    def D_wann_full_matrix(self, ikirr=None, isym=None):
        """
        Returns the full matrix of the Wannier function transformation matrix
        """ 
        if ikirr is None:
            return np.array([self.D_wann_full_matrix(ikirr, isym) for ikirr in range(self.NKirr)])
        if isym is None:
            return np.array([self.D_wann_full_matrix(ikirr, isym) for isym in range(self.Nsym)])
        
        result = np.zeros((self.num_wann, self.num_wann), dtype=complex)
        for (start, end), block in zip(self.D_wann_block_indices, self.D_wann_blocks[ikirr][isym]):
            result[start:end, start:end] = block
        return result
        

    def to_w90_file(self, seedname):
        f = open(seedname + ".dmn", "w")
        print(f"writing {seedname}.dmn:  comment = {self.comment}")
        f.write(f"{self.comment}\n")
        f.write(f"{self.NB} {self.Nsym} {self.NKirr} {self.NK}\n\n")
        f.write(writeints(self.kpt2kptirr + 1) + "\n")
        f.write(writeints(self.kptirr + 1) + "\n")
        for i in range(self.NKirr):
            f.write(writeints(self.kptirr2kpt[i] + 1) + "\n")
            # " ".join(str(x + 1) for x in self.kptirr2kpt[i]) + "\n")
        # f.write("\n".join(" ".join(str(x + 1) for x in l) for l in self.kptirr2kpt) + "\n")
        mat_fun_list = []
        if self.num_wann>0:
            mat_fun_list.append(self.D_wann_full_matrix)
        if self.NB>0:
            mat_fun_list.append(self.d_band_full_matrix)

        for M in mat_fun_list:
            for ik in range(self.NKirr):
                for isym in range(self.Nsym):
                    f.write("\n".join("({:17.12e},{:17.12e})".format(x.real, x.imag) for x in M(ik,isym).flatten(order='F')) + "\n\n")

    def from_irrep(self, bandstructure: BandStructure,
                 grid=None, degen_thresh=1e-2):
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

        self.comment = "Generated by wannierberri with irrep"
        self.D_wann = []
        self.spacegroup = bandstructure.spacegroup
        kpoints = np.array([KP.K  for KP in bandstructure.kpoints])
        self.grid, self.selected_kpoints = grid_from_kpoints(kpoints, grid=grid)
        self.kpoints = kpoints[self.selected_kpoints]
        self._NK = len(self.kpoints)
        self.Nsym = bandstructure.spacegroup.size
        def get_K(ik):
            return bandstructure.kpoints[self.selected_kpoints[ik]]
    
        # First determine which kpoints are irreducible
        # and set mapping from irreducible to full grid and back
        #
        # kptirr2kpt[i,isym]=j means that the i-th irreducible kpoint
        # is transformed to the j-th kpoint of full grid  
        #  by the isym-th symmetry operation.
        #
        # This is consistent with w90 documentations, but seemd to be opposite to what pw2wannier90 does
         
        kpoints_mod1 = UniqueListMod1(self.kpoints)
        assert len(kpoints_mod1) == len(self.kpoints)
        is_irreducible = np.ones(self.NK, dtype=bool)
        self.kptirr = []
        self.kptirr2kpt = []
        self.kpt2kptirr = -np.ones(self.NK, dtype=int)
        self.G = []
        ikirr = -1
        for i,k1 in enumerate(self.kpoints):
            if is_irreducible[i]:
                self.kptirr.append(i)
                self.kptirr2kpt.append(np.zeros(self.Nsym, dtype=int))
                self.G.append(np.zeros((self.Nsym, 3), dtype=int))
                ikirr +=1
                
                for isym, symop in enumerate(bandstructure.spacegroup.symmetries):
                    k1p = symop.transform_k(k1)
                    if k1p not in kpoints_mod1:
                        raise RuntimeError("Symmetry operation maps k-point outside the grid. Maybe the grid is incompatible with the symmetry operations")
                    j = kpoints_mod1.index(k1p)
                    k2= self.kpoints[j]
                    if j!=i:
                        is_irreducible[j] = False
                    self.kptirr2kpt[ikirr][isym] = j
                    # the G vectors mean that 
                    # symop.transform(ki) = kj + G
                    self.G[ikirr][isym] = k1p - k2
                    if self.kpt2kptirr[j] == -1:
                        self.kpt2kptirr[j] = ikirr
                    else:
                        assert self.kpt2kptirr[j] == ikirr, (f"two different irreducible kpoints {ikirr} and {self.kpt2kptirr[j]} are mapped to the same kpoint {j}"
                                                             f"kptirr= {self.kptirr}, \nkpt2kptirr= {self.kpt2kptirr}\n kptirr2kpt= {self.kptirr2kpt}")
        self.kptirr = np.array(self.kptirr)
        self.NKirr = len(self.kptirr)
        self.kptirr2kpt = np.array(self.kptirr2kpt)
        self.G = np.array(self.G)
        del kpoints_mod1

        assert np.all(self.kptirr2kpt>=0)
        assert np.all(self.kpt2kptirr>=0)

        self._NB = bandstructure.num_bands
        self.d_band_blocks = [[[] for _ in range(self.Nsym)] for _ in range(self.NKirr)]
        self.d_band_block_indices = []
        for i, ikirr in enumerate(self.kptirr):
            K1 = get_K(ikirr)
            block_indices = get_block_indices(K1.Energy_raw, thresh=degen_thresh, cyclic=False)
            self.d_band_block_indices.append(block_indices)
            for isym, symop in enumerate(bandstructure.spacegroup.symmetries):
                K2 = get_K(self.kptirr2kpt[i,isym])
                block_list = symm_matrix(
                                        K=K1.k,
                                        K_other=K2.k,
                                        WF=K1.WF,
                                        WF_other=K2.WF,
                                        igall=K1.ig,
                                        igall_other=K2.ig,
                                        A=symop.rotation,
                                        S=symop.spinor_rotation,
                                        T=symop.translation,
                                        spinor = K1.spinor,
                                        block_ind=block_indices,
                                        return_blocks=True
                                        )
                self.d_band_blocks[i][isym] = [np.ascontiguousarray(b.T) for b in block_list] 
                #transposed because in irrep WF is row vector, while in dmn it is column vector
            # check the symmetry of the transformation matrices
            if hasattr(K1,'char'):
                E = K1.Energy_raw
                borders = [0]+ (np.where(E[1:]- E[:-1]>1e-4)[0]+1).tolist() + [len(E)]
                characters = []
                for isym in range(self.Nsym):
                    if self.kptirr2kpt[i, isym] == ikirr:
                        characters.append(self.d_band[i, isym].diagonal())
                characters = np.array(characters).T
                characters = np.array([characters[b1:b2].sum(axis=0) for b1, b2 in zip(borders[:-1], borders[1:])])
                if np.max(np.abs(characters - K1.char)) > 1e-3:
                    print (f"characters = {characters}")
                    raise RuntimeError(f"characters do not match for irreducible kpoint {ikirr}, from dmn {characters} vs from irrep {K1.char}")
                # print (f"characters = {characters}")
                    

    def get_disentangled(self, v_matrix_dagger, v_matrix):
        NBnew = v_matrix.shape[2]
        d_band_new = np.zeros((self.NKirr, self.Nsym, NBnew, NBnew), dtype=complex)
        for ikirr, ik in enumerate(self.kptirr):
            for isym in range(self.Nsym):
                ik2 = self.kptirr2kpt[ikirr, isym]
                d_band_new[ikirr, isym] = v_matrix_dagger[ik2] @ self.d_band[ikirr, isym] @ v_matrix[ik]
        other = deepcopy(self)
        other.d_band = d_band_new
        return other
    
    def set_identiy(self, num_wann, num_bands, nkpt):
        """
        set the object to contain only the  transformation matrices
        """
        self.comment = "only identity"
        self._NB, self.Nsym, self.NKirr, self.NK = num_bands, 1, nkpt, nkpt
        self.num_wann = num_wann
        self.kpt2kptirr = np.arange(self.NK)
        self.kptirr = self.kpt2kptirr
        self.kptirr2kpt = np.array([self.kptirr, self.Nsym])
        self.kpt2kptirr_sym = np.zeros(self.NK, dtype=int)
        self.d_band_block_indices = [np.array([(i,i+1) for i in range (self.NB) ])]*self.NKirr
        self.D_wann_block_indices = np.array([(i,i+1) for i in range (self.num_wann) ])
        self.d_band_blocks = [[[np.eye(end-start) for start,end in self.d_band_block_indicespik]
                              for isym in range(self.Nsym)] for ik in range(self.NKirr)]
        self.D_wann_blocks = [[[np.eye(end-start) for start,end in self.D_wann_block_indices]
                                for isym in range(self.Nsym)] for ik in range(self.NKirr)]
                              
    def select_bands(self, win_index_irr):
        self.d_band = [D[:, wi, :][:, :, wi] for D, wi in zip(self.d_band, win_index_irr)]

    def set_free(self, frozen_irr):
        free = np.logical_not(frozen_irr)
        self.d_band_free = [d[:, f, :][:, :, f] for d, f in zip(self.d_band, free)]

    def write(self):
        print(self.comment)
        print(self.NB, self.Nsym, self.NKirr, self.NK, self.num_wann)
        for i in range(self.NKirr):
            for j in range(self.Nsym):
                print()
                for M in self.D_band[i][j], self.d_wann[i][j]:
                    print("\n".join(" ".join(("X" if abs(x)**2 > 0.1 else ".") for x in m) for m in M) + "\n")

    def rotate_U(self, U, ikirr, isym, forward=True):
        """
        Rotates the umat matrix at the irreducible kpoint
        U = D_band^+ @ U @ D_wann
        """
        d_blocks = self.d_band_blocks[ikirr][isym]
        D_blocks = self.D_wann_blocks[ikirr][isym]
        d_indices = self.d_band_block_indices[ikirr]
        D_indices = self.D_wann_block_indices
        # forward = not forward
        U1 = np.zeros(U.shape, dtype=complex)
        if forward:
            return rotate_block_matrix(U, lblocks=d_blocks, lindices=d_indices,
                                        rblocks=D_blocks, rindices=D_indices,
                                        conj_left=False, conj_right=True,
                                        result=U1)
            # return d @ U @ D.conj().T
        else:
            return rotate_block_matrix(U, lblocks=d_blocks, lindices=d_indices,
                                        rblocks=D_blocks, rindices=D_indices,
                                        conj_left=True, conj_right=False,
                                        result=U1)
            # return d.conj().T @ U @ D

        # if forward:
        #     return self.d_band[ikirr, isym].conj().T @ U @ self.D_wann[ikirr, isym]
        # else:
        #     return self.d_band[ikirr, isym] @ U @ self.D_wann[ikirr, isym].conj().T

    def clear_free_bands(self):
        if hasattr(self, 'free_bands_defined'):
            del self.free_bands_defined
            del self.d_band_blocks_free
            del self.d_band_block_indices_free
        
    def set_free_bands(self, ikirr, free_bands):
        assert free_bands is not None
        if not hasattr(self, 'free_bands_defined'):
            self.free_bands_defined = np.zeros(self.NK, dtype=bool)
            self.d_band_blocks_free = [None for _ in range(self.NKirr)]
            self.d_band_block_indices_free = [None for _ in range(self.NKirr)]
        if not self.free_bands_defined[ikirr]:
            self.free_bands_defined[ikirr] = True
            (
                self.d_band_block_indices_free[ikirr],
                self.d_band_blocks_free[ikirr]
            ) = self.select_window(self.d_band_blocks[ikirr], self.d_band_block_indices[ikirr], free_bands)
            

    def rotate_Z(self, Z, isym, ikirr, free=None):
        """
        Rotates the zmat matrix at the irreducible kpoint
        Z = d_band^+ @ Z @ d_band
        """
        if free is not None:
            self.set_free_bands(ikirr, free)
            blocks = self.d_band_blocks_free[ikirr][isym]
            indices = self.d_band_block_indices_free[ikirr]
        else:
            blocks = self.d_band_blocks[ikirr][isym]
            indices = self.d_band_block_indices[ikirr]

        Z1 = np.zeros(Z.shape, dtype=complex)
        Z1 = rotate_block_matrix(Z, lblocks=blocks, lindices=indices,
                                    rblocks=blocks, rindices=indices,
                                    conj_left=True, conj_right=False,
                                    result=Z1)
        return Z1
        # return d_band.conj().T @ Z @ d_band

    def check_unitary(self):
        """
        Check that the transformation matrices are unitary

        Returns
        -------
        float
            the maximum error for the bands 
        float
            the maximum error for the Wannier functions
        """
        maxerr_band = 0
        maxerr_wann = 0
        for ik in range(self.NK):
            ikirr = self.kpt2kptirr[ik]
            for isym in range(self.Nsym):
                d = self.d_band[ikirr, isym]
                w = self.D_wann[ikirr, isym]
                maxerr_band = max(maxerr_band, np.linalg.norm(d @ d.T.conj() - np.eye(self.NB)))
                maxerr_wann = max(maxerr_wann, np.linalg.norm(w @ w.T.conj() - np.eye(self.num_wann)))
        return maxerr_band, maxerr_wann

    def check_group(self, matrices="wann"):
        """
        check that D_wann is a group

        Parameters
        ----------
        matrices : str
            the type of matrices to be checked, either "wann" or "band"

        Returns
        -------
        float
            the maximum error
        """
        if matrices == "wann":
            check_matrices = self.D_wann
        elif matrices == "band":
            check_matrices = self.d_band
        maxerr = 0
        for ikirr in range(self.NKirr):
            Dw = [check_matrices[ikirr, isym] for isym in self.isym_little[ikirr]]
            print(f'ikirr={ikirr} : {len(Dw)} matrices')
            for i1, d1 in enumerate(Dw):
                for i2, d2 in enumerate(Dw):
                    d = d1 @ d2
                    err = [np.linalg.norm(d - _d)**2 for _d in Dw]
                    j = np.argmin(err)
                    print(f"({i1}) * ({i2}) -> ({j})" + (f"err={err[j]}" if err[j] > 1e-10 else ""))
                    maxerr = max(maxerr, err[j])
        return maxerr

    def apply_window(self, selected_bands):
        if selected_bands is None:
            return
        print (f"applying window to select {sum(selected_bands)} bands from {self.NB}\n",selected_bands)
        for ikirr in range(self.NKirr):
            self.d_band_block_indices[ikirr], self.d_band_blocks[ikirr] = \
                self.select_window(self.d_band_blocks[ikirr], self.d_band_block_indices[ikirr], selected_bands)
        for i,block_ind in enumerate(self.d_band_block_indices):
            if i == 0:
                self._NB = block_ind[-1,-1]
            assert block_ind[0,0] == 0
            assert np.all(block_ind[1:,0]==block_ind[:-1,1])
            assert block_ind[-1,-1] == self.NB
        print (f"new NB = {self.NB}")

        

    def select_window(self, d_band_blocks_ik, d_band_block_indices_ik, selected_bands):
        if selected_bands is None:
            return d_band_blocks_ik, d_band_block_indices_ik
        
        new_block_indices = []
        new_blocks = [[] for _ in range(self.Nsym)]
        st = 0
        for iblock,(start, end) in enumerate(d_band_block_indices_ik):
            select = selected_bands[start:end]
            nsel = np.sum(select)
            if nsel > 0:
                new_block_indices.append((st, st + nsel))
                st += nsel
                for isym in range(self.Nsym):
                    new_blocks[isym].append(
                        np.ascontiguousarray(d_band_blocks_ik[isym][iblock][:, select][select, :]))
        return np.array(new_block_indices), new_blocks
                
    def check_eig(self, eig):
        """
        Check the symmetry of the eigenvlues

        Parameters
        ----------
        eig : EIG object
            the eigenvalues

        Returns
        -------
        float
            the maximum error
        """
        maxerr = 0
        for ik in range(self.NK):
            ikirr = self.kpt2kptirr[ik]
            e1 = eig.data[ik]
            e2 = eig.data[self.kptirr[ikirr]]
            maxerr = max(maxerr, np.linalg.norm(e1 - e2))

        for ikirr in range(self.NKirr):
            for isym in range(self.Nsym):
                e1 = eig.data[self.kptirr[ikirr]]
                e2 = eig.data[self.kptirr2kpt[ikirr, isym]]
                maxerr = max(maxerr, np.linalg.norm(e1 - e2))
        return maxerr

    def check_amn(self, amn, warning_precision=1e-5, ignore_upper_bands=None):
        """
        Check the symmetry of the amn

        Parameters
        ----------
        amn : AMN object of np.ndarray(complex, shape=(NK, NB, NW))
            the amn

        Returns
        -------
        float
            the maximum error
        """
        if isinstance(amn, AMN):
            amn = amn.data
        maxerr = 0
        if ignore_upper_bands is not None:
            assert abs(ignore_upper_bands) < self.num_wann
            ignore_upper_bands = -abs(int(ignore_upper_bands))

        for ikirr in range(self.NKirr):
            for isym in range(self.Nsym):
                ik = self.kptirr2kpt[ikirr, isym]
                a1 = amn[ik]
                a2 = amn[self.kptirr[ikirr]]
                a1p=self.rotate_U(a1, ikirr, isym, forward=False)
                a1=a1[:ignore_upper_bands]
                a1p=a1p[:ignore_upper_bands]
                a2=a2[:ignore_upper_bands]
                diff = a2 - a1p
                diff = np.max(abs(diff))
                maxerr = max(maxerr, np.linalg.norm(diff))
                if diff > warning_precision:
                    print (f"ikirr={ikirr}, isym={isym} : {diff}")
                    for aaa in zip(a1, a1p, a2, a1p-a2,a1p/a2):
                        string = ""
                        for a in aaa:
                            _abs = ", ".join(f"{np.abs(_):.4f}" for _ in a)
                            _angle = ", ".join(f"{np.angle(_)/np.pi*180:7.2f}" for _ in a)   
                            string+= f"[{_abs}] [{_angle}]   |    "
                        print (string)    
        return maxerr

    def symmetrize_amn(self, amn):
        """
        Symmetrize the amn

        Parameters
        ----------
        amn : AMN object of np.ndarray(complex, shape=(NK, NB, NW))
            the amn

        Returns
        -------
        AMN
            the symmetrized amn
        """
        if isinstance(amn, AMN):
            amn = amn.data
        assert amn.shape == (self.NK, self.NB, self.num_wann)

        amn_sym_irr = np.zeros((self.NKirr, self.NB, self.num_wann), dtype=complex)
        for ikirr in range(self.NKirr):
            for isym in range(self.Nsym):
                amn_sym_irr[ikirr] += self.rotate_U(amn[self.kptirr2kpt[ikirr,isym]], ikirr, isym, forward=False)
        amn_sym_irr /= self.Nsym
        lfound = np.zeros(self.NK, dtype=bool)
        amn_sym = np.zeros((self.NK, self.NB, self.num_wann), dtype=complex)
        for ikirr in range(self.NKirr):
            ik = self.kptirr[ikirr]
            amn_sym[ik] = amn_sym_irr[ikirr]
            lfound[ik] = True
            for isym in range(self.Nsym):
                ik = self.kptirr2kpt[ikirr, isym]
                if not lfound[ik]:
                    amn_sym[ik] = self.rotate_U(amn_sym_irr[ikirr], ikirr, isym, forward=True)
                    lfound[ik] = True
        return amn_sym
    
    def get_random_amn(self):
        " generate a random amn file that is comaptible with the symmetries of the Wanier functions in the DMN object"
        shape = (self.NK, self.NB, self.num_wann)
        amn = np.random.random(shape) + 1j * np.random.random(shape)
        return AMN(data=self.symmetrize_amn(amn))

    def set_D_wann(self, D_wann):
        """
        set the D_wann matrix

        Parameters
        ----------
        D_wann : np.array(complex, shape=(NKirr, Nsym, num_wann, num_wann))
            the Wannier function transformation matrix (conjugate transpose
        or list of np.array(complex, shape=(NKirr, Nsym, num_wann, num_wann))
            if it is a list, the matrices are considered to be the diagonal blocks of the D_wann matrix
        Notes
        -----
          also updates the num_wann attribute
        """
        if not isinstance(D_wann, list):
            D_wann = [D_wann]
        print ("D.shape", [D.shape for D in D_wann])
        self.D_wann_block_indices = []
        num_wann = 0
        self.D_wann_blocks = [[[] for s in range(self.Nsym)] for ik in range(self.NKirr)]
        for D in D_wann:
            assert D.shape[0] == self.NKirr
            assert D.shape[1] == self.Nsym
            assert D.shape[2] == D.shape[3]
            self.D_wann_block_indices.append((num_wann, num_wann + D.shape[2]))
            num_wann += D.shape[2]
            for ik in range(self.NKirr):
                for isym in range(self.Nsym):
                    self.D_wann_blocks[ik][isym].append(D[ik, isym])
        self.D_wann_block_indices = np.array(self.D_wann_block_indices)
        self.num_wann = num_wann
        print ("num_wann", num_wann)    
        print ("D_wann_block_indices", self.D_wann_block_indices)
        for ik in range(self.NKirr):
            for isym in range(self.Nsym):
                print ("D_wann_blocks", [b.shape for b in self.D_wann_blocks[ik][isym]])
        
    def set_D_wann_from_projections(self,
                                        spacegroup=None, 
                                        projections=[],
                                        projections_obj=[],
                                        win=None,
                                        kpoints=None,
                                        spinor=False):
        """
        Parameters
        ----------
        spacegroup : SpaceGroup
            the spacegroup of the system
        projections : list( wannierberri.wannierise.projections.Projection)
            the list of projections. Note that if some
        win : WIN object
            the win file, just ot get the k-points
        kpoints : np.array(float, shape=(npoints,3,))
            the kpoints in fractional coordinates. Overrides the kpoints in the win file (if provided)
        projections_obj : ProjectionsSet or list(Projection)
            alternative way to provide the projections

        Note: 
        -----

        """
        from ..system.sym_wann_orbitals import Orbitals
        ORBITALS = Orbitals()
        if spacegroup is None:
            spacegroup = self.spacegroup
        else:
            self.spacegroup = spacegroup
        if kpoints is None:
            if win is not None:
                kpoints = win.data["kpoints"]
        if kpoints is None:
            if hasattr(self, "kpoints"):
                kpoints = self.kpoints
            else:
                raise RuntimeError("kpoints are not provided, neither stored in the object")
        else:
            assert kpoints.shape == (self.NK, 3)
            self.kpoints = kpoints
            self._NK = len(kpoints)
        D_wann_list = []
        if isinstance(projections_obj, ProjectionsSet):
            projections_obj = projections_obj.projections
        for proj in projections_obj:
            orbitals = proj.orbitals
            if len(orbitals) > 1:
                warnings.warn(f"projection {proj} has more than one orbital. it will be split into separate blocks, please order them in the win file consistently")
            for orb in orbitals:
                projections.append((proj.positions, orb))
        for positions, proj in projections:
            _Dwann = Dwann(spacegroup, positions, proj, ORBITALS=ORBITALS, spinor=spinor)
            _dwann = _Dwann.get_on_points_all(kpoints, self.kptirr, self.kptirr2kpt)
            D_wann_list.append(_dwann)
        self.set_D_wann(D_wann_list)



    # def check_mmn(self, mmn, f1, f2):
    #     """
    #     Check the symmetry of data in the mmn file (not working)

    #     Parameters
    #     ----------
    #     mmn : MMN object
    #         the mmn file data

    #     Returns
    #     -------
    #     float
    #         the maximum error
    #     """
    #     assert mmn.NK == self.NK
    #     assert mmn.NB == self.NB

    #     maxerr = 0
    #     neighbours_irr = np.array([self.kpt2kptirr[neigh] for neigh in mmn.neighbours])
    #     for i in range(self.NKirr):
    #         ind1 = np.where(self.kpt2kptirr == i)[0]
    #         kirr1 = self.kptirr[i]
    #         neigh_irr = neighbours_irr[ind1]
    #         for j in range(self.NKirr):
    #             kirr2 = self.kptirr[j]
    #             ind2x, ind2y = np.where(neigh_irr == j)
    #             print(f"rreducible kpoints {kirr1} and {kirr2} are equivalent to {len(ind2x)} points")
    #             ref = None
    #             for x, y in zip(ind2x, ind2y):
    #                 k1 = ind1[x]
    #                 k2 = mmn.neighbours[k1][y]
    #                 isym1 = self.kpt2kptirr_sym[k1]
    #                 isym2 = self.kpt2kptirr_sym[k2]
    #                 d1 = self.d_band[i, isym1]
    #                 d2 = self.d_band[j, isym2]
    #                 assert self.kptirr2kpt[i, isym1] == k1
    #                 assert self.kptirr2kpt[j, isym2] == k2
    #                 assert self.kpt2kptirr[k1] == i
    #                 assert self.kpt2kptirr[k2] == j
    #                 ib = np.where(mmn.neighbours[k1] == k2)[0][0]
    #                 assert mmn.neighbours[k1][ib] == k2
    #                 data = mmn.data[k1, ib]
    #                 data = f1(d1) @ data @ f2(d2)
    #                 if ref is None:
    #                     ref = data
    #                     err = 0
    #                 else:
    #                     err = np.linalg.norm(data - ref)
    #                 print(f"   {k1} -> {k2} : {err}")
    #                 maxerr = max(maxerr, err)
    #     return maxerr


# @njit
def rotate_block_matrix(Z, lblocks, lindices, rblocks, rindices, conj_left, conj_right, result):
    """
    Rotates the matrix Z using the block-diagonal rotation matrices

    Parameters
    ----------
    Z : np.array(complex, shape=(M,N))
        the matrix to be rotated
    lblocks : list(np.array(complex, shape=(m,m)))
        the blocks of hte left matrix. sum(m) = M
    lindices : list(tuple(int))
        the indices of the blocks of the left matrix
    rblocks : list(np.array(complex, shape=(n,n)))
        the blocks of hte right matrix. sum(n) = N
    rindices : list(tuple(int))
        the indices of the blocks of the right matrix
    conj_left : bool
        whether to Hermitean conjugate the left blocks
    conj_right : bool
        whether to Hermitean conjugate the right blocks

    Returns
    -------
    np.array(complex, shape=(M,N))
        the rotated matrix
    """
    if conj_left:
        for (start, end), block in zip(lindices, lblocks):
            result[start:end,:] = block.T.conj() @ Z[start:end, :]
    else:
        for (start, end), block in zip(lindices, lblocks):
            result[start:end,:] = block @ Z[start:end, :]
    
    if conj_right:
        for (start, end), block in zip(rindices, rblocks):
            result[:,start:end] = result[:, start:end] @ block.T.conj()
    else:
        for (start, end), block in zip(rindices, rblocks):
            result[:,start:end] = result[:, start:end] @ block

    return result
        