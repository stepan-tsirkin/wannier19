from copy import deepcopy
import numpy as np

DEGEN_THRESH = 1e-2  # for safety - avoid splitting (almost) degenerate states between free/frozen  inner/outer subspaces  (probably too much)


def disentangle(w90data,
                froz_min=np.Inf,
                froz_max=-np.Inf,
                num_iter=100,
                conv_tol=1e-9,
                num_iter_converge=10,
                mix_ratio=0.5,
                print_progress_every=10
                ):
    r"""
    Performs disentanglement of the bands recorded in w90data, following the procedure described in
    `Souza et al., PRB 2001 <https://doi.org/10.1103/PhysRevB.65.035109>`__
    At the end writes `w90data.chk.v_matrix` and sets `w90data.wannierised = True`

    Parameters
    ----------
    w90data: :class:`~wannierberri.system.Wannier90data`
        the data
    froz_min : float
        lower bound of the frozen window
    froz_max : float
        upper bound of the frozen window
    num_iter : int
        maximal number of iteration for disentanglement
    conv_tol : float
        tolerance for convergence of the spread functional  (in :math:`\mathring{\rm A}^{2}`)
    num_iter_converge : int
        the convergence is achieved when the standard deviation of the spread functional over the `num_iter_converge`
        iterations is less than conv_tol
    mix_ratio : float
        0 <= mix_ratio <=1  - mixing the previous itertions. 1 for max speed, smaller values are more stable
    print_progress_every
        frequency to print the progress

    Returns
    -------
    w90data.chk.v_matrix : numpy.ndarray
    """
    assert 0 < mix_ratio <= 1

                    # frozen_irr=[frozen_nondegen(ik) for ik in self.Dmn.kptirr]
                    # self.frozen=np.array([ frozen_irr[ik] for ik in self.Dmn.kpt2kptirr ])

    # frozen is an 2D boolean array of shape (nk,nb) where True means frozen
 
        # self.Dmn.set_free(frozen_irr)

    free, frozen = get_free_frozen(w90data, froz_min, froz_max)
    num_bands_free = np.array([np.sum(fr) for fr in free]) # number of free bands at each k-point
    num_bands_frozen = np.array([np.sum(fr) for fr in frozen]) # number of free bands at each k-point
    nWfree = np.array([w90data.chk.num_wann - np.sum(frz) for frz in frozen]) # number of free Wannier functions at each k-point

                # irr=self.Dmn.kptirr

                # initial guess : eq 27 of SMV2001
                # U_opt_free_irr=self.get_max_eig(  [ self.Amn[ik][free,:].dot(self.Amn[ik][free,:].T.conj())
                # for ik,free in zip(irr,self.free[irr])]  ,self.nWfree[irr],self.chk.num_bandsfree[irr]) # nBfee x nWfree marrices
                # U_opt_free=self.symmetrize_U_opt(U_opt_free_irr,free=True)

    mmn_list = w90data.mmn.data
    amn_list = w90data.amn.data
    eig_list = w90data.eig.data

    # the initial guess
    U_opt_free = get_max_eig([amn_list[ik][fr, :].dot(amn_list[ik][fr, :].T.conj())
                              for ik, fr in enumerate(free)], nWfree, num_bands_free)  # nBfee x nWfree marrices

    Mmn_FF = MmnFreeFrozen(mmn_list, free, frozen, w90data.mmn.neighbours, w90data.mmn.wk, w90data.chk.num_wann)
    Z_frozen = calc_Z(Mmn_FF('free', 'frozen'))

    # main loop of the disentanglement
    Omega_I_list = []
    Z_old = None
    for i_iter in range(num_iter):
        Zfreefree = calc_Z(Mmn_FF('free', 'free'), w90data, U_opt_free)
        Z = [(z + zfr) for z, zfr in zip(Zfreefree, Z_frozen)]  
        if i_iter > 0 and mix_ratio < 1:
            Z = [(mix_ratio * z + (1 - mix_ratio) * zo) for z, zo in zip(Z, Z_old)]  # only for irreducible
        #            U_opt_free_irr=self.get_max_eig(Z,self.nWfree[irr],self.chk.num_bandsfree[irr]) #  only for irreducible
        #            U_opt_free=self.symmetrize_U_opt(U_opt_free_irr,free=True)
        U_opt_free = get_max_eig(Z, nWfree, num_bands_free)  #
        # Symmetrize_U_opt_free 
        # TODO : implement symmetrization : first symmetrize on the irreducible points, then distribute to the reducible points
        # later, advantage can be taken of using only irreducible points
        Omega_I = sum(Mmn_FF.Omega_I(U_opt_free))
        Omega_I_list.append(Omega_I)
        delta_std = print_progress(Omega_I_list, i_iter, num_iter_converge, print_progress_every)

        if delta_std < conv_tol:
            break
        Z_old = deepcopy(Z)
    del Z_old, Zfreefree, Z

    U_opt_full = rotate_to_projections(w90data, U_opt_free)  # temporary, withour symmetries
    w90data.chk.v_matrix = np.array(U_opt_full).transpose((0, 2, 1))
    # TODO : calculate only diagonal elements
    w90data.chk._wannier_centers = w90data.chk.get_AA_q(w90data.mmn, transl_inv=True).diagonal(axis1=1, axis2=2).sum(
        axis=0).real.T / w90data.chk.num_kpts
    w90data.chk._wannier_centers = w90data.chk.get_wannier_centers(w90data.mmn)
    w90data.chk.spreads = w90data.chk.get_wannier_spreads(w90data.mmn)

    w90data.wannierised = True
    return w90data.chk.v_matrix, w90data.chk._wannier_centers, w90data.chk.spreads


def rotate_to_projections(w90data, U_opt_free):
    """
    rotate the U matrix to the projections of the bands
    to better match the initial guess

    Parameters
    ----------
    w90data : Wannier90data
        the data (inputs of wannier90)
    U_opt_free : list of numpy.ndarray(nBfree,nW)
        the optimized U matrix for the free bands and wannier functions

    Returns
    -------
    list of numpy.ndarray(nBfree,nW)
        the rotated U matrix
    """
    U_opt_full = []
    for ik in w90data.iter_kpts:
        nband = w90data.eig.data[ik].shape[0]
        nfrozen = sum(w90data.frozen[ik])
        U = np.zeros((nband, w90data.chk.num_wann), dtype=complex)
        U[w90data.frozen[ik], range(nfrozen)] = 1.
        U[w90data.free[ik], nfrozen:] = U_opt_free[ik]
        Z, _, V = np.linalg.svd(U.T.conj().dot(w90data.amn.data[ik]))
        U_opt_full.append(U.dot(Z.dot(V)))
    return U_opt_full

def print_progress(Omega_I_list, i_iter, num_iter_converge, print_progress_every):
    """
    print the progress of the disentanglement

    Parameters
    ----------
    i_iter : int
        the current iteration
    Omega_I_list : list of float
        the list of the spread functional
    num_iter_converge : int
        the number of iterations to check the convergence
    print_progress_every : int
        the frequency to print the progress

    Returns
    -------
    float
        the standard deviation of the spread functional over the last `num_iter_converge` iterations
    """
    Omega_I = Omega_I_list[-1]
    if i_iter > 0:
        delta = f"{Omega_I - Omega_I_list[-2]:15.8e}"
    else:
        delta = "--"

    if i_iter >= num_iter_converge:
        delta_std = np.std(Omega_I_list[-num_iter_converge:])
        delta_std_str = f"{delta_std:15.8e}"
    else:
        delta_std = np.Inf
        delta_std_str = "--"

    if i_iter % print_progress_every == 0:
        print(f"iteration {i_iter:4d} Omega_I = {Omega_I:15.10f}  delta={delta}, delta_std={delta_std_str}")

    return delta_std


def calc_Z(w90data, mmn_ff, U_loc=None):
    """
    calculate the Z matrix for the given Mmn matrix and U matrix

    Z = \sum_{b,k} w_{b,k} M_{b,k} M_{b,k}^{\dagger}
    where M_{b,k} = M_{b,k}^{loc} U_{b,k}

    Parameters
    ----------
    w90data : Wannier90data
        the data (inputs of wannier90)
    mmn_ff : list of numpy.ndarray(nnb,nb,nb)
        the Mmn matrix (either free-free or free-frozen)

    U_loc : list of numpy.ndarray(nBfree,nW)
        the U matrix

    Returns
    -------
    list of numpy.ndarray(nW,nW)
        the Z matrix
    """
    if U_loc is None:
        # Mmn_loc_opt=[Mmn_loc[ik] for ik in w90data.Dmn.kptirr]
        Mmn_loc_opt = [mmn_ff[ik] for ik in w90data.iter_kpts]
    else:
        # mmnff=[mmnff[ik] for ik in w90data.Dmn.kptirr]
        # mmnff = [mmnff[ik] for ik in w90data.iter_kpts]
        # Mmn_loc_opt=[[Mmn[ib].dot(U_loc[ikb]) for ib,ikb in enumerate(neigh)] for Mmn,neigh in zip(mmnff,self.mmn.neighbours[irr])]
        Mmn_loc_opt = [[Mmn[ib].dot(U_loc[ikb]) for ib, ikb in enumerate(neigh)] for Mmn, neigh in
                        zip(mmn_ff, w90data.mmn.neighbours)]
    return [sum(wb * mmn.dot(mmn.T.conj()) for wb, mmn in zip(wbk, Mmn)) for wbk, Mmn in
            zip(w90data.mmn.wk, Mmn_loc_opt)]

def get_free_frozen(w90data, froz_min, froz_max):
    frozen = np.array([frozen_nondegen(E=w90data.eig.data[ik],
                                        froz_min=froz_min,
                                        froz_max=froz_max) for ik in w90data.iter_kpts])
    free = np.logical_not(frozen)
    for ik in w90data.iter_kpts:
        nfrozen = sum(frozen[ik])
        nfree = sum(free[ik])
        assert nfree + nfrozen == w90data.eig.NB, f"number of free bands {nfree} + number of frozen bands {nfrozen} "
        assert nfrozen <= w90data.chk.num_wann, (f"number of frozen bands {nfrozen} at k-point {ik + 1}"
                                                 f"is greater than number of wannier functions {w90data.chk.num_wann}")
    return free, frozen

def frozen_nondegen(E, thresh=DEGEN_THRESH, froz_min=np.inf, froz_max=-np.inf):
    """define the indices of the frozen bands, making sure that degenerate bands were not split
    (unfreeze the degenerate bands together)

    Parameters
    ----------
    E : numpy.ndarray(nb, dtype=float)
        the energies of the bands
    thresh : float
        the threshold for the degeneracy

    Returns
    -------
    numpy.ndarray(bool)
        the boolean array of the frozen bands  (True for frozen)
    """
    ind = list(np.where((E <= froz_max) * (E >= froz_min))[0])
    while len(ind) > 0 and ind[0] > 0 and E[ind[0]] - E[ind[0] - 1] < thresh:
        del ind[0]
    while len(ind) > 0 and ind[0] < len(E) and E[ind[-1] + 1] - E[ind[-1]] < thresh:
        del ind[-1]
    froz = np.zeros(E.shape, dtype=bool)
    froz[ind] = True
    return froz
# now rotating to the optimized space
#        self.Hmn=[]
#        print (self.Amn.shape)
#        for ik in self.iter_kpts:
#            U=U_opt_full[ik]
#            Ud=U.T.conj()
# hamiltonian is not diagonal anymore
#            self.Hmn.append(Ud.dot(np.diag(self.Eig[ik])).dot(U))
#            self.Amn[ik]=Ud.dot(self.Amn[ik])
#            self.Mmn[ik]=[Ud.dot(M).dot(U_opt_full[ibk]) for M,ibk in zip (self.Mmn[ik],self.mmn.neighbours[ik])]


# def symmetrize_U_opt(self,U_opt_free_irr,free=False):
#     # TODO : first symmetrize by the little group
#     # Now distribute to reducible points
#     d_band=self.Dmn.d_band_free if free else self.Dmn.d_band
#     U_opt_free=[d_band[ikirr][isym] @ U_opt_free_irr[ikirr] @ self.Dmn.D_wann_dag[ikirr][isym] for isym,ikirr in zip(self.Dmn.kpt2kptirr_sym,self.Dmn.kpt2kptirr)  ]
#     return U_opt_free
#
# def rotate(self,mat,ik1,ik2):
#     # data should be of form NBxNBx ...   - any form later
#     if len(mat.shape)==1:
#         mat=np.diag(mat)
#     assert mat.shape[:2]==(self.num_bands,)*2
#     shape=mat.shape[2:]
#     mat=mat.reshape(mat.shape[:2]+(-1,)).transpose(2,0,1)
#     mat=mat[self.win_min[ik1]:self.win_max[ik1],self.win_min[ik2]:self.win_max[ik2]]
#     v1=self.v_matrix[ik1].conj()
#     v2=self.v_matrix[ik2].T
#     return np.array( [v1.dot(m).dot(v2) for m in mat]).transpose( (1,2,0) ).reshape( (self.num_wann,)*2+shape )


# def write_files(self,seedname="wannier90"):
#    "Write the disentangled files , where num_wann==num_bands"
#    Eig=[]
#    Uham=[]
#    Amn=[]
#    Mmn=[]
#    for H in self.Hmn:
#        E,U=np.linalg.eigh(H)
#        Eig.append(E)
#        Uham.append(U)
#    EIG(data=Eig).write(seedname)
#    for ik in self.iter_kpts:
#        U=Uham[ik]
#        Ud=U.T.conj()
#        Amn.append(Ud.dot(self.Amn[ik]))
#        Mmn.append([Ud.dot(M).dot(Uham[ibk]) for M,ibk in zip (self.Mmn[ik],self.mmn.neighbours[ik])])
#    MMN(data=Mmn,G=self.G,bk_cart=self.mmn.bk_cart,wk=self.mmn.wk,neighbours=self.mmn.neighbours).write(seedname)
#    AMN(data=Amn).write(seedname)

def get_max_eig(matrix, nvec, nBfree):
    """ return the nvec column-eigenvectors of matrix with maximal eigenvalues.
    Both matrix and nvec are lists by k-points with arbitrary size of matrices

    Parameters
    ----------
    matrix : list of numpy.ndarray(n,n)
        list of matrices
    nvec : list of int
        number of eigenvectors to return at each k-point
    nBfree : list of int
        number of free bands at each k-point

    Returns
    -------
    list of numpy.ndarray(n,nvec)
        list of eigenvectors
    """
    assert len(matrix) == len(nvec) == len(nBfree)
    assert np.all([m.shape[0] == m.shape[1] for m in matrix])
    assert np.all([m.shape[0] >= nv for m, nv in zip(matrix, nvec)]), \
        f"nvec={nvec}, m.shape={[m.shape for m in matrix]}"
    EV = [np.linalg.eigh(M) for M in matrix]
    return [ev[1][:, np.argsort(ev[0])[nf - nv:nf]] for ev, nv, nf in zip(EV, nvec, nBfree)]


class MmnFreeFrozen:
    # TODO : make use of irreducible kpoints (maybe)
    """ a class to store and call the Mmn matrix between/inside the free and frozen subspaces,
        as well as to calculate the spreads

        Parameters
        ----------
        Mmn : list of numpy.ndarray(nnb,nb,nb)
            list of Mmn matrices
        free : list of numpy.ndarray(nk,nb)
            list of free bands at each k-point
        frozen : list of numpy.ndarray(nk,nb)
            list of frozen bands at each k-point
        neighbours : list of list of tuple
            list of neighbours for each k-point
        wb : list of numpy.ndarray(nnb)
            list of weights for each neighbour (b-vector)
        NW : int
            number of Wannier functions

        Attributes
        ----------
        Omega_I_0 : float
            the constant term of the spread functional
        Omega_I_frozen : float
            the spread of the frozen bands
        data : dict((str,str),list of numpy.ndarray(nnb,nf,nf)
            the data for the Mmn matrix for each pair of subspaces (free/frozen)
        spaces : dict
            the spaces (free/frozen)
        neighbours : list of list of tuple
            list of neighbours for each k-point
        wk : list of numpy.ndarray(nnb)
            list of weights for each neighbour (b-vector)
        NK : int
            number of k-points
        """

    def __init__(self, Mmn, free, frozen, neighbours, wb, NW):
        self.NK = len(Mmn)
        self.wk = wb
        self.neighbours = neighbours
        self.data = {}
        self.spaces = {'free': free, 'frozen': frozen}
        for s1, sp1 in self.spaces.items():
            for s2, sp2 in self.spaces.items():
                self.data[(s1, s2)] = [[Mmn[ik][ib][sp1[ik], :][:, sp2[ikb]]
                                        for ib, ikb in enumerate(neigh)] for ik, neigh in enumerate(self.neighbours)]
        self.Omega_I_0 = NW * self.wk[0].sum()
        self.Omega_I_frozen = -sum(sum(wb * np.sum(abs(mmn[ib]) ** 2) for ib, wb in enumerate(WB)) for WB, mmn in
                                   zip(self.wk, self('frozen', 'frozen'))) / self.NK

    def __call__(self, space1, space2):
        """
        return the Mmn matrix between the given subspaces

        Parameters
        ----------
        space1, space2 : str
            the two subspaces (free/frozen)
        
        Returns
        -------
        list of numpy.ndarray(nnb,nf,nf)
            the Mmn matrix
        """
        assert space1 in self.spaces
        assert space2 in self.spaces
        return self.data[(space1, space2)]

    def Omega_I_free_free(self, U_opt_free):
        """
        calculate the spread of the free bands

        Parameters
        ----------
        U_opt_free : list of numpy.ndarray(nBfree,nW)
            the optimized U matrix for the free bands

        Returns
        -------
        float
            the spread of the free bands (eq. 27 of SMV2001)
        """
        U = U_opt_free
        Mmn = self('free', 'free')
        return -sum(self.wk[ik][ib] * np.sum(abs(U[ik].T.conj().dot(Mmn[ib]).dot(U[ikb])) ** 2)
                    for ik, Mmn in enumerate(Mmn) for ib, ikb in enumerate(self.neighbours[ik])) / self.NK

    def Omega_I_free_frozen(self, U_opt_free):
        """
        calculate the spread between the free and frozen bands
        
        Parameters
        ----------
        U_opt_free : list of numpy.ndarray(nBfree,nW)
            the optimized U matrix for the free bands

        Returns
        -------
        float
            the spread between the free and frozen bands (eq. 27 of SMV2001)
        """
        U = U_opt_free
        Mmn = self('free', 'frozen')
        return -sum(self.wk[ik][ib] * np.sum(abs(U[ik].T.conj().dot(Mmn[ib])) ** 2)
                    for ik, Mmn in enumerate(Mmn) for ib, ikb in enumerate(self.neighbours[ik])) / self.NK * 2

    def Omega_I(self, U_opt_free):
        """
        calculate the spread of the optimized U matrix

        Parameters
        ----------
        U_opt_free : list of numpy.ndarray(nBfree,nW)
            the optimized U matrix for the free bands

        Returns
        -------
        float, float, float, float
            the spreads: Omega_I_0, Omega_I_frozen, Omega_I_free_frozen, Omega_I_free_free
        """
        return self.Omega_I_0, self.Omega_I_frozen, self.Omega_I_free_frozen(U_opt_free), self.Omega_I_free_free(
            U_opt_free)
