from time import time
import numpy as np
from .utility import readstr
from ..__utility import FortranFileR, points_to_mp_grid
import gc
from ..__utility import alpha_A, beta_A


class CheckPoint:
    """
    A class to store the data about wannierisation, written by Wannier90

    Parameters
    ----------
    seedname : str
        the prefix of the file (including relative/absolute path, but not including the extension `.chk`)
    kmesh_tol : float
        tolerance to distinguish different/same k-points
    bk_complete_tol : float
        tolerance for the completeness relation for finite-difference scheme
    """

    def __init__(self, seedname, kmesh_tol=1e-7, bk_complete_tol=1e-5):

        self.kmesh_tol = kmesh_tol  # will be used in set_bk
        self.bk_complete_tol = bk_complete_tol  # will be used in set_bk
        t0 = time()
        seedname = seedname.strip()
        FIN = FortranFileR(seedname + '.chk')
        readint = lambda: FIN.read_record('i4')
        readfloat = lambda: FIN.read_record('f8')

        def readcomplex():
            a = readfloat()
            return a[::2] + 1j * a[1::2]

        print('Reading restart information from file ' + seedname + '.chk :')
        self.comment = readstr(FIN)
        self.num_bands = readint()[0]
        num_exclude_bands = readint()[0]
        self.exclude_bands = readint()
        assert len(self.exclude_bands) == num_exclude_bands, f"read exclude_bands are {self.exclude_bands}, length={len(self.exclude_bands)} while num_exclude_bands={num_exclude_bands}"
        self.real_lattice = readfloat().reshape((3, 3), order='F')
        self.recip_lattice = readfloat().reshape((3, 3), order='F')
        assert np.linalg.norm(self.real_lattice.dot(self.recip_lattice.T) / (2 * np.pi) - np.eye(3)) < 1e-14, f"the read real and reciprocal lattices are not consistent {self.real_lattice.dot(self.recip_lattice.T) / (2 * np.pi)}!=identiy"
        self.num_kpts = readint()[0]
        self.mp_grid = readint()
        assert len(self.mp_grid) == 3
        assert self.num_kpts == np.prod(self.mp_grid), f"the number of k-points is not consistent with the mesh {self.num_kpts}!={np.prod(self.mp_grid)}"
        self.kpt_latt = readfloat().reshape((self.num_kpts, 3))
        self.nntot = readint()[0]
        self.num_wann = readint()[0]
        self.checkpoint = readstr(FIN)
        self.have_disentangled = bool(readint()[0])
        print(f"have_disentangled={self.have_disentangled}")
        if self.have_disentangled:
            self.omega_invariant = readfloat()[0]
            lwindow = np.array(readint().reshape((self.num_kpts, self.num_bands)), dtype=bool)
            ndimwin = readint()
            print(f"ndimwin={ndimwin}")
            u_matrix_opt = readcomplex().reshape((self.num_kpts, self.num_wann, self.num_bands)).swapaxes(1, 2)
            self.win_min = np.array([np.where(lwin)[0].min() for lwin in lwindow])
            self.win_max = np.array([wm + nd for wm, nd in zip(self.win_min, ndimwin)])
        else:
            self.win_min = np.array([0] * self.num_kpts)
            self.win_max = np.array([self.num_wann] * self.num_kpts)

        u_matrix = readcomplex().reshape((self.num_kpts, self.num_wann, self.num_wann)).swapaxes(1, 2)
        m_matrix = readcomplex().reshape((self.num_kpts, self.nntot, self.num_wann, self.num_wann)).swapaxes(2, 3)
        if self.have_disentangled:
            self.v_matrix = [u_opt[:nd, :].dot(u) for u, u_opt, nd in zip(u_matrix, u_matrix_opt, ndimwin)]
            # self.v_matrix = [u_opt.dot(u) for u, u_opt in zip(u_matrix, u_matrix_opt)]
        else:
            self.v_matrix = u_matrix
        self._wannier_centers = readfloat().reshape((self.num_wann, 3))
        self.wannier_spreads = readfloat().reshape((self.num_wann))
        del u_matrix, m_matrix
        gc.collect()
        print(f"Time to read .chk : {time() - t0}")

    def spin_order_block_to_interlace(self):
        """
        If the chk was obtain from a block ordering (like in old VASP versions), the ordering should be changed to interlace
        """
        v_matrix = np.zeros((self.num_kpts, self.num_bands, self.num_wann), dtype=complex)
        v_matrix[:, :, 0::2] = self.v_matrix[:, :, :self.num_wann // 2]
        v_matrix[:, :, 1::2] = self.v_matrix[:, :, self.num_wann // 2:]
        self.v_matrix = v_matrix

    def spin_order_interlace_to_block(self):
        """
        If the chk was obtain from an interlace ordering, one may want to change the ordering to block
        """
        v_matrix = np.zeros((self.num_kpts, self.num_bands, self.num_wann), dtype=complex)
        v_matrix[:, :, :self.num_wann // 2] = self.v_matrix[:, :, 0::2]
        v_matrix[:, :, self.num_wann // 2:] = self.v_matrix[:, :, 1::2]
        self.v_matrix = v_matrix


    def wannier_gauge(self, mat, ik1, ik2):
        """
        Returns the matrix elements in the Wannier gauge

        Parameters
        ----------
        mat : np.ndarray
            the matrix elements in the Hamiltonian gauge
        ik1, ik2 : int
            the indices of the k-points

        Returns
        -------
        np.ndarray
            the matrix elements in the Wannier gauge
        """
        # data should be of form NBxNBx ...   - any form later
        if len(mat.shape) == 1:
            mat = np.diag(mat)
        assert mat.shape[:2] == (self.num_bands,) * 2, f"mat.shape={mat.shape}, num_bands={self.num_bands}"
        mat = mat[self.win_min[ik1]:self.win_max[ik1], self.win_min[ik2]:self.win_max[ik2]]
        v1 = self.v_matrix[ik1].conj().T
        v2 = self.v_matrix[ik2]
        return np.tensordot(np.tensordot(v1, mat, axes=(1, 0)), v2, axes=(1, 0)).transpose(
            (0, -1) + tuple(range(1, mat.ndim - 1)))

    def get_HH_q(self, eig):
        """
        Returns the Hamiltonian matrix in the Wannier gauge

        Parameters
        ----------
        eig : `~wannierberri.system.w90_files.EIG`
            the eigenvalues of the Hamiltonian

        Returns
        -------
        np.ndarray
            the Hamiltonian matrix in the Wannier gauge
        """
        assert (eig.NK, eig.NB) == (self.num_kpts, self.num_bands), f"eig file has NK={eig.NK}, NB={eig.NB}, while the checkpoint has NK={self.num_kpts}, NB={self.num_bands}"
        HH_q = np.array([self.wannier_gauge(E, ik, ik) for ik, E in enumerate(eig.data)])
        return 0.5 * (HH_q + HH_q.transpose(0, 2, 1).conj())

    def get_SS_q(self, spn):
        """
        Returns the spin matrix in the Wannier gauge

        Parameters
        ----------
        spn : `~wannierberri.system.w90_files.SPN`
            the spin matrix  

        Returns
        -------
        np.ndarray
            the spin matrix in the Wannier gauge
        """

        assert (spn.NK, spn.NB) == (self.num_kpts, self.num_bands), f"spn file has NK={spn.NK}, NB={spn.NB}, while the checkpoint has NK={self.num_kpts}, NB={self.num_bands}"
        SS_q = np.array([self.wannier_gauge(S, ik, ik) for ik, S in enumerate(spn.data)])
        return 0.5 * (SS_q + SS_q.transpose(0, 2, 1, 3).conj())

    #########
    # Oscar #
    ###########################################################################

    # Depart from the original matrix elements in the ab initio mesh
    # (Hamiltonian gauge) to obtain the corresponding matrix elements in the
    # Wannier gauge. The last constitute the basis to construct the real-space
    # matrix elements for Wannier interpolation, independently of the
    # finite-difference scheme used.

    def get_AABB_qb(self, mmn, transl_inv=False, eig=None, phase=None, sum_b=False):
        """
        Returns the matrix elements AA or BB(if eig is not Flase) in the Wannier gauge

        Parameters
        ----------
        mmn : `~wannierberri.system.w90_files.MMN`
            the overlap matrix elements between the Wavefunctions at neighbouring k-points
        transl_inv : bool
            if True, the band-diagonal matrix elements are calculated using the Marzari & Vanderbilt 
            translational invariant formula
        eig : `~wannierberri.system.w90_files.EIG`
            the eigenvalues of the Hamiltonian, needed to calculate BB (if None, the matrix elements are AA)
        phase : np.ndarray(shape=(num_wann, num_wann, nnb), dtype=complex)
            the phase factors to be applied to the matrix elements (if None, no phase factors are applied)
        sum_b : bool
            if True, the matrix elements are summed over the neighbouring k-points. Otherwise, the matrix elements are stored in a 5D array of shape (num_kpts, num_wann, num_wann, nnb, 3)

        Returns
        -------
        np.ndarray(shape=(num_kpts, num_wann, num_wann, nnb, 3), dtype=complex) (if sum_b=False)
        or np.ndarray(shape=(num_kpts, num_wann, num_wann, nnb, 3), dtype=complex) (if sum_b=True)
            the q-resolved matrix elements AA or BB in the Wannier gauge
        """
        assert (not transl_inv) or eig is None, "transl_inv cannot be used for BB matrix elements"
        if sum_b:
            AA_qb = np.zeros((self.num_kpts, self.num_wann, self.num_wann, 3), dtype=complex)
        else:
            AA_qb = np.zeros((self.num_kpts, self.num_wann, self.num_wann, mmn.NNB, 3), dtype=complex)
        for ik in range(self.num_kpts):
            for ib in range(mmn.NNB):
                iknb = mmn.neighbours[ik, ib]
                ib_unique = mmn.ib_unique_map[ik, ib]
                # Matrix < u_k | u_k+b > (mmn)
                data = mmn.data[ik, ib]                   # Hamiltonian gauge
                if eig is not None:
                    data = data * eig.data[ik, :, None]  # Hamiltonian gauge (add energies)
                AAW = self.wannier_gauge(data, ik, iknb)  # Wannier gauge
                # Matrix for finite-difference schemes
                AA_q_ik_ib = 1.j * AAW[:, :, None] * mmn.wk[ik, ib] * mmn.bk_cart[ik, ib, None, None, :]
                # Marzari & Vanderbilt formula for band-diagonal matrix elements
                if transl_inv:
                    AA_q_ik_ib[range(self.num_wann), range(self.num_wann)] = -np.log(
                        AAW.diagonal()).imag[:, None] * mmn.wk[ik, ib] * mmn.bk_cart[ik, ib, None, :]
                if phase is not None:
                    AA_q_ik_ib *= phase[:, :, ib_unique, None]
                if sum_b:
                    AA_qb[ik] += AA_q_ik_ib
                else:
                    AA_qb[ik, :, :, ib_unique, :] = AA_q_ik_ib
        return AA_qb


    # --- A_a(q,b) matrix --- #


    def get_AA_qb(self, mmn, transl_inv=False, phase=None, sum_b=False):
        """	
         A wrapper for get_AABB_qb with eig=None
         see '~wannierberri.system.w90_files.CheckPoint.get_AABB_qb' for more details  
         """
        return self.get_AABB_qb(mmn, transl_inv=transl_inv, phase=phase, sum_b=sum_b)

    def get_AA_q(self, mmn, transl_inv=False):
        """
        A wrapper for get_AA_qb with sum_b=True
        see '~wannierberri.system.w90_files.CheckPoint.get_AA_qb' for more details
        """
        return self.get_AA_qb(mmn=mmn, transl_inv=transl_inv).sum(axis=3)

    def get_wannier_centers(self, mmn, spreads=False):
        """
        calculate wannier centers  with the Marzarri-Vanderbilt translational invariant formula
        and optionally the spreads

        Parameters
        ----------
        mmn : `~wannierberri.system.w90_files.MMN`
            the overlap matrix elements between the Wavefunctions at neighbouring k-points
        spreads : bool
            if True, the spreads are calculated

        Returns
        -------
        np.ndarray(shape=(num_wann, 3), dtype=float)
            the wannier centers
        np.ndarray(shape=(num_wann,), dtype=float)
            the wannier spreads (in Angstrom^2) (if spreads=True)
        """
        wcc = np.zeros((self.num_wann, 3), dtype=float)
        if spreads:
            r2 = np.zeros(self.num_wann, dtype=float)
        for ik in range(mmn.NK):
            for ib in range(mmn.NNB):
                iknb = mmn.neighbours[ik, ib]
                mmn_loc = self.wannier_gauge(mmn.data[ik, ib], ik, iknb)
                mmn_loc = mmn_loc.diagonal()
                log_loc = np.angle(mmn_loc)
                wcc += -log_loc[:, None] * mmn.wk[ik, ib] * mmn.bk_cart[ik, ib]
                if spreads:
                    r2 += (1 - np.abs(mmn_loc) ** 2 + log_loc ** 2) * mmn.wk[ik, ib]
        wcc /= mmn.NK
        if spreads:
            return wcc, r2 / mmn.NK - np.sum(wcc**2, axis=1)
        else:
            return wcc


    # --- B_a(q,b) matrix --- #


    def get_BB_qb(self, mmn, eig, phase=None, sum_b=False):
        """	
        a wrapper for get_AABB_qb to evaluate BB matrix elements. (transl_inv is disabled)
        see '~wannierberri.system.w90_files.CheckPoint.get_AABB_qb' for more details
        """
        return self.get_AABB_qb(mmn, eig=eig, phase=phase, sum_b=sum_b)


    def get_CCOOGG_qb(self, mmn, uhu, antisym=True, phase=None, sum_b=False):
        """
        Returns the matrix elements CC, OO or GG in the Wannier gauge

        Parameters
        ----------
        mmn : `~wannierberri.system.w90_files.MMN`
            the overlap matrix elements between the Wavefunctions at neighbouring k-points
        uhu : `~wannierberri.system.w90_files.UHU` or `~wannierberri.system.w90_files.UIU`
            the matrix elements uhu or uiu produced by pw2wannier90
        antisym : bool
            if True, the antisymmetric piece of the matrix elements is calculated. Otherwise, the full matrix is calculated
        phase : np.ndarray(shape=(num_wann, num_wann, nnb), dtype=complex)
            the phase factors to be applied to the matrix elements (if None, no phase factors are applied)
        sum_b : bool
            if True, the matrix elements are summed over the neighbouring k-points. Otherwise, the matrix elements are stored in a 6D array of shape (num_kpts, num_wann, num_wann, nnb, nnb, 3)

        Returns
        -------
        np.ndarray(shape=(num_kpts, num_wann, num_wann, nnb, nnb, 3), dtype=complex) (if sum_b=False)
        or np.ndarray(shape=(num_kpts, num_wann, num_wann, nnb, nnb, 3), dtype=complex) (if sum_b=True)
            the q-resolved matrix elements CC, OO or GG in the Wannier gauge
        """
        nd_cart = 1 if antisym else 2
        shape_NNB = () if sum_b else (mmn.NNB, mmn.NNB)
        shape = (self.num_kpts, self.num_wann, self.num_wann) + shape_NNB + (3,) * nd_cart
        CC_qb = np.zeros(shape, dtype=complex)
        if phase is not None:
            phase = np.reshape(phase, np.shape(phase)[:4] + (1,) * nd_cart)
        for ik in range(self.num_kpts):
            for ib1 in range(mmn.NNB):
                iknb1 = mmn.neighbours[ik, ib1]
                ib1_unique = mmn.ib_unique_map[ik, ib1]
                for ib2 in range(mmn.NNB):
                    iknb2 = mmn.neighbours[ik, ib2]
                    ib2_unique = mmn.ib_unique_map[ik, ib2]

                    # Matrix < u_k+b1 | H_k | u_k+b2 > (uHu)
                    data = uhu.data[ik, ib1, ib2]                 # Hamiltonian gauge
                    CCW = self.wannier_gauge(data, iknb1, iknb2)  # Wannier gauge

                    if antisym:
                        # Matrix for finite-difference schemes (takes antisymmetric piece only)
                        CC_q_ik_ib = 1.j * CCW[:, :, None] * (
                            mmn.wk[ik, ib1] * mmn.wk[ik, ib2] * (
                                mmn.bk_cart[ik, ib1, alpha_A] * mmn.bk_cart[ik, ib2, beta_A] -
                                mmn.bk_cart[ik, ib1, beta_A] * mmn.bk_cart[ik, ib2, alpha_A]))[None, None, :]
                    else:
                        # Matrix for finite-difference schemes (takes symmetric piece only)
                        CC_q_ik_ib = CCW[:, :, None, None] * (
                            mmn.wk[ik, ib1] * mmn.wk[ik, ib2] * (
                                mmn.bk_cart[ik, ib1, :, None] *
                                mmn.bk_cart[ik, ib2, None, :]))[None, None, :, :]
                    if phase is not None:
                        CC_q_ik_ib *= phase[:, :, ib1_unique, ib2_unique]
                    if sum_b:
                        CC_qb[ik] += CC_q_ik_ib
                    else:
                        CC_qb[ik, :, :, ib1_unique, ib2_unique] = CC_q_ik_ib
        return CC_qb

    # --- C_a(q,b1,b2) matrix --- #
    def get_CC_qb(self, mmn, uhu, phase=None, sum_b=False):
        """
        A wrapper for get_CCOOGG_qb with antisym=True
        see '~wannierberri.system.w90_files.CheckPoint.get_CCOOGG_qb' for more details
        """
        return self.get_CCOOGG_qb(mmn, uhu, phase=phase, sum_b=sum_b)

    # --- O_a(q,b1,b2) matrix --- #
    def get_OO_qb(self, mmn, uiu, phase=None, sum_b=False):
        """
        A wrapper for get_CCOOGG_qb with antisym=False
        see '~wannierberri.system.w90_files.CheckPoint.get_CCOOGG_qb' for more details
        (actually, the same as "~wannierberri.system.w90_files.CheckPoint.get_CC_qb")
        """
        return self.get_CCOOGG_qb(mmn, uiu, phase=phase, sum_b=sum_b)

    # Symmetric G_bc(q,b1,b2) matrix
    def get_GG_qb(self, mmn, uiu, phase=None, sum_b=False):
        """
        A wrapper for get_CCOOGG_qb with antisym=False 
        see '~wannierberri.system.w90_files.CheckPoint.get_CCOOGG_qb' for more details
        """
        return self.get_CCOOGG_qb(mmn, uiu, antisym=False, phase=phase, sum_b=sum_b)
    ###########################################################################



    def get_SH_q(self, spn, eig):
        SH_q = np.zeros((self.num_kpts, self.num_wann, self.num_wann, 3), dtype=complex)
        assert (spn.NK, spn.NB) == (self.num_kpts, self.num_bands), f"spn file has NK={spn.NK}, NB={spn.NB}, while the checkpoint has NK={self.num_kpts}, NB={self.num_bands}"
        assert (eig.NK, eig.NB) == (self.num_kpts, self.num_bands), f"eig file has NK={eig.NK}, NB={eig.NB}, while the checkpoint has NK={self.num_kpts}, NB={self.num_bands}"
        for ik in range(self.num_kpts):
            SH_q[ik, :, :, :] = self.wannier_gauge(spn.data[ik, :, :, :] * eig.data[ik, None, :, None], ik, ik)
        return SH_q

    def get_SHA_q(self, shu, mmn, phase=None, sum_b=False):
        """
        SHA or SA (if siu is used instead of shu)
        """
        mmn.set_bk_chk(self)
        if sum_b:
            SHA_qb = np.zeros((self.num_kpts, self.num_wann, self.num_wann, 3, 3), dtype=complex)
        else:
            SHA_qb = np.zeros((self.num_kpts, self.num_wann, self.num_wann, mmn.NNB, 3, 3), dtype=complex)
        assert shu.NNB == mmn.NNB, f"shu.NNB={shu.NNB}, mmn.NNB={mmn.NNB} - mismatch"
        for ik in range(self.num_kpts):
            for ib in range(mmn.NNB):
                iknb = mmn.neighbours[ik, ib]
                ib_unique = mmn.ib_unique_map[ik, ib]
                SHAW = self.wannier_gauge(shu.data[ik, ib], ik, iknb)
                SHA_q_ik_ib = 1.j * SHAW[:, :, None, :] * mmn.wk[ik, ib] * mmn.bk_cart[ik, ib, None, None, :, None]

                if phase is not None:
                    SHA_q_ik_ib *= phase[:, :, ib_unique, None, None]
                if sum_b:
                    SHA_qb[ik] += SHA_q_ik_ib
                else:
                    SHA_qb[ik, :, :, ib_unique, :, :] = SHA_q_ik_ib

        return SHA_qb



    def get_SHR_q(self, spn, mmn, eig=None, phase=None):
        """
        SHR or SR(if eig is None)
        """
        mmn.set_bk_chk(self)
        SHR_q = np.zeros((self.num_kpts, self.num_wann, self.num_wann, 3, 3), dtype=complex)
        assert (spn.NK, spn.NB) == (self.num_kpts, self.num_bands), f"spn file has NK={spn.NK}, NB={spn.NB}, while the checkpoint has NK={self.num_kpts}, NB={self.num_bands}"
        assert (mmn.NK, mmn.NB) == (self.num_kpts, self.num_bands), f"mmn file has NK={mmn.NK}, NB={mmn.NB}, while the checkpoint has NK={self.num_kpts}, NB={self.num_bands}"
        for ik in range(self.num_kpts):
            SH = spn.data[ik, :, :, :]
            if eig is not None:
                SH = SH * eig.data[ik, None, :, None]
            SHW = self.wannier_gauge(SH, ik, ik)
            for ib in range(mmn.NNB):
                iknb = mmn.neighbours[ik, ib]
                ib_unique = mmn.ib_unique_map[ik, ib]
                SHM = np.tensordot(SH, mmn.data[ik, ib], axes=((1,), (0,))).swapaxes(-1, -2)
                SHRW = self.wannier_gauge(SHM, ik, iknb)
                if phase is not None:
                    SHRW = SHRW * phase[:, :, ib_unique, None]
                SHRW = SHRW - SHW
                SHR_q[ik, :, :, :, :] += 1.j * SHRW[:, :, None] * mmn.wk[ik, ib] * mmn.bk_cart[ik, ib, None, None, :, None]
        return SHR_q


    @property
    def wannier_centers(self):
        return self._wannier_centers

    def apply_window(self, selected_bands):
        if selected_bands is not None:
            self.num_bands = sum(selected_bands)
            for ik in range(self.num_kpts):
                if hasattr(self, "v_matrix"):
                    self.v_matrix[ik] = self.v_matrix[ik][:, selected_bands]
            print(np.min(np.where(selected_bands)[0]))
            win_min = np.min(np.where(selected_bands)[0])
            win_max = np.max(np.where(selected_bands)[0]) + 1
            self.win_min = np.max([self.win_min - win_min, [0] * self.num_kpts], axis=0)
            self.win_max = self.num_bands - np.max([win_max - self.win_max, [0] * self.num_kpts], axis=0)




class CheckPoint_bare(CheckPoint):
    """
            Class to store data from Wanierisationm obtained internally
            Initialize without the v_matrix (to be written later by `~wannierberri.system.disentangle`)

            Parameters
            ----------
            win : `~wannierberri.system.w90_files.WIN`
            eig : `~wannierberri.system.w90_files.EIG`
            mmn : `~wannierberri.system.w90_files.MMN`
            """

    def __init__(self, win=None, 
                 kpt_latt=None, 
                 real_lattice=None,
                 num_bands=None,
                 num_wann=None):
        if win is not None:
            if kpt_latt is None:
                try:
                    kpt_latt = win.get_kpoints()
                except AttributeError:
                    pass
            if real_lattice is None:
                real_lattice = win.get_unit_cell_cart_ang()
            if num_wann is None:
                try:
                    num_wann = win["num_wann"]
                except KeyError:
                    pass
            if num_bands is None:
                try:
                    num_bands = win["num_bands"]
                except KeyError:
                    pass
        self.num_bands = num_bands
        self.num_wann = num_wann    
        self.real_lattice = real_lattice
        self.kpt_latt = kpt_latt
        if self.kpt_latt is not None:
            self.num_kpts = self.kpt_latt.shape[0]
            self.mp_grid = np.array( points_to_mp_grid(self.kpt_latt))
                
        self.win_min = np.array([0] * self.num_kpts)
        self.win_max = np.array([self.num_bands] * self.num_kpts)
        self.recip_lattice = 2 * np.pi * np.linalg.inv(self.real_lattice).T
