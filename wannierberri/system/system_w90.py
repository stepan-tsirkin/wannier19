#                                                            #
# This file is distributed as part of the WannierBerri code  #
# under the terms of the GNU General Public License. See the #
# file `LICENSE' in the root directory of the WannierBerri   #
# distribution, or http://www.gnu.org/copyleft/gpl.txt       #
#                                                            #
# The WannierBerri code is hosted on GitHub:                 #
# https://github.com/stepan-tsirkin/wannier-berri            #
#                     written by                             #
#           Stepan Tsirkin, University of Zurich             #
#   some parts of this file are originate                    #
# from the translation of Wannier90 code                     #
# ------------------------------------------------------------#

import numpy as np
import os
import functools
import multiprocessing
import warnings
from ..__utility import real_recip_lattice, fourier_q_to_R, alpha_A, beta_A
from .system_R import System_R
from ..w90files import Wannier90data
from .ws_dist import wigner_seitz


class System_w90(System_R):
    """
    System initialized from the Wannier functions generated by `Wannier90 <http://wannier.org>`__ code.
    Reads the ``.chk``, ``.eig`` and optionally ``.mmn``, ``.spn``, ``.uHu``, ``.sIu``, and ``.sHu`` files

    Parameters
    ----------
    seedname : str
        the seedname used in Wannier90
    w90data : `~wannierberri.system.Wannier90data`
        object that contains all Wannier90 input files and chk all together. If provided, overrides the `seedname`
    transl_inv_MV : bool
        Use Eq.(31) of `Marzari&Vanderbilt PRB 56, 12847 (1997) <https://journals.aps.org/prb/abstract/10.1103/PhysRevB.56.12847>`_ for band-diagonal position matrix elements
    transl_inv_JM : bool
        translational-invariant scheme for diagonal and off-diagonal matrix elements for all matrices. Follows method of Jae-Mo Lihm
    wcc_phase_fin_diff : bool
        Use the phase factors associated with the WCC in the finite-difference scheme (a "cheap" way to get translational invariance) 
    guiding_centers : bool
        If True, enable overwriting the diagonal elements of the AA_R matrix at R=0 with the
        Wannier centers calculated from Wannier90.
    npar : int
        number of processes used in the constructor
    fft : str
        library used to perform the fast Fourier transform from **q** to **R**. ``fftw`` or ``numpy``. (practically does not affect performance,
        anyway mostly time of the constructor is consumed by reading the input files)
    kmesh_tol : float
        tolerance to consider the b_k vectors (connecting to neighbouring k-points on the grid) belonging to the same shell
    bk_complete_tol : float
        tolerance to consider the set of b_k shells as complete.
    read_npz : bool
    write_npz_list : tuple(str)
    write_npz_formatted : bool
        see `~wannierberri.system.w90_files.Wannier90data`
    overwrite_npz : bool
        see `~wannierberri.system.w90_files.Wannier90data`
    formatted : tuple(str)
        see `~wannierberri.system.w90_files.Wannier90data`
    **parameters
        see `~wannierberri.system.system.System`

    Notes
    -----
    The R-matrices are evaluated in the nearest-neighbor vectors of the finite-difference scheme chosen.

    Attributes
    ----------
    seedname : str
        the seedname used in Wannier90
    npar : int
        number of processes used in the constructor
    real_lattice : np.ndarray(shape=(3, 3))
        real-space lattice vectors
    recip_lattice : np.ndarray(shape=(3, 3))    
        reciprocal-space lattice vectors
    iRvec : np.ndarray(shape=(num_R, 3), dtype=int)
        set R-vectors in the Wigner-Seitz supercell (in the basis of the real-space lattice vectors)
    cRvec : np.ndarray(shape=(num_R, 3))
        set R-vectors in the Wigner-Seitz supercell (in the cartesian coordinates)
    nRvec0 : int
        number of R-vectors, before applying the Wigner-Seitz distance
    num_wann : int
        number of Wannier functions
    wannier_centers_cart : np.ndarray(shape=(num_wann, 3))
        Wannier centers in Cartesian coordinates
    _NKFFT_recommended : int
        recommended size of the FFT grid in the interpolation
    use_ws : bool
        whether the Wigner-Seitz distance is applied
    use_wcc_phase : bool
        whether the phase factors associated with the WCC are used
    needed_R_matrices : list(str)
        list of the R-matrices, which will be evaluated

    See Also
    --------
    `~wannierberri.system.system.System_R`
    """

    def __init__(
            self,
            seedname="wannier90",
            w90data=None,
            transl_inv_MV=True,
            transl_inv_JM=False,
            guiding_centers=False,
            fftlib='fftw',
            npar=multiprocessing.cpu_count(),
            kmesh_tol=1e-7,
            bk_complete_tol=1e-5,
            wcc_phase_fin_diff=True,
            read_npz=True,
            write_npz_list=("eig", "mmn"),
            write_npz_formatted=True,
            overwrite_npz=False,
            formatted=tuple(),
            **parameters
    ):

        if "name" not in parameters:
            parameters["name"] = os.path.split(seedname)[-1]
        super().__init__(**parameters)

        use_wcc_phase_findiff = self.use_wcc_phase and wcc_phase_fin_diff
        assert not (transl_inv_JM and use_wcc_phase_findiff)
        if not (transl_inv_JM or transl_inv_MV):
            warnings.warn("It is highly recommended to use translational invairiance "
                          "(either transl_inv_JM or transl_inv_MV) ")
        if transl_inv_JM:
            assert self.use_wcc_phase, "transl_inv_JM is implemented only with convention I"
            known = ['Ham', 'AA', 'BB', 'CC', 'OO', 'GG', 'SS', 'SH', 'SA', 'SHA']
            unknown = set(self.needed_R_matrices) - set(known)
            if len(unknown) > 0:
                raise NotImplementedError(f"transl_inv_JM for {list(unknown)} is not implemented")
            # Deactivate transl_inv_MV if Jae-Mo's scheme is used
            if transl_inv_MV:
                warnings.warn("Jae-Mo's scheme does not apply Marzari & Vanderbilt formula for"
                              "the band-diagonal matrix elements of the position operator.")
                transl_inv_MV = False
        if self.use_wcc_phase and not wcc_phase_fin_diff:
            warnings.warn("converting convention II to convention I is not recommended."
                          "Better use 'wcc_phase_fin_dif=True' or `transl_inv_JM=True`")
        if use_wcc_phase_findiff:
            known = ['Ham', 'AA', 'BB', 'CC', 'OO', 'GG', 'SS', 'SH', 'SHR', 'SHA', 'SA', 'SR']
            unknown = set(self.needed_R_matrices) - set(known)
            if len(unknown) > 0:
                raise NotImplementedError(f"wcc_phase_fin_diff for {list(unknown)} is not implemented")

        self.npar = npar
        self.seedname = seedname
        if w90data is None:
            w90data = Wannier90data(self.seedname, read_chk=True, kmesh_tol=kmesh_tol, bk_complete_tol=bk_complete_tol,
                                    write_npz_list=write_npz_list, read_npz=read_npz, overwrite_npz=overwrite_npz,
                                    write_npz_formatted=write_npz_formatted,
                                    formatted=formatted)
        w90data.check_wannierised(msg="creation of System_w90")
        chk = w90data.chk
        self.real_lattice, self.recip_lattice = real_recip_lattice(chk.real_lattice, chk.recip_lattice)
        if hasattr(w90data, 'pointgroup'):
            self.set_pointgroup(pointgroup=w90data.pointgroup)
        elif hasattr(w90data, 'spacegroup'):
            self.set_pointgroup(spacegroup=w90data.spacegroup)
        
        
        mp_grid = chk.mp_grid
        self._NKFFT_recommended = mp_grid
        self.iRvec, Ndegen = wigner_seitz(real_lattice=self.real_lattice, mp_grid=chk.mp_grid)
        self.nRvec0 = len(self.iRvec)
        self.num_wann = chk.num_wann
        self.wannier_centers_cart = w90data.wannier_centers

        kpt_mp_grid = [
            tuple(k) for k in np.array(np.round(chk.kpt_latt * np.array(chk.mp_grid)[None, :]), dtype=int) % chk.mp_grid
        ]
        if (0, 0, 0) not in kpt_mp_grid:
            raise ValueError(
                "the grid of k-points read from .chk file is not Gamma-centered. Please, use Gamma-centered grids in the ab initio calculation"
            )

        fourier_q_to_R_loc = functools.partial(
            fourier_q_to_R,
            mp_grid=chk.mp_grid,
            kpt_mp_grid=kpt_mp_grid,
            iRvec=self.iRvec,
            ndegen=Ndegen,
            numthreads=npar,
            fftlib=fftlib)

        #########
        # Oscar #
        #######################################################################

        # Compute the Fourier transform of matrix elements in the original
        # ab-initio mesh (Wannier gauge) to real-space. These matrices are
        # resolved in b, i.e. in the nearest-neighbor vectors of the
        # finite-difference scheme chosen. After ws_dist is applied, phase
        # factors depending on the lattice vectors R can be added, and the sum
        # over nearest-neighbor vectors can be finally performed.

        w90data.mmn.set_bk_chk(chk)

        # H(R) matrix
        HHq = chk.get_HH_q(w90data.eig)
        self.set_R_mat('Ham', fourier_q_to_R_loc(HHq))

        # Wannier centers
        centers = chk.wannier_centers
        # Unique set of nearest-neighbor vectors (cartesian)
        bk_cart_unique = w90data.mmn.bk_cart_unique

        if use_wcc_phase_findiff or transl_inv_JM:  # Phase convention I
            if use_wcc_phase_findiff:
                _r0 = centers[None, :, :]
                sum_b = True
            elif transl_inv_JM:
                _r0 = 0.5 * (centers[:, None, :] + centers[None, :, :])
                sum_b = False
            expjphase1 = np.exp(1j * np.einsum('ba,ija->ijb', bk_cart_unique, _r0))
            print(f"expjphase1 {expjphase1.shape}")
            expjphase2 = expjphase1.swapaxes(0, 1).conj()[:, :, :, None] * expjphase1[:, :, None, :]
        else:
            expjphase1 = None
            expjphase2 = None
            sum_b = True

        # A_a(R,b) matrix
        if self.need_R_any('AA'):
            AA_qb = chk.get_AA_qb(w90data.mmn, transl_inv=transl_inv_MV, sum_b=sum_b, phase=expjphase1)
            AA_Rb = fourier_q_to_R_loc(AA_qb)
            self.set_R_mat('AA', AA_Rb, Hermitian=True)
            # Checking Wannier_centers
            if True:
                AA_q = chk.get_AA_qb(w90data.mmn, transl_inv=True, sum_b=True, phase=None)
                AA_R0 = AA_q.sum(axis=0) / np.prod(mp_grid)
                wannier_centers_cart_new = np.diagonal(AA_R0, axis1=0, axis2=1).T
                if not np.all(abs(wannier_centers_cart_new - self.wannier_centers_cart) < 1e-6):
                    if guiding_centers:
                        print(
                            f"The read Wannier centers\n{self.wannier_centers_cart}\n"
                            f"are different from the evaluated Wannier centers\n{wannier_centers_cart_new}\n"
                            "This can happen if guiding_centres was set to true in Wannier90.\n"
                            "Overwrite the evaluated centers using the read centers.")
                        for iw in range(self.num_wann):
                            self.get_R_mat('AA')[iw, iw, self.iR0, :] = self.wannier_centers_cart[iw, :]
                    else:
                        raise ValueError(
                            f"the difference between read\n{self.wannier_centers_cart}\n"
                            f"and evaluated \n{wannier_centers_cart_new}\n wannier centers is\n"
                            f"{self.wannier_centers_cart - wannier_centers_cart_new}\n"
                            "If guiding_centres was set to true in Wannier90, pass guiding_centers = True to System_w90."
                        )

        # B_a(R,b) matrix
        if 'BB' in self.needed_R_matrices:
            BB_qb = chk.get_BB_qb(w90data.mmn, w90data.eig, sum_b=sum_b, phase=expjphase1)
            BB_Rb = fourier_q_to_R_loc(BB_qb)
            self.set_R_mat('BB', BB_Rb)

        # C_a(R,b1,b2) matrix
        if 'CC' in self.needed_R_matrices:
            CC_qb = chk.get_CC_qb(w90data.mmn, w90data.uhu, sum_b=sum_b, phase=expjphase2)
            CC_Rb = fourier_q_to_R_loc(CC_qb)
            self.set_R_mat('CC', CC_Rb, Hermitian=True)

        # O_a(R,b1,b2) matrix
        if 'OO' in self.needed_R_matrices:
            OO_qb = chk.get_OO_qb(w90data.mmn, w90data.uiu, sum_b=sum_b, phase=expjphase2)
            OO_Rb = fourier_q_to_R_loc(OO_qb)
            self.set_R_mat('OO', OO_Rb, Hermitian=True)

        # G_bc(R,b1,b2) matrix
        if 'GG' in self.needed_R_matrices:
            GG_qb = chk.get_GG_qb(w90data.mmn, w90data.uiu, sum_b=sum_b, phase=expjphase2)
            GG_Rb = fourier_q_to_R_loc(GG_qb)
            self.set_R_mat('GG', GG_Rb, Hermitian=True)

        #######################################################################

        if self.need_R_any('SS'):
            self.set_R_mat('SS', fourier_q_to_R_loc(chk.get_SS_q(w90data.spn)))
        if self.need_R_any('SR'):
            self.set_R_mat('SR', fourier_q_to_R_loc(chk.get_SHR_q(spn=w90data.spn, mmn=w90data.mmn, phase=expjphase1)))
        if self.need_R_any('SH'):
            self.set_R_mat('SH', fourier_q_to_R_loc(chk.get_SH_q(w90data.spn, w90data.eig)))
        if self.need_R_any('SHR'):
            self.set_R_mat('SHR', fourier_q_to_R_loc(
                chk.get_SHR_q(spn=w90data.spn, mmn=w90data.mmn, eig=w90data.eig, phase=expjphase1)))

        if 'SA' in self.needed_R_matrices:
            self.set_R_mat('SA',
                           fourier_q_to_R_loc(chk.get_SHA_q(w90data.siu, w90data.mmn, sum_b=sum_b, phase=expjphase1)))
        if 'SHA' in self.needed_R_matrices:
            self.set_R_mat('SHA',
                           fourier_q_to_R_loc(chk.get_SHA_q(w90data.shu, w90data.mmn, sum_b=sum_b, phase=expjphase1)))

        del expjphase1, expjphase2

        if self.use_ws:
            self.do_ws_dist(mp_grid=mp_grid)

        if transl_inv_JM:
            self.recenter_JM(centers, bk_cart_unique)

        self.do_at_end_of_init()
        if (not transl_inv_JM) and self.use_wcc_phase and (not wcc_phase_fin_diff):
            self.convention_II_to_I()
        self.check_AA_diag_zero(msg="after conversion of conventions with "
                                    f"transl_inv_MV={transl_inv_MV}, transl_inv_JM={transl_inv_JM}",
                                set_zero=transl_inv_MV or transl_inv_JM)

    ###########################################################################
    def recenter_JM(self, centers, bk_cart_unique):
        """"
        Recenter the matrices in the Jae-Mo scheme
        (only in convention I)

        Parameters
        ----------
        centers : np.ndarray(shape=(num_wann, 3))
            Wannier centers in Cartesian coordinates
        bk_cart_unique : np.ndarray(shape=(num_bk, 3))
            set of unique nearest-neighbor vectors (cartesian)

        Notes
        -----
        The matrices are recentered in the following way:
        - A_a(R) matrix: no recentering
        - B_a(R) matrix: recentered by the Hamiltonian
        - C_a(R) matrix: recentered by the B matrix
        - O_a(R) matrix: recentered by the A matrix
        - G_bc(R) matrix: no recentering
        - S_a(R) matrix: recentered by the S matrix
        - SH_a(R) matrix: recentered by the S matrix
        - SR_a(R) matrix: recentered by the S matrix
        - SA_a(R) matrix: recentered by the S matrix
        - SHA_a(R) matrix: recentered by the S matrix
        """
        assert self.use_wcc_phase
        #  Here we apply the phase factors associated with the
        # JM scheme not accounted above, and perform the sum over
        # nearest-neighbor vectors to finally obtain the real-space matrix
        # elements.

        # Optimal center in Jae-Mo's implementation
        phase = np.einsum('ba,Ra->Rb', bk_cart_unique, - 0.5 * self.cRvec)
        expiphase1 = np.exp(1j * phase)
        expiphase2 = expiphase1[:, :, None] * expiphase1[:, None, :]

        def _reset_mat(key, phase, axis, Hermitian=True):
            if self.need_R_any(key):
                XX_Rb = self.get_R_mat(key)
                phase = np.reshape(phase, (1, 1) + np.shape(phase) + (1,) * (XX_Rb.ndim - 2 - np.ndim(phase)))
                XX_R = np.sum(XX_Rb * phase, axis=axis)
                self.set_R_mat(key, XX_R, reset=True, Hermitian=Hermitian)

        _reset_mat('AA', expiphase1, 3)
        _reset_mat('BB', expiphase1, 3, Hermitian=False)
        _reset_mat('CC', expiphase2, (3, 4))
        _reset_mat('SA', expiphase1, 3, Hermitian=False)
        _reset_mat('SHA', expiphase1, 3, Hermitian=False)
        _reset_mat('OO', expiphase2, (3, 4))
        _reset_mat('GG', expiphase2, (3, 4))

        del expiphase1, expiphase2
        r0 = 0.5 * (centers[:, None, None, :] + centers[None, :, None, :] + self.cRvec[None, None, :, :])

        # --- A_a(R) matrix --- #
        if self.need_R_any('AA'):
            AA_R0 = self.get_R_mat('AA').copy()
        # --- B_a(R) matrix --- #
        if self.need_R_any('BB'):
            BB_R0 = self.get_R_mat('BB').copy()
            HH_R = self.get_R_mat('Ham')
            rc = (r0 - self.cRvec[None, None, :, :] - centers[None, :, None, :]) * HH_R[:, :, :, None]
            self.set_R_mat('BB', rc, add=True)
        # --- C_a(R) matrix --- #
        if self.need_R_any('CC'):
            assert BB_R0 is not None, 'Recentered B matrix is needed in Jae-Mo`s implementation of C'
            BB_R0_conj = self.conj_XX_R(BB_R0)
            rc = 1j * (r0[:, :, :, :, None] - centers[:, None, None, :, None]) * (BB_R0 + BB_R0_conj)[:, :, :, None, :]
            CC_R_add = rc[:, :, :, alpha_A, beta_A] - rc[:, :, :, beta_A, alpha_A]
            self.set_R_mat('CC', CC_R_add, add=True, Hermitian=True)
        if self.need_R_any('SA'):
            SS_R = self.get_R_mat('SS')
            rc = (r0[:, :, :, :, None] - self.cRvec[None, None, :, :, None] - centers[None, :, None, :, None]
                  ) * SS_R[:, :, :, None, :]
            self.set_R_mat('SA', rc, add=True)
        if self.need_R_any('SHA'):
            SH_R = self.get_R_mat('SH')
            rc = (r0[:, :, :, :, None] - self.cRvec[None, None, :, :, None] -
                  centers[None, :, None, :, None]) * SH_R[:, :, :, None, :]
            self.set_R_mat('SHA', rc, add=True)
        # --- O_a(R) matrix --- #
        if self.need_R_any('OO'):
            assert AA_R0 is not None, 'Recentered A matrix is needed in Jae-Mo`s implementation of O'
            rc = 1.j * (r0[:, :, :, :, None] - centers[:, None, None, :, None]) * AA_R0[:, :, :, None, :]
            OO_R_add = rc[:, :, :, alpha_A, beta_A] - rc[:, :, :, beta_A, alpha_A]
            self.set_R_mat('OO', OO_R_add, add=True, Hermitian=True)
        # --- G_bc(R) matrix --- #
        if self.need_R_any('GG'):
            pass
