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
from ..__utility import real_recip_lattice, fourier_q_to_R, alpha_A, beta_A
from .system_R import System_R
from .w90_files import Wannier90data
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
    transl_inv : bool
        Use Eq.(31) of `Marzari&Vanderbilt PRB 56, 12847 (1997) <https://journals.aps.org/prb/abstract/10.1103/PhysRevB.56.12847>`_ for band-diagonal position matrix elements
    transl_inv_JM : bool
        translational-invariant scheme for diagonal and off-diagonal matrix elements for all matrices. Follows method of Jae-Mo Lihm
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

    Notes
    -----
    see also  parameters of the :class:`~wannierberri.system.System`
    """

    def __init__(
            self,
            seedname="wannier90",
            w90data=None,
            transl_inv=True,
            transl_inv_JM=False,
            guiding_centers=False,
            fft='fftw',
            npar=multiprocessing.cpu_count(),
            kmesh_tol=1e-7,
            bk_complete_tol=1e-5,
            wcc_phase_fin_diff=True,
            **parameters):

        if "name" not in parameters:
            parameters["name"] = os.path.split(seedname)[-1]
        super().__init__(**parameters)
        self.npar = npar
        self.seedname = seedname
        if w90data is None:
            w90data = Wannier90data(self.seedname, read_chk=True, kmesh_tol=kmesh_tol, bk_complete_tol=bk_complete_tol)
        w90data.check_wannierised(msg="creation of System_Wannierise")
        chk = w90data.chk
        self.real_lattice, self.recip_lattice = real_recip_lattice(chk.real_lattice, chk.recip_lattice)
        mp_grid = chk.mp_grid
        self._NKFFT_recommended = mp_grid
        self.iRvec, Ndegen = wigner_seitz(real_lattice=self.real_lattice, mp_grid=chk.mp_grid)
        self.nRvec0 = len(self.iRvec)
        self.num_wann = chk.num_wann
        self.wannier_centers_cart = w90data.wannier_centers

        # Deactivate transl_inv if Jae-Mo's scheme is used
        if transl_inv_JM:
            if transl_inv:
                print("WARNING : Jae-Mo's scheme does not apply Marzari & Vanderbilt formula for"
                      "the band-diagonal matrix elements of the position operator.")
                transl_inv = False
        #######################################################################

        use_wcc_phase_findiff = self.use_wcc_phase and wcc_phase_fin_diff
        assert not (transl_inv_JM and use_wcc_phase_findiff)
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
            fft=fft)

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

        # A_a(R,b) matrix
        if self.need_R_any('AA'):
            AA_qb = chk.get_AA_qb(w90data.mmn, transl_inv=transl_inv)
            AA_Rb = fourier_q_to_R_loc(AA_qb)
            self.set_R_mat('AA', AA_Rb)
            if transl_inv:
                wannier_centers_cart_new = np.diagonal(AA_Rb[:, :, self.iR0, :], axis1=0,
                                                       axis2=1).sum(axis=0).T
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
            BB_qb = chk.get_BB_qb(w90data.mmn, w90data.eig)
            BB_Rb = fourier_q_to_R_loc(BB_qb)
            self.set_R_mat('BB', BB_Rb)

        # C_a(R,b1,b2) matrix
        if 'CC' in self.needed_R_matrices:
            CC_qb = chk.get_CC_qb(w90data.mmn, w90data.uhu)
            CC_Rb = fourier_q_to_R_loc(CC_qb)
            self.set_R_mat('CC', CC_Rb)

        # O_a(R,b1,b2) matrix
        if 'OO' in self.needed_R_matrices:
            OO_qb = chk.get_OO_qb(w90data.mmn, w90data.uiu)
            OO_Rb = fourier_q_to_R_loc(OO_qb)
            self.set_R_mat('OO', OO_Rb)

        # G_bc(R,b1,b2) matrix
        if 'GG' in self.needed_R_matrices:
            GG_qb = chk.get_GG_qb(w90data.mmn, w90data.uiu)
            GG_Rb = fourier_q_to_R_loc(GG_qb)
            self.set_R_mat('GG', GG_Rb)

        #######################################################################

        if self.need_R_any('SS'):
            self.set_R_mat('SS', fourier_q_to_R_loc(chk.get_SS_q(w90data.spn)))
        if self.need_R_any('SR'):
            self.set_R_mat('SR', fourier_q_to_R_loc(chk.get_SR_q(w90data.spn, w90data.mmn)))
        if self.need_R_any('SH'):
            self.set_R_mat('SH', fourier_q_to_R_loc(chk.get_SH_q(w90data.spn, w90data.eig)))
        if self.need_R_any('SHR'):
            self.set_R_mat('SHR', fourier_q_to_R_loc(chk.get_SHR_q(w90data.spn, w90data.mmn, w90data.eig)))

        if 'SA' in self.needed_R_matrices:
            self.set_R_mat('SA', fourier_q_to_R_loc(chk.get_SA_q(w90data.siu, w90data.mmn)))
        if 'SHA' in self.needed_R_matrices:
            self.set_R_mat('SHA', fourier_q_to_R_loc(chk.get_SHA_q(w90data.shu, w90data.mmn)))

        if self.use_ws:
            self.do_ws_dist(mp_grid=mp_grid)
        print("Real-space lattice:\n", self.real_lattice)

        #########
        # Oscar #
        #######################################################################

        # After the minimal-distance replica selection method (ws_dist) has
        # been applied -- called in 'do_at_end_of_init()' -- the b-resolved
        # matrix elements in real-space correspond to the final list of lattice
        # vectors {R}. Here we apply the phase factors associated with the
        # chosen finite-difference scheme, and perform the sum over
        # nearest-neighbor vectors to finally obtain the real-space matrix
        # elements.

        # Wannier centers
        centers = chk.wannier_centers
        # Unique set of nearest-neighbor vectors (cartesian)
        bk_cart_unique = w90data.mmn.bk_cart_unique

        if use_wcc_phase_findiff:  # Phase convention I
            _expiphase = np.exp(1j * np.einsum('ba,ja->jb', bk_cart_unique, centers))
            expiphase1 = _expiphase[None, :, None, :, None]
            expiphase2 = (_expiphase[:, None, :, None].conj() * _expiphase[None, :, None, :]
                                    )[:, :, None, :, :, None]
            del _expiphase
        elif transl_inv_JM:
            # Optimal center in Jae-Mo's implementation
            r0 = 0.5 * (centers[:, None, None, :] + centers[None, :, None, :] + self.cRvec[None, None, :, :])
            phase = np.einsum('ba,ijRa->ijRb', bk_cart_unique, r0 - self.cRvec[None, None, :, :])
            _expiphase = np.exp(1j * phase)
            expiphase1 = _expiphase[:, :, :, :, None]
            phase_1 = -np.einsum('ba,ijRa->ijRb', bk_cart_unique, r0)
            expiphase2 = (np.exp(1j * phase_1)[:, :, :, :, None] * _expiphase[:, :, :, None, :]
                          )[:, :, :, :, :, None]
            del phase_1, phase, _expiphase
        else:
            expiphase1 = 1
            expiphase2 = 1

        def _reset_mat(key, phase, axis, Hermitian=True):
            if self.need_R_any(key):
                XX_Rb = self.get_R_mat(key)
                phase = np.reshape(phase, np.shape(phase) + (1,) * (XX_Rb.ndim - np.ndim(phase)))
                XX_R = np.sum(XX_Rb * phase, axis=axis)
                self.set_R_mat(key, XX_R, reset=True, Hermitian=Hermitian)

        _reset_mat('AA', expiphase1, 3)
        _reset_mat('BB', expiphase1, 3, Hermitian=False)
        _reset_mat('CC', expiphase2, (3, 4))
        _reset_mat('OO', expiphase2, (3, 4))
        _reset_mat('GG', expiphase2, (3, 4))

        del expiphase1, expiphase2

        if transl_inv_JM:
            self.recenter_JM(r0, centers)

        self.do_at_end_of_init(
            convert_convention=((not transl_inv_JM) and self.use_wcc_phase and (not wcc_phase_fin_diff)))

    ###########################################################################
    def recenter_JM(self, r0, centers):
        known = ['Ham', 'AA', 'BB', 'CC', 'OO', 'GG', 'SS']
        unknown = set(self._XX_R.keys()) - set(known)
        if len(unknown) > 0:
            raise NotImplementedError(f"transl_inv_JM for {list(unknown)} is not implemented")
        if self.use_wcc_phase:
            self.recenter_JM_I(r0, centers)
        else:
            self.recenter_JM_II(r0, centers)

    def recenter_JM_I(self, r0, centers):
        """convention I"""
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
        # --- O_a(R) matrix --- #
        if self.need_R_any('OO'):
            assert AA_R0 is not None, 'Recentered A matrix is needed in Jae-Mo`s implementation of O'
            rc = 1.j * (r0[:, :, :, :, None] - centers[:, None, None, :, None]) * AA_R0[:, :, :, None, :]
            OO_R_add = rc[:, :, :, alpha_A, beta_A] - rc[:, :, :, beta_A, alpha_A]
            self.set_R_mat('OO', OO_R_add, add=True, Hermitian=True)

        # --- G_bc(R) matrix --- #
        if self.need_R_any('GG'):
            pass

    def recenter_JM_II(self, r0, centers):
        """convention II - will be eventually removed"""
        # --- A_a(R) matrix --- #
        if self.need_R_any('AA'):
            AA_R0 = self.get_R_mat('AA').copy()
            self.set_R_mat('AA', centers, R=[0, 0, 0], add=True, diag=True)
        # --- B_a(R) matrix --- #
        if self.need_R_any('BB'):
            BB_R0 = self.get_R_mat('BB').copy()
            HH_R = self.get_R_mat('Ham')
            rc = (r0 - self.cRvec[None, None, :, :]) * HH_R[:, :, :, None]
            self.set_R_mat('BB', rc, add=True)
        # --- C_a(R) matrix --- #
        if self.need_R_any('CC'):
            assert BB_R0 is not None, 'Recentered B matrix is needed in Jae-Mo`s implementation of C'
            BB_R0_conj = self.conj_XX_R(BB_R0)
            rc = 1j * r0[:, :, :, :, None] * BB_R0[:, :, :, None, :]
            rc -= 1j * (r0[:, :, :, :, None] - self.cRvec[None, None, :, :, None]) * BB_R0_conj[:, :, :, None, :]
            rc -= 0.5j * (centers[:, None, None, :, None] + centers[None, :, None, :, None]
                          ) * self.cRvec[None, None, :, None, :] * HH_R[:, :, :, None, None]
            CC_R_add = rc[:, :, :, alpha_A, beta_A] - rc[:, :, :, beta_A, alpha_A]
            self.set_R_mat('CC', CC_R_add, add=True, Hermitian=True)
        # --- O_a(R) matrix --- #
        if self.need_R_any('OO'):
            assert AA_R0 is not None, 'Recentered A matrix is needed in Jae-Mo`s implementation of O'
            rc = 1.j * self.cRvec[None, None, :, :, None] * AA_R0[:, :, :, None, :]
            OO_R_add = rc[:, :, :, alpha_A, beta_A] - rc[:, :, :, beta_A, alpha_A]
            self.set_R_mat('OO', OO_R_add, add=True, Hermitian=True)
        # --- G_bc(R) matrix --- #
        if self.need_R_any('GG'):
            assert AA_R0 is not None, 'Recentered A matrix is needed in Jae-Mo`s implementation of G'
            rc = (centers[:, None, None, :, None] + centers[None, :, None, :, None]) * AA_R0[:, :, :, None, :]
            rc[self.range_wann, self.range_wann, self.iR0] += (
                    centers[range(self.num_wann), :, None] * centers[range(self.num_wann), None, :])
            self.set_R_mat('GG', 0.5 * (rc + rc.swapaxes(3, 4)),
                           add=True, Hermitian=True)
