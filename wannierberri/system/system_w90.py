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
#------------------------------------------------------------#

import numpy as np
import functools
import multiprocessing
from ..__utility import iterate3dpm, real_recip_lattice, fourier_q_to_R, alpha_A, beta_A
from .system import System
from .__w90_files import EIG, MMN, CheckPoint, SPN, UHU, UIU, SIU, SHU
from time import time


class System_w90(System):
    """
    System initialized from the Wannier functions generated by `Wannier90 <http://wannier.org>`__ code.
    Reads the ``.chk``, ``.eig`` and optionally ``.mmn``, ``.spn``, ``.uHu``, ``.sIu``, and ``.sHu`` files

    Parameters
    ----------
    seedname : str
        the seedname used in Wannier90
    transl_inv : bool
        Use Eq.(31) of `Marzari&Vanderbilt PRB 56, 12847 (1997) <https://journals.aps.org/prb/abstract/10.1103/PhysRevB.56.12847>`_ for band-diagonal position matrix elements
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
    see also  parameters of the :class:`~wannierberri.System`
    """

    def __init__(
            self,
            seedname="wannier90",
            transl_inv=True,      
            guiding_centers=False,
            fft='fftw',
            npar=multiprocessing.cpu_count(),
            kmesh_tol=1e-7,
            bk_complete_tol=1e-5,
            **parameters):

        self.set_parameters(**parameters)
        self.npar = npar
        self.seedname = seedname

        chk = CheckPoint(self.seedname, kmesh_tol=kmesh_tol, bk_complete_tol=bk_complete_tol)
        self.real_lattice, self.recip_lattice = real_recip_lattice(chk.real_lattice, chk.recip_lattice)
        if self.mp_grid is None:
            self.mp_grid = chk.mp_grid
        self.iRvec, self.Ndegen = self.wigner_seitz(chk.mp_grid)
        self.nRvec0 = len(self.iRvec)
        self.num_wann = chk.num_wann
        self.wannier_centers_cart_auto = chk.wannier_centers

        #########
        # Oscar #
        #######################################################################

        # Deactivate transl_inv if Jae-Mo's scheme is used
        if self.transl_inv_JM:
            transl_inv = False

        # Necessary ab initio matrices
        eig = EIG(seedname)
        if self.need_R_any(['AA', 'BB']):
            mmn = MMN(seedname, npar=npar)
        if self.need_R_any(['CC']):
            uhu = UHU(seedname)
        if self.need_R_any(['GG']):
            uiu = UIU(seedname)

        #######################################################################

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
            ndegen=self.Ndegen,
            numthreads=npar,
            fft=fft)

        #########
        # Oscar #
        ############################################################################################################

        centers = chk.wannier_centers
        transl_inv_JM = self.transl_inv_JM

        # H(R) matrix
        timeFFT = 0
        HHq = chk.get_HH_q(eig)
        t0 = time()
        self.set_R_mat('Ham', fourier_q_to_R_loc(HHq))
        timeFFT += time() - t0

        # A_a(R) matrix
        if self.need_R_any('AA'):
            AA_qb, bk_latt_unique = chk.get_AA_qb(
                mmn, centers=centers, transl_inv=transl_inv, transl_inv_JM=transl_inv_JM)
            t0 = time()
            AA_Rb = fourier_q_to_R_loc(AA_qb)
            timeFFT += time() - t0

            # Naive finite-difference scheme
            if not transl_inv_JM:
                AA_R = np.sum(AA_Rb, axis=3)

            # Following Jae-Mo's scheme, keep b-resolved real-space matrix and sum after ws_dist is used
            if transl_inv_JM:
                AA_R = AA_Rb
                self.NNB = mmn.NNB
                self.bk_latt_unique = bk_latt_unique

            self.set_R_mat('AA', AA_R)

            # Using Marzari & Vanderbilt formula
            if transl_inv:
                wannier_centers_cart_new = np.diagonal(self.get_R_mat('AA')[:, :, self.iR0, :], axis1=0, axis2=1).transpose()
                if not np.all(abs(wannier_centers_cart_new - self.wannier_centers_cart_auto) < 1e-6):
                    if guiding_centers:
                        print(
                            f"The read Wannier centers\n{self.wannier_centers_cart_auto}\n"
                            f"are different from the evaluated Wannier centers\n{wannier_centers_cart_new}\n"
                            "This can happen if guiding_centres was set to true in Wannier90.\n"
                            "Overwrite the evaluated centers using the read centers.")
                        for iw in range(self.num_wann):
                            self.get_R_mat('AA')[iw, iw, self.iR0, :] = self.wannier_centers_cart_auto[iw, :]
                    else:
                        raise ValueError(
                            f"the difference between read\n{self.wannier_centers_cart_auto}\n"
                            f"and evluated \n{wannier_centers_cart_new}\n wannier centers is\n"
                            f"{self.wannier_centers_cart_auto-wannier_centers_cart_new}\n"
                            "If guiding_centres was set to true in Wannier90, pass guiding_centers = True to System_w90."
                        )

        # B_a(R) matrix
        if 'BB' in self.needed_R_matrices:
            BB_qb, bk_latt_unique = chk.get_BB_qb(mmn, eig, centers=centers, transl_inv_JM=transl_inv_JM)
            t0 = time()
            BB_Rb = fourier_q_to_R_loc(BB_qb)
            timeFFT += time() - t0

            # Naive finite-difference scheme
            if not transl_inv_JM:
                BB_R = np.sum(BB_Rb, axis=3)

            # Following Jae-Mo's scheme, keep b-resolved real-space matrix and sum after ws_dist is used
            if transl_inv_JM:
                BB_R = BB_Rb
                self.NNB = mmn.NNB
                self.bk_latt_unique = bk_latt_unique

            self.set_R_mat('BB', BB_R)

        # C_a(R) matrix
        if 'CC' in self.needed_R_matrices:
            CC_qb, b1k_latt_unique, b2k_latt_unique = chk.get_CC_qb(mmn, uhu, centers=centers, transl_inv_JM=transl_inv_JM)
            t0 = time()
            CC_Rb = fourier_q_to_R_loc(CC_qb)
            timeFFT += time() - t0

            # Naive finite-difference scheme
            if not transl_inv_JM:
                CC_R = np.sum(CC_Rb, axis=(3, 4))

            # Following Jae-Mo's scheme, keep b-resolved real-space matrix and sum after ws_dist is used
            if transl_inv_JM:
                CC_R = CC_Rb
                self.NNB = mmn.NNB
                self.bk_latt_unique = bk_latt_unique

            self.set_R_mat('CC', CC_R)

        # G_bc(R) matrix
        if 'GG' in self.needed_R_matrices:
            GG_qb, b1k_latt_unique, b2k_latt_unique = chk.get_GG_qb(mmn, uiu, centers=centers, transl_inv_JM=transl_inv_JM)
            t0 = time()
            GG_Rb = fourier_q_to_R_loc(GG_qb)
            timeFFT += time() - t0

            # Naive finite-difference scheme
            if not transl_inv_JM:
                GG_R = np.sum(GG_Rb, axis=(3, 4))

            # Following Jae-Mo's scheme, keep b-resolved real-space matrix and sum after ws_dist is used
            if transl_inv_JM:
                GG_R = GG_Rb
                self.NNB = mmn.NNB
                self.bk_latt_unique = bk_latt_unique

            self.set_R_mat('GG', GG_R)

        try:
            del uhu
        except NameError:
            pass

        try:
            del uiu
        except NameError:
            pass

        #######################################################################

        if self.need_R_any(['SS', 'SR', 'SH', 'SHR']):
            spn = SPN(seedname)
            t0 = time()
            if self.need_R_any('SS'):
                self.set_R_mat('SS' ,fourier_q_to_R_loc(chk.get_SS_q(spn)))
            if self.need_R_any('SR'):
                self.set_R_mat('SR' , fourier_q_to_R_loc(chk.get_SR_q(spn, mmn)))
            if self.need_R_any('SH'):
                self.set_R_mat('SH' , fourier_q_to_R_loc(chk.get_SH_q(spn, eig)))
            if self.need_R_any('SHR'):
                self.set_R_mat('SHR' , fourier_q_to_R_loc(chk.get_SHR_q(spn, mmn, eig)))
            timeFFT += time() - t0
            del spn


        if 'SA' in self.needed_R_matrices:
            siu = SIU(seedname)
            t0 = time()
            self.set_R_mat('SA', fourier_q_to_R_loc(chk.get_SA_q(siu, mmn)) )
            timeFFT += time() - t0
            del siu

        if 'SHA' in self.needed_R_matrices:
            shu = SHU(seedname)
            t0 = time()
            self.set_R_mat('SHA', fourier_q_to_R_loc(chk.get_SHA_q(shu, mmn)) )
            timeFFT += time() - t0
            del shu

        print("time for FFT_q_to_R : {} s".format(timeFFT))

        self.do_at_end_of_init()
        print("Real-space lattice:\n", self.real_lattice)

        #########
        # Oscar #
        #######################################################################

        # Perform the b-sums after the ws_dist is applied for Jae-Mo's scheme, adding the phase factors and the recentered matrices
        if transl_inv_JM:
            t0 = time()
            print("Completing real-space matrix elements obtained from Jae-Mo's approach...")

            # Basic quantities to simplify notation
            num_wann = self.num_wann                  # Number of Wannier functions
            wc_cart = self.wannier_centers_cart       # Wannier centers in cartesian coordinates
            nRvec = self.nRvec                        # Number of R vectors after ws_dist
            iRvec = self.iRvec                        # List of R vector indices after ws_dist
            cRvec = self.cRvec                        # List of R vector cartesian coordinates after ws_dist
            NNB = self.NNB                            # Number of nearest-neighbor b vectors
            bk_latt_unique = self.bk_latt_unique      # List of nearest-neighbor b vectors

            # A_a(R) matrix
            AA_R = np.zeros((num_wann, num_wann, nRvec, 3), dtype=complex)
            AA_Rb = self.get_R_mat('AA')

            for iw in range(num_wann):
                for jw in range(num_wann):
                    for iR in range(nRvec):
                        for ib in range(NNB):
                            phase = np.exp(-2.j * np.pi * np.dot(bk_latt_unique[ib, :], iRvec[iR, :]) / 2)
                            AA_R[iw, jw, iR] += AA_Rb[iw, jw, iR, ib] * phase

            self.set_R_mat('AA', AA_R, reset=True)

            # B_a(R) matrix
            BB_R = np.zeros((num_wann, num_wann, nRvec, 3), dtype=complex)
            BB_Rb = self.get_R_mat('BB')

            BB_R_rc = np.zeros((num_wann, num_wann, nRvec, 3), dtype=complex)  # Recentered B_a(R) matrix
            for iw in range(num_wann):
                for jw in range(num_wann):
                    for iR in range(nRvec):
                        for ib in range(NNB):
                            phase = np.exp(-2j * np.pi * np.dot(bk_latt_unique[ib, :], iRvec[iR, :]) / 2)
                            BB_R_rc[iw, jw, iR, :] += BB_Rb[iw, jw, iR, ib, :] * phase

            HH_R = self.get_R_mat('Ham')  # Hamiltonian matrix in the new real-space mesh
            for iw in range(num_wann):    # Matrices relating recentered B_a(R) matrix to original B_a(R) matrix
                for jw in range(num_wann):
                    for iR in range(nRvec):
                        rc = cRvec[iR, :] + wc_cart[jw, :] - wc_cart[iw, :]

                        rc_to_H = -0.5 * rc * HH_R[iw, jw, iR]

                        BB_R[iw, jw, iR] = BB_R_rc[iw, jw, iR] + rc_to_H

            self.set_R_mat('BB', BB_R, reset=True)

            # C_a(R) matrix
            CC_R = np.zeros((num_wann, num_wann, nRvec, 3), dtype=complex)
            CC_Rb = self.get_R_mat('CC')

            CC_R_rc = np.zeros((num_wann, num_wann, nRvec, 3), dtype=complex)  # Recentered C_a(R) matrix
            for iw in range(num_wann):
                for jw in range(num_wann):
                    for iR in range(nRvec):
                        for ib1 in range(NNB):
                            for ib2 in range(NNB):
                                phase_b1 = np.exp(-2j * np.pi * np.dot(bk_latt_unique[ib1, :], iRvec[iR, :]) / 2)
                                phase_b2 = np.exp(-2j * np.pi * np.dot(bk_latt_unique[ib2, :], iRvec[iR, :]) / 2)
                                CC_R_rc[iw, jw, iR, :] += CC_Rb[iw, jw, iR, ib1, ib2, :] * phase_b1 * phase_b2

            HH_R = self.get_R_mat('Ham')  # Matrices relating recentered C_a(R) matrix to original C_a(R) matrix
            for iw in range(num_wann):
                for jw in range(num_wann):
                    for iR in range(nRvec):
                        # Find the inverse R vector
                        for jR in range(nRvec):
                            if all(iRvec[iR] == -iRvec[jR]):
                                iRinv = jR

                        rc_1 = cRvec[iR, :] + wc_cart[jw, :] - wc_cart[iw, :]
                        rc_2 = (wc_cart[jw, :, None] - wc_cart[iw, :, None]) * cRvec[iR, None, :]

                        rc_to_H = 0.5j * rc_1[alpha_A] * (BB_R_rc[iw, jw, iR, beta_A] + BB_R_rc[jw, iw, iRinv, beta_A].conj())
                        rc_to_H -= 0.5j * rc_1[beta_A] * (BB_R_rc[iw, jw, iR, alpha_A] + BB_R_rc[jw, iw, iRinv, alpha_A].conj())
                        rc_to_H += 0.5j * rc_2[alpha_A, beta_A] * HH_R[iw, jw, iR]
                        rc_to_H -= 0.5j * rc_2[beta_A, alpha_A] * HH_R[iw, jw, iR]

                        CC_R[iw, jw, iR] = CC_R_rc[iw, jw, iR] + rc_to_H

            self.set_R_mat('CC', CC_R, reset=True)

            # G_bc(R) matrix
            GG_R = np.zeros((num_wann, num_wann, nRvec, 3, 3), dtype=complex)
            GG_Rb = self.get_R_mat('GG')

            GG_R_rc = np.zeros((num_wann, num_wann, nRvec, 3, 3), dtype=complex)  # Recentered F_bc(R) matrix
            for iw in range(num_wann):
                for jw in range(num_wann):
                    for iR in range(nRvec):
                        for ib1 in range(NNB):
                            for ib2 in range(NNB):
                                phase_b1 = np.exp(-2j * np.pi * np.dot(bk_latt_unique[ib1, :],iRvec[iR, :]) / 2)
                                phase_b2 = np.exp(-2j * np.pi * np.dot(bk_latt_unique[ib2, :],iRvec[iR, :]) / 2)
                                GG_R_rc[iw, jw, iR, :, :] += GG_Rb[iw, jw, iR, ib1, ib2, :, :] * phase_b1 * phase_b2

            # Matrices relating recentered F_bc(R) matrix to original F_bc(R) matrix
            for iw in range(num_wann):
                for jw in range(num_wann):
                    for iR in range(nRvec):
                        rc = cRvec[iR, :] + wc_cart[jw, :] - wc_cart[iw, :]

                        rc_to_H = -0.5 * AA_R[iw, jw, iR, :, None] * rc[None, :]
                        rc_to_H += 0.5 * rc[:, None] * AA_R[iw, jw, iR, None, :]

                        GG_R[iw, jw, iR] = GG_R_rc[iw, jw, iR] + rc_to_H

            self.set_R_mat('GG', GG_R, reset=True)

            print(time()-t0)

    ###########################################################################

    def wigner_seitz(self, mp_grid):
        ws_search_size = np.array([1] * 3)
        dist_dim = np.prod((ws_search_size + 1) * 2 + 1)
        origin = divmod((dist_dim + 1), 2)[0] - 1
        real_metric = self.real_lattice.dot(self.real_lattice.T)
        mp_grid = np.array(mp_grid)
        irvec = []
        ndegen = []
        for n in iterate3dpm(mp_grid * ws_search_size):
            dist = []
            for i in iterate3dpm((1, 1, 1) + ws_search_size):
                ndiff = n - i * mp_grid
                dist.append(ndiff.dot(real_metric.dot(ndiff)))
            dist_min = np.min(dist)
            if abs(dist[origin] - dist_min) < 1.e-7:
                irvec.append(n)
                ndegen.append(np.sum(abs(dist - dist_min) < 1.e-7))

        return np.array(irvec), np.array(ndegen)
