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
import functools
import multiprocessing
from ..__utility import iterate3dpm, real_recip_lattice, fourier_q_to_R, alpha_A, beta_A
from .system import System
from .w90_files import Wannier90data


class System_w90(System):
    """
    System initialized from the Wannier functions generated by `Wannier90 <http://wannier.org>`__ code.
    Reads the ``.chk``, ``.eig`` and optionally ``.mmn``, ``.spn``, ``.uHu``, ``.sIu``, and ``.sHu`` files

    Parameters
    ----------
    seedname : str
        the seedname used in Wannier90
    w90data : `~wannierberri.system.Wannier90data`
        object that contains all Wanier90 input files and chk all together. If provided, overrides the `seedname`
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
            w90data=None,
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
        if w90data is None:
            w90data = Wannier90data(self.seedname, read_chk=True, kmesh_tol=kmesh_tol, bk_complete_tol=bk_complete_tol)
        w90data.check_wannierised(msg="creation of System_Wannierise")
        chk = w90data.chk
        self.real_lattice, self.recip_lattice = real_recip_lattice(chk.real_lattice, chk.recip_lattice)
        if self.mp_grid is None:
            self.mp_grid = chk.mp_grid
        self.iRvec, self.Ndegen = self.wigner_seitz(chk.mp_grid)
        self.nRvec0 = len(self.iRvec)
        self.num_wann = chk.num_wann
        self.wannier_centers_cart_auto = w90data.wannier_centers

        #########
        # Oscar #
        #######################################################################

        # Deactivate transl_inv if Jae-Mo's scheme is used
        if self.transl_inv_JM:
            if transl_inv:
                print("WARNING : Jae-Mo's scheme does not apply Marzari & Vanderbilt formula for"
                      "the band-diagonal matrix elements of the position operator.")
                transl_inv = False

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

        self.do_at_end_of_init()
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
        # Optimal center in Jae-Mo's implementation
        r0 = 0.5 * (centers[:,None,None,:] + centers[None,:,None,:] + self.cRvec[None,None,:,:])
        # Unique set of nearest-neighbor vectors (cartesian)
        bk_cart_unique = w90data.mmn.bk_cart_unique

        # --- A_a(R) matrix --- #
        if self.need_R_any('AA'):
            AA_Rb = self.get_R_mat('AA')

            # Naive finite-difference scheme
            if not self.transl_inv_JM:
                if not self.use_wcc_phase: # Phase convention II
                    pass
                else:                      # Phase convention I
                    phase  = np.einsum('ba,ja->jb', bk_cart_unique, centers)
                    AA_Rb *= np.exp(1.j * phase[None,:,None,:,None])

                AA_R = np.sum(AA_Rb, axis=3)

                # Hermiticity is not preserved, but it can be enforced
                AA_R = 0.5 * (AA_R + self.conj_XX_R(AA_R))

            # Jae-Mo's finite-difference scheme
            else:
                # Recentered matrix
                phase  = np.einsum('ba,ijRa->ijRb', bk_cart_unique, r0 - self.cRvec[None,None,:,:])
                AA_R0b = AA_Rb * np.exp(1.j * phase[:,:,:,:,None])
                AA_R0  = np.sum(AA_R0b, axis=3)

                # Original matrix
                AA_R = AA_R0
                if not self.use_wcc_phase: # Phase convention II
                    AA_R[range(self.num_wann),range(self.num_wann),self.iR0] += centers
                else:                      # Phase convention I
                    pass

            self.set_R_mat('AA', AA_R, reset=True)

            # Check wannier centers if Marzari & Vanderbilt formula is used
            if (transl_inv and not self.use_wcc_phase):
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

        # --- B_a(R) matrix --- #
        if 'BB' in self.needed_R_matrices:
            BB_Rb = self.get_R_mat('BB')

            # Naive finite-difference scheme
            if not self.transl_inv_JM:
                if not self.use_wcc_phase: # Phase convention II
                    pass
                else:                      # Phase convention I
                    phase  = np.einsum('ba,ja->jb', bk_cart_unique, centers)
                    BB_Rb *= np.exp(1.j * phase[None,:,None,:,None])

                BB_R = np.sum(BB_Rb, axis=3)

            # Jae-Mo's finite-difference scheme
            else:
                # Recentered matrix
                phase  = np.einsum('ba,ijRa->ijRb', bk_cart_unique, r0 - self.cRvec[None,None,:,:])
                BB_R0b = BB_Rb * np.exp(1.j * phase[:,:,:,:,None])
                BB_R0  = np.sum(BB_R0b, axis=3)

                # Original matrix
                HH_R = self.get_R_mat('Ham')
                if not self.use_wcc_phase: # Phase convention II
                    rc = (r0 - self.cRvec[None,None,:,:]) * HH_R[:,:,:,None]
                else:                      # Phase convention I
                    rc = (r0 - self.cRvec[None,None,:,:] - centers[None,:,None,:]) * HH_R[:,:,:,None]

                BB_R = BB_R0 + rc

            self.set_R_mat('BB', BB_R, reset=True)

        # --- C_a(R) matrix --- #
        if 'CC' in self.needed_R_matrices:
            CC_Rb = self.get_R_mat('CC')

            # Naive finite-difference scheme
            if not self.transl_inv_JM:
                if not self.use_wcc_phase: # Phase convention II
                    pass
                else:                      # Phase convention I
                    phase_1 = -np.einsum('ba,ia->ib', bk_cart_unique, centers)
                    phase_2 =  np.einsum('ba,ja->jb', bk_cart_unique, centers)
                    phase  = phase_1[:,None,:,None] + phase_2[None,:,None,:]
                    CC_Rb *= np.exp(1.j * phase[:,:,None,:,:,None])

                CC_R = np.sum(CC_Rb, axis=(3,4))

                # Hermiticity is not preserved, but it can be enforced
                CC_R = 0.5 * (CC_R + self.conj_XX_R(CC_R))

            # Jae-Mo's finite-difference scheme
            else:
                # Recentered matrix
                phase_1 = -np.einsum('ba,ijRa->ijRb', bk_cart_unique, r0)
                phase_2 =  np.einsum('ba,ijRa->ijRb', bk_cart_unique, r0 - self.cRvec[None,None,:,:])
                phase  = phase_1[:,:,:,:,None] + phase_2[:,:,:,None,:]
                CC_R0b = CC_Rb * np.exp(1.j * phase[:,:,:,:,:,None])
                CC_R0  = np.sum(CC_R0b, axis=(3,4))

                # Original matrix
                if BB_R0 is None:
                    raise ValueError('Recentered B matrix is needed in Jae-Mo`s implementation of C')
                BB_R0_conj = self.conj_XX_R(BB_R0)
                if not self.use_wcc_phase: # Phase convention II
                    rc  = 0.5j * r0[:,:,:,:,None] * BB_R0[:,:,:,None,:]
                    rc -= 0.5j * (r0[:,:,:,:,None] - self.cRvec[None,None,:,:,None]) *  BB_R0_conj[:,:,:,None,:]
                    rc -= 0.5j * (centers[:,None,None,:,None] + centers[None,:,None,:,None]) * self.cRvec[None,None,:,None,:] * HH_R[:,:,:,None,None]
                else:                      # Phase convention I
                    rc   = 0.5j * (r0[:,:,:,:,None] - centers[:,None,None,:,None]) * (BB_R0 + BB_R0_conj)[:,:,:,None,:]

                CC_R = CC_R0 + rc[:,:,:,alpha_A,beta_A] - rc[:,:,:,beta_A,alpha_A]

            self.set_R_mat('CC', CC_R, reset=True)

        # --- O_a(R) matrix --- #
        if 'OO' in self.needed_R_matrices:
            OO_Rb = self.get_R_mat('OO')

            # Naive finite-difference scheme
            if not self.transl_inv_JM:
                if not self.use_wcc_phase: # Phase convention II
                    pass
                else:                      # Phase convention I
                    phase_1 = -np.einsum('ba,ia->ib', bk_cart_unique, centers)
                    phase_2 =  np.einsum('ba,ja->jb', bk_cart_unique, centers)
                    phase  = phase_1[:,None,:,None] + phase_2[None,:,None,:]
                    OO_Rb *= np.exp(1.j * phase[:,:,None,:,:,None])

                OO_R = np.sum(OO_Rb, axis=(3,4))

                # Hermiticity is not preserved, but it can be enforced
                OO_R = 0.5 * (OO_R + self.conj_XX_R(OO_R))

            # Jae-Mo's finite-difference scheme
            else:
                # Recentered matrix
                phase_1 = -np.einsum('ba,ijRa->ijRb', bk_cart_unique, r0)
                phase_2 =  np.einsum('ba,ijRa->ijRb', bk_cart_unique, r0 - self.cRvec[None,None,:,:])
                phase  = phase_1[:,:,:,:,None] + phase_2[:,:,:,None,:]
                OO_R0b = OO_Rb * np.exp(1.j * phase[:,:,:,:,:,None])
                OO_R0  = np.sum(OO_R0b, axis=(3,4))

                # Original matrix
                if AA_R0 is None:
                    raise ValueError('Recentered A matrix is needed in Jae-Mo`s implementation of O')
                if not self.use_wcc_phase: # Phase convention II
                    rc = 1.j * self.cRvec[None,None,:,:,None] * AA_R0[:,:,:,None,:]
                else:                      # Phase convention I
                    rc = 1.j * (r0[:,:,:,:,None] - centers[:,None,None,:,None]) * AA_R0[:,:,:,None,:]

                OO_R = OO_R0 + rc[:,:,:,alpha_A,beta_A] - rc[:,:,:,beta_A,alpha_A]

            self.set_R_mat('OO', OO_R, reset=True)

        # --- G_bc(R) matrix --- #
        if 'GG' in self.needed_R_matrices:
            GG_Rb = self.get_R_mat('GG')

            # Naive finite-difference scheme
            if not self.transl_inv_JM:
                if not self.use_wcc_phase: # Phase convention II
                    pass
                else:                      # Phase convention I
                    phase_1 = -np.einsum('ba,ia->ib', bk_cart_unique, centers)
                    phase_2 =  np.einsum('ba,ja->jb', bk_cart_unique, centers)
                    phase  = phase_1[:,None,:,None] + phase_2[None,:,None,:]
                    GG_Rb *= np.exp(1.j * phase[:,:,None,:,:,None,None])

                GG_R = np.sum(GG_Rb, axis=(3,4))

                # Hermiticity is not preserved, but it can be enforced
                GG_R = 0.5 * (GG_R + self.conj_XX_R(GG_R))

            # Jae-Mo's finite-difference scheme
            else:
                # Recentered matrix
                phase_1 = -np.einsum('ba,ijRa->ijRb', bk_cart_unique, r0)
                phase_2 =  np.einsum('ba,ijRa->ijRb', bk_cart_unique, r0 - self.cRvec[None,None,:,:])
                phase  = phase_1[:,:,:,:,None] + phase_2[:,:,:,None,:]
                GG_R0b = GG_Rb * np.exp(1.j * phase[:,:,:,:,:,None,None])
                GG_R0  = np.sum(GG_R0b, axis=(3,4))

                # Original matrix
                if AA_R0 is None:
                    raise ValueError('Recentered A matrix is needed in Jae-Mo`s implementation of G')
                if not self.use_wcc_phase: # Phase convention II
                    rc = (centers[:,None,None,:,None] + centers[None,:,None,:,None]) * AA_R0[:,:,:,None,:]
                    rc[range(self.num_wann),range(self.num_wann),self.iR0] += centers[range(self.num_wann),:,None] * centers[range(self.num_wann),None,:]
                else:                      # Phase convention I
                    rc = np.zeros((self.num_wann,self.num_wann,self.nRvec,3,3), dtype=complex)

                GG_R = GG_R0 + 0.5 * (rc + rc.swapaxes(3,4))

            self.set_R_mat('GG', GG_R, reset=True)

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
