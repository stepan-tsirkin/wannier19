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
#                                                            #
# ------------------------------------------------------------

import numpy as np

from ..__utility import str2bool, real_recip_lattice
from termcolor import cprint
from .system_R import System_R
from collections import defaultdict
from scipy.constants import physical_constants, angstrom

bohr = physical_constants['Bohr radius'][0] / angstrom


class System_fplo(System_R):
    """
    System initialized from the `+hamdata` file written by `FPLO <https://www.fplo.de/>`__ code,

    Parameters
    ----------
    hamdata : str
        name (and path) of the "+hamdata" file to be read
    mp_grid : [nk1,nk2,nk3]
        size of Monkhorst-Pack frid used in ab initio calculation. Needed when use_ws=True, which is default and highly
        recommended


    Notes
    -----
    see also  parameters of the :class:`~wannierberri.System`
    """

    def __init__(self, hamdata="+hamdata",
                 mp_grid=None,
                 **parameters):

        super().__init__(**parameters)
        if not self.use_wcc_phase:
            print(
                "WARNING: It is highly recommended to use `use_wcc_phase=True` with System_fplo"
                ", and further set parameters={`external_terms':False}"
                "in any case, the external terms are evaluated using the diagonal approximation for position matrix elements (Tight-binding-like)"
            )

        self.seedname = hamdata.split("/")[-1].split("_")[0]
        f = open(hamdata, "r")
        allread = False
        while not allread:
            l = next(f)
            if l.startswith("end spin:"):
                break
            elif l.startswith("lattice_vectors:"):
                real_lattice_bohr = np.array([next(f).split() for _ in range(3)], dtype=float)
                inv_real_lattice = np.linalg.inv(real_lattice_bohr)
            elif l.startswith("nwan:"):
                self.num_wann = int(next(f))
            elif l.startswith("nspin:"):
                nspin = int(next(f))
                assert nspin == 1, "spin-polarized calculations arte not supported yeet"
            elif l.startswith("have_spin_info:"):
                have_spin = str2bool(next(f))
                if (not have_spin) and self.spin:
                    raise ValueError("spin info required, but not contained in the file")
            elif l.startswith("wancenters:"):
                self.wannier_centers_cart = np.array([next(f).split() for i in range(self.num_wann)], dtype=float)
            elif l.startswith("spin:"):
                ispin = int(next(f))
                assert ispin == 1, f"spin = 1 expected, got {ispin}"
                Ham_R = defaultdict(lambda: np.zeros((self.num_wann, self.num_wann), dtype=complex))
                if self.need_R_any('SS'):
                    SS_R = defaultdict(lambda: np.zeros((self.num_wann, self.num_wann, 3), dtype=complex))
                while True:
                    l = next(f)
                    if l.startswith("end spin:"):
                        allread = True
                        break
                    if l.startswith("Tij, Hij"):
                        iw, jw = [int(x) for x in next(f).split()]
                        iw -= 1
                        jw -= 1
                        arread = []
                        while True:
                            l = next(f)
                            if l.startswith("end Tij, Hij"):
                                break
                            arread.append(l.split())
                        if len(arread) == 0:
                            continue
                        arread = np.array(arread, dtype=float)
                        Rvec = arread[:, :3] + (
                                self.wannier_centers_cart[None, iw] - self.wannier_centers_cart[None, jw])
                        Rvec = Rvec.dot(inv_real_lattice)  # should be integer now
                        iRvec = np.array(np.round(Rvec), dtype=int)
                        assert (abs(iRvec - Rvec).max() < 1e-8)
                        iRvec = [tuple(ir) for ir in iRvec]
                        for iR, a in zip(iRvec, arread):
                            Ham_R[iR][iw, jw] = a[3] + 1j * a[4]
                            if self.need_R_any('SS'):
                                SS_R[iR][iw, jw, :] = a[5:11:2] + 1j * a[6:11:2]
        f.close()
        # Reading of file finished

        self.real_lattice, self.recip_lattice = real_recip_lattice(real_lattice=real_lattice_bohr * bohr)
        iRvec = list(Ham_R.keys())
        self.set_R_mat('Ham', np.array([Ham_R[iR] for iR in iRvec]).transpose((1, 2, 0)))
        if self.need_R_any('SS'):
            self.set_R_mat('SS', np.array([SS_R[iR] for iR in iRvec]).transpose((1, 2, 0, 3)))

        self.nRvec0 = len(iRvec)
        self.iRvec = np.array(iRvec, dtype=int)

        self.getXX_only_wannier_centers(getSS=False)

        self.do_at_end_of_init()
        if self.use_ws:
            self.do_ws_dist(mp_grid=mp_grid)

        cprint("Reading the FPLO Wannier system from {} finished successfully".format(hamdata), 'green', attrs=['bold'])
