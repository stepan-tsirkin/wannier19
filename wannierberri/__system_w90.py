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
import copy
import lazy_property
import functools
import multiprocessing 
#from pathos.multiprocessing import ProcessingPool as Pool
from .__utility import str2bool, alpha_A, beta_A, iterate3dpm, real_recip_lattice,fourier_q_to_R
from colorama import init
from termcolor import cprint 
from .__system import System
from .__w90_files import EIG,MMN,CheckPoint,SPN,UHU,SIU,SHU
from time import time
import pickle
from itertools import repeat
from .__sym_wann import sym_wann
np.set_printoptions(precision=4,threshold=np.inf,linewidth=500)


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
    npar : int
        number of processes used in the constructor
    fft : str
        library used to perform the fast Fourier transform from **q** to **R**. ``fftw`` or ``numpy``. (practically does not affect performance, 
        anyway mostly time of the constructor is consumed by reading the input files)

    Notes
    -----
    see also  parameters of the :class:`~wannierberri.System` 
    """

    def __init__(self,seedname="wannier90",
                    transl_inv=True,
                    fft='fftw',
                    npar=multiprocessing.cpu_count()  , 
                    **parameters
                    ):

        self.set_parameters(**parameters)
        self.seedname=seedname

        chk=CheckPoint(self.seedname)
        self.real_lattice,self.recip_lattice=real_recip_lattice(chk.real_lattice,chk.recip_lattice)
        if self.mp_grid is None:
            self.mp_grid=chk.mp_grid
        self.iRvec,self.Ndegen=self.wigner_seitz(chk.mp_grid)
        self.nRvec0=len(self.iRvec)
        self.num_wann=chk.num_wann
        self.wannier_centers_cart_auto = chk.wannier_centers

        
        eig=EIG(seedname)
        if self.getAA or self.getBB:
            mmn=MMN(seedname,npar=npar)

        kpt_mp_grid=[tuple(k) for k in np.array( np.round(chk.kpt_latt*np.array(chk.mp_grid)[None,:]),dtype=int)%chk.mp_grid]
        if (0,0,0) not in kpt_mp_grid:
            raise ValueError("the grid of k-points read from .chk file is not Gamma-centered. Please, use Gamma-centered grids in the ab initio calculation")
        
        fourier_q_to_R_loc=functools.partial(fourier_q_to_R, mp_grid=chk.mp_grid,kpt_mp_grid=kpt_mp_grid,iRvec=self.iRvec,ndegen=self.Ndegen,numthreads=npar,fft=fft)

        timeFFT=0
        HHq=chk.get_HH_q(eig)
        t0=time()
        self.Ham_R=fourier_q_to_R_loc( HHq )
        timeFFT+=time()-t0

        if self.getAA:
            AAq=chk.get_AA_q(mmn,transl_inv=transl_inv)
            t0=time()
            self.AA_R=fourier_q_to_R_loc(AAq)
            timeFFT+=time()-t0
            if transl_inv:
                wannier_centers_cart_new = np.diagonal(self.AA_R[:,:,self.iR0,:],axis1=0,axis2=1).transpose()
                assert np.all(abs(wannier_centers_cart_new-self.wannier_centers_cart_auto)<1e-6), (
                  "the difference between read\n{}\n and evluated \n{}\n wannier centers is \n{}\n".format(self.wannier_centers_cart_auto,
                        wannier_centers_cart_new,self.wannier_centers_cart_auto-wannier_centers_cart_new))

        if self.getBB:
            t0=time()
            self.BB_R=fourier_q_to_R_loc(chk.get_AA_q(mmn,eig))
            timeFFT+=time()-t0

        if self.getCC:
            uhu=UHU(seedname)
            t0=time()
            self.CC_R=fourier_q_to_R_loc(chk.get_CC_q(uhu,mmn))
            timeFFT+=time()-t0
            del uhu

        if self.getSS:
            spn=SPN(seedname)
            t0=time()
            self.SS_R=fourier_q_to_R_loc(chk.get_SS_q(spn))
            if self.getSHC:
                self.SR_R=fourier_q_to_R_loc(chk.get_SR_q(spn,mmn))
                self.SH_R=fourier_q_to_R_loc(chk.get_SH_q(spn,eig))
                self.SHR_R=fourier_q_to_R_loc(chk.get_SHR_q(spn,mmn,eig))
            timeFFT+=time()-t0
            del spn

        if self.getSA:
            siu=SIU(seedname)
            t0=time()
            self.SA_R=fourier_q_to_R_loc(chk.get_SA_q(siu,mmn))
            timeFFT+=time()-t0
            del siu

        if self.getSHA:
            shu=SHU(seedname)
            t0=time()
            self.SHA_R=fourier_q_to_R_loc(chk.get_SHA_q(shu,mmn))
            timeFFT+=time()-t0
            del shu

        print ("time for FFT_q_to_R : {} s".format(timeFFT))
        
        if self.symmetrization:
            XX_R={'Ham':self.Ham_R}
            for X in ['AA','BB','CC','SS','FF','SA','SHA','SR','SH','SHR']:
                try:
                    XX_R[X] = vars(self)[X+'_R']
                except KeyError:
                    pass
            symmetrize_wann = sym_wann(num_wann=self.num_wann,lattice=self.real_lattice,positions=self.positions,atom_name=self.atom_name,
                proj=self.proj,iRvec=self.iRvec,XX_R=XX_R,spin=self.soc,magmom=self.magmom)
            XX_R,self.iRvec = symmetrize_wann.symmetrize()
            for X in ['Ham','AA','BB','CC','SS','FF','SA','SHA','SR','SH','SHR']:
                try:
                    vars(self)[X+'_R'] = XX_R[X]
                except KeyError:
                    pass

        self.do_at_end_of_init()
        print ("Real-space lattice:\n",self.real_lattice)

    def wigner_seitz(self,mp_grid):
        ws_search_size=np.array([1]*3)
        dist_dim=np.prod((ws_search_size+1)*2+1)
        origin=divmod((dist_dim+1),2)[0]-1
        real_metric=self.real_lattice.dot(self.real_lattice.T)
        mp_grid=np.array(mp_grid)
        irvec=[]
        ndegen=[]
        for n in iterate3dpm(mp_grid*ws_search_size):
            dist=[]
            for i in iterate3dpm((1,1,1)+ws_search_size):
                ndiff=n-i*mp_grid
                dist.append(ndiff.dot(real_metric.dot(ndiff)))
            dist_min = np.min(dist)
            if  abs(dist[origin] - dist_min) < 1.e-7 :
                irvec.append(n)
                ndegen.append(np.sum( abs(dist - dist_min) < 1.e-7 ))
    
        return np.array(irvec),np.array(ndegen)


