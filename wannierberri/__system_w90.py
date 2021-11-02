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
        self.mp_grid=chk.mp_grid
        self.iRvec,self.Ndegen=self.wigner_seitz(chk.mp_grid)
        self.nRvec0=len(self.iRvec)
        self.num_wann=chk.num_wann
        self.wannier_centers_cart_auto = chk.wannier_centers

        if  self.use_ws:
            print ("using ws_distance")
            ws_map=ws_dist_map(self.iRvec,chk.wannier_centers, chk.mp_grid,self.real_lattice, npar=npar)
        
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
                assert( np.all(abs(wannier_centers_cart_new-self.wannier_centers_cart))<1e-6)


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
        if  self.use_ws:
            for X in ['Ham','AA','BB','CC','SS','FF','SA','SHA','SR','SH','SHR']:
                XR=X+'_R'
                if hasattr(self,XR) :
                    print ("using ws_dist for {}".format(XR))
                    vars(self)[XR]=ws_map(vars(self)[XR])
            self.iRvec=np.array(ws_map._iRvec_ordered,dtype=int)

        self.do_at_end_of_init()
        print ("Real-space lattice:\n",self.real_lattice)

    @property
    def NKFFT_recommended(self):
        return self.mp_grid

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


class ws_dist_map():

    def __init__(self,iRvec,wannier_centers, mp_grid,real_lattice,npar=multiprocessing.cpu_count()):
    ## Find the supercell translation (i.e. the translation by a integer number of
    ## supercell vectors, the supercell being defined by the mp_grid) that
    ## minimizes the distance between two given Wannier functions, i and j,
    ## the first in unit cell 0, the other in unit cell R.
    ## I.e., we find the translation to put WF j in the Wigner-Seitz of WF i.
    ## We also look for the number of equivalent translation, that happen when w_j,R
    ## is on the edge of the WS of w_i,0. The results are stored 
    ## a dictionary shifts_iR[(iR,i,j)]
        t0=time()
        ws_search_size=np.array([2]*3)
        ws_distance_tol=1e-5
        cRvec=iRvec.dot(real_lattice)
        mp_grid=np.array(mp_grid)
        shifts_int_all = np.array([ijk  for ijk in iterate3dpm(ws_search_size+1)])*np.array(mp_grid[None,:])
        self.num_wann=wannier_centers.shape[0]
        self._iRvec_new=dict()
        param=(shifts_int_all,wannier_centers,real_lattice, ws_distance_tol, wannier_centers.shape[0])
        p=multiprocessing.Pool(npar)
        t1=time()
        irvec_new_all=p.starmap(functools.partial(ws_dist_stars,param=param),zip(iRvec,cRvec))
        print('irvec_new_all shape',np.shape(irvec_new_all))
        t2=time()
        for ir,iR in enumerate(iRvec):
          for ijw,irvec_new in irvec_new_all[ir].items():
              self._add_star(ir,irvec_new,ijw[0],ijw[1])
        t3=time()
        self._iRvec_ordered=sorted(self._iRvec_new)
        for ir,R  in enumerate(iRvec):
            chsum=0
            for irnew in self._iRvec_new:
                if ir in self._iRvec_new[irnew]:
                    chsum+=self._iRvec_new[irnew][ir]
            chsum=np.abs(chsum-np.ones( (self.num_wann,self.num_wann) )).sum() 
            if chsum>1e-12: print ("WARNING: Check sum for {0} : {1}".format(ir,chsum))
        t4=time()
        print ("time for ws_dist_map : ",t4-t0, t1-t0,t2-t1,t3-t2,t4-t3)


    def __call__(self,matrix):
        ndim=len(matrix.shape)-3
        num_wann=matrix.shape[0]
        reshaper=(num_wann,num_wann)+(1,)*ndim
        matrix_new=np.array([ sum(matrix[:,:,ir]*self._iRvec_new[irvecnew][ir].reshape(reshaper)
                                  for ir in self._iRvec_new[irvecnew] ) 
                                       for irvecnew in self._iRvec_ordered]).transpose( (1,2,0)+tuple(range(3,3+ndim)) )
        assert ( np.abs(matrix_new.sum(axis=2)-matrix.sum(axis=2)).max()<1e-12)
        return matrix_new

    def _add_star(self,ir,irvec_new,iw,jw):
        weight=1./irvec_new.shape[0]
        for irv in irvec_new:
            self._add(ir,irv,iw,jw,weight)


    def _add(self,ir,irvec_new,iw,jw,weight):
        irvec_new=tuple(irvec_new)
        if not (irvec_new in self._iRvec_new):
             self._iRvec_new[irvec_new]=dict()
        if not ir in self._iRvec_new[irvec_new]:
             self._iRvec_new[irvec_new][ir]=np.zeros((self.num_wann,self.num_wann),dtype=float)
        self._iRvec_new[irvec_new][ir][iw,jw]+=weight



def ws_dist_stars(iRvec,cRvec,param):
          shifts_int_all,wannier_centers,real_lattice, ws_distance_tol, num_wann = param
          irvec_new={}
          for jw in range(num_wann):
            for iw in range(num_wann):
              # function JW translated in the Wigner-Seitz around function IW
              # and also find its degeneracy, and the integer shifts needed
              # to identify it
              R_in=-wannier_centers[iw] +cRvec + wannier_centers[jw]
              dist=np.linalg.norm( R_in[None,:]+shifts_int_all.dot(real_lattice),axis=1)
              irvec_new[(iw,jw)]=iRvec+shifts_int_all[ dist-dist.min() < ws_distance_tol ].copy()
          return irvec_new



