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
#------------------------------------------------------------

import numpy as np
import copy

from .__utility import str2bool, alpha_A, beta_A ,real_recip_lattice
from colorama import init
from termcolor import cprint 
from .__system import System
from .__sym_wann import sym_wann
import pickle

class System_tb(System):
    """
    System initialized from the `*_tb.dat` file, which can be written either by  `Wannier90 <http://wannier.org>`__ code, 
    or composed by the user based on some tight-binding model. 
    See Wannier90 `code <https://github.com/wannier-developers/wannier90/blob/2f4aed6a35ab7e8b38dbe196aa4925ab3e9deb1b/src/hamiltonian.F90#L698-L799>`_
    for details of the format. 
    
    Parameters
    ----------
    tb_file : str
        name (and path) of file to be read

    Notes
    -----
    see also  parameters of the :class:`~wannierberri.System` 
    """

    def __init__(self,tb_file="wannier90_tb.dat",**parameters):

        self.set_parameters(**parameters)
        if self.morb : raise ValueError("System_tb class cannot be used for evaluation of orbital magnetic moments")
        if self.spin : raise ValueError("System_tb class cannot be used for evaluation of spin properties")
 
        self.seedname=tb_file.split("/")[-1].split("_")[0]
        f=open(tb_file,"r")
        l=f.readline()
        cprint ("reading TB file {0} ( {1} )".format(tb_file,l.strip()),'green', attrs=['bold'])
        real_lattice=np.array([f.readline().split()[:3] for i in range(3)],dtype=float)
        self.real_lattice,self.recip_lattice= real_recip_lattice(real_lattice=real_lattice)
        self.num_wann=int(f.readline())
        nRvec=int(f.readline())
        self.nRvec0=nRvec
        self.Ndegen=[]
        while len(self.Ndegen)<nRvec:
            self.Ndegen+=f.readline().split()
        self.Ndegen=np.array(self.Ndegen,dtype=int)
        
        self.iRvec=[]
        
        self.Ham_R=np.zeros( (self.num_wann,self.num_wann,nRvec) ,dtype=complex)
        
        for ir in range(nRvec):
            f.readline()
            self.iRvec.append(f.readline().split())
            hh=np.array( [[f.readline().split()[2:4] 
                             for n in range(self.num_wann)] 
                                for m in range(self.num_wann)],dtype=float).transpose( (1,0,2) )
            self.Ham_R[:,:,ir]=(hh[:,:,0]+1j*hh[:,:,1])/self.Ndegen[ir]
        
        self.iRvec=np.array(self.iRvec,dtype=int)

        if self.getAA:
            self.AA_R=np.zeros( (self.num_wann,self.num_wann,nRvec,3) ,dtype=complex)
            for ir in range(nRvec):
                f.readline()
                assert (np.array(f.readline().split(),dtype=int)==self.iRvec[ir]).all()
                aa=np.array( [[f.readline().split()[2:8] 
                             for n in range(self.num_wann)] 
                                for m in range(self.num_wann)],dtype=float)
                self.AA_R[:,:,ir,:]=(aa[:,:,0::2]+1j*aa[:,:,1::2]).transpose( (1,0,2) ) /self.Ndegen[ir]
            self.wannier_centers_cart_auto =  np.diagonal(self.AA_R[:,:,self.iR0,:],axis1=0,axis2=1).T 
        else: 
            self.AA_R = None
        f.close()
        pickle.dump(self.AA_R,open("AA.pickle","wb"))
        pickle.dump(self.HH_R,open("HH.pickle","wb"))
        pickle.dump(self.iRvec,open("iRvec.pickle","wb"))

        if self.symmetrization: 
            XX_R={'HH':self.HH_R}
            XX_R['AA'] = self.AA_R
            symmetrize_wann = sym_wann(num_wann=self.num_wann,lattice=self.real_lattice,positions=self.positions,atom_name=self.atom_name,
                proj=self.proj,iRvec=self.iRvec,XX_R=XX_R,spin=True,TR=True)
            XX_R,self.iRvec = symmetrize_wann.symmetrize() 
            self.HH_R = XX_R['HH']
            self.AA_R = XX_R['AA']
        self.do_at_end_of_init()

        cprint ("Reading the system from {} finished successfully".format(tb_file),'green', attrs=['bold'])

        
