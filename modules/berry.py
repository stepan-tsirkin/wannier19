#------------------------------------------------------------#
# This file is distributed as part of the Wannier19 code     #
# under the terms of the GNU General Public License. See the #
# file `LICENSE' in the root directory of the Wannier19      #
# distribution, or http://www.gnu.org/copyleft/gpl.txt       #
#                                                            #
# this file initially was  an adapted translation of         #
# the corresponding Fortran90 code from  Wannier 90 project  #
#                                                            #
# with significant modifications for better performance      #
#   it is now a lot different                                #
#                                                            #
# The Wannier19 code is hosted on GitHub:                    #
# https://github.com/stepan-tsirkin/wannier19                #
#                                                            #
# The webpage of the Wannier90 code is www.wannier.org       #
# The Wannier90 code is hosted on GitHub:                    #
# https://github.com/wannier-developers/wannier90            #
#------------------------------------------------------------#
#                                                            #
#  Translated to python and adapted for wannier19 project by #
#           Stepan Tsirkin, University ofZurich              #
#                                                            #
#------------------------------------------------------------#

import numpy as np
from scipy import constants as constants
from collections import Iterable

from utility import  print_my_name_start,print_my_name_end,voidsmoother
import parent


class AHCresult(parent.Result):

    def __init__(self,Efermi,AHC,AHCsmooth=None,smoother=None):
        assert (Efermi.shape[0]==AHC.shape[0])
        assert (AHC.shape[-1]==3)
        self.Efermi=Efermi
        self.AHC=AHC
        if not (AHCsmooth is None):
            self.AHCsmooth=AHCsmooth
        elif not (smoother is None):
            self.AHCsmooth=smoother(self.AHC)
        else:
             raise ValueError("either  AHCsmooth or smoother should be defined")

    def __mul__(self,other):
        if isinstance(other,int) or isinstance(other,float) :
            return AHCresult(self.Efermi,self.AHC*other,self.AHCsmooth*other)
        else:
            raise TypeError("result can only be multilied by a number")
    
    def __add__(self,other):
        if other == 0:
            return self
        if np.linalg.norm(self.Efermi-other.Efermi)>1e-8:
            raise RuntimeError ("Adding results with different Fermi energies - not allowed")
        return AHCresult(self.Efermi,self.AHC+other.AHC,self.AHCsmooth+other.AHCsmooth)

    def write(self,name):
        open(name,"w").write(
           "    ".join("{0:^15s}".format(s) for s in ["EF",]+
                [b for b in ("x","y","z")*2])+"\n"+
          "\n".join(
           "    ".join("{0:15.6f}".format(x) for x in [ef]+[x for x in ahc]) 
                      for ef,ahc in zip (self.Efermi,np.hstack( (self.AHC,self.AHCsmooth) ) ))
               +"\n") 

    def transform(self,sym):
        return AHCresult(self.Efermi,sym.transform_axial_vector(self.AHC),sym.transform_axial_vector(self.AHCsmooth) )

    def _maxval(self):
        return self.AHCsmooth.max() 

    def _norm(self):
        return np.linalg.norm(self.AHCsmooth)

    def _normder(self):
        return np.linalg.norm(self.AHCsmooth[1:]-self.AHCsmooth[:-1])

    def max(self):
        return np.array([self._maxval(),self._norm(),self._normder()])


alpha=np.array([1,2,0])
beta =np.array([2,0,1])

#              -1.0e8_dp*elem_charge_SI**2/(hbar_SI*cell_volume)
fac_ahc  = -1.0e8*constants.elementary_charge**2/constants.hbar
bohr= constants.physical_constants['Bohr radius'][0]/constants.angstrom
eV_au=constants.physical_constants['electron volt-hartree relationship'][0] 
fac_morb =  -eV_au/bohr**2





def eval_J0(A,occ):
    return A[occ].sum(axis=0)

def eval_J12(B,UnoccOcc):
    return -2*B[UnoccOcc].sum(axis=0)

def eval_J3(B,UnoccUnoccOcc):
    return -2*B[UnoccUnoccOcc].sum(axis=0)

def get_occ(E_K,Efermi):
    return (E_K< Efermi)





def calcAHC(data,Efermi=None,occ_old=None,smoother=voidsmoother):
    if occ_old is None: 
        occ_old=np.zeros((data.NKFFT_tot,data.num_wann),dtype=bool)

    if isinstance(Efermi, Iterable):
#        print ("iterating over Fermi levels")
        nFermi=len(Efermi)
        AHC=np.zeros( ( nFermi,3) ,dtype=float )
        for iFermi in range(nFermi):
#            print ("iFermi={}".format(iFermi))
            AHC[iFermi]=calcAHC(data,Efermi=Efermi[iFermi],occ_old=occ_old)
        return AHCresult(Efermi,np.cumsum(AHC,axis=0),smoother=smoother)
    
    # now code for a single Fermi level:
    AHC=np.zeros(3)

#    print ("  calculating occ matrices")
    occ_new=get_occ(data.E_K,Efermi)
    unocc_new=np.logical_not(occ_new)
    unocc_old=np.logical_not(occ_old)
    selectK=np.where(np.any(occ_old!=occ_new,axis=1))[0]
    occ_old_selk=occ_old[selectK]
    occ_new_selk=occ_new[selectK]
    unocc_old_selk=unocc_old[selectK]
    unocc_new_selk=unocc_new[selectK]
    delocc=occ_new_selk!=occ_old_selk
    unoccocc_plus=unocc_new_selk[:,:,None]*delocc[:,None,:]
    unoccocc_minus=delocc[:,:,None]*occ_old_selk[:,None,:]
#    print ("  calculating occ matrices - done")

#    print ("evaluating J0")
    AHC=eval_J0(data.OOmegaUU_K_rediag[selectK], delocc)
#    print ("evaluating B")
    B=data.delHH_dE_AA_delHH_dE_SQ_K[selectK]
#    print ("evaluating J12")
    AHC+=eval_J12(B,unoccocc_plus)-eval_J12(B,unoccocc_minus)
#    print ("evaluating J12-done")
    occ_old[:,:]=occ_new[:,:]
    return AHC*fac_ahc/(data.NKFFT_tot*data.cell_volume)


## Not working with the new "result" class yet
def calcMorb(data,Efermi=None,occ_old=None, evalJ0=True,evalJ1=True,evalJ2=True):
    if not isinstance(Efermi, Iterable):
        Efermi=np.array([Efermi])
    imfgh=calcImfgh(data,Efermi=Efermi,occ_old=occ_old, evalJ0=evalJ0,evalJ1=evalJ1,evalJ2=evalJ2)
    imf=imfgh[:,0,:,:]
    img=imfgh[:,1,:,:]
    imh=imfgh[:,2,:,:]
    LCtil=fac_morb*(img-Efermi[:,None,None]*imf)
    ICtil=fac_morb*(imh-Efermi[:,None,None]*imf)
    Morb = LCtil + ICtil
    return np.array([Morb,LCtil,ICtil])





### routines for a band-resolved mode

def eval_Jo_deg(A,degen):
    return np.array([A[ib1:ib2].sum(axis=0) for ib1,ib2 in degen])

def eval_Juo_deg(B,degen):
    return np.array([B[:ib1,ib1:ib2].sum(axis=(0,1)) + B[ib2:,ib1:ib2].sum(axis=(0,1)) for ib1,ib2 in degen])

def eval_Joo_deg(B,degen):
    return np.array([B[ib1:ib2,ib1:ib2].sum(axis=(0,1))  for ib1,ib2 in degen])

def eval_Juuo_deg(B,degen):
    return np.array([   sum(C.sum(axis=(0,1,2)) 
                          for C in  (B[:ib1,:ib1,ib1:ib2],B[:ib1,ib2:,ib1:ib2],B[ib2:,:ib1,ib1:ib2],B[ib2:,ib2:,ib1:ib2]) )  
                                      for ib1,ib2 in degen])

def eval_Juoo_deg(B,degen):
    return np.array([   sum(C.sum(axis=(0,1,2)) 
                          for C in ( B[:ib1,ib1:ib2,ib1:ib2],B[ib2:,ib1:ib2,ib1:ib2])  )  
                                      for ib1,ib2 in degen])


def calcImf_K(data,degen_bands,ik):
    A=data.OOmegaUU_K_rediag[ik]
    B=data.delHH_dE_AA_delHH_dE_SQ_K[ik]
    return eval_Jo_deg(A,degen_bands)-2*eval_Juo_deg(B,degen_bands) 


def calcImfgh(data,Efermi=None,occ_old=None, evalJ0=True,evalJ1=True,evalJ2=True):

    if occ_old is None: 
        occ_old=np.zeros((data.NKFFT_tot,data.num_wann),dtype=bool)

    if isinstance(Efermi, Iterable):
        nFermi=len(Efermi)
        imfgh=np.zeros( ( nFermi,3,4,3) ,dtype=float )
        for iFermi in range(nFermi):
            imfgh[iFermi]=calcImfgh(data,Efermi=Efermi[iFermi],occ_old=occ_old, evalJ0=evalJ0,evalJ1=evalJ1,evalJ2=evalJ2)
        return np.cumsum(imfgh,axis=0)
    
    # now code for a single Fermi level:
    imfgh=np.zeros((3,4,3))

    occ_new=get_occ(data.E_K,Efermi)
    unocc_new=np.logical_not(occ_new)
    unocc_old=np.logical_not(occ_old)
    selectK=np.where(np.any(occ_old!=occ_new,axis=1))[0]
    occ_old_selk=occ_old[selectK]
    occ_new_selk=occ_new[selectK]
    unocc_old_selk=unocc_old[selectK]
    unocc_new_selk=unocc_new[selectK]
    delocc=occ_new_selk!=occ_old_selk
    UnoccOcc_plus=unocc_new_selk[:,:,None]*delocc[:,None,:]
    UnoccOcc_minus=delocc[:,:,None]*occ_old_selk[:,None,:]

    UnoccUnoccOcc_new=unocc_new_selk[:,:,None,None]*unocc_new_selk[:,None,:,None]*occ_new_selk[:,None,None,:]
    UnoccUnoccOcc_old=unocc_old_selk[:,:,None,None]*unocc_old_selk[:,None,:,None]*occ_old_selk[:,None,None,:]

    UnoccOccOcc_new=unocc_new_selk[:,:,None,None]*  occ_new_selk[:,None,:,None]*occ_new_selk[:,None,None,:]
    UnoccOccOcc_old=unocc_old_selk[:,:,None,None]*  occ_old_selk[:,None,:,None]*occ_old_selk[:,None,None,:]
    
    UnoccUnoccOcc_plus =UnoccUnoccOcc_new*np.logical_not(UnoccUnoccOcc_old)
    UnoccUnoccOcc_minus=UnoccUnoccOcc_old*np.logical_not(UnoccUnoccOcc_new)

    UnoccOccOcc_plus =UnoccOccOcc_new*np.logical_not(UnoccOccOcc_old)
    UnoccOccOcc_minus=UnoccOccOcc_old*np.logical_not(UnoccOccOcc_new)

    OccOcc_new=occ_new_selk[:,:,None]*occ_new_selk[:,None,:]
    OccOcc_old=occ_old_selk[:,:,None]*occ_old_selk[:,None,:]
    OccOcc_plus = OccOcc_new * np.logical_not(OccOcc_old)
    
    if evalJ0:
        imfgh[0,0]= eval_J0(data.OOmegaUU_K_rediag[selectK], delocc)
        s=-eval_J12(data.HHAAAAUU_K[selectK],OccOcc_plus)
        imfgh[1,0]= eval_J0(data.CCUU_K_rediag[selectK], delocc)-s
        imfgh[2,0]= eval_J0(data.HHOOmegaUU_K[selectK] , delocc)+s
    if evalJ1:
        B=data.delHH_dE_AA_K[selectK]
        C=data.delHH_dE_BB_K[selectK]
        D=data.delHH_dE_HH_AA_K[selectK]
        imfgh[0,1]=eval_J12(B,UnoccOcc_plus)-eval_J12(B,UnoccOcc_minus)
        imfgh[1,1]=eval_J12(C,UnoccOcc_plus)-eval_J12(C,UnoccOcc_minus)
        imfgh[2,1]=eval_J12(D,UnoccOcc_plus)-eval_J12(D,UnoccOcc_minus)
    if evalJ2:
        B=data.delHH_dE_SQ_K[selectK]
        C,D=data.delHH_dE_SQ_HH_K
        C=C[selectK]
        D=D[selectK]
        imfgh[0,2]=eval_J12(B,UnoccOcc_plus)-eval_J12(B,UnoccOcc_minus)
        imfgh[1,2]=eval_J3(C,UnoccUnoccOcc_plus)-eval_J3(C,UnoccUnoccOcc_minus)
        imfgh[2,2]=eval_J3(D,UnoccOccOcc_plus)-eval_J3(D,UnoccOccOcc_minus)

    imfgh[:,3,:]=imfgh[:,:3,:].sum(axis=1)

    occ_old[:,:]=occ_new[:,:]
    return imfgh/(data.NKFFT_tot)


def calcImfgh_K(data,degen,ik):
    
    imf= calcImf_K(data,degen,ik)

    s=2*eval_Joo_deg(data.HHAAAAUU_K[ik],degen)   
    img=eval_Jo_deg(data.CCUU_K_rediag[ik],degen)-s
    imh=eval_Jo_deg(data.HHOOmegaUU_K[ik],degen)+s


    C=data.delHH_dE_BB_K[ik]
    D=data.delHH_dE_HH_AA_K[ik]
    img+=-2*eval_Juo_deg(C,degen)
    imh+=-2*eval_Juo_deg(D,degen)

    C,D=data.delHH_dE_SQ_HH_K
    img+=-2*eval_Juuo_deg(C[ik],degen) 
    imh+=-2*eval_Juoo_deg(D[ik],degen)

    return imf,img,imh








def calcAHC_uIu(data,Efermi=None,occ_old=None, evalJ0=True,evalJ1=True,evalJ2=True):

    raise NotImplementedError()
    if occ_old is None: 
        occ_old=np.zeros((data.NKFFT_tot,data.num_wann),dtype=bool)

    if isinstance(Efermi, Iterable):
        nFermi=len(Efermi)
        AHC=np.zeros( ( nFermi,4,3) ,dtype=float )
        for iFermi in range(nFermi):
            AHC[iFermi]=calcAHC_uIu(data,Efermi=Efermi[iFermi],occ_old=occ_old, evalJ0=evalJ0,evalJ1=evalJ1,evalJ2=evalJ2)
        return np.cumsum(AHC,axis=0)
    
    # now code for a single Fermi level:
    AHC=np.zeros(3)

    occ_new=get_occ(data.E_K,Efermi)
    selectK=np.where(np.any(occ_old!=occ_new,axis=1))[0]
    occ_old_selk=occ_old[selectK]
    occ_new_selk=occ_new[selectK]
    delocc=occ_new_selk!=occ_old_selk

    AHC= eval_J0(data.FF[selectK], delocc)

    return AHC*fac_ahc/(data.NKFFT_tot*data.cell_volume)

