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

import multiprocessing 
import functools
import numpy as np
from collections import Iterable
import lazy_property
from copy import copy
from time import time
import pickle
import glob

from .__Data_K import Data_K
from . import __symmetry as SYM
from  .__Kpoint import KpointBZ,exclude_equiv_points
from . import __utility as utility
   

def process(paralfunc,K_list,nproc,symgroup=None):
    t0=time()
    selK=[ik for ik,k in enumerate(K_list) if k.res is None]
    dK_list=[K_list[ik].Kp_fullBZ for ik in selK]
    if len(dK_list)==0:
        print ("nothing to process now")
        return 0
    print ("processing {0}  points :".format(len(dK_list)) )
    if nproc<=0:
        res = [paralfunc(k) for k in dK_list]
        nproc_=1
    else:
        p=multiprocessing.Pool(nproc)
        res= p.map(paralfunc,dK_list)
        p.close()
        nproc_=nproc
    if not (symgroup is None):
        res=[symgroup.symmetrize(r) for r in res]
    for i,ik in enumerate(selK):
        K_list[ik].set_res(res[i])
    t=time()-t0
    print ("time for processing {0:6d} K-points : {1:10.4f} ; per K-point {2:15.4f} ; proc-sec per K-point : {3:15.4f}".format(len(selK),t,t/len(selK),t*nproc_/len(selK)) )
    return len(dK_list)
        



def one2three(nk):
    if isinstance(nk, Iterable):
        if len(nk)!=3 :
            raise RuntimeError("nk should be specified either a on number or 3numbers. found {}".format(nk))
        return nk
    return (nk,)*3



def autonk(nk,nkfftmin):
    if nk<nkfftmin:
        return 1,nkfftmin
    else:
        lst=[]
        for i in range(nkfftmin,nkfftmin*2):
            if nk%i==0:
                return nk//i,i
            j=nk//i
            lst.append( min( abs( j*i-nk),abs(j*i+i-nk)))
    i=nkfftmin+np.argmin(lst)
    j=nk//i
    j=[j,j+1][np.argmin([abs( j*i-nk),abs(j*i+i-nk)])]
    return j,i
#    return int(round(nk/nkfftmin)),nkfftmin)


def determineNK(NKdiv,NKFFT,NK,NKFFTmin):
    if ((NKdiv is None) or (NKFFT is None)) and ((NK is None) or (NKFFTmin is None)  ):
        raise ValueError("you need to specify either  (NK,NKFFTmin) or a pair (NKdiv,NKFFT). found ({},{}) and ({},{}) ".format(NK,NKFFTmin,NKdiv,NKFFT))
    if not ((NKdiv is None) or (NKFFT is None)):
        return np.array(one2three(NKdiv)),np.array(one2three(NKFFT))
    lst=[autonk(nk,nkfftmin) for nk,nkfftmin in zip(one2three(NK),one2three(NKFFTmin))]
    return np.array([l[0] for l in lst]),np.array([l[1] for l in lst])



def evaluate_K(func,system,NK=None,NKdiv=None,nproc=0,NKFFT=None,
            adpt_mesh=2,adpt_num_iter=0,adpt_nk=1,fout_name="result",
             symmetry_gen=[SYM.Identity],suffix="",
             GammaCentered=True,file_Klist="K_list.pickle",restart=False,start_iter=0):
    """This function evaluates in parallel or serial an integral over the Brillouin zone 
of a function func, which whould receive only one argument of type Data_K, and return 
a numpy.array of whatever dimensions

the user has to provide 2 grids:  of K-points - NKdiv and FFT grid (k-points) NKFFT

The parallelisation is done by K-points

As a result, the integration will be performed over NKFFT x NKdiv
"""
    
    if file_Klist is not None:
        if not file_Klist.endswith(".pickle"):
            file_Klist+=".pickle"
    cnt_exclude=0
    
    NKdiv,NKFFT=determineNK(NKdiv,NKFFT,NK,system.NKFFTmin)

    print ("using NKdiv={}, NKFFT={}, NKtot={}".format( NKdiv,NKFFT,NKdiv*NKFFT))
    
    symgroup=SYM.Group(symmetry_gen,basis=system.recip_lattice)

    paralfunc=functools.partial(
        _eval_func_k, func=func,system=system,NKFFT=NKFFT )

    if GammaCentered :
        shift=(NKdiv%2-1)/(2*NKdiv)
    else :
        shift=np.zeros(3)
    print ("shift={}".format(shift))

    if restart:
        try:
            K_list=pickle.load(open(file_Klist,"rb"))
            print ("{0} K-points were read from {1}".format(len(K_list),file_Klist))
            if len(K_list)==0:
                print ("WARNING : {0} contains zero points starting from scrath".format(file_Klist))
                restart=False
        except Exception as err:
            restart=False
            print ("WARNING: {}".format( err) )
            print ("WARNING : reading from {0} failed, starting from scrath".format(file_Klist))
            
    if not restart:
        print ("generating K_list")
        K_list=[KpointBZ(K=shift, NKFFT=NKFFT,symgroup=symgroup )]
        K_list+=K_list[0].divide(NKdiv)
        print ("Done, sum of weights:{}".format(sum(Kp.factor for Kp in K_list)))
        start_iter=0

    suffix="-"+suffix if len(suffix)>0 else ""

    if restart:
        print ("searching for start_iter")
        try:
            start_iter=int(sorted(glob.glob(fout_name+"*"+suffix+"_iter-*.dat"))[-1].split("-")[-1].split(".")[0])
        except Exception as err:
            print ("WARNING : {0} : failed to read start_iter. Setting to zero".format(err))
            start_iter=0

    if adpt_num_iter<0:
        adpt_num_iter=-adpt_num_iter*np.prod(NKdiv)/np.prod(adpt_mesh)/adpt_nk/3
    adpt_num_iter=int(round(adpt_num_iter))


    if (adpt_mesh is None) or np.max(adpt_mesh)<=1:
        adpt_num_iter=0
    else:
        if not isinstance(adpt_mesh, Iterable):
            adpt_mesh=[adpt_mesh]*3
        adpt_mesh=np.array(adpt_mesh)
    
    counter=0


    for i_iter in range(adpt_num_iter+1):
        print ("iteration {0} - {1} points. New points are:".format(i_iter,len([K for K in  K_list if K.res is None])) ) 
        for i,K in enumerate(K_list):
          if not K.evaluated:
            print (" K-point {0} : {1} ".format(i,K))
        counter+=process(paralfunc,K_list,nproc,symgroup=symgroup)
        
        try:
            if file_Klist is not None:
                pickle.dump(K_list,open(file_Klist,"wb"))
        except Exception as err:
            print ("Warning: {0} \n the K_list was not pickled".format(err))
            
        result_all=sum(kp.get_res for kp in K_list)
        
        if not (restart and i_iter==0):
            result_all.write(fout_name+"-{}"+suffix+"_iter-{0:04d}.dat".format(i_iter+start_iter))
        
        if i_iter >= adpt_num_iter:
            break
             
        # Now add some more points
        Kmax=np.array([K.max for K in K_list]).T
        select_points=set().union( *( np.argsort( Km )[-adpt_nk:] for Km in Kmax )  )
        
        l1=len(K_list)
        for iK in select_points:
            K_list+=K_list[iK].divide(adpt_mesh)
        print ("checking for equivalent points in all points (of new  {} points)".format(len(K_list)-l1))
        nexcl=exclude_equiv_points(K_list,new_points=len(K_list)-l1)
        print (" excluded {0} points".format(nexcl))
        print ("sum of weights now :{}".format(sum(Kp.factor for Kp in K_list)))
        
    
    print ("Totally processed {0} K-points ".format(counter))
    return result_all
       


def _eval_func_k(K,func,system,NKFFT):
    data=Data_K(system,K,NKFFT=NKFFT)
    return func(data)

