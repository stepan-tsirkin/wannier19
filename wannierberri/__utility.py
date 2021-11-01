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



__debug = False

import inspect
import numpy as np

from lazy_property import LazyProperty as Lazy
from time import time
from termcolor import cprint 
import functools,fortio,scipy.io

try: 
    import pyfftw
    PYFFTW_IMPORTED=True
except Exception as err:
    PYFFTW_IMPORTED=False
    print ("WARNING : error importing  `pyfftw` : {} \n will use numpy instead \n".format(err))


# inheriting just in order to have posibility to change default values, without changing the rest of the code
class FortranFileR(fortio.FortranFile):
    def __init__(self,filename):
        print ( "using fortio to read" )
        try: 
            super().__init__( filename, mode='r',  header_dtype='uint32'  , auto_endian=True, check_file=True )
        except ValueError :
            print ("File '{}' contains subrecords - using header_dtype='int32'".format(filename))
            super().__init__( filename, mode='r',  header_dtype='int32'  , auto_endian=True, check_file=True )


class FortranFileW(scipy.io.FortranFile):
    def __init__(self,filename):
        print ( "using scipy.io to write" )
        super().__init__( filename, mode='w')


alpha_A=np.array([1,2,0])
beta_A =np.array([2,0,1])
TAU_UNIT=1E-9 # tau in nanoseconds
TAU_UNIT_TXT="ns"


def print_my_name_start():
    if __debug: 
        print("DEBUG: Running {} ..".format(inspect.stack()[1][3]))


def print_my_name_end():
    if __debug: 
        print("DEBUG: Running {} - done ".format(inspect.stack()[1][3]))

    
def conjugate_basis(basis):
    return 2*np.pi*np.linalg.inv(basis).T

def warning(message,color="yellow"):
    cprint ("\n WARNING!!!!! {} \n".format(message),color)


def real_recip_lattice(real_lattice=None,recip_lattice=None):
    if recip_lattice is None:
        if real_lattice is None : 
            cprint ("\n WARNING!!!!! usually need to provide either with real or reciprocal lattice. If you only want to generate a random symmetric tensor - that it fine \n","yellow")
            return None,None
        recip_lattice=conjugate_basis(real_lattice)
    else: 
        if real_lattice is not None:
            assert np.linalg.norm(real_lattice.dot(recip_lattice.T)/(2*np.pi)-np.eye(3))<=1e-8 , "real and reciprocal lattice do not match"
        else:
            real_lattice=conjugate_basis(recip_lattice)
    return real_lattice, recip_lattice



from scipy.constants import Boltzmann,elementary_charge,hbar
import abc

class AbstractSmoother(abc.ABC):
    """ Smoother for smoothing an array by convolution.
    This is an abstract class which cannot by instantiated. Only the specific child classes
    can be instantiated. Each child class should implement its own version of _params,
    __init__, and _broaden.
    - _params : list of parameters that uniquely define the smoother.
    - _broaden : function that defines the broadening method.
    - __init__ : initialize Smoother parameters

    Parameters
    -----------
    E : 1D array
        The energies on which the data are calculated at.
    smear : float
        Smearing parameter in eV.
    maxdE : int
        Determines the width of the convoluting function as (-smear * maxdE, smear * maxdE).
    """
    @property
    @abc.abstractmethod
    def _params(self):
        pass

    @abc.abstractmethod
    def __init__(self, E, smear, maxdE):
        self.smear = smear
        self.E = np.copy(E)
        self.maxdE = maxdE
        self.dE = E[1] - E[0]
        self.Emin = E[0]
        self.Emax = E[-1]
        self.NE1 = int(self.maxdE * self.smear / self.dE)
        self.NE = E.shape[0]
        self.smt = self._broaden(np.arange(-self.NE1,self.NE1+1)*self.dE)*self.dE

    @abc.abstractmethod
    def _broaden(self, E):
        pass

    def __str__(self):
        return f"<{type(self).__name__}>"

    def __eq__(self,other):
        if type(self) != type(other):
            return False
        else:
            for param in self._params:
                if not np.allclose(getattr(self, param), getattr(other, param)):
                    return False
        return True

    def __call__(self, A, axis=0):
        assert self.E.shape[0]==A.shape[axis]
        A=A.transpose((axis,)+tuple(range(0,axis))+tuple(range(axis+1,A.ndim)))
        res=np.zeros(A.shape, dtype=A.dtype)
        #TODO maybe change tensordot to numba
        for i in range(self.NE):
            start=max(0,i-self.NE1)
            end=min(self.NE,i+self.NE1+1)
            start1=self.NE1-(i-start)
            end1=self.NE1+(end-i)
            res[i]=np.tensordot(A[start:end],self.smt[start1:end1],axes=(0,0))/self.smt[start1:end1].sum()
        return res.transpose( tuple(range(1,axis+1))+ (0,)+tuple(range(axis+1,A.ndim)) )


class FermiDiracSmoother(AbstractSmoother):
    """ Smoother that uses the derivative of Fermi-Dirac function.

    Parameters
    -----------
    E : 1D array
        The energies on which the data are calculated at.
    T_Kelvin : float
        Temperature in Kelvin. Transformed into self.smear, which is in eV.
    maxdE : int
        Determines the width of the convoluting function as (-T * maxdE, T * maxdE)
    """
    _params = ['smear', 'E', 'maxdE', 'NE1']

    def __init__(self, E, T_Kelvin, maxdE=8):
        self.T_Kelvin = T_Kelvin
        smear = T_Kelvin * Boltzmann / elementary_charge  # convert K to eV
        super().__init__(E, smear, maxdE)

    def _broaden(self,E):
        return 0.25 / self.smear / np.cosh(E/(2*self.smear))**2

    def __str__(self):
        return f"<FermiDiracSmoother T={self.smear} ({self.T_Kelvin:.1f} K), NE={self.NE}, NE1={self.NE1}, E={self.Emin}..{self.Emax}, step {self.dE}>"


class GaussianSmoother(AbstractSmoother):
    """ Smoother that uses Gaussian function.

    Parameters
    -----------
    E : 1D array
        The energies on which the data are calculated at.
    smear : float
        Smearing parameter in eV.
    maxdE : int
        Determines the width of the convoluting function as (-smear * maxdE, smear * maxdE)
    """
    _params = ['smear', 'E', 'maxdE', 'NE1']

    def __init__(self, E, smear, maxdE=8):
        super().__init__(E, smear, maxdE)

    def _broaden(self,E):
        return np.exp(-(E / self.smear)**2) / self.smear / np.sqrt(np.pi)

    def __str__(self):
        return f"<GaussianSmoother smear={self.smear}, NE={self.NE}, NE1={self.NE1}, E={self.Emin}..{self.Emax}, step {self.dE}>"


class VoidSmoother(AbstractSmoother):
    """ Void smoother. When called, do nothing and return the original array."""
    _params = []

    def __init__(self):
        pass

    def _broaden(self, E):
        pass

    def __call__(self, A, axis=0):
        return A


def getSmoother(energy, smear, mode=None):
    if energy is None:
        return VoidSmoother()
    if smear is None or smear<=0:
        return VoidSmoother()
    if len(energy) <= 1:
        return VoidSmoother()
    if mode == "Fermi-Dirac":
        return FermiDiracSmoother(energy, smear)
    elif mode == "Gaussian":
        return GaussianSmoother(energy, smear)
    else:
        raise ValueError("Smoother mode not recognized.")


def str2bool(v):
    if v[0] in "fF" :
        return False
    elif v[0] in "tT":
        return True
    else :
        raise RuntimeError(" unrecognized value of bool parameter :{0}".format(v) )


def fft_W(inp,axes,inverse=False,destroy=True,numthreads=1):
    assert inp.dtype==complex
    t0=time()
    fft_in  = pyfftw.empty_aligned(inp.shape, dtype='complex128')
    fft_out = pyfftw.empty_aligned(inp.shape, dtype='complex128')
    t01=time()
    fft_object = pyfftw.FFTW(fft_in, fft_out,axes=axes, 
            flags=('FFTW_ESTIMATE',)+(('FFTW_DESTROY_INPUT',)  if destroy else () ), 
            direction='FFTW_BACKWARD' if inverse else 'FFTW_FORWARD',
            threads=numthreads)
    t1=time()
    fft_object(inp)
    t2=time()
    return fft_out



    def getHead(n):
       if n<=0:
          return ['  ']
       else:
          return [a+b for a in 'xyz' for b in getHead(n-1)]



def fft_np(inp,axes,inverse=False):
    assert inp.dtype==complex
    if inverse:
        return np.fft.ifftn(inp,axes=axes)
    else:
        return np.fft.fftn(inp,axes=axes)

def FFT(inp,axes,inverse=False,destroy=True,numthreads=1,fft='fftw' ):
    if fft=='fftw' and not PYFFTW_IMPORTED:
        fft='numpy'
    if fft=='fftw':
        return fft_W(inp,axes,inverse=inverse,destroy=destroy,numthreads=numthreads)
    elif fft=='numpy':
        return fft_np(inp,axes,inverse=inverse)
    else:
        raise ValueError("unknown type of fft : {}".format(fft))


def fourier_q_to_R(AA_q,mp_grid,kpt_mp_grid,iRvec,ndegen,numthreads=1,fft='fftw'):
    print_my_name_start()
    mp_grid=tuple(mp_grid)
    shapeA=AA_q.shape[1:]  # remember the shapes after q
    AA_q_mp=np.zeros(tuple(mp_grid)+shapeA,dtype=complex)
    for i,k in enumerate(kpt_mp_grid):
        AA_q_mp[k]=AA_q[i]
    AA_q_mp = FFT(AA_q_mp,axes=(0,1,2),numthreads=numthreads,fft=fft,destroy=False)
    AA_R=np.array([AA_q_mp[tuple(iR%mp_grid)]/nd for iR,nd in zip(iRvec,ndegen)])/np.prod(mp_grid)
    AA_R=AA_R.transpose((1,2,0)+tuple(range(3,AA_R.ndim)))
    print_my_name_end()
    return AA_R




class FFT_R_to_k():
    
    def __init__(self,iRvec,NKFFT,num_wann,numthreads=1,lib='fftw',name=None):
        t0=time()
        print_my_name_start()
        self.NKFFT=tuple(NKFFT)
        self.num_wann=num_wann
        assert lib in ('fftw','numpy','slow') , "fft lib '{}' is not known/supported".format(lib)
        if lib=='fftw' and not PYFFTW_IMPORTED:
            lib='numpy'
        self.lib = lib
        if lib == 'fftw':
            shape=self.NKFFT+(self.num_wann,self.num_wann)
            fft_in  = pyfftw.empty_aligned(shape, dtype='complex128')
            fft_out = pyfftw.empty_aligned(shape, dtype='complex128')
            self.fft_plan = pyfftw.FFTW(fft_in, fft_out,axes=(0,1,2), 
                flags=('FFTW_ESTIMATE','FFTW_DESTROY_INPUT'),
                direction='FFTW_BACKWARD' ,
                threads=numthreads  )
        self.iRvec=iRvec%self.NKFFT
        self.nRvec=iRvec.shape[0]
        self.time_init=time()-t0
        self.time_call=0
        self.n_call=0

    def execute_fft(self,A):
        return self.fft_plan(A)

    def transform(self,AAA_K):
        if self.lib=='numpy':
            AAA_K[...] = np.fft.ifftn(AAA_K,axes=(0,1,2))
        elif self.lib=='fftw':
        # do recursion if array has cartesian indices. The recursion should not be very deep
            if AAA_K.ndim>5:
                for i in range(AAA_K.shape[-1]):
                    AAA_K[...,i]=self.transform(AAA_K[...,i])
            else:
                AAA_K[...]=self.execute_fft(AAA_K[...])
            return AAA_K
        elif self.lib=='slow':
            raise RuntimeError("FFT.transform should not be called for slow FT")
        else :
            raise ValueError("Unknown type of Fourier transform :''".format(self.lib)) 
    
    @Lazy
    def exponent(self):
        '''
        exponent for Fourier transform exp(1j*k*R)
        '''
        return [np.exp(2j*np.pi/self.NKFFT[i])**np.arange(self.NKFFT[i]) for i in range(3)]


    def __call__(self,AAA_R,hermitean=False,antihermitean=False,reshapeKline=True):
        t0=time()
    #  AAA_R is an array of dimension (  num_wann x num_wann x nRpts X... ) (any further dimensions allowed)
        if  hermitean and antihermitean :
            raise ValueError("A matrix cannot be both hermitean and anti-hermitean, unless it is zero")
        AAA_R=AAA_R.transpose((2,0,1)+tuple(range(3,AAA_R.ndim)))
        shapeA=AAA_R.shape
        if self.lib=='slow':
            t0=time()
            k=np.zeros(3,dtype=int)
            AAA_K=np.array([[[
                 sum( np.prod([self.exponent[i][(k[i]*R[i])%self.NKFFT[i]] for i in range(3)])  *  A    for R,A in zip( self.iRvec, AAA_R) )
                    for k[2] in range(self.NKFFT[2]) ] for k[1] in range(self.NKFFT[1]) ] for k[0] in range(self.NKFFT[0])  ] )
            t=time()-t0
        else:
            t0=time()
            assert  self.nRvec==shapeA[0]
            assert  self.num_wann==shapeA[1]==shapeA[2]
            AAA_K=np.zeros( self.NKFFT+shapeA[1:], dtype=complex )
            ### TODO : place AAA_R to FFT grid from beginning, even before multiplying by exp(dkR)
            for ir,irvec in enumerate(self.iRvec):
                AAA_K[tuple(irvec)]+=AAA_R[ir]
            self.transform(AAA_K)
            AAA_K*=np.prod(self.NKFFT)
            t=time()-t0


        ## TODO - think if fft transform of half of matrix makes sense
        if hermitean:
            AAA_K= 0.5*(AAA_K+AAA_K.transpose((0,1,2,4,3)+tuple(range(5,AAA_K.ndim))).conj())
        elif antihermitean:
            AAA_K=0.5*(AAA_K-AAA_K.transpose((0,1,2,4,3)+tuple(range(5,AAA_K.ndim))).conj())

        if reshapeKline:
            AAA_K=AAA_K.reshape( (np.prod(self.NKFFT),)+shapeA[1:])
        self.time_call+=time()-t0
        self.n_call+=1
        return AAA_K

def iterate3dpm(size):
    return ( np.array([i,j,k]) for i in range(-size[0],size[0]+1)
                     for j in range(-size[1],size[1]+1)
                     for k in range(-size[2],size[2]+1) )

def iterate3d(size):
    return ( np.array([i,j,k]) for i in range(0,size[0])
                     for j in range(0,size[1])
                     for k in range(0,size[2]) )

def find_degen(arr,degen_thresh):
    """ finds shells of 'almost same' values in array arr, and returns a list o[(b1,b2),...]"""
    A=np.where(arr[1:]-arr[:-1]>degen_thresh)[0]+1 
    A=[0,]+list(A)+[len(arr)] 
    return [(ib1,ib2) for ib1,ib2 in zip(A,A[1:]) ] 


def is_round(A,prec=1e-14):
     """ returns true if all values in A are integers, at least within machine precision"""
     return( np.linalg.norm(A-np.round(A))<prec )
