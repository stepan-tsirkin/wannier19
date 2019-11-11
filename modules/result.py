#------------------------------------------------------------#
# This file is distributed as part of the Wannier19 code     #
# under the terms of the GNU General Public License. See the #
# file `LICENSE' in the root directory of the Wannier19      #
# distribution, or http://www.gnu.org/copyleft/gpl.txt       #
#                                                            #
# The Wannier19 code is hosted on GitHub:                    #
# https://github.com/stepan-tsirkin/wannier19                #
#                     written by                             #
#           Stepan Tsirkin, University ofZurich              #
#                                                            #
#------------------------------------------------------------#
#
#  The purpose of this module is to provide Result classes for  
#  different types of  calculations. 
#  child classes can be defined specifically in each module

import numpy as np
from lazy_property import LazyProperty as Lazy
from utility import voidsmoother
from copy import deepcopy
## A class to contain results or a calculation:
## For any calculation there should be a class with the samemethods implemented

class Result():

    def __init__(self):
        raise NotImprementedError()

#  multiplication by a number 
    def __mul__(self,other):
        raise NotImprementedError()

# +
    def __add__(self,other):
        raise NotImprementedError()

# writing to a file
    def write(self,name):
        raise NotImprementedError()

#  how result transforms under symmetry operations
    def transform(self,sym):
        raise NotImprementedError()

# a list of numbers, by each of those the refinement points will be selected
    @property
    def max(self):
        raise NotImprementedError()


### these methods do no need re-implementation: 
    def __rmul__(self,other):
        return self*other

    def __radd__(self,other):
        return self+other
        
    def __truediv__(self,number):
        return self*(1./number)


# a class for data defined for a set of Fermi levels
#Data is stored in an array data, where first dimension indexes the Fermi level

class EnergyResult(Result):

    def __init__(self,Energy,data,smoother=voidsmoother()):
        assert (Energy.shape[0]==data.shape[0])
        self.Energy=Energy
        self.data=data
        self.smoother=smoother

    @Lazy
    def dataSmooth(self):
        return self.smoother(self.data)

    def __mul__(self,other):
        if isinstance(other,int) or isinstance(other,float) :
            return EnergyResult(self.Energy,self.data*other,self.smoother)
        else:
            raise TypeError("result can only be multilied by a number")

    def __add__(self,other):
        if other == 0:
            return self
        if np.linalg.norm(self.Energy-other.Energy)>1e-8:
            raise RuntimeError ("Adding results with different Fermi energies - not allowed")
        if self.smoother != other.smoother:
            raise RuntimeError ("Adding results with different smoothers ={} and {}".format(self.smoother,other.smoother))
        return EnergyResult(self.Energy,self.data+other.data,self.smoother)

    def write(self,name):
        # assule, that the dimensions starting from first - are cartesian coordinates       
        def getHead(n):
           if n<=0:
              return ['  ']
           else:
              return [a+b for a in 'xyz' for b in getHead(n-1)]
        rank=len(self.data.shape[1:])

        open(name,"w").write(
           "    ".join("{0:^15s}".format(s) for s in ["EF",]+
                [b for b in getHead(rank)*2])+"\n"+
          "\n".join(
           "    ".join("{0:15.6f}".format(x) for x in [ef]+[x for x in data.reshape(-1)]+[x for x in datasm.reshape(-1)]) 
                      for ef,data,datasm in zip (self.Energy,self.data,self.dataSmooth)  )
               +"\n") 

    @property
    def _maxval(self):
        return self.dataSmooth.max() 

    @property
    def _norm(self):
        return np.linalg.norm(self.dataSmooth)

    @property
    def _normder(self):
        return np.linalg.norm(self.dataSmooth[1:]-self.dataSmooth[:-1])
    
    @property
    def max(self):
        return np.array([self._maxval,self._norm,self._normder])




class ScalarResult(EnergyResult):
    def transform(self,sym):
        return self 

class AxialVectorResult(EnergyResult):
    def transform(self,sym):
        return AxialVectorResult(self.Energy,sym.transform_axial_vector(self.data),self.smoother )

class PolarVectorResult(EnergyResult):
    def transform(self,sym):
        return PolarVectorResult(self.Energy,sym.transform_polar_vector(self.data),self.smoother )


#a more general class. Scalar,polar and axial vectors may be derived as particular cases of the tensor class
class TensorResult(EnergyResult):

    def __init__(self,Energy,data,dataSmooth=None,smoother=None,TRodd=False,Iodd=False):
        shape=data.shape[1:]
        assert  len(shape)==len(trueVector)
        assert np.all(np.array(shape)==3)
        super(TensorResult,self).__init__(Energy,data,smoother=smoother)
        self.TRodd=TRodd
        self.Iodd=Iodd
        self.rank=len(data.shape[1:]) if rank is None else eank
 
    def transform(self,sym):
        return TensorResult(self.Energy,sym.transform(self.data,sym,TRodd=self.TRodd,Iodd=self.Iodd),self.smoother,self.TRodd,self.Iodd)


    def __mul__(self,other):
        res=super(TensorResult,self).__mul__(other)
        return TensorResult(res.Energy,res.data, res.smoother ,self.TRodd,self.Iodd)

    def __add__(self,other):
        assert self.TRodd == other.TRodd
        assert self.Iodd  == other.Iodd
        res=super(TensorResult,self).__add__(other)
        return TensorResult(res.Energy,res.data, res.smoother ,self.TRodd,self.Iodd)




class KBandResult(Result):

    def __init__(self,data,TRodd,Iodd):
        self.data=data
        self.TRodd=TRodd
        self.Iodd=Iodd
        
    def fit(self,other):
        for var in ['TRodd','Iodd','rank','nband']:
            if getattr(self,var)!=getattr(other,var):
                return False
        return True

    
    @property
    def rank(self):
       return len(self.data.shape)-2

    @property
    def nband(self):
       return self.data.shape[1]

    @property
    def nk(self):
       return self.data.shape[0]

    def __add__(self,other):
        assert self.fit(other)
        data=np.vstack( (self.data,other.data) )
        return KBandResult(data,self.TRodd,self.Iodd) 

    def to_grid(self,k_map):
        data=np.array( [sum(self.data[ik] for ik in km)/len(km)   for km in k_map])
        return KBandResult(data,self.TRodd,self.Iodd) 


    def select_bands(self,ibands):
        return KBandResult(self.data[:,ibands],self.TRodd,self.Iodd)


    def average_deg(self,deg):
        for i,D in enumerate(deg):
           for ib1,ib2 in D:
              self.data[i,ib1:ib2]=self.data[i,ib1:ib2].mean(axis=0)
        return self


    def transform(self,sym):
        data=sym.transform_tensor(self.data,rank=self.rank,TRodd=self.TRodd,Iodd=self.Iodd)
        return KBandResult(data,self.TRodd,self.Iodd)

    def write(self,name):
        # assule, that the dimensions starting from first - are cartesian coordinates       
        def getHead(n):
           if n<=0:
              return ['  ']
           else:
              return [a+b for a in 'xyz' for b in getHead(n-1)]
        rank=len(self.data.shape[1:])

        open(name,"w").write(
           "    ".join("{0:^15s}".format(s) for s in ["EF",]+
                [b for b in getHead(rank)*2])+"\n"+
          "\n".join(
           "    ".join("{0:15.6f}".format(x) for x in [ef]+[x for x in data.reshape(-1)]+[x for x in datasm.reshape(-1)]) 
                      for ef,data,datasm in zip (self.Energy,self.data,self.dataSmooth)  )
               +"\n") 


    def get_component(self,component=None):
        if component is None:
            return None
        xyz={"x":0,"y":1,"z":2}
        dim=self.data.shape[2:]
        try:
            if not  np.all(np.array(dim)==3):
                raise RuntimeError("dimensions of all components should be 3, found {}".format(dim))
                
            dim=len(dim)
            component=component.lower()
            if dim==0:
                Xnk=self.data
            elif dim==1:
                if component  in "xyz":
                    return self.data[:,:,xyz[component]]
                elif component=='norm':
                    return np.linalg.norm(self.data,axis=-1)
                elif component=='sq':
                    return np.linalg.norm(self.data,axis=-1)**2
                else:
                    raise RuntimeError("Unknown component {} for vectors".format(component))
            elif dim==2:
                if component=="trace":
                    return sum([self.data[:,:,i,i] for i in range(3)])
                else:
                    try :
                        return self.data[:,:,xyz[component[0]],xyz[component[1]]]
                    except IndexError:
                        raise RuntimeError("Unknown component {} for rank-2  tensors".format(component))
            elif dim==3:
                if component=="trace":
                    Xnk = sum([self.data[:,:,i,i,i] for i in range(3)])
                else:
                    try :
                        return self.data[:,:,xyz[component[0]],xyz[component[1]],xyz[component[2]]]
                    except IndexError:
                        raise RuntimeError("Unknown component {} for rank-3  tensors".format(component))
            elif dim==4:
                if component=="trace":
                    return sum([self.data[:,:,i,i,i,i] for i in range(3)])
                else:
                    try :
                        return self.data[:,:,xyz[component[0]],xyz[component[1]],xyz[component[2]],xyz[component[3]]]
                    except IndexError:
                        raise RuntimeError("Unknown component {} for rank-4  tensors".format(component))
            else: 
                raise RuntimeError("writing tensors with rank >4 is not implemented. But easy to do")
        except RuntimeError as err:
            print ("WARNING: {} - printing only energies".format(err) )
            return None

