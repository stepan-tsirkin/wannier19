import numpy as np
import lazy_property
from .__utility import  alpha_A,beta_A
from numba import njit

#####################################################
#####################################################

INNER_IND="lmnpqrsto"
OUTER_IND="LMNPRRSTP"
INDICES=INNER_IND+OUTER_IND

@njit
def inner_outer(ind,inner,outer):
    if   ind in INNER_IND:
       return inner
    elif ind in OUTER_IND:
       return outer
    else:
       return None
#       raise ValueError("unknown type of index : '{}'".formta(ind))

class FormulaProduct():

    """a class to store a product of several formulae"""
    def __init__(self,formula_list,subscripts=None,hermitian=False,name='unknown'):
        if type(formula_list) not in (list,tuple):
            formula_list=[formula_list]
        self.TRodd=bool(sum(f.TRodd for f in formula_list)%2)
        self.Iodd =bool(sum(f.TRodd for f in formula_list)%2)
        self.ndim = sum(f.ndim for f in formula_list)
        self.name=name
        self.formulae=formula_list
        self.hermitian=hermitian # not sure if it will ever be needed
        self.einline =  { tr:self.__einline(subscripts,trace=tr) for tr in (True,False) }

    def __call__(self,ik,ib_in_start,ib_in_end,trace=True):
        # not sure if we will ever use trace=False ?
        res = np.einsum( self.einline[trace],*( frml(ik,ib_in_start,ib_in_end,trace=False) 
                        for frml in self.formulae )  )
        if trace: 
            return res.real
        else:
            if self.hermitian:
                res=0.5*(res+swapaxes(res,0,1).conj())
            return np.array(res,dtype=complex)
    

    def __einline(self,subscripts,trace):
        """to get the string for the einsum"""
        if subscripts is None:
            ind_cart="abcdefghijk"
            left=[]
            right=""
            for frml in self.formulae:
                d=frml.ndim
                left.append ( ind_cart[ :d])
                right   +=    ind_cart[ :d]
                ind_cart =    ind_cart[d: ]
        else:
            left,right=subscripts.split("->")
            left=left.split(",")

        for frml,l in zip(self.formulae,left):
            d=frml.ndim
            if d!=len(l):
                raise ValueError("The number of subscripts in '{}' does not correspond to dimention '{}' of formula '{}' ".format(l,d,frml.name))
        ind_bands="lmnopqrstuvwxyz"[:len(self.formulae)]
        ind_bands+= ind_bands[0]  if trace else "Z"
        einleft=[]
        if not trace:
            right=ind_bands[0]+ind_bands[-1]+right
        for l in left:
            einleft.append(ind_bands[:2]+l)
            ind_bands=ind_bands[1:]
        return ",".join(einleft)+"->"+right

#    print ("using line '{}' for einsum".format(einline))



class Formula():
    """a class to write a formula as trace of matrix product,
     where inner summation is either over 'inner'/'occupied'  or 'outer'/'unoccupied' states
    as input takes a list of entriies like 
    ('nm,mL,L,Ln',A,B,C,D)  
    which will mean :math:`\sum_nm&{\rm in}\sum_L^{\rm out} A){nm}B_{mL}C_{L}D_{Ln}`
    by convention lower case indices correspond to 'inner' states, upper case --- to 'outer' states
    indices  of neighbouring matrices should match (i.e. 'mn,qn' is forbidden), 
    first and last index should be different, if more then one matrix is given.
    """
    def __init__(self,TRodd=False,Iodd=False,ndim=0,hermitian=True,name="unknown"):
        self.name=name
        self.TRodd=TRodd
        self.Iodd=Iodd
        self.ndim=ndim
        self.term_list=[]
        self.hermitian=hermitian

    def __call__(self,ik,ib_in_start,ib_in_end,trace):
        res = sum( term(ik,ib_in_start,ib_in_end) for term in self.term_list )
        if trace: 
            return np.einsum('nn...->...',res).real
        else:
            if self.hermitian:
                res=0.5*(res+np.swapaxes(res,0,1).conj())
            return np.array(res,dtype=complex)

    def add_term(self,ind_string ,mat_list ,mult=1.):
        self.term_list.append( 
                  MatrixProductTerm(ind_string , mat_list ,self.ndim , mult)
                             )


class MatrixProductTerm():
    """ a term containing product of matrices, to be evaluated as a matrix or as a trace
        mat_list should contain matrices of shape [ik,ib1,ib2,a,b,...] or [ik,ib,a,b,...]
        same index should appear only on neighbouring positions (or frst and last)
    """

    def __init__(self,ind_string ,mat_list,ndim,mult):
        self.ndim=ndim
        self.mult=mult
        if type(mat_list) not in (list,tuple):
            mat_list=(mat_list,)

        line=ind_string.replace(' ','')  # remove spaces if appear
        line=line.replace('.',',') # allow to use dot instead of comma
        # now goes a long boring check that the string is valid
 
        allowed_symbols=INDICES+",. "
        forbidden_symbols=set(line)-set(allowed_symbols) # everything that is not allowed is forbidden
        if len(forbidden_symbols) > 0 :
            raise FormulaIndicesError(ind_string,"Unrecognized symbols ( {} ) ".format(
                        " , ".join("'{}'".format(s) for s in forbidden_symbols) )  )

        terms=line.split(',')
        if len(terms)>1 and terms[0][1]==terms[-1][-1] :
            raise FormulaIndicesError(ind_string,'first and last indices should be different')
        if terms[0][0] not in INNER_IND:
            raise FormulaIndicesError(ind_string,"first index should be innner i.e. in '{}'".format(INNER_IND))
        if terms[-1][-1] not in INNER_IND:
            raise FormulaIndicesError(ind_string,"last  index should be innner i.e. in '{}'".format(INNER_IND))
        if len(terms)==0: 
            raise FormulaIndicesError(ind_string,"does not contain any terms")
        if len(terms)!=len(mat_list) :
            raise FormulaIndicesError(ind_string,"number of terms in the string {} is not equal to number of matrices supplied {}".format(len(terms),len(mat_list)))
        for ind1,ind2 in zip(terms,terms[1:]):
            if ind1[-1]!=ind2[0]:
                raise FormulaIndicesError(ind_string,"matching indices {} and {} are different".format(ind1,ind2))
        self.mat_list=[]
        self.ind_list=[]
        
        for term,mat in zip(terms,mat_list):
            assert isinstance(mat,np.ndarray) , "inputs should be arrays"
            if len(term)>2 : 
                raise FormulaIndicesError(ind_string,"more then two indices in '{}' ".format(term))
            assert mat.ndim==self.ndim+1+len(term) ,  ( 
                          "shape of matrix {} does not correspond to index '{}' and dimension of the result {}".format(
                               mat.shape,term,self.ndim)  )
            if len(term)==2:
                if term[0]==term[1]:
                    mat=mat[:,np.arange(mat.shape[1]),np.arange(mat.shape[1])]
            self.mat_list.append(mat)
            self.ind_list.append(term)
            
    @lazy_property.LazyProperty
    def nb(self):
        return self.mat_list[0].shape[1]

    @lazy_property.LazyProperty
    def ibrange(self):
        return np.arange(self.nb)


    def __call__(self,ik,ib_in_start,ib_in_end):
        return self.mult*MatrixProductTerm_call(ik,ib_in_start,ib_in_end,self.ind_list,self.mat_list,self.nb)

#@njit
def MatrixProductTerm_call(ik,ib_in_start,ib_in_end,self_ind_list,self_mat_list,self_nb):
#        print ('running in njit')
#        inner=np.arange(ib_in_start,ib_in_end)
#        outer=np.concatenate( (np.arange(0,ib_in_start),np.arange(ib_in_end,self_nb)) )
#        def get_index(i):
#        get_index = lambda i : inner_outer(i,inner,outer)
        mat_list=[]
        ind_list=[]
        for ind,m in zip(self_ind_list,self_mat_list):
            m1=m[ik]
            if ind[0] in INNER_IND:
                m1=m1[ib_in_start:ib_in_end]
            else :
                m1=np.concatenate((m1[0:ib_in_start],m1[ib_in_end:]),axis=0)
            if len(ind)==2 :
                if ind[1] in INNER_IND:
                    m1=m1[:,ib_in_start:ib_in_end]
                else :
                    m1=np.concatenate((m1[:,0:ib_in_start],m1[:,ib_in_end:]),axis=1)
            mat_list.append(m1)
            ind_list.append(ind)
        while len(mat_list)>1:
            ind1=ind_list[-2]
            ind2=ind_list[-1]
            ind_new=ind1[:-1]+ind2[1:]
#            mat_list[-2] = np.einsum(ind1+'...,'+ind2+'...->'+ind_new+'...', mat_list[-2],mat_list[-1])
            mat_list[-2] = matrix_prod_cart(mat_list[-2],mat_list[-1],ind1,ind2)
            del mat_list[-1]
            del ind_list[-1]
            ind_list[-1]=ind_new
        m = mat_list[0]
# Note : this is an inefficient workaround, to maintain uniformity of the code (fermisea vs nonabelian)
# probably it is not an problem anyway  . TODO : optimize if needed.
        if len(ind_list[0])==1 :   
            m1=np.zeros((m.shape[0],)+m.shape, dtype=complex )
            for i in range(m.shape[0]):
                m1[i,i]=m[i]
            m=m1
        return m


def matrix_prod_cart(m1,m2,ind1,ind2):
    if len(ind1)==len(ind2)==2:
        return sum(m1[:,i]*m2[i,:][None,:] for i in range(m1.shape[1]) )
#    if len(ind1)==len(ind2)==2:
#        shape1=m1.shape
#        shape2=m2.shape
#        shape=[shape1[0],shape2[1]]
#        for i in range(len(shape1)): 
#            shape.append(max(shape1[i],shape2[i]),)
#        shapet=tuple(s for s in shape)
#        result=np.zeros(shape,dtype=complex)
#        for i in range(m1.shape[1]):
#            result+=m1[:,i]*m2[i,:][None,:]
    return m1
#    else:
#        return None  # not implemented yet


class FormulaIndicesError(ValueError):
    def __init__(self, ind_string,reason):
        # Call the base class constructor with the parameters it needs
        super().__init__(  
           "The following string of indices '{}' are not valid ".format(ind_string)
            +"because '{}' ".format(reason) )
