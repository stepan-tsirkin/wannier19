import numpy as np
import abc 



"""some basic classes to construct formulae for evaluation"""

class Formula_ln(abc.ABC):

    @abc.abstractmethod
    def __init__(self,data_K,internal_terms=True,external_terms=True):
        self.internal_terms = internal_terms
        self.external_terms = external_terms

    @abc.abstractmethod
    def ln(self,ik,inn,out):
        pass

    @abc.abstractmethod
    def nn(self,ik,inn,out):
        pass

    def nl(self,ik,inn,out):
        return self.ln(ik,out,inn)

    def ll(self,ik,inn,out):
        return self.nn(ik,out,inn)

    @property
    def additive(self):
        """ if Trace_A+Trace_B = Trace_{A+B} holds. 
        needs override for quantities that do not obey this rule (e.g. Orbital magnetization)
        """
        return True

    def trace(self,ik,inn,out):
        return np.einsum("nn...->...",self.nn(ik,inn,out).real)



class Matrix_ln(Formula_ln):
    "anything that can be called just as elements of a matrix"
    @abc.abstractmethod
    def __init__(self,matrix):
        self.matrix=matrix
        self.ndim=len(matrix.shape)-3

    def ln(self,ik,inn,out):
        return self.matrix[ik][out][:,inn]

    def nn(self,ik,inn,out):
        return self.matrix[ik][inn][:,inn]


class Matrix_GenDer_ln(Formula_ln):
    "generalized erivative of MAtrix_ln"
    def __init__(self,matrix,matrix_comader,D):
        self.A  = matrix
        self.dA = matrix_comader
        self.D  =  D
        self.ndim=matrix.ndim+1

    def nn(self,ik,inn,out):
        summ=self.dA.nn(ik,inn,out)
        summ -= np.einsum( "mld,lnb->mnbd" , self.D.nl(ik,inn,out) , self.A.ln(ik,inn,out) )
        summ += np.einsum( "mlb,lnd->mnbd" , self.A.nl(ik,inn,out) , self.D.ln(ik,inn,out) )
        return summ

    def ln(self,ik,inn,out):
        summ= self.dA.ln(ik,inn,out)
        summ -= np.einsum( "mld,lnb->mnbd" , self.D.ln(ik,inn,out) , self.A.nn(ik,inn,out) )
        summ += np.einsum( "mlb,lnd->mnbd" , self.A.ll(ik,inn,out) , self.D.ln(ik,inn,out) )
        return summ



class FormulaProduct(Formula_ln):

    """a class to store a product of several formulae"""
    def __init__(self,formula_list,name="unknown",hermitian=False):
        if type(formula_list) not in (list,tuple):
            formula_list=[formula_list]
        self.TRodd=bool(sum(f.TRodd for f in formula_list)%2)
        self.Iodd =bool(sum(f.TRodd for f in formula_list)%2)
        self.name=name
        self.formulae=formula_list
        self.hermitian = hermitian
        ndim_list = [f.ndim for f in formula_list]
        self.ndim = sum(ndim_list)
        self.einsumlines=[]
        letters = "abcdefghijklmnopqrstuvw"
        dim = ndim_list[0]
        for d in ndim_list[1:]:
            self.einsumlines.append( "LM"+letters[:dim]+",MN"+letters[dim:dim+d]+"->LN"+letters[:dim+d])
            dim+=d

    def nn(self,ik,inn,out):
        matrices = [frml.nn(ik,inn,out) for frml in self.formulae ]
        res=matrices[0]
        for mat,line  in zip(matrices[1:],self.einsumlines):
#            print (line,res.shape,mat.shape)
            res=np.einsum(line,res,mat)
        if self.hermitian:
            res=0.5*(res+res.swapaxes(0,1).conj())
        return np.array(res,dtype=complex)

    def ln(self,ik,inn,out):
        raise NotImplementedError()
    
