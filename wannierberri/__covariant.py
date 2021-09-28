import numpy as np
from .__utility import  alpha_A,beta_A
from .__formula_3 import Formula_ln
from .__formulas_nonabelian_3 import Aln,Vln,Dln,Fln, DerDln, DerA_Hbar_ln,DerF_Hbar_ln
#####################################################
#####################################################

""" The following  Formulue are fundamental. They can be used to construct all 
quantities relatred to Berry curvature and orbital magnetic moment. They are written
in the most explicit form, although probably not the most efecient. 
Foe practical reasons more eficient formulae may be constructed (e.g. by  excluding
some terms that cancel out). However, the following may be used as benchmark.
"""


#############################
###   Berry curvature    ####
#############################

class tildeFab(Formula_ln):

    def __init__(self,data_K,**parameters):
        super().__init__(data_K,**parameters)
        self.D=Dln(data_K)

#        print (f"tildeFab evaluating: internal({self.internal_terms}) and external({self.external_terms})")
        if self.external_terms:
            self.A=Aln(data_K)
            self.V=Vln(data_K)
            self.F=Fln(data_K)
        print ("Done")
        self.ndim=2
#        self.Iodd=False
#        self.TRodd=True

    def nn(self,ik,inn,out):
        summ = np.zeros( (len(inn),len(inn),3,3),dtype=complex )
        Dnl = self.D.nl(ik,inn,out)
        Dln = self.D.ln(ik,inn,out)
        if self.internal_terms:
            summ+= -np.einsum("mla,lnb->mnab",Dnl,Dln)

        if self.external_terms:
            summ += self.F.nn(ik,inn,out)
            summ += 2j * np.einsum("mla,lnb->mnab", Dnl,self.A.ln(ik,inn,out)    )  
#            summ += 1j * np.einsum("mla,lnb->mnab", self.A.nl (ik,inn,out),Dln   )
            summ +=  -1* np.einsum("mla,lnb->mnab",self.A.nn(ik,inn,out),self.A.nn(ik,inn,out))

#  Terms (a<->b, m<-n> )*   are accounted above by factor 2
        summ =  0.5*(summ+summ.transpose((1,0,3,2)).conj())
        return summ

    def ln(self,ik,inn,out):
        raise NotImplementedError()


#############################
###   derivative of      ####
###   Berry curvature    ####
#############################


class tildeFab_d(Formula_ln):

    def __init__(self,data_K,**parameters):
        super().__init__(data_K,**parameters)
        self.dD = DerDln(data_K)
        self.D  = Dln(data_K)

#        print (f"derOmega evaluating: internal({self.internal_terms}) and external({self.external_terms})")

        if self.external_terms:
            self.A  = Aln(data_K)
            self.dA = DerA_Hbar_ln(data_K)
            self.dF  = DerF_Hbar_ln(data_K)
        self.ndim=3
#        self.Iodd=True
#        self.TRodd=False

    def nn(self,ik,inn,out):
        summ = np.zeros( (len(inn),len(inn),3,3,3),dtype=complex )
        Dnl = self.D.nl(ik,inn,out)
        dDln = self.dD.ln(ik,inn,out)

        if self.internal_terms:
            summ+= -2* np.einsum("mla,lnbd->mnabd",Dnl,dDln)

        if self.external_terms:
            summ += self.dF.nn(ik,inn,out)
            summ += 2j * np.einsum("mla,lnbd->mnabd",Dnl , self.dA.ln(ik,inn,out))
            summ += 2j * np.einsum("mla,lnbd->mnabd",self.A.nl (ik,inn,out) ,dDln       )
            summ += -2 * np.einsum("mla,lnbd->mnabd",self.A.nn (ik,inn,out) , self.dA.nn(ik,inn,out))

#  Terms (a<->b, m<-n> )*   are accounted above by factor 2
        summ = 0.5*(summ+summ.transpose((1,0,3,2,4)).conj())
        return summ


    def ln(self,ik,inn,out):
        raise NotImplementedError()


#####################################################
##   Now define their anitsymmetric combinations   ##
#####################################################

class AntiSymmetric(Formula_ln):
    def __init__(self,full,data_K,**parameters):
        self.full = full(data_K,**parameters)
        self.ndim=self.full.ndim-1

    def nn(self,ik,inn,out):
        fab =self.full.nn(ik,inn,out)
        return 1j*(fab[:,:,alpha_A,beta_A] -  fab[:,:,beta_A,alpha_A]) 

    def ln(self,ik,inn,out):
        fab =self.full.ln(ik,inn,out)
        return 1j*(fab[:,:,alpha_A,beta_A] -  fab[:,:,beta_A,alpha_A]) 


class Symmetric(Formula_ln):
    def __init__(self,full,data_K,axes=[0,1],**parameters):
        self.full = full(data_K,**parameters)
        self.ndim=self.full.ndim

    def nn(self,ik,inn,out):
        fab =self.full.nn(ik,inn,out)
        return fab+fab.swapaxes(axes[0]+2,axes[1]+2) 

    def ln(self,ik,inn,out):
        fab =self.full.nn(ik,inn,out)
        return fab+fab.swapaxes(axes[0]+2,axes[1]+2) 


class tildeFc(AntiSymmetric):

    def __init__(self,data_K,**parameters):
        super().__init__(tildeFab,data_K,**parameters)
        self.Iodd=False
        self.TRodd=True

class tildeFc_d(AntiSymmetric):

    def __init__(self,data_K,**parameters):
        super().__init__(tildeFab_d,data_K,**parameters)
        self.Iodd=True
        self.TRodd=False

