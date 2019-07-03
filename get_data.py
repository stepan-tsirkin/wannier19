import numpy as np
from scipy.io import FortranFile as FF
from aux import str2bool
import wan_ham as wham
import copy


class Data():

    def __init__(self,seedname="wannier90",getAA=False,getBB=False,getCC=False,getSS=False,NK=None):
        f=open(seedname+"_HH_save.info","r")
        l=f.readline().split()[:3]
        self.seedname=seedname
        self.num_wann,nRvec,self.spinors=int(l[0]),int(l[1]),str2bool(l[2])
        self.real_lattice=np.array([f.readline().split()[:3] for i in range(3)],dtype=float)
        iRvec=np.array([f.readline().split()[:4] for i in range(nRvec)],dtype=int)
        f.close()
        self.cell_volume=np.linalg.det(self.real_lattice)
        self.Ndegen=iRvec[:,3]
        self.iRvec=iRvec[:,:3]
        self.cRvec=self.iRvec.dot(self.real_lattice)
        if NK is None:
            self.NK=np.abs(self.iRvec).max(axis=0)*2+1
        else:
            self.NK=NK

        print ("Number of wannier functions:",self.num_wann)
        print ("Number of R points:", self.nRvec)
        print ("Number of K points:", self.NK)
        print ("Real-space lattice:\n",self.real_lattice)
        #print ("R - points and dege=neracies:\n",iRvec)
        
        self.HH_R=self.__getMat('HH')
        
        if getAA:
            self.AA_R=self.__getMat('AA')
            
        if getBB:
            self.BB_R=self.__getMat('BB')

        if getCC:
            self.CC_R=self.__getMat('CC')

        if getSS:
            self.SS_R=self.__getMat('SS')
   
    def get_nRvec(self):
        return self.iRvec.shape[0]
    
    nRvec=property(get_nRvec)

    def __getMat(self,suffix):

        f=FF(self.seedname+"_" + suffix+"_R.dat")
        MM_R=np.array([[np.array(f.read_record('2f8'),dtype=float) for m in range(self.num_wann)] for n in range(self.num_wann)])
        MM_R=MM_R[:,:,:,0]+1j*MM_R[:,:,:,1]
        f.close()
        ncomp=MM_R.shape[2]/self.nRvec
        if ncomp==1:
            print "reading 0d for ",suffix
            return MM_R/self.Ndegen[None,None,:]
        elif ncomp==3:
            print "reading 1d for ",suffix
            return MM_R.reshape(self.num_wann, self.num_wann, 3, self.nRvec).transpose(0,1,3,2)/self.Ndegen[None,None,:,None]
        elif ncomp==9:
            print "reading 2d for ",suffix
            return MM_R.reshape(self.num_wann, self.num_wann, 3,3, self.nRvec).transpose(0,1,4,3,2)/self.Ndegen[None,None,:,None,None]
        
    def get_AA_K(self,NK):
        return wham.fourier_R_to_k( self.AA_R,self.iRvec, NK )
    
    def write_tb(self,f=None):
        if f is None: f=self.seedname+"_tb.dat"
        f=open(f,"w")
        f.write("written from saved HH_R")
        f.write("".join( ("  ".join("{0:15.12f}".format(x) for x in y)+"\n") for y in self.real_lattice))
        f.write("   {0}   \n".format(self.num_wann))
        f.write("   {0}   \n".format(len(self.iRvec)))
        for i in range(int(len(self.iRvec)/15)+1):
            try:
                a=self.Ndegen[15*i:min(len(self.Ndegen),15*i+15)]
                if len(a)>0:
                    f.write(" ".join("{0:5d}".format(i) for i in a)+"\n")
            except:
                pass
        
        for i,iR in enumerate(self.iRvec):
            f.write("\n {0} {1} {2}\n".format(i,iR))
#            f.write("\n".join(" {0:5d} {1:5d}  {2:


class Data_dk(Data):
    def __init__(self,data,dk=None,AA=None,BB=None,CC=None,SS=None,NK=None):
        self.num_wann=data.num_wann
        self.spinors=data.spinors
        self.iRvec=data.iRvec
        self.cRvec=data.cRvec
        self.cell_volume=data.cell_volume
        self.NK=data.NK if NK is None else NK
        
        if dk is not None:
            expdk=np.exp(2j*np.pi*self.iRvec.dot(dk))
        else:
            expdk=np.ones(self.nRvec)


        self.HH_R=data.HH_R[:,:,:]*expdk[None,None,:]
        
        if AA in (None,True):
            try:
                self.AA_R=data.AA_R[:,:,:,:]*expdk[None,None,:,None]
            except AttributeError:
                if AA : raise AttributeError("AA_R is not defined")


    def get_AA_K(self):
        try:
            return self.AA_K
        except AttributeError:
            self.AA_K=wham.fourier_R_to_k( self.AA_R,self.iRvec,self.NK)
            return self.AA_K
            
    def get_OOmega_K(self):
        try:
            return self.OOmega_K
        except AttributeError:
            self.OOmega_K=    -1j* wham.fourier_R_to_k( 
                        self.AA_R[:,:,:,wham.alpha]*self.cRvec[None,None,:,wham.beta ] - 
                        self.AA_R[:,:,:,wham.beta ]*self.cRvec[None,None,:,wham.alpha]   , self.iRvec, self.NK )
             
            return self.OOmega_K
    
    
