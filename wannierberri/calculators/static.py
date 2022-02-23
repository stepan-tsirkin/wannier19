from .classes import StaticCalculator
from wannierberri import covariant_formulak as frml
from wannierberri import fermiocean
from wannierberri.formula import FormulaProduct
from scipy.constants import  elementary_charge, hbar, electron_mass, physical_constants, angstrom
from wannierberri.__utility import  TAU_UNIT

##################################################
######                                     #######
######         integration (Efermi-only)   #######
######                                     #######
##################################################

#  TODO: Ideally, a docstring of every calculator should contain the equation that it implements
#        and references (with urls) to the relevant papers"""


class AHC(StaticCalculator):

    def __init__(self,**kwargs):
        self.Formula = frml.Omega
        self.factor =  fermiocean.fac_ahc
        self.fder = 0
        super().__init__(**kwargs)

class Ohmic(StaticCalculator):

    def __init__(self,**kwargs):
        self.Formula = frml.InvMass
        self.factor =  fermiocean.factor_ohmic
        self.fder = 0
        super().__init__(**kwargs)



class BerryDipole_FermiSurf(StaticCalculator):

    def __init__(self,**kwargs):
        self.Formula = frml.VelOmega
        self.factor =  1
        self.fder = 1
        super().__init__(**kwargs)


class BerryDipole_FermiSea(StaticCalculator):

    def __init__(self,**kwargs):
        self.Formula = frml.DerOmega
        self.factor =  1
        self.fder = 0
        super().__init__(**kwargs)

    def  __call__(self,data_K):
        res = super().__call__(data_K)
        res.data= res.data.swapaxes(1,2)  # swap axes to be consistent with the eq. (29) of DOI:10.1038/s41524-021-00498-5
        return res


class DOS(StaticCalculator):

    def __init__(self,**kwargs):
        self.Formula = frml.Identity
        self.factor =  1
        self.fder = 1
        super().__init__(**kwargs)

    def __call__(self,data_K):
        return super().__call__(data_K)*data_K.cell_volume

class CumDOS(DOS):

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.fder = 0




class Frml_VVO(FormulaProduct):

    def __init__(self,data_K,**kwargs_formula):
        super().__init__( [data_K.covariant('Ham',commader=1),data_K.covariant('Ham',commader=1),frml.Omega(data_K,**kwargs_formula)], name='VelVelOmega')

class VelVelOmega_f1(StaticCalculator):

    def __init__(self,**kwargs):
        self.Formula = Frml_VVO
        # The formula after integration gives eV*Ang
        # multiply by e*1e-10 to get J*m
        # multiply by tau*e^3/hbar^3 to get 
        # J*m*s*(A*s)^3/(J*s)^3 = m*s*A^3/J^2
        # We need units A/m^2/(V/m)/T
        # V = J/(A*s) , T = J/ (A*m^2)
        # So, we need A/m^2 *A*s*m/J * A*m^2/J = m*s*A^3/J^2  - what we get.
        self.factor =  elementary_charge**4*angstrom*TAU_UNIT/hbar**3
        self.fder = 1
        super().__init__(**kwargs)




class Frml_OM(FormulaProduct):
        def __init__(self,data_K,**kwargs_formula):
            super().__init__( [frml.Omega(data_K,**kwargs_formula),frml.morb(data_K,**kwargs_formula)], name='Omega-morb')

class FieldInducedAHC1_orb(StaticCalculator):

    def __init__(self,**kwargs):
        self.Formula = Frml_OM
        # we get the integral in A^-1. first convert to SI (m), 
        # the pre-factor of morb is e/(2*hbar)
        # and the factor  e^2/hbar
        self.factor =  elementary_charge**3*angstrom/(2*hbar**2)
        self.fder = 1
        super().__init__(**kwargs)
