#
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
#------------------------------------------------------------
#
#  This is the main file of the library, and the user is suposed 
#  to import only it. 


import functools 
from .__evaluate import evaluate_K, determineNK
from .__utility import smoother 
from . import __integrate 
from . import __tabulate  
from . import __symmetry as symmetry
from .__system import System
from .__version import __version__
from .__result import NoComponentError
from collections import Iterable
integrate_options=__integrate.calculators.keys()
tabulate_options =__tabulate.calculators.keys()





import sys,glob


from colorama import init
from termcolor import cprint 
from pyfiglet import figlet_format

def figlet(text,font='cosmike',col='red'):
    init(strip=not sys.stdout.isatty()) # strip colors if stdout is redirected
    letters=[figlet_format(X, font=font).rstrip("\n").split("\n") for X in text]
#    print (letters)
    logo=[]
    for i in range(len(letters[0])):
        logo.append("".join(L[i] for L in letters))
    cprint("\n".join(logo),col, attrs=['bold'])


def print_options():
    cprint ("Options available to integrate:",'green', attrs=['bold'])
    print("\n".join("{0:10s}  :  {1} ".format(key,__integrate.descriptions[key]) for key in integrate_options)+"\n\n")
    cprint ("Options available to tabulate:",'green', attrs=['bold'])
    print("\n".join("{0:10s}  :  {1} ".format(key,__tabulate.descriptions[key]) for key in tabulate_options)+"\n\n")
      


figlet("Wannier 19",font='speed',col='yellow')
figlet("    by Stepan Tsirkin",font='straight',col='green')

cprint( "\nVersion: {}\n".format( __version__),'cyan', attrs=['bold'])
print_options()

#for font in ['twopoint','contessa','tombstone','thin','straight','stampatello','slscript','short','pepper']:
#    __figlet("by Stepan Tsirkin",font=font,col='green')

    

def check_option(quantities,avail,tp):
    for opt in quantities:
      if opt not in avail:
        raise RuntimeError("Quantity {} is not available for {}. Available options are : \n{}\n".format(opt,tp,avail) )


def integrate(system,NK=None,NKdiv=None,NKFFT=None,Efermi=None,omega=None, Ef0=0,
                        smearEf=10,smearW=10,quantities=[],adpt_num_iter=0,
                        fout_name="w19",symmetry_gen=[],
                GammaCentered=True,restart=False,numproc=0,file_klist="klist_int"):

    cprint ("\nIntegrating the following qantities: "+", ".join(quantities)+"\n",'green', attrs=['bold'])
    check_option(quantities,integrate_options,"integrate")
    smooth=smoother(Efermi,10)
    eval_func=functools.partial(  __integrate.intProperty, Efermi=Efermi, smootherEf=smooth,quantities=quantities )
    res=evaluate_K(eval_func,system,NK=NK,NKdiv=NKdiv,NKFFT=NKFFT,nproc=numproc,
            adpt_num_iter=adpt_num_iter,adpt_nk=1,
                fout_name=fout_name,symmetry_gen=symmetry_gen,
                GammaCentered=GammaCentered,restart=restart,file_klist=file_klist)
    cprint ("Integrating finished successfully",'green', attrs=['bold'])
    return res



def tabulate(system,NK=None,NKdiv=None,NKFFT=None,omega=None, quantities=[],symmetry_gen=[],
                  fout_name="w19",ibands=None,file_klist="klist_tab",
                      restart=False,numproc=0,Ef0=0):

    cprint ("\nTabulating the following qantities: "+", ".join(quantities)+"\n",'green', attrs=['bold'])
    NKdiv,NKFFT=determineNK(NKdiv,NKFFT,NK,system.NKFFTmin)
    NK=NKdiv*NKFFT
    print ("swebwtbwt",NKdiv,NKFFT,NK,system.NKFFTmin)

    check_option(quantities,tabulate_options,"tabulate")
    eval_func=functools.partial(  __tabulate.tabXnk, ibands=ibands,quantities=quantities )

    res=evaluate_K(eval_func,system,NK=NK,NKdiv=NKdiv,NKFFT=NKFFT,nproc=numproc,
            adpt_num_iter=0 ,symmetry_gen=symmetry_gen,  GammaCentered=True ,restart=restart,file_klist=file_klist)
            
    res=res.to_grid(NKFFT*NKdiv)
        
    open("{0}_E.frmsf".format(fout_name),"w").write(
         res.fermiSurfer(quantity=None,efermi=Ef0) )
    
    for Q in quantities:
#     for comp in ["x","y","z","sq","norm"]:
     for comp in ["x","y","z","xx","yy","zz","xy","yx","xz","zx","yz","zy"]:
        try:
            txt=res.fermiSurfer(quantity=Q,component=comp,efermi=Ef0)
            open("{2}_{1}-{0}.frmsf".format(comp,Q,fout_name),"w").write(txt)
        except NoComponentError:
            pass

    cprint ("Tabulating finished successfully",'green', attrs=['bold'])
    return res



