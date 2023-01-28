"""
The module describes calculators - objects that 
receive :calss:`~wannierberri.data_K.Data_K` objects and yield
:class:`~wannierberri.result.Result`
"""

import abc
import numpy as np
from wannierberri.result import KBandResult,TABresult
from termcolor import cprint

class Calculator():

    def __init__(self, degen_thresh=1e-4, degen_Kramers=False, save_mode="bin+txt", print_comment=True):
        self.degen_thresh = degen_thresh
        self.degen_Kramers = degen_Kramers
        self.save_mode = save_mode
        if not hasattr(self, 'comment'):
            if self.__doc__ is not None:
                self.comment = self.__doc__
            else:
                self.comment = "calculator not described"
        if print_comment:
            cprint("{}\n".format(self.comment), 'cyan', attrs=['bold'])

    @property
    def allow_path(self):
        return False    # change for those who can be calculated on a path instead of a grid

    @property
    def allow_sym(self):
        return True

from . import static, dynamic, tabulate, tabulateOD

class TabulatorAll(Calculator):
    """    Calculator that wraps all tabulators
    Parameters
    ----------
    tabulators : dict str : :class:`~wannierberri.calculators.Tabulator` or :class:`~wannierberri.calculators.TabulatorOD`
        one of them should be "Energy" 
    ibands : list
        select band indices "i" in X_{ij}
    jbands : list
        select band indices "j" in X_{ij}
    mode : str
        "grid" or "path"
    save_mode : 
        "npz" or "npy" or "frmsf" (later for grid mode only)
    """

    def __init__(self, tabulators, ibands=None, jbands=None, mode="grid", save_mode="npz"):

        self.tabulators = {}
        self.tabulators.update(tabulators)
        mode = mode.lower()
        assert mode in ("grid","path")
        self.mode = mode
        self.save_mode = save_mode
        if "frmsf in save_mode":
            if "_Energy" not in self.tabulators.keys():
                self.tabulators["_Energy"] = tabulate.Energy()
#        self.tabulators["_Energy"].ibands = None  # tabulate all energies for FermiSurfer
        if ibands is not None:
            ibands = np.array(ibands)
        if jbands is not None:
            jbands = np.array(jbands)
        for k, v in self.tabulators.items():
            if v.ibands is None:
                v.ibands = ibands
            if hasattr(v,'jbands'):
                if v.jbands is None:
                    v.jbands = jbands


    def __call__(self, data_K):
        return TABresult(
            kpoints=data_K.kpoints_all.copy(),
            mode=self.mode,
            recip_lattice=data_K.system.recip_lattice,
            save_mode=self.save_mode,
            results={k: v(data_K)
                     for k, v in self.tabulators.items()} )


    @property
    def allow_path(self):
        return self.mode == "path"

    @property
    def allow_sym(self):
        return all([t.allow_sym for t in self.tabulators.values()])


