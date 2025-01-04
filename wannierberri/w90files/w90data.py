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
#   some parts of this file are originate                    #
# from the translation of Wannier90 code                     #
# ------------------------------------------------------------#

from functools import cached_property
from copy import copy
import numpy as np
from ..wannierise import wannierise

from .win import WIN
from .eig import EIG
from .mmn import MMN
from .amn import AMN
from .xxu import UIU, UHU, SIU, SHU
from .spn import SPN
from .dmn import DMN
from .chk import CheckPoint, CheckPoint_bare

FILES_CLASSES = {'win': WIN,
                'eig': EIG,
                'mmn': MMN,
                'amn': AMN,
                'uiu': UIU,
                'uhu': UHU,
                'siu': SIU,
                'shu': SHU,
                'spn': SPN,
                'dmn': DMN,
                'chk': CheckPoint,
                }


class Wannier90data:
    """A class to describe all input files of wannier90, and to construct the Wannier functions
     via disentanglement procedure

    Parameters:
        seedname : str
            the prefix of the file (including relative/absolute path, but not including the extensions, like `.chk`, `.mmn`, etc)	
        formatted : tuple(str)
            list of files which should be read as formatted files (uHu, uIu, etc)
        read_npz : bool
            if True, try to read the files converted to npz (e.g. wanier90.mmn.npz instead of wannier90.
        write_npz_list : list(str)
            for which files npz will be written
        write_npz_formatted : bool
            write npz for all formatted files
        overwrite_npz : bool
            overwrite existing npz files  (incompatinble with read_npz)
        read_chk : bool
            if True, read the checkpoint file, otherwise create a '~wannierberri.w90files.CheckPoint_bare' object and prepare for disentanglement
        kmesh_tol : float
            see `~wannierberri.w90files.CheckPoint`
        bk_complete_tol : float
            see `~wannierberri.w90files.CheckPoint`

    Attributes
    ----------
    chk : `~wannierberri.w90files.CheckPoint` or `~wannierberri.w90files.CheckPoint_bare`
        the checkpoint file
    seedname : str
        the prefix of the file (including relative/absolute path, but not including the extensions, like `.chk`, `.mmn`, etc)
    wannierised : bool
        if True, the data is already wannierised (so, it can be used to create a System_w90 object)
    read_npz : bool
        if True, try to read the files converted to npz (e.g. wanier90.mmn.npz instead of wannier90.
    write_npz_list : set(str)
        for which files npz will be written
    write_npz_formatted : bool
        write npz for all formatted files
    formatted_list : list(str)
        list of files which should be read as formatted files (uHu, uIu, etc)
    _files : dict(str, `~wannierberri.w90files.W90_file`)
        the dictionary of the files (e.g. the keys are 'mmn', 'eig', 'amn', 'uiu', 'uhu', 'siu', 'shu', 'spn')
    """
    # todo :  rotate uHu and spn
    # todo : symmetry

    def __init__(self, seedname="wannier90",
                 read_npz=True,
                 write_npz_list=('mmn', 'eig', 'amn'),
                 write_npz_formatted=True,
                 overwrite_npz=False,
                 formatted=tuple(),
                 files=[]
                 ):  # ,sitesym=False):
        assert not (read_npz and overwrite_npz), "cannot read and overwrite npz files"
        self.seedname = copy(seedname)
        self.read_npz = read_npz
        self.write_npz_list = set([s.lower() for s in write_npz_list])
        formatted = [s.lower() for s in formatted]
        if write_npz_formatted:
            self.write_npz_list.update(formatted)
            self.write_npz_list.update(['mmn', 'eig', 'amn'])
        self.formatted_list = formatted
        self._files = {}
        for f in files:
            self.set_file(f)


    def get_spacegroup(self):
        """
        Get the spacegroup of the system from the symmetrizer object
        if the dmn file is not set and cannot be read from the npz file, return None

        Returns
        -------
        `~irrep.spacegroup.SpaceGroupBare`
            the spacegroup of the system
        """
        if hasattr(self, "symmetrizer") and self.symmetrizer is not None:
            return self.symmetrizer.spacegroup
        else:
            return None

    def set_symmetrizer(self, symmetrizer):
        """
        Set the symmetrizer of the system

        Parameters
        ----------
        symmetrizer : `~wanierberri.symmetry.symmetrizer_sawf.SymmetrizerSAWF`
            the symmetrizer object
        """
        self.symmetrizer = symmetrizer

    def set_file(self, key, val=None, overwrite=False,
                 **kwargs):
        """
        Set the file with the key `key` to the value `val`

        Parameters
        ----------
        key : str
            the key of the file, e.g. 'mmn', 'eig', 'amn', 'uiu', 'uhu', 'siu', 'shu', 'spn'
        val : `~wannierberri.w90files.W90_file`
            the value of the file
        overwrite : bool
            if True, overwrite the file if it was already set, otherwise raise an error
        kwargs : dict
            the keyword arguments to be passed to the constructor of the file
            see `~wannierberri.w90files.W90_file`, 
            `~wannierberri.w90files.MMN`, `~wannierberri.w90files.EIG`, `~wannierberri.w90files.AMN`, `~wannierberri.w90files.UIU`, `~wannierberri.w90files.UHU`, `~wannierberri.w90files.SIU`, `~wannierberri.w90files.SHU`, `~wannierberri.w90files.SPN`
            for more details        
        """
        if key == "chk":
            self.set_chk(val=val, overwrite=overwrite, **kwargs)
            return
        kwargs_auto = self.auto_kwargs_files(key)
        kwargs_auto.update(kwargs)
        if not overwrite and self.has_file(key):
            raise RuntimeError(f"file '{key}' was already set")
        if val is None:
            val = FILES_CLASSES[key](self.seedname, **kwargs_auto)
        self.check_conform(key, val)
        self._files[key] = val

    def has_file(self, key):
        """
        Check if the file with the key `key` is set

        Parameters
        ----------
        key : str
            the key of the file, e.g. 'mmn', 'eig', 'amn', 'uiu', 'uhu', 'siu', 'shu', 'spn'

        Returns
        -------
        bool
            True if the file is set, False otherwise
        """
        return key in self._files

    def set_chk(self, val=None, kmesh_tol=1e-7, bk_complete_tol=1e-5, read=False, overwrite=False):
        if not overwrite and self.has_file("chk"):
            raise RuntimeError("chk file was already set")
        if not read:
            val = CheckPoint_bare(win=self.win)
        elif val is None:
            val = CheckPoint(self.seedname, kmesh_tol=kmesh_tol, bk_complete_tol=bk_complete_tol)
        self._files['chk'] = val
        self.wannierised = read
        self.kpt_mp_grid = [tuple(k) for k in
                            np.array(np.round(self.chk.kpt_latt * np.array(self.chk.mp_grid)[None, :]),
                                     dtype=int) % self.chk.mp_grid]
        self.mmn.set_bk(mp_grid=self.chk.mp_grid, kpt_latt=self.chk.kpt_latt, recip_lattice=self.chk.recip_lattice)
        self.win_index = [np.arange(self.eig.NB)] * self.chk.num_kpts

    def set_amn(self, val=None, **kwargs):
        self.set_file("amn", val=val, **kwargs)

    def set_eig(self, val=None, **kwargs):
        self.set_file("eig", val=val, **kwargs)

    def set_mmn(self, val=None, **kwargs):
        self.set_file("mmn", val=val, **kwargs)

    def set_uiu(self, val=None, **kwargs):
        self.set_file("uiu", val=val, **kwargs)

    def set_uhu(self, val=None, **kwargs):
        self.set_file("uhu", val=val, **kwargs)

    def set_siu(self, val=None, **kwargs):
        self.set_file("siu", val=val, **kwargs)

    def set_shu(self, val=None, **kwargs):
        self.set_file("shu", val=val, **kwargs)

    def set_spn(self, val=None, **kwargs):
        self.set_file("spn", val=val, **kwargs)

    def set_win(self, val=None, **kwargs):
        self.set_file("win", val=val, **kwargs)


    def write(self, seedname, files=None):
        """
        writes the files on disk

        Parameters
        ----------
        seedname : str
            the seedname of the files (including relative/absolute path, but not including the extensions, like `.chk`, `.mmn`, etc)
        files : list(str)
            the list of files extensions to be written. If None, all files are written
        """
        if files is None:
            files = self._files.keys()
        for key in files:
            self.get_file(key).to_w90_file(seedname)

    def auto_kwargs_files(self, key):
        """
        Returns the default keyword arguments for the file with the key `key`

        Parameters
        ----------
        key : str
            the key of the file, e.g. 'mmn', 'eig', 'amn', 'uiu', 'uhu', 'siu', 'shu', 'spn'

        Returns
        -------
        dict(str, Any)
            the keyword arguments for the file
        """
        kwargs = {}
        if key in ["uhu", "uiu", "shu", "siu"]:
            kwargs["formatted"] = key in self.formatted_list
        if key not in ["chk", "win"]:
            kwargs["read_npz"] = self.read_npz
            kwargs["write_npz"] = key in self.write_npz_list
        if key not in ["win", "chk"]:
            kwargs["selected_bands"] = self.selected_bands
        if key == "chk":
            kwargs["bk_complete_tol"] = 1e-5
            kwargs["kmesh_tol"] = 1e-7
        print(f"kwargs for {key} are {kwargs}")
        return kwargs


    def get_file(self, key, **kwargs):
        """
        Get the file with the key `key`

        Parameters
        ----------
        key : str
            the key of the file, e.g. 'mmn', 'eig', 'amn', 'uiu', 'uhu', 'siu', 'shu', 'spn'
        kwargs : dict
            the keyword arguments to be passed to the constructor of the file
            see `~wannierberri.w90files.W90_file`, 
            `~wannierberri.w90files.MMN`, `~wannierberri.w90files.EIG`, `~wannierberri.w90files.AMN`, `~wannierberri.w90files.UIU`, `~wannierberri.w90files.UHU`, `~wannierberri.w90files.SIU`, `~wannierberri.w90files.SHU`, `~wannierberri.w90files.SPN`
            for more details

        Returns
        -------
        `~wannierberri.w90files.W90_file`
            the file with the key `key`
        """
        if key not in self._files:
            self.set_file(key, **kwargs)
        return self._files[key]

    def check_conform(self, key, this):
        """
        Check if the file `this` conforms with the other files

        Parameters
        ----------
        key : str
            the key of the file, e.g. 'mmn', 'eig', 'amn', 'uiu', 'uhu', 'siu', 'shu', 'spn'
        this : `~wannierberri.w90files.W90_file`
            the file to be checked

        Raises
        ------
        AssertionError
            if the file `this` does not conform with the other files
        """
        for key2, other in self._files.items():
            for attr in ['NK', 'NB', 'NW', 'NNB']:
                if hasattr(this, attr) and hasattr(other, attr):
                    a = getattr(this, attr)
                    b = getattr(other, attr)
                    if None not in (a, b):
                        assert a == b, f"files {key} and {key2} have different attribute {attr} : {a} and {b} respectively"

    @property
    def win(self):
        """
        Returns the WIN file
        """
        return self.get_file('win')

    @property
    def amn(self):
        """	
        Returns the AMN file
        """
        return self.get_file('amn')

    @property
    def eig(self):
        """
        Returns the EIG file
        """
        return self.get_file('eig')

    @property
    def mmn(self):
        """	
        Returns the MMN file
        """
        return self.get_file('mmn')

    @property
    def uhu(self):
        """	
        Returns the UHU file
        """
        return self.get_file('uhu')

    @property
    def uiu(self):
        """	
        Returns the UIU file
        """
        return self.get_file('uiu')

    @property
    def spn(self):
        """
        Returns the SPN file
        """
        return self.get_file('spn')

    @property
    def siu(self):
        """
        Returns the SIU file
        """
        return self.get_file('siu')

    @property
    def chk(self):
        """
        Returns the checkpoint file
        """
        return self.get_file('chk')

    @property
    def shu(self):
        """
        Returns the SHU file
        """
        return self.get_file('shu')

    @property
    def iter_kpts(self):
        """
        Returns the iterator over the k-points
        """
        return range(self.chk.num_kpts)

    @cached_property
    def wannier_centers(self):
        """
        Returns the Wannier centers stored in the checkpoint file
        """
        return self.chk.wannier_centers

    def check_wannierised(self, msg=""):
        """	
        Check if the system was wannierised

        Parameters
        ----------
        msg : str
            the message to be printed in case of error

        Raises
        ------
        RuntimeError
            if the system was not wannierised
        """
        if not (self.wannierised):
            raise RuntimeError(f"no wannierisation was performed on the w90 input files, cannot proceed with {msg}")

    def wannierise(self, **kwargs):
        """
        Perform the disentanglement procedure calling `~wannierberri.system.disentangle`

        Parameters
        ----------
        kwargs : dict
            the keyword arguments to be passed to `~wannierberri.system.disentangle`     
        """
        wannierise(self, **kwargs)

    @property
    def selected_bands(self):
        if hasattr(self, '_selected_bands'):
            return self._selected_bands
        else:
            return None

    @selected_bands.setter
    def selected_bands(self, value):
        self._selected_bands = value

    def apply_window(self, win_min=-np.inf, win_max=np.inf, band_start=None, band_end=None):
        """
        Apply the window to the system
        Unlike w90, only bands are removed which are fully out of window (at all k-points)

        Parameters
        ----------
        win_min, win_max : float
            the minimum and maximum energy of the window
        band_start, band_end : int
            the range of bands to be included (the numeration in the current system, not in the original (if window is applied more than once))
        """
        new_selected_bands = np.all((self.eig.data >= win_min) * (self.eig.data <= win_max), axis=0)
        print(f"new_selected_bands = {new_selected_bands}")
        if band_end is not None:
            new_selected_bands[band_end:] = False
        if band_start is not None:
            new_selected_bands[:band_start] = False
        print(f"new_selected_bands = {new_selected_bands}")

        print(f"new_selected_bands = {new_selected_bands.shape}")
        _selected_bands = self.selected_bands
        print(f"_selected_bands = {_selected_bands}")
        # if window is applied for the firstr time, the selected bands are all bands
        # and self.selected_bands is bool (=True)
        print("eig", self.eig.data.shape)
        if _selected_bands is None:
            _selected_bands = np.ones(self.eig.NB, dtype=bool)
        print(f"_selected_bands = {_selected_bands}")
        _tmp_selected_bands = _selected_bands[_selected_bands].copy()
        print(f"_tmp_selected_bands = {_tmp_selected_bands}")
        _tmp_selected_bands[np.logical_not(new_selected_bands)] = False
        self.selected_bands = _tmp_selected_bands
        assert np.sum(self.selected_bands) == np.sum(new_selected_bands), "error in applying window"
        print(f"self.selected_bands = {self.selected_bands}")
        for key, val in self._files.items():
            if key != 'win' and key != 'chk':
                print(f"key = {key} ,number of bands = {val.NB}")
            if key == 'chk':
                print(f"key = {key} ,number of bands = {val.num_bands}")
        self.chk.apply_window(new_selected_bands)
        for key in FILES_CLASSES:
            if key in self._files:
                if key == 'win':
                    win = self._files['win']
                    win['dis_win_min'] = win_min
                    win['dis_win_max'] = win_max
                else:
                    print(f"applying window to {key} {val}")
                    self.get_file(key).apply_window(new_selected_bands)
        if hasattr(self, 'symmetrizer'):
            self.symmetrizer.apply_window(new_selected_bands)
        for key, val in self._files.items():
            if key != 'win' and key != 'chk':
                print(f"key = {key} ,number of bands = {val.NB}")
                if hasattr(val, 'data'):
                    print(f"key = {key} ,number of bands = {val.data.shape}")
            if key == 'chk':
                print(f"key = {key} ,number of bands = {val.num_bands}")


    # def get_disentangled(self, files=[]):
    #     """
    #     after disentanglement, get the Wannier90data object with
    #     num_wann == num_bands

    #     Parameters
    #     ----------
    #     files : list(str)
    #         the extensions of the files to be read (e.g. 'mmn', 'eig', 'amn', 'uiu', 'uhu', 'siu', 'shu', 'spn', 'dmn')
    #     """
    #     assert self.wannierised, "wannierisation was not performed"
    #     new_files = {}

    #     v = self.chk.v_matrix
    #     print("v.shape", v.shape)
    #     ham_tmp = np.einsum('kml,km,kmn->kln', v.conj(), self.eig.data, v)
    #     print("ham_tmp.shape", ham_tmp.shape)
    #     EV = [np.linalg.eigh(h_tmp) for h_tmp in ham_tmp]
    #     eig_new = EIG(data=np.array([ev[0] for ev in EV]))
    #     # v_right = np.array([_v @ ev[1].T.conj() for ev, _v in zip(EV,v)])
    #     v_right = np.array([_v @ ev[1] for ev, _v in zip(EV, v)])

    #     v_left = v_right.conj().transpose(0, 2, 1)


    #     for file in files:
    #         if file == 'eig':
    #             new_files['eig'] = eig_new
    #         else:
    #             new_files[file] = self.get_file(file).get_disentangled(v_left, v_right)
    #             if file == 'mmn':
    #                 new_files[file].neighbours = self.mmn.neighbours
    #                 new_files[file].ib_unique_map = self.mmn.ib_unique_map
    #                 new_files[file].G = self.mmn.G
    #     win = deepcopy(self.win)
    #     win.NB = self.chk.num_wann
    #     new_files['win'] = win
    #     other = Wannier90data(read_chk=False, files=new_files)
    #     return other


    def check_symmetry(self, silent=False):
        """
        Check the symmetry of the system
        """

        err_eig = self.symmetrizer.check_eig(self.eig)
        err_amn = self.symmetrizer.check_amn(self.amn)
        if not silent:
            print(f"eig symmetry error : {err_eig}")
            print(f"amn symmetry error : {err_amn}")
        return err_eig, err_amn

    def set_random_symmetric_projections(self):
        """
        Set random symmetric projections for the system,
        according to the dmn file
        """
        self.set_file("amn", self.dmn.get_random_amn(), overwrite=True)

    # def set_d_band(self, bandstructure, overwrite=False):
    #     """
    #     Set the d-band from the bandstructure object

    #     Parameters
    #     ----------
    #     bandstructure : irrep.bandstructure.BandStructure
    #         the bandstructure object
    #     overwrite : bool
    #         if False and the dmn file already exists, raise an error
    #     """
    #     dmn_new = DMN(empty=True)
    #     dmn_new.from_irrep(bandstructure)
    #     self.set_file("dmn", dmn_new, overwrite=overwrite)

    # def set_D_wann_from_projections(self, projections):
    #     """
    #     Set the Dwann matrix from the projections

    #     Parameters
    #     ----------
    #     projections : list(tuple(np.array(3), str))
    #         the list of projections, each element is a tuple of the position of the projection and the orbital type
    #     """
    #     if "dmn" not in self._files:
    #         raise RuntimeError("First set the bands part of dmn file")
    #     self.dmn.set_D_wann_from_projections(projections=projections)




    # TODO : allow k-dependent window (can it be useful?)
    # def apply_outer_window(self,
    #                  win_min=-np.inf,
    #                  win_max=np.inf ):
    #     raise NotImplementedError("outer window does not work so far")
    #     "Excludes the bands from outside the outer window"
    #
    #     def win_index_nondegen(ik,thresh=DEGEN_THRESH):
    #         "define the indices of the selected bands, making sure that degenerate bands were not split"
    #         E=self.Eig[ik]
    #         ind=np.where( ( E<=win_max)*(E>=win_min) )[0]
    #         while ind[0]>0 and E[ind[0]]-E[ind[0]-1]<thresh:
    #             ind=[ind[0]-1]+ind
    #         while ind[0]<len(E) and E[ind[-1]+1]-E[ind[-1]]<thresh:
    #             ind=ind+[ind[-1]+1]
    #         return ind
    #
    #     # win_index_irr=[win_index_nondegen(ik) for ik in self.Dmn.kptirr]
    #     # self.excluded_bands=[list(set(ind)
    #     # self.Dmn.select_bands(win_index_irr)
    #     # win_index=[win_index_irr[ik] for ik in self.Dmn.kpt2kptirr]
    #     win_index=[win_index_nondegen(ik) for ik in self.iter_kpts]
    #     self._Eig=[E[ind] for E, ind in zip(self._Eig,win_index)]
    #     self._Mmn=[[self._Mmn[ik][ib][win_index[ik],:][:,win_index[ikb]] for ib,ikb in enumerate(self.mmn.neighbours[ik])] for ik in self.iter_kpts]
    #     self._Amn=[self._Amn[ik][win_index[ik],:] for ik in self.iter_kpts]
