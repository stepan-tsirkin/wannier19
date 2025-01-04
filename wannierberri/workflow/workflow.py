import copy
import os
import pickle
import shutil
import warnings
from irrep.bandstructure import BandStructure
from matplotlib import pyplot as plt
import numpy as np
from wannierberri.symmetry.symmetrizer_sawf import SymmetrizerSAWF
from wannierberri.w90files.amn import amn_from_bandstructure
from wannierberri.w90files.w90data import FILES_CLASSES
from ..wannierise.projections import num_orbitals
from ..w90files import WIN, Wannier90data
from .. parallel import Serial as wbSerial
from .ase import write_espresso_in
from ..grid import Path
from ..system import System_w90
from .. import calculators as wbcalculators
from .. import run as wbrun


def message(msg):
    l = len(msg) + 4
    print("#" * l)
    print("# " + msg + " #")
    print("#" * l)


class Executables:
    """
    Class to keep track of the paths, executables and parallel options for
    the Quantum Espresso and Wannier90 executables

    Parameters
    ----------
    path_to_pw: str
        Path to the Quantum Espresso executables
    path_to_w90: str
        Path to the Wannier90 executables
    parallel: bool
        If True, run the calculations in parallel
    npar: int
        Number of parallel processes
    npar_k: int
        Number of parallel processes for k-points
    mpi: str
        Command to run mpi
    parallel_wb: wannierberri.parallel.Parallel
        Object to run the wannier-berri calculations in parallel

    Attributes
    ----------
    pwx: str
        Path to pw.x
    pw2wanx: str
        Path to pw2wannier90.x
    wannierx: str
        Path to wannier90.x
    bandsx: str
        Path to bands.x
    parallel_wb: wannierberri.parallel.Parallel
        Object to run the wannier-berri calculations in parallel
    """



    def __init__(self, path_to_pw='', path_to_w90='',
                 parallel=False, npar=1, npar_k=None, mpi="mpirun",
                 parallel_wb=None):
        self.pwx = os.path.join(path_to_pw, 'pw.x')
        self.pw2wanx = os.path.join(path_to_pw, 'pw2wannier90.x')
        self.wannierx = os.path.join(path_to_w90, 'wannier90.x')
        self.wannierx_serial = np.copy(self.wannierx)
        self.bandsx = os.path.join(path_to_pw, 'bands.x')
        if parallel:
            if npar_k is not None:
                park = f" -nk {npar_k} "
            else:
                park = ""
            self.pwx = f'{mpi} -np {npar} {self.pwx} {park}'
            self.pw2wanx = f'{mpi} -np {npar} {self.pw2wanx} '
            self.bandsx = f'{mpi} -np {npar} {self.bandsx}'
        self.parallel_wb = parallel_wb
        if parallel_wb is None:
            self.parallel_wb = wbSerial()


        print("pw.x : ", self.pwx)
        print("pw2wannier90.x : ", self.pw2wanx)
        print("wannier90.x : ", self.wannierx)
        print("wannier90.x serial : ", self.wannierx_serial)
        print("bands.x : ", self.bandsx)


    def run_pw(self, f_in, f_out):
        os.system(f'{self.pwx} -i {f_in} > {f_out}')

    def run_pw2wannier(self, f_in, f_out):
        os.system(f'{self.pw2wanx} < {f_in}  > {f_out}')

    def run_wannier(self, prefix, pp=False):
        if pp:
            os.system(f'{self.wannierx_serial} -pp {prefix}')
        else:
            os.system(f'{self.wannierx}  {prefix}')

    def run_bands(self, f_in, f_out):
        os.system(f'{self.bandsx} -i {f_in} > {f_out}')


# class FlagsCalculated:
#     """
#     Class to keep track of the quantities that have been calculated

#     Parameters
#     ----------
#     flags: list(str)
#         List of flags
#     depend: dict(str, list(str))
#         Dictionary with the flags that each flag depends on.
#         i.e. if the other flags are not calculated, the current flag cannot be calculated
#         fro the dependance calculation will be triggered
#     map_forward: dict(str, list(str))
#         Dictionary with the flags that each flag sets to False when it is set to True
#         i.e. which objects need to be recalculated after the current flag is calculated
#     """

#     def __init__(self, flags=[], depend={}, map_forward={}):
#         self.allflags = flags
#         self.map_backward = defaultdict(list)
#         self.map_backward.update(depend)
#         self.map_forward = defaultdict(list)
#         self.map_forward.update(map_forward)
#         for map in self.map_forward, self.map_backward:
#             for key, value in map.items():
#                 assert key in self.allflags, f"Flag {key} not in {self.allflags}"
#                 for v in value:
#                     assert v in self.allflags, f"Flag {v} not in {self.allflags}"
#         self.calculated = {flag: False for flag in flags}

#     def __getitem__(self, flag):
#         assert flag in self.allflags, f"Flag {flag} not in {self.allflags}"
#         return self.calculated[flag]

#     def on(self, flag):
#         """
#         Set the flag to True and all the flags that depend on it 
#         (via map_forward) to False

#         Parameters
#         ----------
#         flag: str
#             The flag to set to True
#         """
#         assert flag in self.allflags, f"Flag {flag} not in {self.allflags}"
#         self.calculated[flag] = True
#         for f in self.map_forward[flag]:
#             self.off(f)

#     def off(self, flag):
#         """
#         Set the flag to False and all the flags that depend on it
#         (via map_backward) to False

#         Parameters
#         ----------
#         flag: str
#             The flag to set to False
#         """
#         assert flag in self.allflags, f"Flag {flag} not in {self.allflags}"
#         self.calculated[flag] = False
#         for f in self.map_forward[flag]:
#             self.off(f)

#     def check(self, flag):
#         """
#         Check if the flag is calculated, and if all the flags that it depends on
#         are calculated

#         Parameters
#         ----------
#         flag: str
#             The flag to check

#         Returns
#         -------
#         bool

#         Raises
#         ------
#         AssertionError
#             If the flag is not in the list of flags
#             If a flag that is needed for the calculation is not calculated
#         """
#         assert flag in self.allflags, f"Flag {flag} not in {self.allflags}"
#         for f in self.map_backward.get(flag, []):
#             assert self.calculated[f], f"Flag {f} not calculated, needed for {flag}"
#         return self.calculated[flag]


# class FlagsCalculatedVoid(FlagsCalculated):

#     def __init__(self):
#         self.calculated = {"anything": False}
#         pass

#     def on(self, flag):
#         pass

#     def off(self, flag):
#         pass

#     def check(self, flag):
#         return False

#     def __getitem__(self, flag):
#         return False




class WorkflowQE:
    """
    Class to keep track of the workflow for Quantum Espresso 
    calculations with Wannier90

    Parameters
    ----------
    atoms: ase.Atoms object
        The system to calculate
    pseudopotentials: dict(str, str)
        Dictionary with the pseudopotentials for the elements in the system
        {element: pseudopotential_file}
    prefix: str
        Prefix for the calculations
    pseudo_dir: str
        Directory with the pseudopotential files
    executables: Executables
        Object with the paths to the executables
    num_bands: int
        Number of bands to calculate
    pickle_file: str
        File to save the object with pickle

    """

    def __init__(self,
                 atoms=None,
                 pseudopotentials={},
                 prefix='crystal',
                 pseudo_dir='./',
                 executables=None,
                 spinor=False,
                num_bands=20,
                pickle_file=None,
                try_unpickle=True,
                k_nodes=[[0, 0, 0], [0.5, 0.5, 0.5]],
                sitesym=True,
                kwargs_gen={}, kwargs_gs={}, kwargs_nscf={}, kwargs_bands={}, kwargs_wannier={},
                ):

        warnings.warn("Workflow is not fully implemented yet. It is experimental and may not work as expected. Use it on your own risk")
        self.sitesym = sitesym
        path_to_files = os.path.dirname(prefix)
        if not os.path.exists(path_to_files):
            os.mkdir(path_to_files)
        if pickle_file is None:
            pickle_file = f'{prefix}.pkl'
        self.pickle_file = pickle_file
        if try_unpickle:
            try:
                self.unpickle(self.pickle_file)
                # print(f"unpickled {self.pickle_file}, flags are {self.flags.calculated}")
                print(f"unpickled from {self.pickle_file}")
                return
            except Exception as e:
                print(f"Could not unpickle : {e}")
                pass

        self.atoms = atoms
        self.pseudopotentials = pseudopotentials
        self.prefix = prefix
        self.executables = Executables() if executables is None else executables
        self.k_nodes = k_nodes
        self.num_bands = num_bands
        self.bandstructure = None
        self.projections_set = None

        self.kwargs_gen = dict(
            ecutwfc=20,
            lspinorb=False,
            noncolin=False,
            occupations='smearing',
            smearing='cold',
            degauss=0.02,
            restart_mode='from_scratch',
            outdir='./',
            verbosity='low',
            startingwfc='random',
            diagonalization='cg',
            conv_thr=1.0e-10,
            pseudo_dir=pseudo_dir,
        )
        self.kwargs_gen.update(kwargs_gen)
        self.kwargs_gen["prefix"] = prefix
        self.kwargs_wannier = copy.copy(kwargs_wannier)
        if spinor:
            self.kwargs_gen["lspinorb"] = True
            self.kwargs_gen["noncolin"] = True
            self.kwargs_wannier["spinors"] = True
        self.spinor = spinor

        self.kwargs_gs = dict(calculation='scf',
                            kpts=(6, 6, 4),
                        )

        self.kwargs_gs.update(self.kwargs_gen)
        self.kwargs_gs.update(kwargs_gs)

        self.kwargs_bands = dict(calculation='bands',
                                nbnd=num_bands,
                                )
        self.kwargs_bands.update(self.kwargs_gen)
        self.kwargs_bands.update(kwargs_bands)

        self.pw2wannier_full_list = ['eig', 'mmn', 'amn', 'spn', 'uhu', 'uiu', 'unk', 'dmn']

        self.kwargs_nscf = dict(calculation='nscf',
                                nosym=True,
                                )
        self.kwargs_nscf.update(self.kwargs_gen)
        self.kwargs_nscf.update(kwargs_nscf)
        self.kwargs_nscf["nbnd"] = num_bands


        # if use_flags:
        #     self.flags = FlagsCalculated(
        #         ['gs', 'nscf', 'pw2wannier', 'wannier_w90',
        #          'bands_qe', 'bands_wannier_w90',
        #          'wannierise_wberri', 'bands_wannier_wberri',
        #          'win', 'projections', 'dmn'],
        #         depend=dict(nscf=['gs'],
        #                     pw2wannier=['nscf', 'win'],
        #                     win=['nscf', 'projections'],
        #                     wannier_w90=['pw2wannier', 'win'],
        #                     bands_qe=['gs'],
        #                     bands_wannier_w90=['wannier_w90'],
        #                     wannierise_wberri=['pw2wannier'],
        #                     bands_wannier_wberri=['wannierise_wberri'],
        #                     ),
        #         map_forward=dict(gs=['nscf', 'bands_qe'],
        #                          nscf=['pw2wannier'],
        #                          pw2wannier=['wannier_w90', 'wannierise_wberri'],
        #                          wannier_w90=['bands_wannier_w90'],
        #                          wannierise_wberri=['bands_wannier_wberri'],
        #                          projections=['win', 'pw2wannier'],
        #                          win=['wannier_w90'],
        #                             ))
        # else:
        #     self.flags = FlagsCalculatedVoid()

        self.pickle()


    def ground_state(self, kpts=(8, 8, 8), run=True, **kwargs):
        """
        Do the self-consistent calculation

        Parameters
        ----------
        kpts: array-like(3)
            The k-points grid
        run: bool
            If True, run the calculation, if False, just write the input file
        kwargs: dict
            Additional parameters for the calculation (inputs to the Quantum Espresso)
        """
        message("Ground state")
        self.kwargs_gs.update(kwargs)
        f_in = f'{self.prefix}.scf.in'
        f_out = f'{self.prefix}.scf.out'
        write_espresso_in(f_in, self.atoms, kpts=kpts,
                          pseudopotentials=self.pseudopotentials,
                          input_data=self.kwargs_gs)
        if run:
            self.executables.run_pw(f_in, f_out)
        message("Done")
        self.pickle()

    def unpickle(self, file):
        self.__dict__.update(pickle.load(open(file, 'rb')).__dict__)

    def pickle(self, file=None):
        if file is None:
            file = self.pickle_file
        pickle.dump(self, open(file, 'wb'))

    def nscf(self, mp_grid=(4, 4, 4), run=True, set_bandstructure=True, Ecut_loc=100, **kwargs):
        """
        Do the non-self-consistent calculation

        Parameters
        ----------
        mp_grid: array-like(3)
            The Monkhorst-Pack grid
        run: bool
            If True, run the calculation, if False, just write the input file
        kwargs: dict
            Additional parameters for the calculation (inputs to the Quantum Espresso)
        """
        message("NSCF")
        self.mp_grid = mp_grid
        self.kwargs_nscf.update(kwargs)
        self.kpoints_nscf = np.array([[i, j, k] for i in range(mp_grid[0]) for j in range(mp_grid[1]) for k in range(mp_grid[2])]) / np.array(mp_grid)[None, :]
        f_in = f'{self.prefix}.nscf.in'
        f_out = f'{self.prefix}.nscf.out'
        write_espresso_in(f_in, self.atoms, kpoints_array=self.kpoints_nscf,
                          pseudopotentials=self.pseudopotentials,
                          input_data=self.kwargs_nscf)
        if run:
            self.executables.run_pw(f_in, f_out)
        message("Done")
        self.pickle()
        if set_bandstructure:
            return self.set_bandstructure(Ecut_loc = Ecut_loc)
        
    def set_projections(self, projections_set):
        """
        set projections for future w90 calculations, also updates num_wann and projections_str

        Parameters
        ----------
        projections: ProjectionsSet
            the projections to set
        """
        self.projections_set = projections_set.as_numeric().split_orbitals()
        print("projections are\n", projections_set)
        self.num_wann = self.projections_set.num_wann
        self.projections_str = projections_set.write_wannier90(mod1=False, beginend=False, numwann=False)
        self.set_amn_internally()
        self.set_symmetrizer(bands=False, proj=True)
        self.pickle()

    def clear(self, *args):
        for arg in args:
            try:
                delattr(self, arg)
            except AttributeError:
                pass

    def write_win(self,**kwargs):
        data = dict(kpoints=self.kpoints_nscf,
                unit_cell_cart=self.atoms.get_cell(),
                atoms_frac=self.atoms.get_scaled_positions(),
                atoms_names=self.atoms.get_chemical_symbols(),
                num_wann=self.num_wann,
                num_bands=self.num_bands,
                projections=self.projections_str,
                num_iter=0,
                dis_num_iter=0,
                spinors=self.spinor,
                )
        data.update(self.kwargs_wannier)
        data.update(kwargs)
        win = WIN(seedname=None, data=data)
        win.write(self.prefix)
        self.pickle()

    def pw2wannier(self, targets=['eig', 'mmn'], run=True, **kwargs):
        message("pw2wannier")
        self.executables.run_wannier(self.prefix, pp=True)
        f_in = f'{self.prefix}.pw2wan.in'
        f_out = f'{self.prefix}.pw2wan.out'
        open(f_in, 'w').write(f"""&inputpp\n  outdir = './'\n  prefix = '{self.prefix}'\n  seedname = '{self.prefix}'\n""" +
                          "\n".join([f"write_{x} = .{x in targets}." for x in self.pw2wannier_full_list]) + "\n/\n"
                          )
        if run:
            self.executables.run_pw2wannier(f_in, f_out)
        message("Done")
        self.pickle()

    def wannierise_w90(self):
        message("Wannier-w90")
        self.executables.run_wannier(self.prefix)
        message("Done")
        self.pickle()

    def wannierise_wberri(self, kwargs_system={}, kwargs_window={}, init="amn-internal", **kwargs):
        w90data = Wannier90data(seedname=self.prefix, files=["eig", "mmn",])
        w90data.set_symmetrizer(self.symmetrizer)
        if init=="amn-internal":
            w90data.set_amn(self.amn_int)
            init_w="amn"
        elif init=="amn-external":
            w90data.set_amn()
            init_w="amn"
        else:
            assert init=="random", f"init should be 'amn-internal', 'amn-external' or 'random'. found  '{init}'"
            init_w="random"
        w90data.apply_window(**kwargs_window)
        # for key in ["amn", "mmn", "eig", "dmn"]:
        #     print (f"w90data[{key}].NB = {w90data.get_file(key).NB}")
        w90data.wannierise(init=init_w, **kwargs)
        self.system_wberri = System_w90(w90data=w90data, **kwargs_system)
        self.pickle()

    def set_bandstructure(self, Ecut_loc=100):
        self.bandstructure = BandStructure(code='espresso',
                                      prefix=self.prefix,
                                      normalize=False,
                                      Ecut=Ecut_loc
                                    )
        self.set_symmetrizer()
        self.set_amn_internally()
        self.pickle()
        return self.bandstructure

    def set_symmetrizer(self, bands=True, proj=True):
        if bands:
            self.symmetrizer = SymmetrizerSAWF().from_irrep(self.bandstructure)
        if self.projections_set is not None and proj:
            self.symmetrizer.set_D_wann_from_projections(projections_obj=self.projections_set)
    
    def set_amn_internally(self):
        if self.projections_set is not None:
            self.amn_int = amn_from_bandstructure(bandstructure=self.bandstructure, projections_set=self.projections_set)

    @property
    def spacegroup(self):
        return self.bandstructure.spacegroup

    def calc_bands_wannier_w90(self, kdensity=1000, enforce=False):
        if self.flags.check('bands_wannier_w90') and not enforce:
            return
        system = System_w90(self.prefix)
        self.path_wannier, self.bands_wannier_w90 = get_wannier_band_structure(system, self.k_nodes, length=kdensity,
                                                                           parallel=self.executables.parallel_wb)
        self.flags.on('bands_wannier_w90')
        self.pickle()

    def calc_bands_wannier_wberri(self, kdensity=1000, enforce=False):
        system = self.system_wberri
        self.path_wannier, self.bands_wannier_wberri = get_wannier_band_structure(system, self.k_nodes, length=kdensity,
                                                                           parallel=self.executables.parallel_wb)
        self.pickle()


    def calc_bands_qe(self, kdensity,  run=True, **kargs):
        self.kwargs_bands.update(kargs)
        message("Band structure QE")
        prefix_tmp = self.prefix + '_bands'
        self.kwargs_bands["prefix"] = prefix_tmp
        dir1 = self.prefix + '.save'
        dir2 = prefix_tmp + '.save'
        if os.path.exists(dir2):
            shutil.rmtree(dir2)
        os.mkdir(dir2)
        for f in ["charge-density.dat", "data-file-schema.xml"]:
            shutil.copy(dir1 + '/' + f, dir2 + '/' + f)
        if os.path.exists(dir1 + '/paw.txt'):
            shutil.copy(dir1 + '/paw.txt', dir2 + '/paw.txt')
        self.path_qe = Path(system=self.atoms.get_cell(), k_nodes=self.k_nodes, length=kdensity)
        f_in = f'{self.prefix}.bands.in'
        f_out = f'{self.prefix}.bands.out'
        write_espresso_in(f_in, self.atoms, kpoints_array=self.path_qe.K_list,
                          pseudopotentials=self.pseudopotentials,
                          input_data=self.kwargs_bands)
        if run:
            self.executables.run_pw(f_in, f_out)
        f_in = f'{self.prefix}.bandsx.in'
        f_out = f'{self.prefix}.bandsx.out'
        open(f_in, 'w').write(f"""&bands\n  prefix = '{prefix_tmp}'\n  outdir = './'\n  lsym = .false.\n  filband = '{self.prefix}.bands.dat'\n/\n""")
        self.executables.run_bands(f_in, f_out)
        self.kline_qe = self.path_qe.getKline()
        nk = len(self.kline_qe)
        bands_qe = np.loadtxt(f'{self.prefix}.bands.dat.gnu')
        self.bands_qe = bands_qe[:, 1].reshape(-1, nk)
        message("Done")
        self.pickle()

    def plot(self, show=True, savefig=None, ylim=None):
        try:
            for band in self.bands_qe:
                plt.scatter(self.kline_qe, band, c='g', s=4)
        except AttributeError:
            pass

        try:
            self.bands_wannier_w90.plot_path_fat(self.path_wannier, show_fig=False, close_fig=False, linecolor='b', label="w90")
        except AttributeError:
            pass

        try:
            self.bands_wannier_wberri.plot_path_fat(self.path_wannier, show_fig=False, close_fig=False, linecolor='r', label="wberri")
        except AttributeError:
            pass

        if ylim is not None:
            plt.ylim(ylim)
        if savefig is not None:
            plt.savefig(savefig)
        if show:
            plt.show()


def get_wannier_band_structure(system, k_nodes, length=1000, npar=0, parallel=None):
    """
    Calculate or (try to) read the band structure using wannierberri

    Parameters
    ----------

    system: wb.system.System object
        The system with the Wannier functions
    k_nodes: array-like(num_points,3)
        Nodes of the k-point path

    Returns
    -------
    wb.Path object
    wb.reslut.TABresult object
    """
    path = Path(system, k_nodes=k_nodes, length=length)
    if parallel is None:
        parallel = parallel.Serial()
    calculators = dict(tabulate=wbcalculators.TabulatorAll(tabulators={}, mode='path'))
    result = wbrun(system, grid=path, calculators=calculators, parallel=parallel)
    return path, result.results['tabulate']






def run_pw2wannier(projections=[],
                   win=None,
                   prefix='pwscf',
                   outdir='./',
                   seedname='pwscf',
                   targets=['eig', 'mmn', 'amn'],
                   pw2wan_cmd='pw2wannier90.x',
                   w90_cmd='wannier90.x',
                   kwargs_wannier={},
                   return_dict=False):
    pw2wan_full_list = ['eig', 'mmn', 'amn', 'spn', 'uhu', 'uiu', 'unk', 'dmn']
    if win is None:
        win = WIN(seedname)

    if projections is not None:
        win["projections"] = projections
    num_wann = 0
    for x in projections:
        num_wann += num_orbitals(x.split(":")[1])
    win["num_wann"] = num_wann
    win.update(kwargs_wannier)
    win.write(seedname)
    os.system(f'{w90_cmd} -pp {seedname}')
    fname = f'{seedname}-{"+".join(targets)}.pw2wan'
    f_in = fname + '.in'
    f_out = fname + '.out'
    f_in_txt = f"""&inputpp\n  outdir = '{outdir}'\n  prefix = '{prefix}'\n  seedname = '{seedname}'\n"""
    for x in pw2wan_full_list:
        f_in_txt += f"write_{x} = .{x in targets}.\n"
    f_in_txt += "/\n"
    open(f_in, 'w').write(f_in_txt)
    os.system(f'{pw2wan_cmd} < {f_in}  > {f_out}')
    return_dict = {}
    for x in targets:
        return_dict[x] = FILES_CLASSES[x](seedname)
    if len(targets) == 1 and not return_dict:
        return return_dict[targets[0]]
    return return_dict
