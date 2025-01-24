"""Create system objects."""

import os
import tarfile
import pytest
import numpy as np
import pickle
import wannierberri as wberri
from wannierberri import point_symmetry as SYM
from pathlib import Path
from wannierberri import models as wb_models

from .common import ROOT_DIR

symmetries_Fe = [SYM.C4z, SYM.C2x * SYM.TimeReversal, SYM.Inversion]
symmetries_Te = ["C3z", "C2x", "TimeReversal"]
symmetries_GaAs = [SYM.C4z * SYM.Inversion, SYM.TimeReversal, SYM.Rotation(3, [1, 1, 1])]
symmetries_Si = ["C4z", "C4x", "TimeReversal"]
symmetries_Mn3Sn = ["C3z"]

Efermi_Fe = np.linspace(17, 18, 11)
Efermi_Te_gpaw = np.linspace(4, 8, 11)
Efermi_Te_sparse = np.linspace(4, 8, 11)
Efermi_Fe_FPLO = np.linspace(-0.5, 0.5, 11)
Efermi_GaAs = np.linspace(7, 9, 11)
Efermi_Haldane = np.linspace(-3, 3, 11)
Efermi_CuMnAs_2d = np.linspace(-2, 2, 11)
Efermi_Chiral = np.linspace(-5, 8, 27)
omega_chiral = np.linspace(0, 1., 11)
omega_phonon = np.linspace(-0.01, 0.1, 23)
Efermi_Mn3Sn = np.linspace(2, 3, 11)


def create_W90_files(seedname, tags_needed, data_dir, tags_untar=["mmn", "amn"]):
    """
    Extract the compressed amn and mmn data files.
    Create files listed in tags_needed using utils.mmn2uHu.
    """

    # Extract files if is not already done
    for tag in tags_untar:
        if not os.path.isfile(os.path.join(data_dir, f"{seedname}.{tag}")):
            tar = tarfile.open(os.path.join(data_dir, f"{seedname}.{tag}.tar.gz"))
            for tarinfo in tar:
                tar.extract(tarinfo, data_dir)

    # Compute tags only if the corresponding files do not exist
    tags_compute = []
    for tag in tags_needed:
        if not os.path.isfile(os.path.join(data_dir, f"{seedname}.{tag}")):
            tags_compute.append(tag)

    if len(tags_compute) > 0:
        kwargs = {}
        for tag in tags_compute:
            kwargs["write" + tag.upper()] = True

        nb_out_list = wberri.utils.mmn2uHu.run_mmn2uHu(
            seedname, INPUTDIR=data_dir, OUTDIR=str(data_dir) + "/reduced", **kwargs)
        nb_out = nb_out_list[0]

        for tag in tags_compute:
            result_dir = os.path.join(data_dir, f"reduced_NB={nb_out}")
            os.rename(
                os.path.join(result_dir, f"{seedname}_nbs={nb_out}.{tag}"),
                os.path.join(data_dir, f"{seedname}.{tag}"))




@pytest.fixture(scope="session")
def create_files_Fe_W90():
    """Create data files for Fe: uHu, uIu, sHu, and sIu"""

    seedname = "Fe"
    tags_needed = ["uHu", "uIu", "sHu", "sIu"]  # Files to calculate if they do not exist
    data_dir = os.path.join(ROOT_DIR, "data", "Fe_Wannier90")

    create_W90_files(seedname, tags_needed, data_dir)

    return data_dir


@pytest.fixture(scope="session")
def create_files_Fe_W90_npz(create_files_Fe_W90, system_Fe_W90):
    """Create symbolic links to the npz files"""

    seedname = "Fe"
    data_dir = Path(create_files_Fe_W90)
    data_dir_new = data_dir.joinpath("NPZ")
    data_dir_new.mkdir(exist_ok=True)

    def _link(ext):
        f = seedname + "." + ext
        try:
            data_dir_new.joinpath(f).symlink_to(data_dir.joinpath(f))
        except FileExistsError:
            pass

    _link("chk")
    for ext in ["eig", "mmn", "spn", "uHu", "sHu", "sIu"]:
        _link(ext + ".npz")
    return data_dir_new


@pytest.fixture(scope="session")
def create_files_GaAs_W90():
    """Create data files for Fe: uHu, uIu, sHu, and sIu"""

    seedname = "GaAs"
    tags_needed = ["uHu", "uIu", "sHu", "sIu"]  # Files to calculate if they do not exist
    data_dir = os.path.join(ROOT_DIR, "data", "GaAs_Wannier90")

    create_W90_files(seedname, tags_needed, data_dir)

    return data_dir


def create_files_tb(dir, file):
    data_dir = os.path.join(ROOT_DIR, "data", dir)
    path_tb_file = os.path.join(data_dir, file)
    if not os.path.isfile(path_tb_file):
        tar = tarfile.open(os.path.join(data_dir, file + ".tar.gz"))
        for tarinfo in tar:
            tar.extract(tarinfo, data_dir)
    return path_tb_file


@pytest.fixture(scope="session")
def create_files_Si_W90():
    """Create data files for Si: uHu, and uIu"""

    seedname = "Si"
    tags_needed = []  # Files to calculate if they do not exist
    data_dir = os.path.join(ROOT_DIR, "data", "Si_Wannier90")

    create_W90_files(seedname, tags_needed, data_dir)

    return data_dir


@pytest.fixture(scope="session")
def system_Fe_W90(create_files_Fe_W90):
    """Create system for Fe using Wannier90 data"""

    data_dir = create_files_Fe_W90

    # Load system
    seedname = os.path.join(data_dir, "Fe")
    system = wberri.system.System_w90(
        seedname, berry=True, morb=True, SHCqiao=True, SHCryoo=True, transl_inv_MV=False, use_wcc_phase=False,
        read_npz=False, overwrite_npz=True, write_npz_list=["uHu", "uIu", "spn", "sHu", "sIu"],
        write_npz_formatted=True)
    system.set_pointgroup(symmetries_Fe)
    return system


@pytest.fixture(scope="session")
def system_Fe_W90_npz(create_files_Fe_W90_npz):
    """Create system for Fe using Wannier90 data"""

    data_dir = create_files_Fe_W90_npz

    # Load system
    seedname = os.path.join(data_dir, "Fe")
    system = wberri.system.System_w90(
        seedname, berry=True,
        morb=True, SHCqiao=True, SHCryoo=True,
        transl_inv_MV=False, use_wcc_phase=False,
        read_npz=True, write_npz_list=[], overwrite_npz=False, write_npz_formatted=False)
    system.set_pointgroup(symmetries_Fe)
    return system


@pytest.fixture(scope="session")
def system_Fe_W90_sparse(create_files_Fe_W90, system_Fe_W90):
    """Create convert to sparse format (keeping all matrix elements) and back, to test interface"""

    params = system_Fe_W90.get_sparse({X: -1 for X in system_Fe_W90._XX_R.keys()})
    system = wberri.system.SystemSparse(**params)

    print("use_wcc", params['use_wcc_phase'], system.use_wcc_phase)
    system.set_pointgroup(symmetries_Fe)
    return system


@pytest.fixture(scope="session")
def system_Fe_W90_wcc(create_files_Fe_W90):
    """Create system for Fe using Wannier90 data"""

    data_dir = create_files_Fe_W90

    # Load system
    seedname = os.path.join(data_dir, "Fe")
    system = wberri.system.System_w90(seedname, morb=True, spin=True, SHCqiao=False, SHCryoo=False, transl_inv_MV=False,
                                      use_wcc_phase=True, wcc_phase_fin_diff=False)
    system.set_pointgroup(symmetries_Fe)
    return system


def get_system_Fe_sym_W90(symmetrize=False, use_wcc_phase=True, wcc_phase_fin_diff=False, use_ws=False,
                          extra_tags=[],
                          sym_wann_method="old",
                          **kwargs):
    """Create system for Fe symmetrization using Wannier90 data"""

    data_dir = os.path.join(ROOT_DIR, "data", "Fe_sym_Wannier90")
    create_W90_files('Fe_sym', ['uHu', 'uIu', 'sHu', 'sIu'], data_dir)

    # Load system
    seedname = os.path.join(data_dir, "Fe_sym")
    system = wberri.system.System_w90(seedname, berry=True, morb=True, spin=True, SHCryoo=True, use_ws=use_ws,
                                      use_wcc_phase=use_wcc_phase, wcc_phase_fin_diff=wcc_phase_fin_diff,
                                      **kwargs)
    system.set_pointgroup(symmetries_Fe)
    if symmetrize:
        system.symmetrize(
            proj=['Fe:sp3d2;t2g'],
            atom_name=['Fe'],
            positions=np.array([[0, 0, 0]]),
            magmom=[[0., 0., -2.31]],
            soc=True,
            spin_ordering='interlace',
            method=sym_wann_method)

    return system


@pytest.fixture(scope="session")
def system_Fe_sym_W90_wcc():
    return get_system_Fe_sym_W90(symmetrize=True)


@pytest.fixture(scope="session")
def system_Fe_sym_W90_wcc_new():
    return get_system_Fe_sym_W90(symmetrize=True, sym_wann_method="new")


@pytest.fixture(scope="session")
def system_Fe_sym_W90_wcc_fd():
    return get_system_Fe_sym_W90(symmetrize=True, use_ws=True,
                                 OSD=True, SHCqiao=True,
                                 extra_tags=['sIu', 'sHu'],
                                 wcc_phase_fin_diff=True
                                 )


@pytest.fixture(scope="session")
def system_Fe_W90_proj_set_spin(create_files_Fe_W90):
    system = get_system_Fe_sym_W90(use_wcc_phase=False)
    system.set_spin_from_projections(spin_ordering="interlace")
    return system


@pytest.fixture(scope="session")
def system_Fe_W90_proj(create_files_Fe_W90):
    return get_system_Fe_sym_W90(SHCqiao=True, use_wcc_phase=False, use_ws=False)


@pytest.fixture(scope="session")
def system_Fe_W90_proj_ws(create_files_Fe_W90):
    return get_system_Fe_sym_W90(use_ws=True, use_wcc_phase=False)


@pytest.fixture(scope="session")
def system_GaAs_W90(create_files_GaAs_W90):
    """Create system for GaAs using Wannier90 data"""

    data_dir = create_files_GaAs_W90

    # Load system
    seedname = os.path.join(data_dir, "GaAs")
    system = wberri.system.System_w90(seedname, berry=True, morb=True, spin=True, transl_inv_MV=False,
                                      use_wcc_phase=False)
    system.set_pointgroup(symmetries_GaAs)

    return system


@pytest.fixture(scope="session")
def system_GaAs_W90_wcc(create_files_GaAs_W90):
    """Create system for GaAs using Wannier90 data with wcc phases"""

    data_dir = create_files_GaAs_W90
    # Load system
    seedname = os.path.join(data_dir, "GaAs")
    system = wberri.system.System_w90(seedname, morb=True,
                                      transl_inv_MV=False, spin=True,
                                      wcc_phase_fin_diff=False)
    system.set_pointgroup(symmetries_GaAs)
    return system


@pytest.fixture(scope="session")
def system_GaAs_W90_wccFD(create_files_GaAs_W90):
    """Create system for GaAs using Wannier90 data with wcc phases"""

    data_dir = create_files_GaAs_W90
    # Load system
    seedname = os.path.join(data_dir, "GaAs")
    system = wberri.system.System_w90(seedname, berry=True,
                                      morb=True,
                                      transl_inv_MV=True, spin=True,
                                      OSD=True,
                                      SHCqiao=True, SHCryoo=True,
                                      wcc_phase_fin_diff=True)
    system.set_pointgroup(symmetries_GaAs)
    return system


@pytest.fixture(scope="session")
def system_GaAs_W90_wccJM(create_files_GaAs_W90):
    """Create system for GaAs using Wannier90 data with wcc phases"""

    data_dir = create_files_GaAs_W90
    # Load system
    seedname = os.path.join(data_dir, "GaAs")
    system = wberri.system.System_w90(seedname, morb=True,
                                      transl_inv_JM=True, spin=True,
                                      OSD=True, SHCryoo=True,
                                      wcc_phase_fin_diff=False)
    system.set_pointgroup(symmetries_GaAs)
    return system


def get_system_GaAs_tb(use_wcc_phase=True, use_ws=False, symmetrize=True, berry=True,
                       sym_wann_method="old", reorder_spin=False):
    """Create system for GaAs using sym_tb.dat data"""
    seedname = create_files_tb(dir="GaAs_Wannier90", file=f"GaAs{'_sym' if symmetrize else ''}_tb.dat")
    system = wberri.system.System_tb(seedname, berry=berry, use_ws=use_ws, use_wcc_phase=use_wcc_phase)
    if not reorder_spin and sym_wann_method == "new":
        raise ValueError("sum_wann_method='new' requires reorder_spin=True (because the original ordering is block, but new method requires interlace)")
    if reorder_spin:
        system.spin_block2interlace()
    if symmetrize:
        system.symmetrize(
            positions=np.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]),
            atom_name=['Ga', 'As'],
            proj=['Ga:sp3', 'As:sp3'],
            soc=True,
            spin_ordering='interlace' if reorder_spin else 'block',
            method=sym_wann_method
        )
    if reorder_spin:
        system.spin_interlace2block()
    if use_ws:
        system.do_ws_dist(mp_grid=(2, 2, 2))
    system.set_pointgroup(symmetries_GaAs)
    return system


@pytest.fixture(scope="session")
def system_GaAs_tb():
    """Create system for GaAs using _tb.dat data"""
    return get_system_GaAs_tb(use_wcc_phase=False, use_ws=False, symmetrize=False)


@pytest.fixture(scope="session")
def system_GaAs_sym_tb_wcc():
    """Create system for GaAs using sym_tb.dat data"""
    return get_system_GaAs_tb(use_ws=False, symmetrize=True)


@pytest.fixture(scope="session")
def system_GaAs_sym_tb_wcc_reorder():
    """Create system for GaAs using sym_tb.dat data"""
    return get_system_GaAs_tb(use_ws=False, symmetrize=True, sym_wann_method="old", reorder_spin=True)


@pytest.fixture(scope="session")
def system_GaAs_sym_tb_wcc_reorder_new():
    """Create system for GaAs using sym_tb.dat data"""
    return get_system_GaAs_tb(use_ws=False, symmetrize=True, sym_wann_method="new", reorder_spin=True)


@pytest.fixture(scope="session")
def system_GaAs_tb_wcc():
    """Create system for GaAs using _tb_dat data"""
    return get_system_GaAs_tb(use_ws=False, symmetrize=False)


@pytest.fixture(scope="session")
def system_GaAs_tb_wcc_ws():
    """Create system for GaAs using _tb_dat data"""
    return get_system_GaAs_tb(use_ws=True, symmetrize=False)


@pytest.fixture(scope="session")
def system_GaAs_tb_wcc_ws_noAA():
    """Create system for GaAs using _tb_dat data"""
    return get_system_GaAs_tb(use_ws=True, symmetrize=False, berry=False)


def get_system_Si_W90_JM(data_dir, transl_inv=False, transl_inv_JM=False, wcc_phase_fin_diff=False,
                         matrices=dict(OSD=True),
                         symmetrize=False):
    """Create system for Si using Wannier90 data with Jae-Mo's approach for real-space matrix elements"""

    for tag in ('uHu', 'uIu'):
        if not os.path.isfile(os.path.join(data_dir, f"Si.{tag}")):
            tar = tarfile.open(os.path.join(data_dir, f"Si.{tag}.tar.gz"))
            for tarinfo in tar:
                tar.extract(tarinfo, data_dir)
    # Load system
    seedname = os.path.join(data_dir, "Si")
    system = wberri.system.System_w90(seedname, use_ws=True, use_wcc_phase=True,
                                      transl_inv_MV=transl_inv,
                                      transl_inv_JM=transl_inv_JM,
                                      wcc_phase_fin_diff=wcc_phase_fin_diff,
                                      guiding_centers=True,
                                      **matrices)
    if symmetrize:
        system.symmetrize(
            positions=np.array([[-0.125, -0.125, 0.375],
                                [0.375, -0.125, -0.125],
                                [-0.125, 0.375, -0.125],
                                [-0.125, -0.125, -0.125]]),
            atom_name=['bond'] * 4,
            proj=['bond:s'],
            soc=False,
            spin_ordering='interlace',
            method='old')

    return system


@pytest.fixture(scope="session")
def system_Si_W90_JM(create_files_Si_W90):
    """Create system for Si using Wannier90 data with Jae-Mo's approach for real-space matrix elements"""
    data_dir = create_files_Si_W90
    return get_system_Si_W90_JM(data_dir, transl_inv_JM=True)


@pytest.fixture(scope="session")
def system_Si_W90_wccFD(create_files_Si_W90):
    """Create system for Si using Wannier90 data with Jae-Mo's approach for real-space matrix elements"""
    data_dir = create_files_Si_W90
    return get_system_Si_W90_JM(data_dir, transl_inv=True, wcc_phase_fin_diff=True)


@pytest.fixture(scope="session")
def system_Si_W90_wccFD_sym(create_files_Si_W90):
    """Create system for Si using Wannier90 data with Jae-Mo's approach for real-space matrix elements"""
    data_dir = create_files_Si_W90
    system = get_system_Si_W90_JM(data_dir, transl_inv=True,
                                  wcc_phase_fin_diff=True, matrices=dict(OSD=True),
                                  symmetrize=True)
    # system = get_system_Si_W90_JM(data_dir, transl_inv_JM=True, matrices=dict(berry=True) )
    return system


@pytest.fixture(scope="session")
def system_Si_W90_wccJM_sym(create_files_Si_W90):
    """Create system for Si using Wannier90 data with Jae-Mo's approach for real-space matrix elements"""
    data_dir = create_files_Si_W90
    system = get_system_Si_W90_JM(data_dir, transl_inv_JM=True,
                                matrices=dict(berry=True),
                                  symmetrize=True)
    return system

# Haldane model from TBmodels


@pytest.fixture(scope="session")
def system_Haldane_TBmodels():
    # Load system
    try:
        import tbmodels
        tbmodels  # just to avoid F401
    except (ImportError, ModuleNotFoundError):
        pytest.xfail("failed to import tbmodels")
    model_tbmodels_Haldane = wb_models.Haldane_tbm(delta=0.2, hop1=-1.0, hop2=0.15)
    system = wberri.system.System_TBmodels(model_tbmodels_Haldane)
    system.set_pointgroup(["C3z"])
    return system




@pytest.fixture(scope="session")
def system_Haldane_PythTB():
    """Create system for Haldane model using PythTB"""
    # Load system
    model_pythtb_Haldane = wb_models.Haldane_ptb(delta=0.2, hop1=-1.0, hop2=0.15)
    system = wberri.system.System_PythTB(model_pythtb_Haldane)
    system.set_pointgroup(["C3z"])
    return system


@pytest.fixture(scope="session")
def system_KaneMele_odd_PythTB():
    """Create system for Haldane model using PythTB"""
    # Load system
    model_pythtb_KaneMele_odd = wb_models.KaneMele_ptb('odd')
    system = wberri.system.System_PythTB(model_pythtb_KaneMele_odd, spin=True)
    system.set_pointgroup(["C3z", "TimeReversal"])
    return system


@pytest.fixture(scope="session")
def system_Chiral_OSD():
    """Create system for Haldane model using PythTB"""
    # Load system
    model_pythtb_Chiral_OSD = wb_models.Chiral_OSD()
    system = wberri.system.System_PythTB(model_pythtb_Chiral_OSD, spin=True)
    # system.set_symmetry(["C3z","TimeReversal"])
    return system


# Chiral model
# A chiral system that also breaks time-reversal. It can be used to test almost any quantity.


@pytest.fixture(scope="session")
def system_Chiral_left():
    model_Chiral_left = wb_models.Chiral(
        delta=2, hop1=1, hop2=1. / 3, phi=np.pi / 10, hopz_left=0.2, hopz_right=0.0, hopz_vert=0)
    system = wberri.system.System_PythTB(model_Chiral_left)
    system.set_pointgroup(["C3z"])
    system.set_spin([1, -1])
    return system


@pytest.fixture(scope="session")
def system_Chiral_left_TR():
    model_Chiral_left_TR = wb_models.Chiral(
        delta=2, hop1=1, hop2=1. / 3, phi=-np.pi / 10, hopz_left=0.2, hopz_right=0.0, hopz_vert=0)
    system = wberri.system.System_PythTB(model_Chiral_left_TR)
    system.set_pointgroup(["C3z"])
    system.set_spin([-1, 1])
    return system


@pytest.fixture(scope="session")
def system_Chiral_right():
    model_Chiral_right = wb_models.Chiral(
        delta=2, hop1=1, hop2=1. / 3, phi=np.pi / 10, hopz_left=0.0, hopz_right=0.2, hopz_vert=0)
    system = wberri.system.System_PythTB(model_Chiral_right)
    system.set_pointgroup(["C3z"])
    system.set_spin([1, -1])
    return system


# Systems from FPLO code interface
@pytest.fixture(scope="session")
def system_Fe_FPLO_wcc():
    """Create system for Fe using  FPLO  data"""
    path = os.path.join(ROOT_DIR, "data", "Fe_FPLO", "+hamdata")
    system = wberri.system.System_fplo(path, morb=True, spin=True, use_ws=False)
    system.set_pointgroup(symmetries_Fe)
    return system


@pytest.fixture(scope="session")
def system_Fe_FPLO_wcc_ws():
    """Create system for Fe using  FPLO  data"""
    path = os.path.join(ROOT_DIR, "data", "Fe_FPLO", "+hamdata")
    system = wberri.system.System_fplo(path, morb=True, spin=True,
                                       use_ws=True, mp_grid=2)
    system.set_pointgroup(symmetries_Fe)
    return system




@pytest.fixture(scope="session")
def system_CuMnAs_2d_broken():
    # CuMnAs 2D model
    # These parameters provide ~0.4eV gap between conduction and valence bands
    # and splitting into subbands is within 0.04 eV
    model_CuMnAs_2d_broken = wb_models.CuMnAs_2d(nx=0, ny=1, nz=0, hop1=1, hop2=0.08, l=0.8, J=1, dt=0.01)
    system = wberri.system.System_PythTB(model_CuMnAs_2d_broken)
    return system


# Systems from ASE+gpaw code interface

@pytest.fixture(scope="session")
def data_Te_ASE():
    """read data for Te from ASE+GPAW"""
    try:
        import gpaw
    except (ImportError, ModuleNotFoundError):
        pytest.xfail("failed to import gpaw")
    import ase.dft.wannier

    path = os.path.join(ROOT_DIR, "data", "Te_ASE")
    calc = gpaw.GPAW(os.path.join(path, "Te.gpw"))
    wan = ase.dft.wannier.Wannier(nwannier=12, calc=calc, file=os.path.join(path, 'wannier-12.json'))
    return wan


@pytest.fixture(scope="session")
def system_Te_ASE_wcc(data_Te_ASE):
    """Create system for Te using  ASE+GPAW data with use_wcc_phase=True"""
    wan = data_Te_ASE
    system = wberri.system.System_ASE(wan)
    system.set_pointgroup(symmetries_Te)
    return system


@pytest.fixture(scope="session")
def system_Te_sparse():
    """Create system for Te using symmetrized Wannier functions through a sparse interface"""
    path = os.path.join(ROOT_DIR, "data", "Te_sparse", "parameters_Te_low.pickle")
    param = pickle.load(open(path, "rb"))
    system = wberri.system.SystemSparse(**param)
    system.set_pointgroup(symmetries_Te)
    return system


@pytest.fixture(scope="session")
def system_Phonons_Si():
    """Create system of phonons of Si using  QE data"""
    path = os.path.join(ROOT_DIR, "data", "Si_phonons/si")
    system = wberri.system.System_Phonon_QE(path, use_ws=True, asr=True)
    system.set_pointgroup(symmetries_Si)
    return system


@pytest.fixture(scope="session")
def system_Phonons_GaAs():
    """Create system of phonons of Si using  QE data"""
    path = os.path.join(ROOT_DIR, "data", "GaAs_phonons/GaAs")
    system = wberri.system.System_Phonon_QE(path, use_ws=True, asr=True)
    system.set_pointgroup(symmetries_GaAs)
    return system




def get_system_Mn3Sn_sym_tb(use_ws=False):
    data_dir = os.path.join(ROOT_DIR, "data", "Mn3Sn_Wannier90")
    if not os.path.isfile(os.path.join(data_dir, "Mn3Sn_tb.dat")):
        tar = tarfile.open(os.path.join(data_dir, "Mn3Sn_tb.dat.tar.gz"))
        for tarinfo in tar:
            tar.extract(tarinfo, data_dir)

    seedname = os.path.join(data_dir, "Mn3Sn_tb.dat")
    system = wberri.system.System_tb(seedname, berry=True, use_ws=use_ws)
    system.symmetrize(
        positions=np.array([
            [0.6666667, 0.8333333, 0],
            [0.1666667, 0.3333333, 0],
            [0.6666667, 0.3333333, 0],
            [0.3333333, 0.1666667, 0.5],
            [0.8333333, 0.6666667, 0.5],
            [0.3333333, 0.6666667, 0.5],
            [0.8333333, 0.1666667, 0.5],
            [0.1666667, 0.8333333, 0]]),
        atom_name=['Mn'] * 6 + ['Sn'] * 2,
        proj=['Mn:s;d', 'Sn:p'],
        soc=True,
        magmom=[
            [0, 2, 0],
            [np.sqrt(3), -1, 0],
            [-np.sqrt(3), -1, 0],
            [0, 2, 0],
            [np.sqrt(3), -1, 0],
            [-np.sqrt(3), -1, 0],
            [0, 0, 0],
            [0, 0, 0]],
        spin_ordering='block',
        method='old'
    )
    return system


@pytest.fixture(scope="session")
def system_Mn3Sn_sym_tb_wcc():
    """Create system for Mn3Sn using _tb.dat data"""
    return get_system_Mn3Sn_sym_tb(use_ws=False)


###################################
# Isotropic effective mas s model #
###################################
mass_kp_iso = 1.912
kmax_kp = 2.123


def ham_mass_iso(k):
    return np.array([[np.dot(k, k) / (2 * mass_kp_iso)]])


def dham_mass_iso(k):
    return np.array(k).reshape(1, 1, 3) / mass_kp_iso


def d2ham_mass_iso(k):
    return np.eye(3).reshape(1, 1, 3, 3) / mass_kp_iso


@pytest.fixture(scope="session")
def system_kp_mass_iso_0():
    return wberri.system.SystemKP(Ham=ham_mass_iso, kmax=kmax_kp)


@pytest.fixture(scope="session")
def system_kp_mass_iso_1():
    return wberri.system.SystemKP(Ham=ham_mass_iso, derHam=dham_mass_iso, kmax=kmax_kp)


@pytest.fixture(scope="session")
def system_kp_mass_iso_2():
    return wberri.system.SystemKP(Ham=ham_mass_iso, derHam=dham_mass_iso, der2Ham=d2ham_mass_iso, kmax=kmax_kp)


###################################
# AnIsotropic effective mas s model #
###################################
kmax_kp_aniso = 2.1

inv_mass_kp_aniso = np.array([[0.86060064, 0.19498375, 0.09798235],
 [0.01270294, 0.77373333, 0.00816169],
    [0.15613272, 0.11770323, 0.71668436]])


def ham_mass_aniso(k):
    e = np.dot(k, np.dot(inv_mass_kp_aniso, k))
    return np.array([[e]])


def dham_mass_aniso(k):
    return (np.dot(k, inv_mass_kp_aniso) + np.dot(inv_mass_kp_aniso, k)).reshape(1, 1, 3)


def d2ham_mass_aniso(k):
    return (inv_mass_kp_aniso + inv_mass_kp_aniso.T).reshape(1, 1, 3, 3)



@pytest.fixture(scope="session")
def system_kp_mass_aniso_0():
    return wberri.system.SystemKP(Ham=ham_mass_aniso, kmax=kmax_kp_aniso)


@pytest.fixture(scope="session")
def system_kp_mass_aniso_1():
    return wberri.system.SystemKP(Ham=ham_mass_aniso, derHam=dham_mass_aniso, kmax=kmax_kp_aniso)


@pytest.fixture(scope="session")
def system_kp_mass_aniso_2():
    return wberri.system.SystemKP(Ham=ham_mass_aniso, derHam=dham_mass_aniso, der2Ham=d2ham_mass_aniso, kmax=kmax_kp_aniso)



def model_1d_pythtb():
    import pythtb
    lat = [[1.0]]
    orb = [[0], [0.5]]
    orb2 = [orb[0]] * 2 + [orb[1]] * 2

    model1d_1 = pythtb.tb_model(1, 1, lat, orb, nspin=2)
    model1d_2 = pythtb.tb_model(1, 1, lat, orb2, nspin=1)
    Delta = 1
    model1d_1.set_onsite([-Delta, Delta])
    model1d_2.set_onsite([-Delta] * 2 + [Delta] * 2)
    a = np.random.random(4)
    b = np.random.random(4)
    model1d_1.set_hop(a, 0, 1, [0])
    model1d_1.set_hop(b, 1, 0, [-1])

    pauli = np.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])
    for i in range(2):
        for j in range(2):
            model1d_2.set_hop(a.dot(pauli[:, i, j]), 0 + i, 2 + j, [0])
            model1d_2.set_hop(b.dot(pauli[:, i, j]), 2 + i, 0 + j, [-1])

    return model1d_1, model1d_2


@pytest.fixture(scope="session")
def system_random():
    system = wberri.system.SystemRandom(num_wann=6, nRvec=20, max_R=4,
                                        use_wcc_phase=True,
                                        berry=True, morb=True, spin=True,
                                        SHCryoo=True, SHCqiao=True, OSD=True)
    # system.save_npz("randomsys")
    return system


@pytest.fixture(scope="session")
def system_random_load_bare():
    system = wberri.system.System_R(berry=True, morb=True, spin=True,
                                    SHCryoo=True, SHCqiao=True, OSD=True)
    system.load_npz(path=os.path.join(ROOT_DIR, "data", "random"))
    return system


@pytest.fixture(scope="session")
def system_random_GaAs():
    return wberri.system.SystemRandom(num_wann=16, nRvec=30, max_R=4,
                                      real_lattice=np.ones(3) - np.eye(3),
                                      berry=True, spin=True, SHCryoo=True,
                                      use_wcc_phase=True)


def get_system_random_GaAs_load_ws_sym(use_ws=False, sym=False):
    system = wberri.system.System_R(berry=True, spin=True, SHCryoo=True)
    system.load_npz(path=os.path.join(ROOT_DIR, "data", "random_GaAs"))
    if use_ws:
        system.do_ws_dist(mp_grid=6)
    if sym:
        system.symmetrize(
            proj=['Ga:sp3', 'As:sp3'],
            atom_name=['Ga', 'As'],
            positions=np.array([[0, 0, 0], [1 / 4, 1 / 4, 1 / 4]]),
            soc=True,
            spin_ordering='interlace',
            method='old'
        )
        system.set_structure(
            atom_labels=['Ga', 'As'],
            positions=np.array([[0, 0, 0], [1 / 4, 1 / 4, 1 / 4]])
        )
        system.set_symmetry_from_structure()
    return system


@pytest.fixture(scope="session")
def system_random_GaAs_load_bare():
    return get_system_random_GaAs_load_ws_sym(use_ws=False, sym=False)


@pytest.fixture(scope="session")
def system_random_GaAs_load_ws():
    return get_system_random_GaAs_load_ws_sym(use_ws=True, sym=False)


@pytest.fixture(scope="session")
def system_random_GaAs_load_ws_sym():
    return get_system_random_GaAs_load_ws_sym(use_ws=True, sym=True)
