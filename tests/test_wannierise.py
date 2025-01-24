import copy
import irrep
from irrep.bandstructure import BandStructure
from pytest import approx, fixture
import pytest
import scipy
import wannierberri as wberri
import numpy as np
from matplotlib import pyplot as plt
import os
import shutil

from wannierberri.w90files.dmn import DMN
from wannierberri.wannierise.projections import Projection
from .common import OUTPUT_DIR, ROOT_DIR, REF_DIR
from wannierberri.symmetry.sawf import SymmetrizerSAWF



def test_wanierise():
    systems = {}

    cwd = os.getcwd()

    tmp_dir = os.path.join(OUTPUT_DIR, "diamond")

    # Check if the directory exists
    if os.path.exists(tmp_dir):
        # Remove the directory and all its contents
        shutil.rmtree(tmp_dir)
        print(f"Directory {tmp_dir} has been removed.")
    os.makedirs(tmp_dir)
    os.chdir(tmp_dir)

    data_dir = os.path.join(ROOT_DIR, "data", "diamond")
    prefix = "diamond"
    for ext in ["mmn", "amn", "eig", "win", "sawf.npz"]:
        shutil.copy(os.path.join(data_dir, prefix + "." + ext),
                    os.path.join(tmp_dir, prefix + "." + ext))
    print("prefix = ", prefix)
    symmetrizer = SymmetrizerSAWF().from_npz(prefix + ".sawf.npz")
    symmetrizer.spacegroup.show()
    symmetrizer.to_w90_file(prefix)
    # Read the data from the Wanier90 inputs
    w90data = wberri.w90files.Wannier90data(seedname=prefix, readfiles=["amn", "mmn", "eig", "win"])
    w90data.set_symmetrizer(symmetrizer=symmetrizer)
    # Now disentangle with sitesym and frozen window (the part that is not implemented in Wanier90)
    w90data.wannierise(
        froz_min=-8,
        froz_max=20,
        num_iter=1000,
        conv_tol=1e-10,
        mix_ratio_z=0.8,
        mix_ratio_u=1,
        print_progress_every=20,
        sitesym=True,
        localise=True
    )
    wannier_centers = w90data.chk._wannier_centers
    wannier_spreads = w90data.chk._wannier_spreads
    wannier_spreads_mean = np.mean(wannier_spreads)
    assert wannier_spreads == approx(wannier_spreads_mean, abs=1e-9)
    assert wannier_spreads == approx(0.39864755, abs=1e-7)
    assert wannier_centers == approx(np.array([[0, 0, 0],
                                               [0, 0, 1],
                                               [0, 1, 0],
                                               [1, 0, 0]
                                               ]).dot(w90data.chk.real_lattice) / 2,
                                     abs=1e-6)

    systems["wberri"] = wberri.system.System_w90(w90data=w90data)


    systems["wberri_symmetrized"] = copy.deepcopy(systems["wberri"])


    systems["wberri_symmetrized"].symmetrize2(symmetrizer=symmetrizer)

    # Now calculate bandstructure for each of the systems
    # for creating a path any of the systems will do the job
    system0 = list(systems.values())[0]
    path = wberri.Path(system0, k_nodes=[[0, 0, 0], [1 / 2, 0, 0]], labels=['G', 'L'], length=100)
    tabulator = wberri.calculators.TabulatorAll(tabulators={}, mode='path')
    calculators = {'tabulate': tabulator}

    linecolors = ['black', 'red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan']
    linestyles = ['-', '--', ':', '-.', ] * 4
    energies = {}
    for key, sys in systems.items():
        result = wberri.run(sys, grid=path, calculators=calculators)
        result.results['tabulate'].plot_path_fat(path, close_fig=False, show_fig=False,
                                                linecolor=linecolors.pop(0), label=key,
                                                kwargs_line={"ls": linestyles.pop(0)})
        energies[key] = result.results['tabulate'].get_data(quantity="Energy", iband=np.arange(0, 4))
    np.save(os.path.join(OUTPUT_DIR, "bands_wannierize.npy"), energies["wberri_symmetrized"])
    energies_ref = np.load(os.path.join(REF_DIR, "bands_wannierize.npy"))
    plt.savefig("bands.png")
    for k1 in energies:
        print(f"comparing {k1} and reference")
        diff = abs(energies[k1] - energies_ref)
        # the precidsion is not very high here, although the two codes are assumed to do the same. Not sure why..
        d, acc = np.max(diff), 0.0005
        assert d < acc, f"the interpolated bands {k1}  differ from reference by max {d}>{acc}"

    # One can see that results do not differ much. Also, the maximal localization does not have much effect.
    os.chdir(cwd)


@fixture
def check_sawf():
    def _inner(sawf_new, sawf_ref):

        for key in ['NB', "num_wann", "NK", "NKirr", "kptirr", "kptirr2kpt", "kpt2kptirr", "time_reversals"]:
            assert np.all(getattr(sawf_ref, key) == getattr(sawf_new, key)), (
                f"key {key} differs between reference and new SymmetrizerSAWF\n"
                f"reference: {getattr(sawf_ref, key)}\n"
                f"new: {getattr(sawf_new, key)}\n"
            )

        assert np.all(sawf_ref.D_wann_block_indices == sawf_new.D_wann_block_indices), (
            f"D_wann_block_indices differs between reference and new SymmetrizerSAWF\n"
            f"reference: {sawf_ref.D_wann_block_indices}\n"
            f"new: {sawf_new.D_wann_block_indices}\n"
        )
        for ikirr in range(sawf_ref.NKirr):
            assert np.all(sawf_ref.d_band_block_indices[ikirr] == sawf_new.d_band_block_indices[ikirr]), (
                f"d_band_block_indices differs  at ikirr={ikirr} between reference and new SymmetrizerSAWF\n"
                f"reference: {sawf_ref.d_band_block_indices}\n"
                f"new: {sawf_new.d_band_block_indices}\n"
            )

        for isym in range(sawf_ref.Nsym):
            try:
                for ikirr in range(sawf_ref.NKirr):
                    for blockref, blocknew in zip(sawf_ref.D_wann_blocks[ikirr][isym], sawf_new.D_wann_blocks[ikirr][isym]):
                        assert blockref == approx(blocknew, abs=1e-6), f"D_wann at ikirr = {ikirr}, isym = {isym} differs between reference and new SymmetrizerSAWF by a maximum of {np.max(np.abs(blockref - blocknew))} > 1e-6"
                    for blockref, blocknew in zip(sawf_ref.d_band_blocks[ikirr][isym], sawf_new.d_band_blocks[ikirr][isym]):
                        assert blockref == approx(blocknew, abs=1e-6), f"d_band at ikirr = {ikirr}, isym = {isym} differs between reference and new SymmetrizerSAWF by a maximum of {np.max(np.abs(blockref - blocknew))} > 1e-6"
            except AssertionError:
                for ikirr in range(sawf_ref.NKirr):
                    for blockref, blocknew in zip(sawf_ref.D_wann_blocks[ikirr][isym], sawf_new.D_wann_blocks[ikirr][isym]):
                        assert blockref == approx(-blocknew, abs=1e-6), f"D_wann at ikirr = {ikirr}, isym = {isym} differs between reference and new SymmetrizerSAWF by a maximum of {np.max(np.abs(blockref - blocknew))} > 1e-6"
                    for blockref, blocknew in zip(sawf_ref.d_band_blocks[ikirr][isym], sawf_new.d_band_blocks[ikirr][isym]):
                        assert blockref == approx(-blocknew, abs=1e-6), f"d_band at ikirr = {ikirr}, isym = {isym} differs between reference and new SymmetrizerSAWF by a maximum of {np.max(np.abs(blockref - blocknew))} > 1e-6"


    return _inner


def test_create_sawf_diamond(check_sawf):
    data_dir = os.path.join(ROOT_DIR, "data", "diamond")

    bandstructure = irrep.bandstructure.BandStructure(prefix=data_dir + "/di", Ecut=100,
                                                      code="espresso",
                                                    include_TR=False,
                                                      )

    projection = Projection(position_num=[[0, 0, 0], [0, 0, 1 / 2], [0, 1 / 2, 0], [1 / 2, 0, 0]], orbital='s', spacegroup=bandstructure.spacegroup)
    sawf_new = SymmetrizerSAWF().from_irrep(bandstructure)
    sawf_new.set_D_wann_from_projections(projections_obj=[projection])

    tmp_sawf_path = os.path.join(OUTPUT_DIR, "diamond")
    sawf_new.to_npz(tmp_sawf_path + ".sawf.npz")
    sawf_ref = SymmetrizerSAWF().from_npz(data_dir + "/diamond.sawf.npz")
    check_sawf(sawf_new, sawf_ref)
    sawf_new.to_w90_file(tmp_sawf_path)
    dmn_new = DMN(seedname=tmp_sawf_path, read_npz=False)

    dmn_ref = DMN(seedname=os.path.join(data_dir, "diamond"), read_npz=False)
    check_sawf(dmn_new, dmn_ref)


@pytest.mark.parametrize("include_TR", [True, False])
def test_create_sawf_Fe(check_sawf, include_TR):
    path_data = os.path.join(ROOT_DIR, "data", "Fe-222-pw")

    bandstructure = BandStructure(code='espresso', prefix=path_data + '/Fe', Ecut=100,
                                normalize=False, magmom=[[0, 0, 1]], include_TR=include_TR)
    sawf_new = SymmetrizerSAWF().from_irrep(bandstructure)
    pos = [[0, 0, 0]]
    sawf_new.set_D_wann_from_projections(projections=[(pos, 's'), (pos, 'p'), (pos, 'd')])
    tmp_sawf_path = os.path.join(OUTPUT_DIR, f"Fe_TR={include_TR}.sawf.npz")
    sawf_new.to_npz(tmp_sawf_path)
    sawf_ref = SymmetrizerSAWF().from_npz(os.path.join(REF_DIR, "sawf", f"Fe_TR={include_TR}.sawf.npz"))
    check_sawf(sawf_new, sawf_ref)



@pytest.mark.parametrize("include_TR", [True, False])
@pytest.mark.parametrize("use_window", [True, False])
def test_sitesym_Fe(include_TR, use_window):
    path_data = os.path.join(ROOT_DIR, "data", "Fe-444-sitesym")
    w90data = wberri.w90files.Wannier90data(seedname=path_data + "/Fe", readfiles=["amn", "eig", "mmn", "win"])

    symmetrizer = SymmetrizerSAWF().from_npz(path_data + f"/Fe_TR={include_TR}.sawf.npz")
    w90data.set_symmetrizer(symmetrizer)
    if use_window:
        w90data.apply_window(win_min=-8, win_max=50)
    froz_max = 30
    w90data.wannierise(init="amn",
                       froz_min=-8,
                    froz_max=froz_max,
                    print_progress_every=20,
                    num_iter=40,
                    conv_tol=1e-6,
                    mix_ratio_z=1.0,
                    localise=True,
                    sitesym=True,
                    )
    assert np.allclose(w90data.wannier_centers, 0, atol=1e-6), f"wannier_centers differ from 0 by {np.max(abs(w90data.wannier_centers))} \n{w90data.wannier_centers}"
    spreads = w90data.chk._wannier_spreads
    assert np.all(spreads < 2)
    atol = 1e-8
    assert spreads[4] == approx(spreads[6], abs=atol)
    assert spreads[5] == approx(spreads[7], abs=atol)
    assert spreads[10] == approx(spreads[12], abs=atol)
    assert spreads[11] == approx(spreads[13], abs=atol)
    system = wberri.system.System_w90(w90data=w90data, berry=True)
    # system.set_symmetry(spacegroup=bandstructure.spacegroup)
    tabulators = {"Energy": wberri.calculators.tabulate.Energy(),
                }


    tab_all_path = wberri.calculators.TabulatorAll(
        tabulators,
        ibands=np.arange(0, 18),
        mode="path"
    )

    # all kpoints given in reduced coordinates
    path = wberri.Path(system,
                    k_nodes=[
                        [0.0000, 0.0000, 0.0000],  # G
                        [0.500, -0.5000, -0.5000],  # H
                        [0.7500, 0.2500, -0.2500],  # P
                        [0.5000, 0.0000, -0.5000],  # N
                        [0.0000, 0.0000, 0.000]
                    ],  # G
        labels=["G", "H", "P", "N", "G"],
        nk=[21] * 5)   # length [ Ang] ~= 2*pi/dk

    result_path = wberri.run(system,
                    grid=path,
                    calculators={"tabulate": tab_all_path},
                    print_Kpoints=False)
    EF = 12.6
    A = np.loadtxt(os.path.join(path_data, "Fe_bands_pw.dat"))
    energies_ref = np.copy(A[:, 1].reshape(-1, 81)[:18].T)

    bohr_ang = scipy.constants.physical_constants['Bohr radius'][0] / 1e-10
    alatt = 5.4235 * bohr_ang
    A[:, 0] *= 2 * np.pi / alatt
    A[:, 1] = A[:, 1] - EF
    plt.scatter(A[:, 0], A[:, 1], c="black", s=5)

    energies = result_path.results["tabulate"].get_data(quantity="Energy", iband=np.arange(0, 18))

    np.save(os.path.join(OUTPUT_DIR, f"Fe_bands-{include_TR}.npy"), energies)
    np.savetxt(os.path.join(OUTPUT_DIR, f"Fe_bands-{include_TR}.dat"), energies)

    atol = 0.7
    nk = energies.shape[0]
    energies_diff = np.abs(energies - energies_ref)
    energies_diff[energies_ref > 13] = 0  # ignore the high energy bands
    for ik in range(nk):
        if ik % 20 == 0:
            _atol = 0.01
        else:
            _atol = atol
        assert np.allclose(energies_diff[ik], 0, atol=_atol), \
            f"energies at ik={ik} differ by {np.max(abs(energies_diff[ik]))} more than {_atol}" +\
            f"energies: {energies[ik]}\nref: {energies_ref[ik]}"

    result_path.results["tabulate"].plot_path_fat(
        path,
        quantity=None,
        Eshift=EF,
        Emin=-10, Emax=50,
        iband=None,
        mode="fatband",
        fatfactor=20,
        cut_k=False,
        linecolor="red",
        close_fig=False,
        show_fig=False,
        label=f"TR={include_TR}"
    )

    plt.ylim(-10, 20)
    plt.hlines(froz_max - EF, 0, A[-1, 0], linestyles="dashed")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, f"Fe_bands-{include_TR}.pdf"))
    plt.close()
