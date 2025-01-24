import os

import numpy as np
from wannierberri.symmetry.sawf import SymmetrizerSAWF as SAWF
from wannierberri.wannierise.projections import Projection, ProjectionsSet


# os.environ['OPENBLAS_NUM_THREADS'] = '1'
# os.environ['MKL_NUM_THREADS'] = '1'

from time import time
import wannierberri as wberri
from pathlib import Path

parallel = False

t0 = time()
path_data = Path("./pwscf/")  # adjust path if needed to point to the data in the tests fo wannier-berri repository

includeTR = False
w90data = wberri.w90files.Wannier90data(seedname=str(path_data / "Ni4W"), readfiles=["amn", "mmn", "eig", "win"])
t1 = time()
sitesym = True

if sitesym:
    try:
        symmetrizer = SAWF().from_npz("Ni4W.sawf.npz")
        assert symmetrizer.num_wann == 34
    except (FileNotFoundError, AssertionError) as e:
        print(f"SAWF file not found ({e}), creating it")
        from irrep.bandstructure import BandStructure
        bandstructure = BandStructure(code='espresso',
                                    prefix=os.path.join(path_data, "Ni4W"),
                                    Ecut=100,
                                    normalize=False, include_TR=includeTR)
        symmetrizer = SAWF().from_irrep(bandstructure)
        spacegroup = symmetrizer.spacegroup


        positions = [[0.200749460000, 0.199526250000, 0.400275700000],
                    [0.799250540000, 0.800473750000, 0.599724300000],
                    [0.599801950000, 0.599724300000, 0.199526240000],
                    [0.400198050000, 0.400275700000, 0.800473760000]]
        projection1 = Projection(position_num=positions, orbital='d', spacegroup=spacegroup)

        positions = [[0, 0, 0],]
        projection2 = Projection(position_num=positions, orbital='d', spacegroup=spacegroup)

        positions = [[0, 1 / 2, 1 / 2],]
        projection3 = Projection(position_num=positions, orbital='s', spacegroup=spacegroup)

        positions = [[0.394108623296, 0.408884501503, 0.432352882355],
                    [0.605891376704, 0.591115498497, 0.567647117645],
                    [0.841237383858, 0.567647117645, 0.038244259059],
                    [0.158762616142, 0.432352882355, 0.961755740941],
                    [0.605891376704, 0.961755740941, 0.197006875201],
                    [0.394108623296, 0.038244259059, 0.802993124799],
                    [0.158762616142, 0.802993124799, 0.591115498497],
                    [0.841237383858, 0.197006875201, 0.408884501503]]
        projection4 = Projection(position_num=positions, orbital='s', spacegroup=spacegroup)


        projections = ProjectionsSet([projection1, projection2, projection3, projection4])
        symmetrizer.set_D_wann_from_projections(projections_obj=projections)
        symmetrizer.to_npz("Ni4W.dmn.npz")
    w90data.set_symmetrizer(symmetrizer)

proj_str = """0.200749460000, 0.199526250000, 0.400275700000: d
              0.799250540000, 0.800473750000, 0.599724300000: d
              0.599801950000, 0.599724300000, 0.199526240000: d
              0.400198050000, 0.400275700000, 0.800473760000: d
              0.000000000000, 0.000000000000, 0.000000000000: d
              0.000000000000, 0.500000000000, 0.500000000000: s
              0.394108623296, 0.408884501503, 0.432352882355: s
              0.605891376704, 0.591115498497, 0.567647117645: s
              0.841237383858, 0.567647117645, 0.038244259059: s
              0.158762616142, 0.432352882355, 0.961755740941: s
              0.605891376704, 0.961755740941, 0.197006875201: s
              0.394108623296, 0.038244259059, 0.802993124799: s
              0.158762616142, 0.802993124799, 0.591115498497: s
              0.841237383858, 0.197006875201, 0.408884501503: s"""



wcc = []
num_wann = {"s": 1, "p": 3, "d": 5}
for line in proj_str.split("\n"):
    pos, orb = line.split(":")
    pos = [float(x) for x in pos.split(",")]
    for _ in range(num_wann[orb.strip()]):
        wcc.append(pos)

wcc = np.array(wcc)
print("wcc_red = \n", wcc)
wcc = wcc @ symmetrizer.spacegroup.lattice
print("wcc_cart = \n", wcc)
print(f"symmetrizer.num_wann = {symmetrizer.num_wann}, symmetrizer.NB = {symmetrizer.NB}")
wcc_sym = symmetrizer.symmetrize_WCC(wcc)
print("wcc_sym = \n", wcc_sym)

diff = abs(wcc_sym - wcc).max()
assert diff < 1e-6, f"diff = {diff}"
print(f"symmetrization OK, diff = {diff}")

symmetrizer.spacegroup.show()



t2 = time()

print(f"dmn.num_wann = {symmetrizer.num_wann}, dmn.NB = {symmetrizer.NB}")
w90data.set_file("dmn", symmetrizer)

# print(f"check amn: {dmn.check_amn(w90data.amn)}")
# exit()

t2a = time()
if parallel:
    import ray
    ray.init(num_gpus=0, num_cpus=16)


froz_max = 24
t3 = time()

w90data.apply_window(win_min=8, )
t4 = time()
w90data.wannierise(init="amn",
                   num_wann=34,
                   froz_min=8,
                   froz_max=froz_max,
                   print_progress_every=10,
                   num_iter=30,
                   conv_tol=1e-6,
                   mix_ratio_z=1.0,
                   sitesym=sitesym,
                   parallel=parallel
                    )
t5 = time()
print("Time elapsed: ", time() - t0)
print("Time elapsed (DMN): ", t2 - t1)
print("Time elapsed (set dmn): ", t2a - t2)
print("Time elapsed (Wannierisation): ", t5 - t4)
print("Time elapsed (ray init): ", t3 - t2a)
print("Time elapsed (Window): ", t4 - t3)
print("Time elapsed (read data): ", t1 - t0)

exit()
system = wberri.system.System_w90(w90data=w90data, silent=True)

path = wberri.Path(system, k_nodes=[[0, 0, 0], [1 / 2, 0, 0], [1, 0, 0]], labels=['G', 'L', 'G'], length=100)
tabulator = wberri.calculators.TabulatorAll(tabulators={}, mode='path')
calculators = {'tabulate': tabulator}

result = wberri.run(system, grid=path, calculators=calculators)
result.results['tabulate'].plot_path_fat(path,
                                         save_file='bands-sym.pdf',
                                         close_fig=True,
                                         show_fig=False,
                                         )
