import wannierberri as wb

path_data = "../../tests/data/Fe-444-sitesym/"  
w90data=wb.w90files.Wannier90data(seedname=path_data+"Fe", readfiles=["mmn","eig", "win"]) 

from irrep.bandstructure import BandStructure
bandstructure = BandStructure(code='espresso', 
                            prefix = path_data+"pwscf/Fe",
                            Ecut=100,
                            normalize=False, 
                            magmom=[[0,0,1]],
                            include_TR=True)
spacegroup = bandstructure.spacegroup
symmetrizer = wb.symmetry.sawf.SymmetrizerSAWF().from_irrep(bandstructure)

from wannierberri.wannierise.projections import Projection, ProjectionsSet
projection_s = Projection(orbital='s', position_num=[0,0,0], spacegroup=spacegroup)
projection_p = Projection(orbital='p', position_num=[0,0,0], spacegroup=spacegroup)
projection_d = Projection(orbital='d', position_num=[0,0,0], spacegroup=spacegroup)
projection_sp3d2 = Projection(orbital='sp3d2', position_num=[0,0,0], spacegroup=spacegroup)
projection_t2g = Projection(orbital='t2g', position_num=[0,0,0], spacegroup=spacegroup)

# projections_set = ProjectionsSet(projections=[projection_s, projection_p, projection_d])
projections_set = ProjectionsSet(projections=[projection_sp3d2, projection_t2g])

symmetrizer.set_D_wann_from_projections(projections_obj=projections_set)
amn = wb.w90files.amn_from_bandstructure(bandstructure, projections_set=projections_set)
w90data.set_symmetrizer(symmetrizer)
w90data.set_file("amn", amn)

w90data.apply_window(win_min=-8,win_max= 50 )

w90data.wannierise( init = "amn",
                    froz_min=-8,
                    froz_max=30,
                    print_progress_every=1,
                    num_iter=11,
                    conv_tol=1e-6,
                    mix_ratio_z=1.0,
                    sitesym=True,
                    parallel=False
                    )


