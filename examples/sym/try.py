from matplotlib import pyplot as plt
import wannierberri as wberri


w90data = wberri.system.Wannier90data(seedname='diamond')

w90data.check_symmetry()

system2 = wberri.system.System_w90('diamond')
system0 = wberri.system.System_w90('ref/diamond')

w90data.disentangle(
                 froz_min=-8,
                 froz_max=10,
                 num_iter=1000,
                 conv_tol=1e-10,
                 mix_ratio=1.0,
                 print_progress_every=20,
                 sitesym=True
                  )
system1 = wberri.system.System_w90(w90data=w90data)
path = wberri.Path(system2, k_nodes=[[0, 0, 0],
                                     [0.5, 0, 0],
                                     [0.5, 0.5, 0],
                                     [0, 0, 0] 
                                     ], labels=['G', 'L', 'X', 'G'
                                                ], length=100)
tabulator = wberri.calculators.TabulatorAll(tabulators={}, mode='path')
calculators = {'tabulate': tabulator}
kwargs = dict(grid=path, calculators=calculators, print_Kpoints=False, file_Klist=None)
result0 = wberri.run(system0, **kwargs)
result1 = wberri.run(system1, **kwargs)
result2 = wberri.run(system2, **kwargs)

# reference
result0.results['tabulate'].plot_path_fat(path, close_fig=False, show_fig=False, linecolor='black')

# wberri disentranglement
result1.results['tabulate'].plot_path_fat(path, close_fig=False, show_fig=False, linecolor='red', kwargs_line={'linestyle': '--'})

# w90 disentranglement
result2.results['tabulate'].plot_path_fat(path, close_fig=False, show_fig=False, linecolor='blue')
plt.ylim(-10, 30)
plt.show()
