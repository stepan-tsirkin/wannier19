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
#                                                            #
#------------------------------------------------------------

import numpy as np
from collections.abc import Iterable
from time import time
import pickle
import glob

from .data_K import get_data_k
from .grid import exclude_equiv_points, Path, Grid, GridTetra
from .parallel import Serial
from .result import ResultDict


def print_progress(count, total, t0, tprev, print_progress_step):
    t = time() - t0
    t_remain = None
    if count == 0:
        t_remain = "unknown"
    else:
        t_rem_s = t / count * (total - count)
        t_remain = "{:22.1f}".format(t_rem_s)
    if t-tprev>print_progress_step:
        print("{:20d}{:17.1f}{:>22s}".format(count, t, t_remain), flush=True)
        tprev=t
    return tprev


def process(paralfunc, K_list, parallel, symgroup=None, remote_parameters={}, print_progress_step=5):
    print(f"symgroup : {symgroup}")
    t0 = time()
    t_print_prev = t0
    selK = [ik for ik, k in enumerate(K_list) if k.res is None]
    numK = len(selK)
    dK_list = [K_list[ik] for ik in selK]
    if len(dK_list) == 0:
        print("nothing to process now")
        return 0

    if parallel.method == 'ray':
        remotes = [paralfunc.remote(dK, **remote_parameters) for dK in dK_list]


    print("processing {0} K points :".format(len(dK_list)), end=" ")
    if parallel.method == 'serial':
        print("in serial.")
    else:
        print("using  {} processes.".format(parallel.npar_K))

    print("# K-points calculated  Wall time (sec)  Est. remaining (sec)", flush=True)
    res = []
    nstep_print = parallel.progress_step(numK, parallel.npar_K)
    if parallel.method == 'serial':
        for count, Kp in enumerate(dK_list):
            res.append(paralfunc(Kp, **remote_parameters))
            if (count + 1) % nstep_print == 0:
                t_print_prev = print_progress(count + 1, numK, t0, t_print_prev, print_progress_step)
    elif parallel.method == 'ray':
        num_remotes = len(remotes)
        num_remotes_calculated = 0
        while True:
            remotes_calculated, _ = parallel.ray.wait(
                remotes, num_returns=min(num_remotes_calculated + nstep_print, num_remotes),
                timeout=60)  # even, if the required number of remotes had not finished,
            # the progress will be printed every minute
            num_remotes_calculated = len(remotes_calculated)
            if num_remotes_calculated >= num_remotes:
                break
            t_print_prev = print_progress(num_remotes_calculated, numK, t0, t_print_prev, print_progress_step)
        res = parallel.ray.get(remotes)
    else:
        raise RuntimeError(f"unsupported parallel method : '{parallel.method}'")

    if not (symgroup is None):
        res = [symgroup.symmetrize(r) for r in res]
    for i, ik in enumerate(selK):
        K_list[ik].set_res(res[i])

    t = time() - t0
    if parallel.method == 'serial':
        print("time for processing {0:6d} K-points in serial: ".format(numK), end="")
        nproc_ = 1
    else:
        print("time for processing {0:6d} K-points on {1:3d} processes: ".format(numK, parallel.npar_K), end="")
        nproc_ = parallel.npar_K
    print(
        "{0:10.4f} ; per K-point {1:15.4f} ; proc-sec per K-point {2:15.4f}".format(t, t / numK, t * nproc_ / numK),
        flush=True)
    return len(dK_list)


def run(
    system,
    grid,
    calculators,
    adpt_num_iter=0,
    use_irred_kpt=True,
    symmetrize=True,
    fout_name="result",
    suffix="",
    parameters_K={},
    file_Klist=None,
    restart=False,
    Klist_part=10,
    parallel=Serial(),  # serial by default
    print_Kpoints=True,
    adpt_mesh=2,
    adpt_fac=1,
    fast_iter=True,
    print_progress_step=5,
):
    """
    The function to run a calculation. Substitutes the old :func:`~wannierberri.integrate` and :func:`~wannierberri.tabulate`
    and allows to integrate and tabulate in one run.

    Parameters
    ----------
    system : :class:`~wannierberri.system.System`
        System under investigation
    grid : :class:`~wannierberri.Grid` or :class:`~wannierberri.Path`
        initial grid for integration. or path for tabulation
    calculators : dict
        a dictionary where keys aare any string identifiers, and the values are of :class:`~wannierberri.calculators.Calculator`
    adpt_num_iter : int
        number of recursive adaptive refinement iterations. See :ref:`sec-refine`
    adpt_mesh : int
        the size of the refinement grid (usuallay no need to change)
    adpt_fac : int
        number of K-points to be refined per quantity and criteria.
    parallel : :class:`~wannierberri.parallel.Parallel`
        object describing parallelization scheme
    use_irred_kpt : bool
        evaluate only symmetry-irreducible K-points
    symmetrize : bool
        symmetrize the result (always `True` if `use_irred_kpt == True`)
    fout_name : str
        beginning of the output files for each quantity after each iteration
    suffix : str
        extra marker inserted into output files to mark this particular calculation run
    print_Kpoints : bool
        print the list of K points
    file_Klist : str or None
        name of file where to store the Kpoint list of each iteration. May be needed to restart a calculation
        to get more iterations. If `None` -- the file is not written
    restart : bool
        if `True` : reads restart information from `file_Klist` and starts from there
    Klist_part : int
        write the file_Klist by portions. Increase for speed, decrease for memory saving
    fast_iter : bool
        if under iterations appear peaks that arte not further removed, set this parameter to False.
    print_progress_step : float or int
        intervals to print progress

    Returns
    --------
    dictionary of  :class:`~wannierberri.result.EnergyResult`

    Notes
    -----
    Results are also printed to ASCII files
    """

    # along a path only tabulating is possible
    if isinstance(grid,Path):
        print ("Calculation along a path - checking calculators for compatibility")
        for key,calc in calculators.items():
            print (key,calc)
            if not calc.allow_path:
                raise ValueError(f"Calculation along a Path is running, but calculator `{key}` is not compatible with a Path")
        print ("All calculators are compatible")
        if symmetrize:
            print ("Symmetrization switched off for Path")
            symmetrize = False
    if isinstance(grid,GridTetra):
        print ("Grid is tetrahedral")
    else:
        print ("Grid is regular")

    if file_Klist is not None:
        do_write_Klist = True
        if not file_Klist.endswith(".pickle"):
            file_Klist += ".pickle"
            file_Klist_factor_changed = file_Klist+".changed_factors.txt"
        else:
            file_Klist_factor_changed = file_Klist[:-7]+".changed_factors.txt"
    else:
        do_write_Klist = False
        file_Klist_factor_changed = None

    print(f"The set of k points is a {grid.str_short}")

    remote_parameters = {'_system': system, '_grid': grid, 'npar_k': parallel.npar_k, '_calculators': calculators}
    if parallel.method == 'ray':
        ray = parallel.ray
        remote_parameters = {k: ray.put(v) for k, v in remote_parameters.items()}

        @ray.remote
        def paralfunc(Kpoint, _system, _grid, _calculators, npar_k):
            data = get_data_k(_system,  Kpoint.Kp_fullBZ, grid=_grid, Kpoint=Kpoint, **parameters_K)
            return ResultDict({k: v(data) for k, v in _calculators.items()})
    else:

        def paralfunc(Kpoint, _system, _grid, _calculators, npar_k):
            data = get_data_k(_system, Kpoint.Kp_fullBZ, grid=_grid, Kpoint=Kpoint, **parameters_K)
            return ResultDict({k: v(data) for k, v in _calculators.items()})

    if restart:
        try:
            fr = open(file_Klist, "rb")
            K_list = []
            while True:
                try:
                    K_list += pickle.load(fr)
                except EOFError:
                    print("Finished reading Klist from file {0}".format(file_Klist))
                    break
            print("{0} K-points were read from {1}".format(len(K_list), file_Klist))
            if len(K_list) == 0:
                print("WARNING : {0} contains zero points starting from scrath".format(file_Klist))
                restart = False
            fr.close()

            nk_prev = len(K_list)

            try:
                # patching the Klist by updating the factors
                fr_div = open(file_Klist_factor_changed, "r")
                factor_changed_K_list = []
                for line in fr_div:
                    line_ = line.split()
                    iK = int(line_[0])
                    fac = float(line_[1])

                    factor_changed_K_list.append(iK)
                    K_list[iK].factor = fac
                print("{0} K-points were read from {1}".format(len(factor_changed_K_list), file_Klist_factor_changed))
                fr_div.close()
            except FileNotFoundError:
                print (f"File with changed factors {file_Klist_factor_changed} not found, assume they were not changed")
        except Exception as err:
            restart = False
#            print("WARNING: {}".format(err))
            raise RuntimeError("{1}: reading from {0} failed, starting from scrath".format(file_Klist,err))
    else:
        K_list = grid.get_K_list(use_symmetry=use_irred_kpt)
        print("Done, sum of weights:{}".format(sum(Kp.factor for Kp in K_list)))
        start_iter = 0
        nk_prev = 0

    if not restart:
        import os
        def remove_file(filename):
            if filename is not None and os.path.exists(filename):
                os.remove(filename)
        remove_file(file_Klist)
        remove_file(file_Klist_factor_changed)


#    suffix="-"+suffix if len(suffix)>0 else ""

    if restart:
        print("searching for start_iter")
        try:
            start_iter = int(
                sorted(glob.glob(fout_name + "*" + suffix + "_iter-*.dat"))[-1].split("-")[-1].split(".")[0])
            print(f"start_iter = {start_iter}")
        except Exception as err:
            print("WARNING : {0} : failed to read start_iter. Setting to zero".format(err))
            start_iter = 0

    if adpt_num_iter < 0:
        adpt_num_iter = -adpt_num_iter * np.prod(grid.div) / np.prod(adpt_mesh) / adpt_fac / 3
    adpt_num_iter = int(round(adpt_num_iter))

    if (adpt_mesh is None) or np.max(adpt_mesh) <= 1:
        adpt_num_iter = 0
    else:
        if not isinstance(adpt_mesh, Iterable):
            adpt_mesh = [adpt_mesh] * 3
        adpt_mesh = np.array(adpt_mesh)

    counter = 0
    result_all = None
    result_excluded = None

    for i_iter in range(adpt_num_iter + 1):
        if print_Kpoints:
            print(
                "iteration {0} - {1} points. New points are:".format(i_iter+start_iter, len([K for K in K_list if K.res is None])))
            for i, K in enumerate(K_list):
                if not K.evaluated:
                    print(" K-point {0} : {1} ".format(i, K))
        counter += process(
            paralfunc,
            K_list,
            parallel,
            symgroup=system.symgroup if symmetrize else None,
            print_progress_step=print_progress_step,
            remote_parameters=remote_parameters)

        nk = len(K_list)
        try:
            if do_write_Klist:
                # append new (refined) k-points only
                fw = open(file_Klist, "ab")
                for ink in range(nk_prev, nk, Klist_part):
                    pickle.dump(K_list[ink:ink + Klist_part], fw)
                fw.close()

        except Exception as err:
            print("Warning: {0} \n the K_list was not pickled".format(err))

        time0 = time()

        if (result_all is None) or (not fast_iter):
            result_all = sum(kp.get_res for kp in K_list)
        else:
            if result_excluded is not None:
                result_all -= result_excluded
            result_all += sum(kp.get_res for kp in K_list[nk_prev:])

        time1 = time()
        print("time1 = ", time1 - time0)
        if not (restart and i_iter == 0):
            result_all.savedata(prefix=fout_name, suffix=suffix, i_iter=i_iter + start_iter)

        if i_iter >= adpt_num_iter:
            break

        # Now add some more points
        Kmax = np.array([K.max for K in K_list]).T
        select_points = set().union(*(np.argsort(Km)[-adpt_fac:] for Km in Kmax))

        time2 = time()
        print("time2 = ", time2 - time1)
        l1 = len(K_list)

        excluded_Klist = []
        result_excluded = None

        nk_prev = nk

        for iK in select_points:
            results = K_list[iK].get_res
            K_list += K_list[iK].divide(adpt_mesh, periodic=system.periodic, use_symmetry=use_irred_kpt)
            if abs(K_list[iK].factor) < 1.e-10:
                excluded_Klist.append(iK)
                if result_excluded is None:
                    result_excluded = results - K_list[iK].get_res
                else:
                    result_excluded += results - K_list[iK].get_res

        if use_irred_kpt and isinstance(grid,Grid):
            print("checking for equivalent points in all points (of new  {} points)".format(len(K_list) - l1))
            nexcl, weight_changed_old = exclude_equiv_points(K_list, new_points=len(K_list) - l1)
            print(" excluded {0} points".format(nexcl))
        else:
            weight_changed_old = {}

        print("sum of weights now :{}".format(sum(Kp.factor for Kp in K_list)))

        for iK, prev_factor in weight_changed_old.items():
            result_excluded += K_list[iK].res * (prev_factor - K_list[iK].factor)

        if do_write_Klist:
            print (f"Writing file_Klist_factor_changed to {file_Klist_factor_changed}")
            fw_changed = open(file_Klist_factor_changed, "a")
            for iK in excluded_Klist:
                fw_changed.write("{0} {1} # refined\n".format(iK, 0.0))
            for iK in weight_changed_old:
                fw_changed.write("{0} {1} # changed\n".format(iK, K_list[iK].factor))
            fw_changed.close()


    print("Totally processed {0} K-points ".format(counter))

    return result_all
