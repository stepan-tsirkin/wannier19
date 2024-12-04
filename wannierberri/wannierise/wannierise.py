from time import time
import warnings
import numpy as np

from .wannierizer import Wannierizer

from .utility import select_window_degen, print_centers_and_spreads, print_centers_and_spreads_chk
from ..__utility import vectorize
from .symmetrizer_SAWF import VoidSymmetrizer, Symmetrizer
# from .spreadfunctional import SpreadFunctional


def wannierise(w90data,
               froz_min=np.inf,
               froz_max=-np.inf,
               num_iter=1000,
               conv_tol=1e-9,
               num_iter_converge=3,
               mix_ratio_z=0.5,
               mix_ratio_u=1,
               print_progress_every=10,
               sitesym=False,
               localise=True,
               kwargs_sitesym={},
               init="amn",
               num_wann=None,
               parallel=True,
               symmetrize_Z=True,
               ):
    r"""
    Performs disentanglement and maximal localization of the bands recorded in w90data.
    The disentanglement is done following the procedure described in `Souza et al., PRB 2001 <https://doi.org/10.1103/PhysRevB.65.035109>`__. 
    The localization is done following a simplified procedure, avoiding gradient descent of the spread functional.

    At the end writes :attr:`w90data.chk.v_matrix` and sets :attr:`w90data.wannierised = True`

    Parameters
    ----------
    w90data: :class:`~wannierberri.w90files.Wannier90data`
        the data
    froz_min : float
        lower bound of the frozen window
    froz_max : float
        upper bound of the frozen window
    num_iter : int
        maximal number of iterations.
    conv_tol : float
        tolerance for convergence of the spread functional  (in :math:`\mathring{\rm A}^{2}`)
    num_iter_converge : int
        the convergence is achieved when the standard deviation of the spread functional over the `num_iter_converge*print_progress_every`
        iterations is less than conv_tol
    mix_ratio_z : float
        0 <= mix_ratio_z <=1  - mixing the Z matrix (disentanglement) from previous itertions. 1 for max speed, smaller values are more stable
    mix_ratio_u : float
        0 <= mix_ratio_u <=1  - mixing the U matrix (localization) from previous itertions. 
        1 for max speed, smaller values are more stable
        WARNING : u<1 at the moment may lead to non-convergence. It is recommended to use mix_ratio_u=1 for now.
    print_progress_every
        frequency to print the progress
    sitesym : bool
        whether to use the site symmetry (Symmetry-adapted Wannier Functions). If True, the dmn attribute should be present in the w90data (either from pw2wannier90 or generated by wabnnierberri
        see  :class:`~wannierberri.w90files.DMN`, :func:`~wannierberri.w90files.DMN.from_irrep` and :func:`~wannierberri.w90files.DMN.set_D_wann_from_projections`
    localise : bool
        whether to perform the localization. If False, only disentanglement and roatation to projections are performed
    kwargs_sitesym : dict
        additional the keyword arguments to be passed to the constructor of the :class:`~wannierberri.wannierise.sitesym.Symmetrizer`
    init : str
        the initialization of the U matrix. "amn" for the current state, "random" for random initialization, "restart" for restarting from the previous state
    num_wann : int
        the number of Wannier functions. Required for random initialization only without sitesymmetry
    parallel : bool
        if True - tries to run in parallel using ray. ray should be initialized before calling this function. If it was not initialized, the function will run in serial mode
    symmetrize_Z : bool
        whether to symmetrize the disentangled Z matrix. If False, the Z matrix is not symmetrized whoch may lead to inaccuracy and slower convergence, but may be a faster calculation. Use it on your own risk.
    Returns
    -------
    w90data.chk.v_matrix : numpy.ndarray
        the optimized U matrices


    Note
    -----
    * Also sets the following attributes of chk:
        - w90data.chk.v_matrix : numpy.ndarray
            the optimized U matrices
        - w90data.wannierised : bool
            True
        - w90data.chk._wannier_centers : numpy.ndarray (nW,3)
            the centers of the Wannier functions
        - w90data.chk._wannier_spreads : numpy.ndarray (nW)
            the spreads of the Wannier functions

    * If the outer window is needed, use :func:`~wannierberri.w90files.Wannier90data.apply_window` of the :class:`~wannierberri.w90files.Wannier90data` before calling this function. 
    * The function is not parallelized yet 
    * Disentanglement and localization are done in the irreducible BZ (if sitesym=True) and then symmetrized to the full BZ
    * Disentanglement and localization are done together, in the same loop. Therefore only one parameter `num_iter` is used for both

    """
    t0 = time()
    if froz_min > froz_max:
        print("froz_min > froz_max, nothing will be frozen")
    assert 0 < mix_ratio_z <= 1
    if sitesym:
        kptirr = w90data.dmn.kptirr
        symmetrizer_dmn = w90data.dmn
    else:
        kptirr = np.arange(w90data.mmn.NK)
        symmetrizer_dmn = None

    frozen = vectorize(select_window_degen, w90data.eig.data[kptirr], to_array=True,
                       kwargs=dict(win_min=froz_min, win_max=froz_max))
    free = vectorize(np.logical_not, frozen, to_array=True)

    if sitesym:
        symmetrizer = Symmetrizer(symmetrizer_dmn, neighbours=w90data.mmn.neighbours,
                                  symmetrize_Z=symmetrize_Z,
                                  **kwargs_sitesym)
    else:
        symmetrizer = VoidSymmetrizer(NK=w90data.mmn.NK)

    if init == "amn":
        amn = w90data.amn.data
    elif init == "random":
        if sitesym:
            num_wann = symmetrizer_dmn.num_wann
        else:
            assert num_wann is not None, "num_wann should be provided for random initialization without sitesymmetry"
        amnshape = (w90data.mmn.NK, w90data.mmn.NB, num_wann)
        amn = np.random.random(amnshape) + 1j * np.random.random(amnshape)
    elif init == "restart":
        assert w90data.wannierised, "The data is not wannierised"
        amn = np.zeros((w90data.mmn.NK, w90data.mmn.NB, w90data.chk.num_wann), dtype=np.complex128)
        for ik in range(w90data.mmn.NK):
            amn[ik][w90data.chk.win_min[ik]:w90data.chk.win_max[ik]] = w90data.chk.v_matrix[ik]
            w90data.chk.win_min[ik] = 0
            w90data.chk.win_max[ik] = w90data.chk.num_bands
        print("Restarting from the previous state", amn.shape)
    else:
        raise ValueError("init should be 'amn' or 'random'")

    neighbours_all = w90data.mmn.neighbours_unique
    neighbours_irreducible = np.array([[symmetrizer.kpt2kptirr[ik] for ik in neigh]
                                       for neigh in w90data.mmn.neighbours_unique[kptirr]])

    # wk = w90data.mmn.wk_unique
    bk_cart = w90data.mmn.bk_cart_unique
    mmn_data_ordered = np.array([data[order] for data, order in zip(w90data.mmn.data, w90data.mmn.ib_unique_map_inverse)])
    t1 = time()
    wannierizer = Wannierizer(parallel=parallel, symmetrizer=symmetrizer_dmn)
    for ik, kpt in enumerate(kptirr):
        wannierizer.add_kpoint(Mmn=mmn_data_ordered[kpt],
                            frozen=frozen[ik],
                            frozen_nb=frozen[neighbours_irreducible[ik]],
                            wb=w90data.mmn.wk_unique,
                            bk=w90data.mmn.bk_cart_unique,
                            symmetrizer_Zirr=symmetrizer.get_symmetrizer_Zirr(ik, free[ik]),
                            symmetrizer_Uirr=symmetrizer.get_symmetrizer_Uirr(ik),
                            ikirr=ik,
                            amn=amn[kpt],
                            weight=symmetrizer.ndegen(ik) / symmetrizer.NK
                            )
    t2 = time()

    # The _IR suffix is used to denote that the U matrix is defined only on k-points in the irreducible BZ
    U_opt_full_IR = wannierizer.get_U_opt_full()
    # the _BZ suffix is used to denote that the U matrix is defined on all k-points in the full BZ
    U_opt_full_BZ = symmetrizer.U_to_full_BZ(U_opt_full_IR, all_k=True)

    wannierizer.update_Unb_all([[U_opt_full_BZ[ib] for ib in neighbours_all[kpt]] for kpt in kptirr])


    # wcc, spreads = print_centers_and_spreads_chk(w90data=w90data, U_opt_full_BZ=U_opt_full_BZ, comment=f"starting WFs (from chk)")
    wcc = wannierizer.wcc
    print_centers_and_spreads(wcc=wcc, spreads=wannierizer.spreads, comment="starting WFs")

    converge_list = []
    delta_std = np.inf
    t_update = 0
    t01 = time()
    for i_iter in range(num_iter):
        wcc_bk_phase = np.exp(1j * wcc.dot(bk_cart.T))
        U_neigh = [[U_opt_full_BZ[ib] for ib in neighbours_all[kpt]] for kpt in kptirr]
        tx = time()
        U_opt_full_IR = wannierizer.update_all(U_neigh, mix_ratio=mix_ratio_z,
                                               mix_ratio_u=mix_ratio_u,
                                               localise=localise,
                                               wcc_bk_phase=wcc_bk_phase)
        t_update += time() - tx

        U_opt_full_BZ = symmetrizer.U_to_full_BZ(U_opt_full_IR, all_k=False)

        wcc = wannierizer.wcc
        spreads = wannierizer.spreads
        converge_list.append(np.hstack((wcc, spreads[:, None])))
        delta_std = np.std(converge_list[-num_iter_converge:], axis=0).max()

        if i_iter % print_progress_every == 0:
            print_centers_and_spreads(wcc=wcc, spreads=spreads, comment=f"Iteration {i_iter} (from wannierizer)", std=delta_std)

        if i_iter > num_iter_converge and delta_std < conv_tol:
            print(f"Converged after {i_iter} iterations")
            break

    t02 = time()

    U_opt_full_BZ = symmetrizer.U_to_full_BZ(U_opt_full_IR, all_k=True)
    print_centers_and_spreads(wcc=wcc, spreads=spreads, comment="Final state (from wannierizer)", std=delta_std)
    wcc_chk, spreads_chk = print_centers_and_spreads_chk(w90data=w90data, U_opt_full_BZ=U_opt_full_BZ, comment="Final state (from chk)")
    if not np.allclose(wcc, wcc_chk, atol=1e-4):
        warnings.warn(f"The Wannier centers from the chk file and the Wannier centers from the wannierizer are not the same. diff = {np.abs(wcc-wcc_chk).max()}")
    if not np.allclose(spreads, spreads_chk, atol=1e-4):
        warnings.warn(f"The Wannier spreads from the chk file and the Wannier spreads from the wannierizer are not the same. diff = {np.abs(spreads-spreads_chk).max()}")


    w90data.wannierised = True
    print(f"time for creating wannierizer {t2-t1}")
    print(f"time for iterations {t02-t01}")
    print(f"time for updating {t_update}")
    print(f"total time for wannierization {time()-t0}")
    return w90data.chk.v_matrix
