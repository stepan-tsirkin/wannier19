from itertools import islice
import multiprocessing
from time import time
import numpy as np
from .utility import convert
from .w90file import W90_file


class MMN(W90_file):
    """
    class to store overlaps between Bloch functions at neighbouring k-points
    the MMN file of wannier90

    MMN.data[ik, ib, m, n] = <u_{m,k}|u_{n,k+b}>

    Parameters
    ----------
    seedname : str
        the prefix of the file (including relative/absolute path, but not including the extension  `.mmn`)
    npar : int
        the number of parallel processes to be used for reading the file

    Attributes
    ----------
    data : np.ndarray(shape=(NK, NNB, NB, NB), dtype=complex)
        the overlap matrix elements between the Wavefunctions at neighbouring k-points
    neighbours : np.ndarray(shape=(NK, NNB), dtype=int)
        the indices of the neighbouring k-points
    G : np.ndarray(shape=(NK, NNB, 3), dtype=int)
        the reciprocal lattice vectors connecting the k-points
    """

    @property
    def n_neighb(self):
        """
        number of nearest neighbours indices
        """
        return 1

    def __init__(self, seedname="wannier90", npar=multiprocessing.cpu_count(), **kwargs):
        self.npz_tags = ["data", "neighbours", "G"]
        super().__init__(seedname, ext="mmn", npar=npar, **kwargs)

    def from_w90_file(self, seedname, npar):
        t0 = time()
        f_mmn_in = open(seedname + ".mmn", "r")
        f_mmn_in.readline()
        NB, NK, NNB = np.array(f_mmn_in.readline().split(), dtype=int)
        block = 1 + NB * NB
        data = []
        headstring = []
        mult = 4

        # FIXME: npar = 0 does not work
        if npar > 0:
            pool = multiprocessing.Pool(npar)
        for j in range(0, NNB * NK, npar * mult):
            x = list(islice(f_mmn_in, int(block * npar * mult)))
            if len(x) == 0:
                break
            headstring += x[::block]
            y = [x[i * block + 1:(i + 1) * block] for i in range(npar * mult) if (i + 1) * block <= len(x)]
            if npar > 0:
                data += pool.map(convert, y)
            else:
                data += [convert(z) for z in y]

        if npar > 0:
            pool.close()
            pool.join()
        f_mmn_in.close()
        t1 = time()
        data = [d[:, 0] + 1j * d[:, 1] for d in data]
        self.data = np.array(data).reshape(NK, NNB, NB, NB).transpose((0, 1, 3, 2))
        headstring = np.array([s.split() for s in headstring], dtype=int).reshape(NK, NNB, 5)
        assert np.all(headstring[:, :, 0] - 1 == np.arange(NK)[:, None])
        self.neighbours = headstring[:, :, 1] - 1
        self.G = headstring[:, :, 2:]
        t2 = time()
        print(f"Time for MMN.__init__() : {t2 - t0} , read : {t1 - t0} , headstring {t2 - t1}")

    def to_w90_file(self, seedname):
        f_mmn_out = open(seedname + ".mmn", "w")
        f_mmn_out.write("MMN file\n")
        f_mmn_out.write(f"{self.NB} {self.NK} {self.NNB}\n")
        for ik in range(self.NK):
            for ib in range(self.NNB):
                f_mmn_out.write(f"{ik + 1} {self.neighbours[ik, ib] + 1} {' '.join(map(str, self.G[ik, ib]))}\n")
                for m in range(self.NB):
                    for n in range(self.NB):
                        f_mmn_out.write(f"{self.data[ik, ib, n, m].real} {self.data[ik, ib, n, m].imag}\n")
        f_mmn_out.close()

    def apply_window(self, selected_bands):
        if selected_bands is not None:
            self.data = self.data[:, :, selected_bands, :][:, :, :, selected_bands]

    def get_disentangled(self, v_left, v_right):
        """
        Reduce number of bands

        Parameters
        ----------
        v_matrix : np.ndarray(NB,num_wann)
            the matrix of column vectors defining the Wannier gauge
        v_matrix_dagger : np.ndarray(num_wann,NB)
            the Hermitian conjugate of `v_matrix`

        Returns
        -------
        `~wannierberri.system.w90_files.MMN`
            the disentangled MMN object
        """
        print(f"v shape {v_left.shape}  {v_right.shape}")
        data = np.zeros((self.NK, self.NNB, v_right.shape[2], v_right.shape[2]), dtype=complex)
        print(f"shape of data {data.shape} , old {self.data.shape}")
        for ik in range(self.NK):
            for ib, iknb in enumerate(self.neighbours[ik]):
                data[ik, ib] = v_left[ik] @ self.data[ik, ib] @ v_right[iknb]
        return self.__class__(data=data)

    def set_bk(self, kpt_latt, mp_grid, recip_lattice, kmesh_tol=1e-7, bk_complete_tol=1e-5):
        try:
            self.bk_cart
            self.wk
            self.wk_unique
            self.bk_latt_unique
            self.bk_cart_unique
            self.ib_unique_map
            self.ib_unique_map_inverse
            self.neighbours_unique
            return
        except AttributeError:
            bk_latt = np.array(
                np.round(
                    [
                        (kpt_latt[nbrs] - kpt_latt + G) * mp_grid[None, :]
                        for nbrs, G in zip(self.neighbours.T, self.G.transpose(1, 0, 2))
                    ]).transpose(1, 0, 2),
                dtype=int)
            bk_latt_unique = np.array([b for b in set(tuple(bk) for bk in bk_latt.reshape(-1, 3))], dtype=int)
            assert len(bk_latt_unique) == self.NNB, f"len(bk_latt_unique) = {len(bk_latt_unique)} != {self.NNB} \n bk_latt_unique={bk_latt_unique}, \nbk_latt={bk_latt}\n neighbours={self.neighbours}\n G={self.G}\n kpt_latt={kpt_latt}\n mp_grid={mp_grid}\n recip_lattice={recip_lattice}"
            bk_cart_unique = bk_latt_unique.dot(recip_lattice / mp_grid[:, None])
            bk_cart_unique_length = np.linalg.norm(bk_cart_unique, axis=1)
            srt = np.argsort(bk_cart_unique_length)
            bk_latt_unique = bk_latt_unique[srt]
            bk_cart_unique = bk_cart_unique[srt]
            bk_cart_unique_length = bk_cart_unique_length[srt]
            brd = [
                0,
            ] + list(np.where(bk_cart_unique_length[1:] - bk_cart_unique_length[:-1] > kmesh_tol)[0] + 1) + [
                self.NNB,
            ]
            shell_mat = np.array([bk_cart_unique[b1:b2].T.dot(bk_cart_unique[b1:b2]) for b1, b2 in zip(brd, brd[1:])])
            shell_mat_line = shell_mat.reshape(-1, 9)
            u, s, v = np.linalg.svd(shell_mat_line, full_matrices=False)
            s = 1. / s
            weight_shell = np.eye(3).reshape(1, -1).dot(v.T.dot(np.diag(s)).dot(u.T)).reshape(-1)
            check_eye = sum(w * m for w, m in zip(weight_shell, shell_mat))
            tol = np.linalg.norm(check_eye - np.eye(3))
            if tol > bk_complete_tol:
                raise RuntimeError(
                    f"Error while determining shell weights. the following matrix :\n {check_eye} \n"
                    f"failed to be identity by an error of {tol}. Further debug information :  \n"
                    f"bk_latt_unique={bk_latt_unique} \n bk_cart_unique={bk_cart_unique} \n"
                    f"bk_cart_unique_length={bk_cart_unique_length}\n shell_mat={shell_mat}\n"
                    f"weight_shell={weight_shell}\n")
            weight = np.array([w for w, b1, b2 in zip(weight_shell, brd, brd[1:]) for i in range(b1, b2)])
            weight_dict = {tuple(bk): w for bk, w in zip(bk_latt_unique, weight)}
            bk_cart_dict = {tuple(bk): bkcart for bk, bkcart in zip(bk_latt_unique, bk_cart_unique)}
            self.bk_cart = np.array([[bk_cart_dict[tuple(bkl)] for bkl in bklk] for bklk in bk_latt])
            self.wk = np.array([[weight_dict[tuple(bkl)] for bkl in bklk] for bklk in bk_latt])

            #############
            ### Oscar ###
            ###################################################################

            # Wannier90 provides a list of nearest-neighbor vectors b for every
            # q point. For Jae-Mo's finite-difference scheme it is convenient
            # to evaluate the Fourier transform of the matrix elements in the
            # original ab-initio mesh before performing the sum over
            # nearest-neighbor vectors. This requires defining a mapping from
            # any pair {q,b} to a unique list of b vectors that is independent
            # of q.

            bk_latt = np.rint((self.bk_cart @ np.linalg.inv(recip_lattice)) * mp_grid[None, None, :]).astype(int)
            bk_latt_unique = np.unique(bk_latt.reshape(-1, 3), axis=0)
            bk_cart_unique = (bk_latt_unique / mp_grid[None, :]) @ recip_lattice
            assert bk_latt_unique.shape == (self.NNB, 3)

            ib_unique_map = np.zeros((self.NK, self.NNB), dtype=int)
            ib_unique_map_inverse = np.zeros((self.NK, self.NNB), dtype=int)

            bk_latt_unique_tuples = [tuple(b) for b in bk_latt_unique]
            for ik in range(self.NK):
                for ib in range(self.NNB):
                    b_latt = np.rint((self.bk_cart[ik, ib, :] @ np.linalg.inv(recip_lattice)) * mp_grid).astype(int)
                    ib_unique = bk_latt_unique_tuples.index(tuple(b_latt))
                    assert np.allclose(bk_cart_unique[ib_unique, :], self.bk_cart[ik, ib, :])
                    ib_unique_map[ik, ib] = ib_unique
                    ib_unique_map_inverse[ik, ib_unique] = ib

            self.bk_latt_unique = bk_latt_unique
            self.bk_cart_unique = bk_cart_unique
            self.ib_unique_map = ib_unique_map
            ###################################################################
            self.ib_unique_map_inverse = ib_unique_map_inverse
            self.wk_unique = self.wk[0, ib_unique_map_inverse[0]]
            self.neighbours_unique = np.array([neigh[order] for neigh, order in
                                               zip(self.neighbours, self.ib_unique_map_inverse)])
            for ik in range(0, self.NK):
                order = ib_unique_map_inverse[ik]
                assert np.allclose(self.wk[ik, order], self.wk_unique)
                assert np.allclose(self.neighbours[ik, order], self.neighbours_unique[ik])

            self.bk_dot_bk = self.bk_cart_unique @ self.bk_cart_unique.T


    def set_bk_chk(self, chk, **argv):
        self.set_bk(chk.kpt_latt, chk.mp_grid, chk.recip_lattice, **argv)
