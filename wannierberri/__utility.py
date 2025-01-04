#                                                            #
# This file is distributed as part of the WannierBerri code  #
# under the terms of the GNU General Public License. See the #
# file 'LICENSE' in the root directory of the WannierBerri   #
# distribution, or http://www.gnu.org/copyleft/gpl.txt       #
#                                                            #
# The WannierBerri code is hosted on GitHub:                 #
# https://github.com/stepan-tsirkin/wannier-berri            #
#                     written by                             #
#           Stepan Tsirkin, University of Zurich             #
#                                                            #
# ------------------------------------------------------------

from random import shuffle
import scipy.io
import fortio
import os
from time import time
from functools import cached_property, lru_cache
import numpy as np
import warnings
from . import PYFFTW_IMPORTED
from collections.abc import Iterable
import datetime
import sys
from typing import Any

if PYFFTW_IMPORTED:
    import pyfftw


def time_now_iso():
    return datetime.datetime.now().isoformat()


# inheriting just in order to have possibility to change default values, without changing the rest of the code
class FortranFileR(fortio.FortranFile):

    def __init__(self, filename):
        print("using fortio to read")
        try:
            super().__init__(filename, mode='r', header_dtype='uint32', auto_endian=True, check_file=True)
        except ValueError:
            print(f"File '{filename}' contains sub-records - using header_dtype='int32'")
            super().__init__(filename, mode='r', header_dtype='int32', auto_endian=True, check_file=True)


class FortranFileW(scipy.io.FortranFile):

    def __init__(self, filename):
        print("using scipy.io to write")
        super().__init__(filename, mode='w')


alpha_A = np.array([1, 2, 0])
beta_A = np.array([2, 0, 1])
delta_f = np.eye(3)


def conjugate_basis(basis):
    return 2 * np.pi * np.linalg.inv(basis).T


def real_recip_lattice(real_lattice=None, recip_lattice=None):
    if recip_lattice is None:
        if real_lattice is None:
            warnings.warn("usually need to provide either with real or reciprocal lattice."
                          "If you only want to generate a random symmetric tensor - that it fine")
            return None, None
        else:
            recip_lattice = conjugate_basis(real_lattice)
    else:
        if real_lattice is not None:
            assert np.linalg.norm(
                np.array(real_lattice).dot(recip_lattice.T) / (2 * np.pi) -
                np.eye(3)) <= 1e-8, "real and reciprocal lattice do not match"
        else:
            real_lattice = conjugate_basis(recip_lattice)
    return np.array(real_lattice), np.array(recip_lattice)


def clear_cached(obj, properties=()):
    for attr in properties:
        if hasattr(obj, attr):
            delattr(obj, attr)


def str2bool(v):
    v1 = v.strip().lower()
    if v1 in ("f", "false", ".false."):
        return False
    elif v1 in ("t", "true", ".true."):
        return True
    else:
        raise ValueError(f"unrecognized value of bool parameter :`{v}`")


def fft_W(inp, axes, inverse=False, destroy=True, numthreads=1):
    try:
        assert inp.dtype == complex
        fft_in = pyfftw.empty_aligned(inp.shape, dtype='complex128')
        fft_out = pyfftw.empty_aligned(inp.shape, dtype='complex128')
        fft_object = pyfftw.FFTW(
            fft_in,
            fft_out,
            axes=axes,
            flags=('FFTW_ESTIMATE',) + (('FFTW_DESTROY_INPUT',) if destroy else ()),
            direction='FFTW_BACKWARD' if inverse else 'FFTW_FORWARD',
            threads=numthreads)
        fft_object(inp)
        return fft_out
    except RuntimeError as err:
        if "This is a bug" in str(err):
            raise RuntimeError(f"{err}\n Probably this can be fixed by importing wannierberri prior to numpy."
                               " See https://docs.wannier-berri.org/en/master/install.html#known-bug-with-pyfftw")
        else:
            raise err


@lru_cache()
def get_head(n):
    if n <= 0:
        return ['  ']
    else:
        return [a + b for a in 'xyz' for b in get_head(n - 1)]


def fft_np(inp, axes, inverse=False):
    assert inp.dtype == complex
    if inverse:
        return np.fft.ifftn(inp, axes=axes)
    else:
        return np.fft.fftn(inp, axes=axes)


def execute_fft(inp, axes, inverse=False, destroy=True, numthreads=1, fftlib='fftw'):
    fftlib = fftlib.lower()
    if fftlib == 'fftw' and not PYFFTW_IMPORTED:
        fftlib = 'numpy'
    if fftlib == 'fftw':
        return fft_W(inp, axes, inverse=inverse, destroy=destroy, numthreads=numthreads)
    elif fftlib == 'numpy':
        return fft_np(inp, axes, inverse=inverse)
    else:
        raise ValueError(f"unknown type of fftlib : {fftlib}")


def fourier_q_to_R(AA_q, mp_grid, kpt_mp_grid, iRvec, ndegen, numthreads=1, fftlib='fftw'):
    mp_grid = tuple(mp_grid)
    shapeA = AA_q.shape[1:]  # remember the shapes after q
    AA_q_mp = np.zeros(tuple(mp_grid) + shapeA, dtype=complex)
    for i, k in enumerate(kpt_mp_grid):
        AA_q_mp[k] = AA_q[i]
    AA_q_mp = execute_fft(AA_q_mp, axes=(0, 1, 2), numthreads=numthreads, fftlib=fftlib, destroy=False)
    AA_R = np.array([AA_q_mp[tuple(iR % mp_grid)] / nd for iR, nd in zip(iRvec, ndegen)]) / np.prod(mp_grid)
    AA_R = AA_R.transpose((1, 2, 0) + tuple(range(3, AA_R.ndim)))
    return AA_R


class FFT_R_to_k:

    def __init__(self, iRvec, NKFFT, num_wann, numthreads=1, fftlib='fftw', name=None):
        t0 = time()
        self.NKFFT = tuple(NKFFT)
        self.num_wann = num_wann
        self.name = name
        fftlib = fftlib.lower()
        assert fftlib in ('fftw', 'numpy', 'slow'), f"fftlib '{fftlib}' is unknown/not supported"
        if fftlib == 'fftw' and not PYFFTW_IMPORTED:
            fftlib = 'numpy'
        self.lib = fftlib
        if fftlib == 'fftw':
            shape = self.NKFFT + (self.num_wann, self.num_wann)
            fft_in = pyfftw.empty_aligned(shape, dtype='complex128')
            fft_out = pyfftw.empty_aligned(shape, dtype='complex128')
            self.fft_plan = pyfftw.FFTW(
                fft_in,
                fft_out,
                axes=(0, 1, 2),
                flags=('FFTW_ESTIMATE', 'FFTW_DESTROY_INPUT'),
                direction='FFTW_BACKWARD',
                threads=numthreads)
        self.iRvec = iRvec % self.NKFFT
        self.nRvec = iRvec.shape[0]
        self.time_init = time() - t0
        self.time_call = 0
        self.n_call = 0

    def execute_fft(self, A):
        return self.fft_plan(A)

    def transform(self, AAA_K):
        assert np.iscomplexobj(AAA_K)
        if self.lib == 'numpy':
            AAA_K[...] = np.fft.ifftn(AAA_K, axes=(0, 1, 2))
        elif self.lib == 'fftw':
            # do recursion if array has cartesian indices. The recursion should not be very deep
            if AAA_K.ndim > 5:
                for i in range(AAA_K.shape[-1]):
                    AAA_K[..., i] = self.transform(AAA_K[..., i])
            else:
                AAA_K[...] = self.execute_fft(AAA_K[...])
            return AAA_K
        elif self.lib == 'slow':
            raise RuntimeError("FFT.transform should not be called for slow FT")

    @cached_property
    def exponent(self):
        """
        exponent for Fourier transform exp(1j*k*R)
        """
        return [np.exp(2j * np.pi / self.NKFFT[i]) ** np.arange(self.NKFFT[i]) for i in range(3)]

    def __call__(self, AAA_R, hermitian=False, antihermitean=False, reshapeKline=True):
        t0 = time()
        # AAA_R is an array of dimension (  num_wann x num_wann x nRpts X... ) (any further dimensions allowed)
        if hermitian and antihermitean:
            raise ValueError("A matrix cannot be both hermitian and anti-hermitian, unless it is zero")
        AAA_R = AAA_R.transpose((2, 0, 1) + tuple(range(3, AAA_R.ndim)))
        shapeA = AAA_R.shape
        if self.lib == 'slow':
            k = np.zeros(3, dtype=int)
            AAA_K = np.array(
                [
                    [
                        [
                            sum(
                                np.prod([self.exponent[i][(k[i] * R[i]) % self.NKFFT[i]] for i in range(3)]) * A
                                for R, A in zip(self.iRvec, AAA_R)) for k[2] in range(self.NKFFT[2])
                        ] for k[1] in range(self.NKFFT[1])
                    ] for k[0] in range(self.NKFFT[0])
                ])
        else:
            assert self.nRvec == shapeA[0]
            assert self.num_wann == shapeA[1] == shapeA[2]
            AAA_K = np.zeros(self.NKFFT + shapeA[1:], dtype=complex)
            # TODO : place AAA_R to FFT grid from beginning, even before multiplying by exp(dkR)
            for ir, irvec in enumerate(self.iRvec):
                AAA_K[tuple(irvec)] += AAA_R[ir]
            self.transform(AAA_K)
            AAA_K *= np.prod(self.NKFFT)

        # TODO - think if fftlib transform of half of matrix makes sense
        if hermitian:
            AAA_K = 0.5 * (AAA_K + AAA_K.transpose((0, 1, 2, 4, 3) + tuple(range(5, AAA_K.ndim))).conj())
        elif antihermitean:
            AAA_K = 0.5 * (AAA_K - AAA_K.transpose((0, 1, 2, 4, 3) + tuple(range(5, AAA_K.ndim))).conj())

        if reshapeKline:
            AAA_K = AAA_K.reshape((np.prod(self.NKFFT),) + shapeA[1:])
        self.time_call += time() - t0
        self.n_call += 1
        return AAA_K


def iterate_nd(size, pm=False):
    a = -size[0] if pm else 0
    b = size[0] + 1 if pm else size[0]
    if len(size) == 1:
        return np.array([(i,) for i in range(a, b)])
    else:
        return np.array([(i,) + tuple(j) for i in range(a, b) for j in iterate_nd(size[1:], pm=pm)])


def iterate3dpm(size):
    assert len(size) == 3
    return iterate_nd(size, pm=True)


#   return (
#       np.array([i, j, k]) for i in range(-size[0], size[0] + 1) for j in range(-size[1], size[1] + 1)
#       for k in range(-size[2], size[2] + 1))


# def iterate3d(size):
#    assert len(size)==3
#    return iterate_nd(size,pm=False)
#    return (np.array([i, j, k]) for i in range(0, size[0]) for j in range(0, size[1]) for k in range(0, size[2]))


def find_degen(arr, degen_thresh):
    """ finds shells of 'almost same' values in array arr, and returns a list o[(b1,b2),...]"""
    A = np.where(arr[1:] - arr[:-1] > degen_thresh)[0] + 1
    A = [0, ] + list(A) + [len(arr)]
    return [(ib1, ib2) for ib1, ib2 in zip(A, A[1:])]


def is_round(A, prec=1e-14):
    # returns true if all values in A are integers, at least within machine precision
    return np.linalg.norm(A - np.round(A)) < prec


def get_angle(sina, cosa):
    """Get angle in radian from sin and cos."""
    if abs(cosa) > 1.0:
        cosa = np.round(cosa, decimals=1)
    alpha = np.arccos(cosa)
    if sina < 0.0:
        alpha = 2.0 * np.pi - alpha
    return alpha


def angle_vectors(vec1, vec2):
    cos = np.dot(vec1, vec2) / np.linalg.norm(vec1) / np.linalg.norm(vec2)
    return np.arccos(cos)


def angle_vectors_deg(vec1, vec2):
    angle = angle_vectors(vec1, vec2)
    return int(round(angle / np.pi * 180))


# smearing functions
def Lorentzian(x, width):
    return 1.0 / (np.pi * width) * width ** 2 / (x ** 2 + width ** 2)


def Gaussian(x, width, adpt_smr):
    """
    Compute 1 / (np.sqrt(pi) * width) * exp(-(x / width) ** 2)
    If the exponent is less than -200, return 0.
    An unoptimized version is the following.
        def Gaussian(x, width, adpt_smr):
            return 1 / (np.sqrt(pi) * width) * np.exp(-np.minimum(200.0, (x / width) ** 2))
    """
    inds = abs(x) < width * np.sqrt(200.0)
    output = np.zeros(x.shape, dtype=float)
    if adpt_smr:
        # width is array
        width_tile = np.tile(width, (x.shape[0], 1, 1))
        output[inds] = 1.0 / (np.sqrt(np.pi) * width_tile[inds]) * np.exp(-(x[inds] / width_tile[inds]) ** 2)
    else:
        # width is number
        output[inds] = 1.0 / (np.sqrt(np.pi) * width) * np.exp(-(x[inds] / width) ** 2)
    return output


# auxillary function"
def FermiDirac(E, mu, kBT):
    """here E is a number, mu is an array"""
    if kBT == 0:
        return 1.0 * (E <= mu)
    else:
        res = np.zeros(mu.shape, dtype=float)
        res[mu > E + 30 * kBT] = 1.0
        res[mu < E - 30 * kBT] = 0.0
        sel = abs(mu - E) <= 30 * kBT
        res[sel] = 1.0 / (np.exp((E - mu[sel]) / kBT) + 1)
        return res


def one2three(nk):
    if nk is None:
        return None
    elif isinstance(nk, Iterable):
        assert len(nk) == 3
    else:
        nk = (nk,) * 3
    assert np.all([isinstance(n, (int, np.integer)) and n > 0 for n in nk])
    return np.array(nk)


def remove_file(filename):
    if filename is not None and os.path.exists(filename):
        os.remove(filename)


def vectorize(func, *args, kwargs={}, sum=False, to_array=False):
    """decorator to vectorize the function over the positional arguments

    TODO : make parallel

    Parameters
    ----------
    func : function
        the function to vectorize over all the arguments
    args : list
        list of arguments
    kwargs : dict
        keyword arguments
    to_array : bool
        if True, return the result as numpy array, otherwise as list
    sum : bool
        if True, sum the results (after transforming to numpy array, if to_array is True)

    Returns
    -------
    list or np.array
        list of results of the function applied to all the arguments
    """
    l = [len(a) for a in args]
    assert all([_ == l[0] for _ in l]), f"length of all arguments should be the same, got {l}"
    lst = [func(*a, **kwargs) for a in zip(*args)]
    if to_array:
        lst = np.array(lst)
    if sum:
        lst = sum(lst)
    return lst


class UniqueList(list):
    """	
    A list that only allows unique elements.
    uniqueness is determined by the == operator.
    Thus, non-hashable elements are also allowed.
    unlike set, the order of elements is preserved.
    """

    def __init__(self, iterator=[], count=False):
        super().__init__()
        self.do_count = count
        if self.do_count:
            self.counts = []
        for x in iterator:
            self.append(x)

    def append(self, item, count=1):
        for j, i in enumerate(self):
            if i == item:
                if self.do_count:
                    self.counts[self.index(i)] += count
                break
        else:
            super().append(item)
            if self.do_count:
                self.counts.append(1)

    def index(self, value: Any, start=0, stop=sys.maxsize) -> int:
        for i in range(start, stop):
            if self[i] == value:
                return i
        raise ValueError(f"{value} not in list")

    def __contains__(self, item):
        for i in self:
            if i == item:
                return True
        return False

    def remove(self, value: Any, all=False) -> None:
        for i in range(len(self)):
            if self[i] == value:
                if all or not self.do_count:
                    del self[i]
                    del self.counts[i]
                else:
                    self.counts[i] -= 1
                    if self.counts[i] == 0:
                        del self[i]
                        del self.counts[i]
                return


class UniqueListMod1(UniqueList):

    def __init__(self, iterator=[], tol=1e-5):
        self.tol = tol
        self.appended_indices = []
        self.last_try_append = -1
        super().__init__(iterator)

    def append(self, item):
        self.last_try_append += 1
        for i in self:
            if all_close_mod1(i, item, tol=self.tol):
                break
        else:
            list.append(self, item)
            self.appended_indices.append(self.last_try_append)

    def __contains__(self, item):
        for i in self:
            if all_close_mod1(i, item, tol=self.tol):
                return True
        return False

    def index(self, value: Any, start=0, stop=sys.maxsize) -> int:
        stop = min(stop, len(self))
        for i in range(start, stop):
            if all_close_mod1(self[i], value):
                return i
        raise ValueError(f"{value} not in list")


def all_close_mod1(a, b, tol=1e-5):
    """check if two vectors are equal modulo 1"""
    if not np.shape(a) == () and not np.shape(b) == () and (np.shape(a) != np.shape(b)):
        return False
    diff = a - b
    return np.allclose(np.round(diff), diff, atol=tol)


def rotate_block_matrix(Z, lblocks, lindices, rblocks, rindices,
                        # inv_left, inv_right,
                        result=None):
    """
    Rotates the matrix Z using the block-diagonal rotation matrices

    Parameters
    ----------
    Z : np.array(complex, shape=(M,N))
        the matrix to be rotated
    lblocks : list(np.array(complex, shape=(m,m)))
        the blocks of hte left matrix. sum(m) = M
    lindices : list(tuple(int))
        the indices of the blocks of the left matrix
    rblocks : list(np.array(complex, shape=(n,n)))
        the blocks of hte right matrix. sum(n) = N
    rindices : list(tuple(int))
        the indices of the blocks of the right matrix

    Returns
    -------
    np.array(complex, shape=(M,N))
        the rotated matrix
    """
    if result is None:
        result = np.zeros(Z.shape, dtype=Z.dtype)
    for (start, end), block in zip(lindices, lblocks):
        result[start:end, :] = block @ Z[start:end, :]

    for (start, end), block in zip(rindices, rblocks):
        result[:, start:end] = result[:, start:end] @ block

    return result


def block_matrix_to_diagonals(blocks, indices=None):
    """
    Extracts the blocks of a block-diagonal matrix

    Parameters
    ----------
    blocks : list(np.array(complex, shape=(m,m)))
        the blocks of hte left matrix. sum(m) = M
    indices : list(tuple(int))
        the indices of the blocks of the left matrix

    Returns
    -------
    list(np.array(complex, shape=(m,m)))
        the blocks of the matrix
    """
    if indices is None:
        indices = []
        start = 0
        for block in blocks:
            indices.append((start, start + block.shape[0]))
            start += block.shape[0]

    size = indices[-1][1] - indices[0][0]
    tmp = np.zeros((size, size), dtype=blocks[0].dtype)
    max_block_size = max(block.shape[0] for block in blocks)
    for (start, end), block in zip(indices, blocks):
        tmp[start:end, start:end] = block
    result = np.zeros((2 * max_block_size - 1, size), dtype=blocks[0].dtype)
    rng = np.arange(size)
    result[0] = tmp[rng, rng]
    for l in range(1, max_block_size):
        result[l, :-l] = tmp[rng[:-l], rng[l:]]
        result[-l, :-l] = tmp[rng[l:], rng[:-l]]
    return result


def get_inverse_block(D):
    """
    Get the inverse of a block-diagonal matrix (given as a nested list of numpy arrays)
    """
    if isinstance(D, list):
        return [get_inverse_block(d) for d in D]
    elif isinstance(D, np.ndarray):
        return np.linalg.inv(D)
    else:
        raise ValueError(f"Unknown type {type(D)}")


def get_max_eig(matrix, nvec, nBfree):
    """ return the nvec column-eigenvectors of matrix with maximal eigenvalues.

    Parameters
    ----------
    matrix : numpy.ndarray(n,n)
        list of matrices
    nvec : int
        number of eigenvectors to return
    nBfree : int
        number of free bands

    Returns
    -------
    numpy.ndarray(n,nvec)
        eigenvectors
    """
    assert matrix.shape[0] == matrix.shape[1]
    assert matrix.shape[0] >= nvec, f"nvec={nvec}, matrix.shape={matrix.shape}"
    e, v = np.linalg.eigh(matrix)
    return v[:, np.argsort(e)[nBfree - nvec:nBfree]]


def orthogonalize(u):
    """
    Orthogonalizes the matrix u using Singular Value Decomposition (SVD).

    Parameters
    ----------
    u : np.ndarray(dtype=complex, shape = (nBfree,nWfree,))
        The input matrix to be orthogonalized.

    Returns
    -------
    u : np.ndarray(dtype=complex, shape = (nBfree,nWfree,))
        The orthogonalized matrix.
    """
    try:
        U, _, VT = np.linalg.svd(u, full_matrices=False)
        return U @ VT
    except np.linalg.LinAlgError as e:
        warnings.warn(f"SVD failed with error '{e}', using non-orthogonalized matrix")
        return u

def points_to_mp_grid_1d(points, prec=1e-5):
    for i in range(1,100):
        x = points*i
        x -= np.round(x)
        if np.all(abs(x))<prec:
            return i
    else:
        raise RuntimeError(f"could not determine the grid for points {points}")


def points_to_mp_grid(points, prec=1e-5):
    """starting from a set of points - find the grid
    
    Returns
    -------
    tuple(int)
        3 numbers defining a grid
    """
    points = np.array(points)
    print(points)
    grid = tuple(points_to_mp_grid_1d(p, prec=prec) for p in points.T)
    grid_arr = np.array(grid)[None,:]
    check = np.zeros( grid, dtype=bool)
    points_int = np.round(points * grid_arr).astype(int) % grid_arr
    check[points_int]=True
    assert np.all(check), f"found grid {grid}, but not all points belong to it"
    return grid
