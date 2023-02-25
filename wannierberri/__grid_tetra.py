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

from collections.abc import Iterable
import numpy as np
from time import time
from . import symmetry
import lazy_property
from .__Kpoint_tetra import KpointBZtetra
from .__grid import GridAbstract
#from .__finite_differences import FiniteDifferences


class GridTetra(GridAbstract):
    """ A class containing information about the k-grid.konsisting of tetrahedra

    Parameters
    -----------
    system : :class:`~wannierberri.system.System`
        which the calculations will be made
    length :  float
        (angstroms) -- in this case the grid is NK[i]=length*||B[i]||/2pi  B- reciprocal lattice
    length_FFT :  float
        (angstroms) -- in this case the FFT grid is NKFFT[i]=length_FFT*||B[i]||/2pi  B- reciprocal lattice
    NKFFT : int
        number of k-points in the FFT grid along each directions
    IBZ_tetra : list
        list of tetrahedra describing the irreducible wedge of the Brillouin zone. By default, the stace is just divided into 5 tetrahedra
    Notes
    -----
     `NKFFT`  may be given as size-3 integer arrays or lists. Also may be just numbers -- in that case the number of kppoints is the same in all directions

    either lewngth_FFT of NKFFT should be provided

    """

    def __init__(self, system, length, NKFFT=None, NK=None, IBZ_tetra = None, 
            refine_by_volume=True,
            refine_by_size=True,
            length_size = None
                ):

        if NKFFT is None:
            self.FFT = system.NKFFT_recommended
        elif isinstance(NKFFT,int):
            self.FFT = np.array([NKFFT]*3)
        else:
            self.FFT = np.array(NKFFT)

        self.recip_lattice_reduced = system.recip_lattice/self.FFT[:,None]
        print ("reduced reciprocal lattice : \n",self.recip_lattice_reduced)
        if IBZ_tetra is None:   # divide the full reciprocal unit cell into 5 tetrahedra - 
            print ("WARNING : irreducible wedge not provided, no use of symmetries")
            tetrahedra = np.array([  [ [0,0,0],[1,0,0],[0,1,0],[0,0,1] ],
                                     [ [1,0,1],[0,0,1],[1,0,0],[1,1,1] ],
                                     [ [1,1,0],[1,0,0],[0,1,0],[1,1,1] ],
                                     [ [0,1,1],[0,0,1],[0,1,0],[1,1,1] ],
                                     [ [0,0,1],[0,1,0],[1,0,0],[1,1,1] ]
                                   ]) - np.array([0.5,0.5,0.5])[None,None,:]
        else :
            tetrahedra = np.array(IBZ_tetra)
        print ("using starting tetrahedra with vertices \n", tetrahedra)
        weights = np.array([tetra_volume(t) for t in tetrahedra])
        print (f"volumes of tetrahedra are {weights}, total = {sum(weights)} (further normalized)")
        weights/=sum(weights)
        self.K_list=[]
        print ("generating starting K_list")
        for tetr,w in zip(tetrahedra,weights):
            K = KpointBZtetra(vertices=tetr, K=0, NKFFT=self.FFT, factor=w, basis=self.recip_lattice_reduced, refinement_level=0, split_level=0)
            print (K)
            print (K.size)
            self.K_list.append(K)

        if refine_by_volume:
            dkmax = 2*np.pi/length
            vmax = dkmax**3/np.linalg.det(self.recip_lattice_reduced)
            self.split_tetra_volume(vmax)
            print ("refinement by volume done")
        if refine_by_size:
            if length_size is None:
                length_size = 0.5*length
            dkmax = (2*np.pi/length_size)*np.sqrt(2)
            self.split_tetra_size(dkmax)
            print ("refinement by size done")


    def split_tetra_size(self,dkmax):
        """split tetrahedra that have at lkeast one edge larger than dkmax"""
        while True:
            print (f"maximal tetrahedron size for now is {self.size_max} ({len(self.K_list)}), we need to refine down to size {dkmax}")
            volumes = [tetra_volume(K.vertices) for K in self.K_list]
            print ("the volume is ",sum(volumes),min(volumes),max(volumes),np.mean(volumes))
#            print ("sizes now are ",self.sizes)
            if self.size_max < dkmax:
                break
            klist = []
            for K in self.K_list:
                if K.size > dkmax:
#                        klist+=K.divide(ndiv=int(K.size/dkmax+1), refine=False)
                    klist+=K.divide(ndiv=2, refine=False)
                else:
                    klist.append(K)
            self.K_list = klist


    def split_tetra_volume(self,vmax):
        """split tetrahedra that have at least one edge larger than dkmax"""
        while True:
            volumes = [tetra_volume(K.vertices) for K in self.K_list]
            print (f"maximal tetrahedron size for now is {max(volumes)} ({len(self.K_list)}), we need to refine down to size {vmax}")
            print ("the volume is ",sum(volumes),min(volumes),max(volumes),np.mean(volumes))
            if max(volumes) < vmax:
                break
            klist = []
            for K,v in zip(self.K_list,volumes):
                if v > vmax:
#                        klist+=K.divide(ndiv=int(K.size/dkmax+1), refine=False)
                    klist+=K.divide(ndiv=2, refine=False)
                else:
                    klist.append(K)
            self.K_list = klist

    @property
    def size_max(self): 
        return self.sizes.max()

    @property
    def sizes(self): 
        return np.array([K.size for K in self.K_list])

    @property
    def str_short(self):
        return "GridTetra() with {} tetrahedrons, NKFFT={}, NKtot={}".format(len(self.K_list), self.FFT, np.prod(self.FFT)*len(self.K_list))

    @property
    def dense(self):
        raise NotImplementedError()

    @lazy_property.LazyProperty
    def points_FFT(self):
        dkx, dky, dkz = 1. / self.FFT
        return np.array(
            [
                np.array([ix * dkx, iy * dky, iz * dkz]) for ix in range(self.FFT[0]) for iy in range(self.FFT[1])
                for iz in range(self.FFT[2])
            ])


    def get_K_list(self, use_symmetry=True):
        """ returns the list of Symmetry-irreducible K-points"""
        return [K.copy() for K in self.K_list]


        dK = 1. / self.div
        factor = 1. / np.prod(self.div)
        print("generating K_list")
        t0 = time()
        K_list = [
            [
                [
                    KpointBZ(
                        K=np.array([x, y, z]) * dK,
                        dK=dK,
                        NKFFT=self.FFT,
                        factor=factor,
                        symgroup=self.symgroup,
                        refinement_level=0) for z in range(self.div[2])
                ] for y in range(self.div[1])
            ] for x in range(self.div[0])
        ]
        print("Done in {} s ".format(time() - t0))
        if use_symmetry:
            t0 = time()
            print("excluding symmetry-equivalent K-points from initial grid")
            for z in range(self.div[2]):
                for y in range(self.div[1]):
                    for x in range(self.div[0]):
                        KP = K_list[x][y][z]
                        if KP is not None:
                            star = KP.star
                            star = [tuple(k) for k in np.array(np.round(KP.star * self.div), dtype=int) % self.div]
                            for k in star:
                                if k != (x, y, z):
                                    KP.absorb(K_list[k[0]][k[1]][k[2]])
                                    K_list[k[0]][k[1]][k[2]] = None
            print("Done in {} s ".format(time() - t0))

        K_list = [K for Kyz in K_list for Kz in Kyz for K in Kz if K is not None]
        print("Done in {} s ".format(time() - t0))
        print(
            "K_list contains {} Irreducible points({}%) out of initial {}x{}x{}={} grid".format(
                len(K_list), round(len(K_list) / np.prod(self.div) * 100, 2), self.div[0], self.div[1], self.div[2],
                np.prod(self.div)))
        return K_list


def tetra_volume( vortices ):
    return abs(np.linalg.det( vortices[1:]-vortices[0][None,:]))/6.


