from numpy import sqrt as sq
import numpy as np
import sympy as sym
from functools import cached_property, lru_cache
from scipy.special import spherical_jn
from scipy.interpolate import CubicSpline
from scipy.integrate import trapezoid
from scipy.constants import physical_constants
bohr_radius_angstrom = physical_constants["Bohr radius"][0] * 1e10

# Note: in the Dwann it is assumed that all orbitals are REAL, so under TR one does not need to take complex conjugate
# in future, if complex orbitals are considered, this should be taken care of


orbitals_sets_dic = {
    's': ['s'],
    'p': ['pz', 'px', 'py'],
    'd': ['dz2', 'dxz', 'dyz', 'dx2-y2', 'dxy'],
    'f': ['fz3', 'fxz2', 'fyz2', 'fzx2-zy2', 'fxyz', 'fx3-3xy2', 'f3yx2-y3'],
    'sp': ['sp-1', 'sp-2'],
    'p2': ['pz', 'py'],
    'sp2': ['sp2-1', 'sp2-2', 'sp2-3'],
    'pz': ['pz'],
    'sp3': ['sp3-1', 'sp3-2', 'sp3-3', 'sp3-4'],
    'sp3d2': ['sp3d2-1', 'sp3d2-2', 'sp3d2-3', 'sp3d2-4', 'sp3d2-5', 'sp3d2-6'],
    'sp3d2_plus': ['sp3d2_plus-1', 'sp3d2_plus-2', 'sp3d2_plus-3', 'sp3d2_plus-4', 'sp3d2_plus-5', 'sp3d2_plus-6'],
    't2g': ['dxz', 'dyz', 'dxy'],
    'eg': ['dx2-y2', 'dz2'],
}

basis_shells_list = ['s', 'p', 'd', 'f']
hybrid_shells_list = ['sp', 'p2', 'sp2', 'pz', 'sp3', 'sp3d2', 't2g', 'eg']

basis_orbital_list = [k for o in basis_shells_list for k in orbitals_sets_dic[o]]


@lru_cache
def orb_to_shell(orb):
    for shell, orbs in orbitals_sets_dic.items():
        if orb in orbs:
            return shell
    raise ValueError(f"orbital {orb} is not in the orbitals_sets_dic")


@lru_cache
def num_orbitals(shell_symbol: str):
    return len(orbitals_sets_dic[shell_symbol])


hybrids_coef = {
    'sp-1': {"s": 1 / sq(2), "px": 1 / sq(2)},
    'sp-2': {"s": 1 / sq(2), "px": -1 / sq(2)},
    'sp2-1': {"s": 1 / sq(3), "px": -1 / sq(6), "py": 1 / sq(2)},
    'sp2-2': {"s": 1 / sq(3), "px": -1 / sq(6), "py": -1 / sq(2)},
    'sp2-3': {"s": 1 / sq(3), "px": 2 / sq(6)},
    'sp3-1': {"s": 1 / 2, "px": 1 / 2, "py": 1 / 2, "pz": 1 / 2},
    'sp3-2': {"s": 1 / 2, "px": 1 / 2, "py": -1 / 2, "pz": -1 / 2},
    'sp3-3': {"s": 1 / 2, "px": -1 / 2, "py": 1 / 2, "pz": -1 / 2},
    'sp3-4': {"s": 1 / 2, "px": -1 / 2, "py": -1 / 2, "pz": 1 / 2},
    'sp3d2-1': {"s": 1 / sq(6), "px": -1 / sq(2), "dz2": -1 / sq(12), "dx2-y2": 1 / 2},
    'sp3d2-2': {"s": 1 / sq(6), "px": 1 / sq(2), "dz2": -1 / sq(12), "dx2-y2": 1 / 2},
    'sp3d2-3': {"s": 1 / sq(6), "py": -1 / sq(2), "dz2": -1 / sq(12), "dx2-y2": -1 / 2},
    'sp3d2-4': {"s": 1 / sq(6), "py": 1 / sq(2), "dz2": -1 / sq(12), "dx2-y2": -1 / 2},
    'sp3d2-5': {"s": 1 / sq(6), "pz": -1 / sq(2), "dz2": 1 / sq(3)},
    'sp3d2-6': {"s": 1 / sq(6), "pz": 1 / sq(2), "dz2": 1 / sq(3)},
}
for k in basis_orbital_list:
    hybrids_coef[k] = {k: 1}







class Orbitals:

    def __init__(self):
        x = sym.Symbol('x')
        y = sym.Symbol('y')
        z = sym.Symbol('z')
        self.xyz = np.transpose([x, y, z])
        orbitals = {}
        orbitals['s'] = lambda x, y, z: 1 + 0 * x

        orbitals['px'] = lambda x, y, z: x
        orbitals['py'] = lambda x, y, z: y
        orbitals['pz'] = lambda x, y, z: z

        orbitals['dz2'] = lambda x, y, z: (2 * z * z - x * x - y * y) / (2 * sym.sqrt(3.0))
        orbitals['dxz'] = lambda x, y, z: x * z
        orbitals['dyz'] = lambda x, y, z: y * z
        orbitals['dx2-y2'] = lambda x, y, z: (x * x - y * y) / 2
        orbitals['dxy'] = lambda x, y, z: x * y

        orbitals['fz3'] = lambda x, y, z: z * (2 * z * z - 3 * x * x - 3 * y * y) / (2 * sym.sqrt(15.0))
        orbitals['fxz2'] = lambda x, y, z: x * (4 * z * z - x * x - y * y) / (2 * sym.sqrt(10.0))
        orbitals['fyz2'] = lambda x, y, z: y * (4 * z * z - x * x - y * y) / (2 * sym.sqrt(10.0))
        orbitals['fzx2-zy2'] = lambda x, y, z: z * (x * x - y * y) / 2
        orbitals['fxyz'] = lambda x, y, z: x * y * z
        orbitals['fx3-3xy2'] = lambda x, y, z: x * (x * x - 3 * y * y) / (2 * sym.sqrt(6.0))
        orbitals['f3yx2-y3'] = lambda x, y, z: y * (3 * x * x - y * y) / (2 * sym.sqrt(6.0))

        self.orb_function_dic = {key: [orbitals[k] for k in val] for key, val in orbitals_sets_dic.items() if key in basis_shells_list}
        self.orb_chara_dic = {
            's': [x],
            'p': [z, x, y],
            'd': [z * z, x * z, y * z, x * x, x * y, y * y],
            'f': [
                z * z * z, x * z * z, y * z * z, z * x * x, x * y * z, x * x * x, y * y * y, z * y * y, x * y * y,
                y * x * x
            ],
        }

        self.hybrid_matrix_dic = {}
        self.hybrid_matrix_shells_dic = {}
        self.hybrid_matrix_shells_start = {}

        for hshell in hybrid_shells_list:
            basis_shells_used = set()
            for orb in orbitals_sets_dic[hshell]:
                if orb in basis_orbital_list:
                    basis_shells_used.add(orb_to_shell(orb))
                elif orb in hybrids_coef:
                    for orb2 in hybrids_coef[orb]:
                        basis_shells_used.add(orb_to_shell(orb2))
                else:
                    raise ValueError(f"orbital {orb} is not in the basis_orbital_list or hybrids_coef")
            basis_shells_used = list(basis_shells_used)
            self.hybrid_matrix_shells_dic[hshell] = basis_shells_used
            basis_shells_start = [0]
            for shell in basis_shells_used:
                basis_shells_start.append(basis_shells_start[-1] + len(orbitals_sets_dic[shell]))
            # print (f"hybrid shell {hshell}, basis_shells = {basis_shells_used}, basis_shells_start = {basis_shells_start}")
            num_orb = len(orbitals_sets_dic[hshell])
            self.hybrid_matrix_shells_start[hshell] = basis_shells_start
            matrix = np.zeros((len(orbitals_sets_dic[hshell]), basis_shells_start[-1]))
            for i, horb in enumerate(orbitals_sets_dic[hshell]):
                for j, borb in enumerate(hybrids_coef[horb]):
                    bshell = orb_to_shell(borb)
                    shell_stat = basis_shells_start[basis_shells_used.index(bshell)]
                    k = shell_stat + orbitals_sets_dic[bshell].index(borb)
                    matrix[i, k] = hybrids_coef[horb][borb]
            self.hybrid_matrix_dic[hshell] = matrix
            # print (f"matrix = \n{matrix}")
            check = np.zeros((num_orb, num_orb))
            for s, e in zip(basis_shells_start[:-1], basis_shells_start[1:]):
                mat_shell = matrix[:, s:e]
                # print (f"check.shape = {check.shape}, mat_shell.shape = {mat_shell.shape}")
                check += mat_shell @ mat_shell.T
            assert np.allclose(check, np.eye(num_orb)), f"check failed for {hshell} : {check}"





    def rot_orb_basis(self, orb_symbol, rot_glb):
        ''' Get rotation matrix of orbitals in each orbital quantum number '''
        orb_dim = num_orbitals(orb_symbol)
        orb_rot_mat = np.zeros((orb_dim, orb_dim), dtype=float)
        xp, yp, zp = np.dot(np.linalg.inv(rot_glb), self.xyz)
        OC = self.orb_chara_dic[orb_symbol]
        OC_len = len(OC)
        for i in range(orb_dim):
            subs = []
            equation = (self.orb_function_dic[orb_symbol][i](xp, yp, zp)).expand()
            for j in range(OC_len):
                eq_tmp = equation.subs(OC[j], 1)
                for j_add in range(1, OC_len):
                    eq_tmp = eq_tmp.subs(OC[(j + j_add) % OC_len], 0)
                subs.append(eq_tmp)
            if orb_symbol in ['s', 'pz']:
                orb_rot_mat[0, 0] = subs[0].evalf()
            elif orb_symbol == 'p':
                orb_rot_mat[0, i] = subs[0].evalf()
                orb_rot_mat[1, i] = subs[1].evalf()
                orb_rot_mat[2, i] = subs[2].evalf()
            elif orb_symbol == 'd':
                orb_rot_mat[0, i] = (2 * subs[0] - subs[3] - subs[5]) / sym.sqrt(3.0)
                orb_rot_mat[1, i] = subs[1].evalf()
                orb_rot_mat[2, i] = subs[2].evalf()
                orb_rot_mat[3, i] = (subs[3] - subs[5]).evalf()
                orb_rot_mat[4, i] = subs[4].evalf()
            elif orb_symbol == 'f':
                orb_rot_mat[0, i] = (subs[0] * sym.sqrt(15.0)).evalf()
                orb_rot_mat[1, i] = (subs[1] * sym.sqrt(10.0) / 2).evalf()
                orb_rot_mat[2, i] = (subs[2] * sym.sqrt(10.0) / 2).evalf()
                orb_rot_mat[3, i] = (2 * subs[3] + 3 * subs[0]).evalf()
                orb_rot_mat[4, i] = subs[4].evalf()
                orb_rot_mat[5, i] = ((2 * subs[5] + subs[1] / 2) * sym.sqrt(6.0)).evalf()
                orb_rot_mat[6, i] = ((-2 * subs[6] - subs[2] / 2) * sym.sqrt(6.0)).evalf()

        return orb_rot_mat


    def rot_orb(self, orb_symbol, rot_glb):
        if orb_symbol in ["s", "p", "d", "f"]:
            return self.rot_orb_basis(orb_symbol, rot_glb)
        elif orb_symbol in hybrid_shells_list:
            nbasis = self.hybrid_matrix_dic[orb_symbol].shape[1]
            rot_orb_loc = np.zeros((nbasis, nbasis))
            matrix_hybrid = self.hybrid_matrix_dic[orb_symbol]
            for s, e, shell  in zip(self.hybrid_matrix_shells_start[orb_symbol], self.hybrid_matrix_shells_start[orb_symbol][1:], self.hybrid_matrix_shells_dic[orb_symbol]):
                rot_orb_loc[s:e, s:e] = self.rot_orb_basis(shell, rot_glb)
            return matrix_hybrid @ rot_orb_loc @ matrix_hybrid.T


@lru_cache
def get_orbitals():
    return Orbitals()


class OrbitalRotator:

    def __init__(self, rotations_cart):
        self.rotations_cart = np.copy(rotations_cart)
        if len(self.rotations_cart.shape) == 2:
            self.rotations_cart = np.array([rotations_cart])
        self.results_dict = {}

    @property
    def orbitals(self):
        return get_orbitals()

    def __call__(self, orb_symbol, isym=0):
        if (isym, orb_symbol) not in self.results_dict:
            rot_glb = self.rotations_cart[isym]
            self.results_dict[(isym, orb_symbol)] = self.orbitals.rot_orb(orb_symbol, rot_glb)
        return self.results_dict[(isym, orb_symbol)]


class Projector:
    """
    a class to calculate the projection of the wavefunctions on the plane vectors
    """

    def __init__(self, gk, bessel, a0=bohr_radius_angstrom):
        self.gk = gk
        self.projectors = {}
        gk_abs = np.linalg.norm(gk, axis=1)
        # print("gk_abs", gk_abs)
        self.gka_abs = gk_abs * a0
        g_costheta = gk[:, 2] / gk_abs
        g_costheta[gk_abs < 1e-8] = 0
        g_phi = np.arctan2(gk[:, 1], gk[:, 0])
        # print("phi", g_phi)
        # print("costheta", g_costheta)
        self.sph = SphericalHarmonics(costheta=g_costheta, phi=g_phi)
        self.bessel = bessel
        self.bessel_l = {}
        self.coef = 4 * np.sqrt(np.pi / a0)

    def get_bessel_l(self, l):
        if l not in self.bessel_l:
            self.bessel_l[l] = self.bessel(l, self.gka_abs) * self.coef * (-1j)**l
        return self.bessel_l[l]

    def __call__(self, orbital):
        if orbital not in self.projectors:
            self.projectors[orbital] = self._projector(orbital)
        return self.projectors[orbital]

    def _projector(self, orbital):
        if orbital in hybrids_coef and orbital not in basis_orbital_list:
            return sum(self(orb) * coef for orb, coef in hybrids_coef[orbital].items())
        else:
            l = {'s': 0, 'p': 1, 'd': 2, 'f': 3}[orbital[0]]
            bessel_j_exp_int = self.get_bessel_l(l)
            spherical = self.sph(orbital)
            return bessel_j_exp_int * spherical



class Bessel_j_exp_int:
    r"""
    a class to evaluate the integral

    :math:`\int_0^{\infty} j_l(k*x) e^{-x} dx`
    """

    def __init__(self,
                 k0=5, kmax=100, dk=0.01, dtk=0.2, kmin=1e-3,
                 x0=5, xmax=100, dx=0.01, dtx=0.2,
                 ):
        self.splines = {}
        self.kmax = kmax
        self.kgrid = self._get_grid(k0, kmax, dk, dtk)
        self.xgrid = self._get_grid(x0, xmax, dx, dtx)
        # print(f"the xgrid has {len(self.xgrid)} points")
        # print(f"the kgrid has {len(self.kgrid)} points")
        self.kmin = kmin

    def _get_grid(self, x0, xmax, dx, dt):
        xgrid = list(np.arange(0, x0, dx))
        t = dt
        x0 = xgrid[-1]
        while xgrid[-1] < xmax:
            xgrid.append(x0 * np.exp(t))
            t += dt
        return np.array(xgrid)

    def set_spline(self, l):
        if l not in self.splines:
            self.splines[l] = self.get_spline(l)
        return self.splines[l]

    def get_spline(self, l):
        e = np.exp(-self.xgrid)
        fourier = []
        for k in self.kgrid:
            if k < self.kmin and l > 0:
                fourier.append(0)
            else:
                j = spherical_jn(l, k * self.xgrid)
                fourier.append(trapezoid(y=j * e, x=self.xgrid))
        return CubicSpline(self.kgrid, fourier)


    def __call__(self, l, k):
        self.set_spline(l)
        res = np.zeros(len(k))
        select = (k <= self.kmax)
        res[select] = self.splines[l](k[select])
        return res


class SphericalHarmonics:

    def __init__(self, costheta, phi):
        self.costheta = costheta
        self.phi = phi
        self.harmonics = {}
        self.sqpi = 1 / np.sqrt(np.pi)

    @cached_property
    def sintheta(self):
        return np.sqrt(1 - self.costheta**2)

    @cached_property
    def cosphi(self):
        return np.cos(self.phi)

    @cached_property
    def sinphi(self):
        return np.sin(self.phi)

    @cached_property
    def sin2phi(self):
        return 2 * self.sinphi * self.cosphi

    @cached_property
    def cos2phi(self):
        return 2 * self.cosphi**2 - 1

    @cached_property
    def cos2theta(self):
        return 2 * self.costheta**2 - 1

    @cached_property
    def sin2theta(self):
        return 2 * self.costheta * self.sintheta

    def __call__(self, orbital):
        if orbital not in self.harmonics:
            self.harmonics[orbital] = self._harmonics(orbital)
        return self.harmonics[orbital]

    def _harmonics(self, orbital):
        """from here https://en.wikipedia.org/wiki/Table_of_spherical_harmonics#Real_spherical_harmonics

        hybrids - according to Wannier90 manual"""
        from numpy import sqrt as sq
        from numpy import pi
        if orbital not in basis_orbital_list:
            assert orbital in hybrids_coef, f"orbital {orbital} not in basis_orbital_list or hybrids_coef"
            return sum(self(orb) * coef for orb, coef in hybrids_coef[orbital].items())
        else:
            match orbital:
                case 's':
                    return 1 / (2 * sq(pi)) * np.ones_like(self.costheta)
                case 'pz':
                    return sq(3 / (4 * pi)) * self.costheta
                case 'px':
                    return sq(3 / (4 * pi)) * self.sintheta * self.cosphi
                case 'py':
                    return sq(3 / (4 * pi)) * self.sintheta * self.sinphi
                case 'dz2':
                    return sq(5 / (16 * pi)) * (3 * self.costheta**2 - 1)
                case 'dx2-y2':
                    return sq(15 / (16 * pi)) * self.sintheta**2 * self.cos2phi
                case 'dxy':
                    return sq(15 / (16 * pi)) * self.sintheta**2 * self.sin2phi
                case 'dxz':
                    return sq(15 / (16 * pi)) * self.sin2theta * self.cosphi
                case 'dyz':
                    return sq(15 / (16 * pi)) * self.sin2theta * self.sinphi
                case _:
                    raise ValueError(f"orbital {orbital} not implemented")
