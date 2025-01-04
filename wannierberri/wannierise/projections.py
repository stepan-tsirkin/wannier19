import copy
from functools import cached_property
import itertools
import numpy as np
from ..symmetry.orbitals import orbitals_sets_dic

try:
    from jax import config
    config.update("jax_enable_x64", True)
    from jax import numpy as jnp
    from jax.scipy.optimize import minimize as jminimize
    from jax import jit as jjit
except ImportError:
    # warnings.warn("jax not found, will use numpy insrtead")
    import numpy as jnp
    from scipy.optimize import minimize as jminimize
    from functools import partial as jjit


from .utility import find_distance_periodic
from wannierberri.__utility import UniqueListMod1
from ..symmetry.orbitals import Orbitals, num_orbitals
from .wyckoff_position import WyckoffPosition, WyckoffPositionNumeric

ORBITALS = Orbitals()


class Projection:
    """	
    A class to store initial projections. 

    Parameters
    ----------
    position_num : np.array(shape=(n,3,), dtype=float) or str
        The position of the projection if fractional coordinates. 
    position_str : str
        comma-separated positions with x,y,z being the free variables, 
        e.g.  "x,y,z", "x,x-y,1/2", etc.
    orbital : str
        The orbital of the projection.  e.g. "s", "p", "sp3 etc. or several separated by a semicolon (e.g. "s;p")	
    spacegroup : irrep.spacegroup.SpaceGroup
        The spacegroup of the structure. All points equivalent to the given ones are also added 
        (not needed if wyckoff_position is provided)
    void : bool
        if true, create an empty object, to be filled later
    wyckoff_position : WyckoffPosition or WyckoffPositionNumeric
        The wyckoff position of the projection. If provided, the position_num and position_str nd spacegroup are ignored
    """

    def __init__(self,
                 position_sym=None,
                 position_num=None,
                 spacegroup=None,
                 wyckoff_position=None,
                 orbital='s',
                 void=False,
                 free_var_values=None,
                 spinor=False):
        if void:
            return
        self.orbitals = orbital.split(";")

        if wyckoff_position is not None:
            self.wyckoff_position = wyckoff_position
        else:
            assert spacegroup is not None, "either wyckoff_position or spacegroup should be provided"
            if position_num is None:
                assert position_sym is not None, "either position_num or position_str should be provided"
                self.wyckoff_position = WyckoffPosition(position_str=position_sym,
                                                    spacegroup=spacegroup,
                                                    free_var_values=free_var_values)
            else:
                assert position_sym is None, "position_num and position_str should NOT be provided together"
                position_num = np.array(position_num)
                if position_num.ndim == 1:
                    position_num = position_num[None, :]
                self.wyckoff_position = WyckoffPositionNumeric(positions=position_num,
                                                    spacegroup=spacegroup)
            spinor = spacegroup.spinor
        self.spinor = spinor

    @property
    def positions(self):
        return self.wyckoff_position.positions

    @property
    def num_wann_per_site(self):
        return sum(num_orbitals(o) for o in self.orbitals) * (2 if self.spinor else 1)

    @property
    def num_points(self):
        return self.wyckoff_position.num_points

    @property
    def num_wann(self):
        return self.num_points * self.num_wann_per_site 

    @property
    def orbitals_str(self):
        return ";".join(self.orbitals)

    def split(self):
        """
        assuming that Projections may contain several orbitals, this function splits them into separate projections
        if there is only one - a list with one element is returned
        """
        return [Projection(wyckoff_position=self.wyckoff_position, orbital=o) for o in self.orbitals]

    def copy(self):
        new = Projection(void=True)
        new.orbitals = self.orbitals
        new.wyckoff_position = self.wyckoff_position
        new.spinor = self.spinor
        return new

    def __add__(self, other):
        new = self.copy()
        if other is not None:
            assert self.wyckoff_position == other.wyckoff_position, f"Cannot add projections from different wyckoff positions {self.wyckoff_position} and {other.wyckoff_position}"
            new.orbitals += other.orbitals
        return new

    def __radd__(self, other):
        return self.__add__(other)

    def __str__(self):
        return (f"Projection {self.wyckoff_position.string}:{self.orbitals} with {self.num_wann} Wannier functions"
                f" on {self.num_points} points ({self.num_wann_per_site} per site)"
               # + self.wyckoff_position.__str__()
        )

    def write_wannier90(self, mod1=False):
        string = ""
        for o in self.orbitals:
            for pos in self.wyckoff_position.positions:
                if mod1:
                    pos = pos % 1
                string += f"f={pos[0]:.12f}, {pos[1]:.12f}, {pos[2]:.12f}: {o}\n"
        return string

    @cached_property
    def str_short(self):
        return f"{self.wyckoff_position.string}:{self.orbitals}"

    def get_positions_and_orbitals(self):
        """
        Returns
        -------
        list(np.ndarray(shape=(3,), dtype=float))
            The positions of the projections
        list(str)
            The orbitals of the projections (each orbital , e.g. pz, sp3-2, dx2-y2, etc.)
        """
        orbitals = []
        positions = []
        for pos in self.positions:
            for orb in self.orbitals:
                for o in orbitals_sets_dic[orb]:
                    orbitals.append(o)
                    positions.append(pos)
        return positions, orbitals


class ProjectionsSet:

    """
    class to store the set of projections and corresponding windows
    """

    def __init__(self,
                 projections=[]):
        self.spinor = None
        for i, p in enumerate(projections):
            assert isinstance(p, Projection), f"element {i} of list 'projections' should be a Projection, not {p}"
            self.set_spinor(p.spinor)
        self.projections = copy.copy(projections)

    def copy(self):
        return ProjectionsSet(projections=[p.copy() for p in self.projections])

    def set_spinor(self, spinor: bool):
        self.spinor = spinor
        if self.spinor is None:
            self.spinor = spinor
        else:
            assert self.spinor == spinor, f"spinor should be the same for all projections. Previously set to {self.spinor}, now trying to set to {spinor}"

    @property
    def num_proj(self):
        return len(self.projections)

    def __len__(self):
        return self.num_proj

    @cached_property
    def num_points(self):
        print(f"finding num points from {self.num_proj} projections")
        return sum([p.wyckoff_position.num_points for p in self.projections])

    @cached_property
    def num_wann(self):
        return sum([p.num_wann for p in self.projections])

    def add(self, projection):
        self.projections.append(projection)
        self.set_spinor(projection.spinor)

    def __add__(self, other):
        new = ProjectionsSet(projections=self.projections + other.projections)
        new.clear_cached_properties()
        return new

    def __str__(self):
        return (f"ProjectionsSet with {self.num_wann} Wannier functions and {self.num_free_vars} free variables\n" +
                "\n".join([str(p) for p in self.projections])
                )

    @cached_property
    def num_free_vars_wyckoff(self):
        return sum([p.wyckoff_position.num_free_vars for p in self.projections])

    def as_numeric(self):
        new = self.copy()
        for p in new.projections:
            p.wyckoff_position = p.wyckoff_position.as_numeric()
        return new

    def split_orbitals(self):
        return ProjectionsSet(sum((p.split() for p in self.projections), []))

    @property
    def map_free_vars(self):
        if not hasattr(self, "map_free_vars_cached"):
            # print ("mapping free vars"	)
            maps = [p.wyckoff_position.map_orbit_on_free_vars for p in self.projections]
            rotations = np.zeros((self.num_points, 3, self.num_free_vars_wyckoff), dtype=float)
            translations = np.zeros((self.num_points, 3), dtype=float)
            rot = [m[0] for m in maps]
            trans = [m[1] for m in maps]
            self._vars_end = np.cumsum([m[0].shape[-1] for m in maps])
            self._vars_start = np.concatenate(([0], self._vars_end[:-1]))
            self._pos_end = np.cumsum([m[0].shape[0] for m in maps])
            self._pos_start = np.concatenate(([0], self._pos_end[:-1]))
            for r, t, vs, ve, ps, pe in zip(rot, trans, self._vars_start, self._vars_end, self._pos_start, self._pos_end):
                rotations[ps:pe, :, vs:ve] = r
                translations[ps:pe] = t
            self.map_free_vars_cached = rotations, translations
        return self.map_free_vars_cached

    @property
    def num_free_vars(self):
        return self.map_free_vars[0].shape[2]


    @cached_property
    def num_wann_per_site(self):
        """	
        Returns:
        --------
        np.array(int, shape=(num_points))
            The number of Wannier functions per site
        """
        return np.array(sum(([p.num_wann_per_site] * p.num_points for p in self.projections), []))

    @property
    def vars_end(self):
        self.map_free_vars
        return self._vars_end

    @property
    def vars_start(self):
        self.map_free_vars
        return self._vars_start

    @property
    def pos_end(self):
        self.map_free_vars
        return self._pos_end

    @property
    def pos_start(self):
        self.map_free_vars
        return self._pos_start

    def get_positions_from_free_vars(self, free_vars):
        rotations, translations = self.map_free_vars
        return rotations @ free_vars + translations

    def get_positions(self):
        return self.get_positions_from_free_vars(self.free_var_values)

    def get_distances(self):
        pos = self.get_positions()
        return find_distance_periodic(pos, self.projections[0].wyckoff_position.spacegroup.Lattice, max_shift=2)

    def get_min_distance(self):
        return np.min([l[i:].min() for i, l in enumerate(self.get_distances())])


    def join_same_wyckoff(self, unmergable=[], use_unmergable_defaults=True):
        """
        merge different projections on the same wyckoff positions 
        """
        if use_unmergable_defaults:
            unmergable_loc = [('s', 'sp3'), ('p', 'sp3')]  # TODO : add more
        else:
            unmergable_loc = []
        unmergable_loc += unmergable

        stick = []
        istick = []
        for i, p in enumerate(self.projections):
            found = False
            for st, ist in zip(stick, istick):
                if st[0].wyckoff_position == p.wyckoff_position:
                    for p2 in st:
                        proj1 = set(p.orbitals)
                        proj2 = set(p2.orbitals)
                        # check if they may be merged
                        print(f"checking if they may be merged {proj1} and {proj2}")
                        merge = True
                        for p1, p2 in itertools.product(proj1, proj2):
                            print(p1, p2)
                            if (p1 == p2 or
                                    (p1, p2) in unmergable_loc or
                                    (p2, p1) in unmergable_loc
                                    ):
                                print("not merging")
                                merge = False
                                break
                        else:
                            print("merging")
                        if not merge:
                            break
                    else:
                        st.append(p)
                        ist.append(i)
                        found = True
                    break
            if not found:
                stick.append([p])
                istick.append([i])
        num_free_vars_new_per_group = [self.vars_end[ist[0]] - self.vars_start[ist[0]] for ist in istick]
        srt = np.argsort(num_free_vars_new_per_group)
        stick = [stick[i] for i in srt]
        istick = [istick[i] for i in srt]
        new_projections = []
        for st in stick:
            projection = sum(st, None)
            new_projections.append(projection)
        self.projections = new_projections
        self.clear_cached_properties()


    def clear_cached_properties(self, attributes=None):
        """
        Clear the cached properties

        Parameters:
        -----------
        attributes: list(str)
            The list of attributes to clear. If None, all cached properties are cleared
        """
        if attributes is None:
            attributes = ["map_free_vars_cached", "_free_vars", "num_wann_per_site",
                     "num_points", "num_wann", "num_free_vars_wyckoff",]
        for attr in attributes:
            if hasattr(self, attr):
                delattr(self, attr)

    def stick_to_atoms(self, atoms=[]):
        """
        NOT SURE IF THIS IS NEEDED NEITHER IF IT WORKS PROPERLY

        Reduces the number of free variables by sticking together the ones that correspond to the same wyckoff position but different projections

        resets the free_vars and clears the cached properties
        and sets map_free_vars to the new map

        Parameters
        ----------
        atoms : np.ndarray(shape=(num_atoms,3), dtype=float)
            List of atomic positions
        """

        fixed = []
        atoms_filled = np.zeros(len(atoms), dtype=bool)
        num_free_vars_new = 0
        for p in self.projections:
            fixed.append(p.wyckoff_position.stick_to_atoms(atoms=atoms, atoms_filled=atoms_filled))
            if fixed[-1] is None:
                num_free_vars_new += p.wyckoff_position.num_free_vars
        new_map = np.zeros((self.num_free_vars, num_free_vars_new), dtype=int)
        new_map_fix = np.zeros(self.num_free_vars, dtype=float)
        start = 0
        for i, p in enumerate(self.projections):
            if fixed[i] is None:
                nvar_loc = self.vars_end[i] - self.vars_start[i]
                end = start + nvar_loc
                new_map[self.vars_start[i]:self.vars_end[i], start:end] = np.eye(nvar_loc, dtype=int)
                start = end
            else:
                new_map_fix[self.vars_start[i]:self.vars_end[i]] = fixed[i]
        rot, trans = self.map_free_vars
        self.clear_cached_properties(["_free_vars"])
        self.map_free_vars_cached = rot @ new_map, rot @ new_map_fix + trans
        # print(f"updated rot,trans  {self.map_free_vars[0].shape}, {self.map_free_vars[1].shape}")
        # print(f"updated rot,trans  {self.map_free_vars_cached[0].shape}, {self.map_free_vars_cached[1].shape}")


    def write_wannier90(self, mod1=False, beginend=True, numwann=True):
        """
        return a string of wannier90 input file
        for projections

        Parameters
        ----------
        mod1 : bool
            If True, the positions are printed modulo 1

        Returns
        -------
        str
            The string for the wannier90 input file
        """
        positions = self.get_positions()
        if mod1:
            positions = positions % 1
        string = ""
        if numwann:
            string += f"num_wann = {self.num_wann}\n"
        if beginend:
            string += "begin projections\n"
        for p in self.projections:
            string += p.write_wannier90(mod1=mod1)
        if beginend:
            string += "end projections\n"
        return string

    def write_with_multiplicities(self, multiplicities=None, orbit=False):
        """
        return a string describing which projections are taken and how many times(if not zero)

        Parameters
        ----------
        multiplicity : np.ndarray(shape=(num_projections), dtype=int)
            The multiplicity of each projection
        orbit : bool
            If True, the orbit of the wyckoff position is also printed

        """
        if multiplicities is None:
            multiplicities = np.ones(self.num_proj, dtype=int)
        assert len(multiplicities) == self.num_proj
        breakline = "-" * 80 + "\n"

        string = breakline
        num_wann = 0
        for m, p in zip(multiplicities, self.projections):
            assert m >= 0, f"multiplicity {m} should be non-negative"
            if m > 0:
                string += f"{m}  X  | {p.str_short}  \n"
                num_wann += m * p.num_wann
                if orbit:
                    string += p.wyckoff_position.orbit_str() + "\n"
        string += f"total number of Wannier functions = {num_wann}\n"
        string += breakline
        return string

    def get_combination(self, multiplicities, dcopy=True):
        """
        get the combination of projections

        Parameters
        ----------
        multiplicities : np.ndarray(shape=(num_projections), dtype=int)
            The multiplicity of each projection

        Returns
        -------
        ProjectionsSet
            The projections set with the given multiplicities
        """
        assert len(multiplicities) == self.num_proj
        new_projections = []
        for m, p in zip(multiplicities, self.projections):
            assert m >= 0, f"multiplicity {m} should be non-negative"
            for _ in range(m):
                new_projections.append(p)
        if dcopy:
            new_projections = [copy.deepcopy(p) for p in new_projections]
        return ProjectionsSet(projections=new_projections)


    def maximize_distance(self, r0=1):
        rot, trans = self.map_free_vars
        num_free_vars = self.num_free_vars
        real_lattice = self.projections[0].wyckoff_position.spacegroup.Lattice
        same_site = np.eye(self.num_points, dtype=bool)
        # not_same_site = np.logical_not(same_site)
        repulsive_potential = RepulsivePotential(rotation=rot, translation=trans,
                                                 weights=self.num_wann_per_site,
                                                 same_site=same_site,
                                                 r0=r0, real_lattice=real_lattice)
        jit_potential = jjit(repulsive_potential.potential_jax)

        if num_free_vars > 0:
            free_var_values = self.free_var_values
            print(f"starting minimization with free vars {free_var_values} ")
            print(f"starting potential {jit_potential(free_var_values)}")
            print(f"minimal distance {self.get_min_distance()}")
            res = jminimize(jit_potential, free_var_values, method='BFGS')
            v = res.x
            print(f"minimized free vars {v}")
            print(f"minimized potential {jit_potential(v)}")
        else:
            v = jnp.zeros(0)
        pot = jit_potential(v)
        self.free_var_values = v
        self.potential = pot
        print(f"minimal distance {self.get_min_distance()}")
        print(f"positions\n {self.get_positions().round(4)}")
        print(f"distances\n {self.get_distances().round(2)}")

    @property
    def free_var_values(self):
        return np.hstack([proj.wyckoff_position.free_var_values for proj in self.projections])

    @free_var_values.setter
    def free_var_values(self, value):
        start = 0
        for proj in self.projections:
            end = start + proj.wyckoff_position.num_free_vars
            proj.wyckoff_position.free_var_values = value[start:end]
            start = end






class RepulsivePotential:

    def __init__(self, rotation, translation,
                 weights=None, same_site=None,
                 r0=1, real_lattice=jnp.eye(3), max_G_r0=5):
        assert rotation.ndim == 3, f"rotation.ndim = {rotation.ndim}, should be 3"
        assert translation.ndim == 2, f"translation.ndim = {translation.ndim}, should be 2"
        assert rotation.shape[0] == translation.shape[0], f"rotation.shape = {rotation.shape}, translation.shape = {translation.shape}"
        # print ("rotation",repr(rotation))
        # print ("translation",repr(translation))
        if weights is None:
            weights = np.ones(rotation.shape[0])
        else:
            weights = np.array(weights)
            assert weights.ndim == 1
            assert weights.shape[0] == rotation.shape[0], f"weights.shape = {weights.shape}, rotation.shape = {rotation.shape}, weights = {weights}"
        self.weights = weights[:, None] * weights[None, :]
        if same_site is None:
            same_site = np.eye(self.weights.shape[0], dtype=bool)
        else:
            assert same_site.shape == self.weights.shape
        self.weights[same_site] = 0
        self.rotation = jnp.array(rotation)
        self.translation = jnp.array(translation)
        self.num_free_vars = self.rotation.shape[2]
        # free_vars_random = jnp.random.rand(self.num_free_vars)
        num_pos = self.rotation.shape[0]
        assert self.rotation.shape[0] == self.translation.shape[0]
        real_lattice = np.array(real_lattice) / abs(np.linalg.det(real_lattice))**(1 / 3)
        reciprocal_lattice = np.linalg.inv(real_lattice).T
        # print (f"reciprocal_lattice = {reciprocal_lattice}, {np.linalg.det(reciprocal_lattice)}")
        r0 = r0 / num_pos**(1 / 3)
        maxG = 10
        max_mod_G = max_G_r0 / r0
        G = np.array([[i, j, k]
                      for i in range(-maxG, maxG + 1)
                      for j in range(-maxG, maxG + 1)
                      for k in range(-maxG, maxG + 1)])
        g = np.linalg.norm(G @ reciprocal_lattice, axis=1)
        select = g <= max_mod_G
        self.G = jnp.array(G[select])
        g = g[select]
        self.Ug = jnp.exp(-g**2 * r0**2 / 2.0)

    def potential_jax(self, free_vars):
        V = (self.rotation @ free_vars + self.translation) % 1
        diff = (V[None, :] - V[:, None])
        return jnp.sum((jnp.cos(2 * np.pi * jnp.dot(diff, self.G.T)) @ self.Ug) * self.weights)


def get_orbit(spacegroup, p, tol=1e-5):
    """
    Get the orbit of a point p under the symmetry operations of a structure.

    Parameters
    ----------
    spacegroup : irrep.spacegroup.SpaceGroup
        The spacegroup of the structure.
    p : np.ndarray(shape=(3,), dtype=float)
        Point for which to calculate the orbit in the reduced coordinates.

    Returns
    -------
    UniqueListMod1 of np.ndarray(shape=(3,), dtype=float)
        The orbit of v under the symmetry operations of the structure.
    """
    return UniqueListMod1([symop.transform_r(p) % 1 for symop in spacegroup.symmetries], tol=tol)


def check_orbit(spacegroup, positions, tol=1e-5):
    """
    check if the positions are in the same orbit of the spacegroup

    Parameters
    ----------
    spacegroup : irrep.spacegroup.SpaceGroup
        The spacegroup of the structure.
    positions : np.ndarray(shape=(N,3), dtype=float)
        Points which are checked to transform into each other under the symmetry operations.

    Returns
    -------
    bool
        True if the points are in the same orbit, False otherwise.
    """
    orbit = get_orbit(spacegroup, positions[0], tol=tol)
    for p in positions[1:]:
        if p not in orbit:
            return False
    return True


def orbit_and_rottrans(spacegroup, p):
    """
    Get the orbit of a point p under the symmetry operations of a structure.
    and the corresponding rotation matrices and translation vectors.

    Parameters
    ----------
    spacegroup : irrep.spacegroup.SpaceGroup
        The spacegroup object.
    p : np.ndarray(shape=(3,), dtype=float)
        Point for which to calculate the orbit in the reduced coordinates.

    Returns
    -------
    np.ndarray(shape=(N, 3)
        The orbit of v under the symmetry operations of the structure.
    np.ndarray(shape=(N, 3, 3)
        The rotation matrices of the symmetry operations
    np.ndarray(shape=(N, 3)
        The translation vectors of the symmetry operations
    """
    orbit = get_orbit(spacegroup, p)
    ind_oper = orbit.appended_indices
    rotations = []
    translations = []
    for i in ind_oper:
        symop = spacegroup.symmetries[i]
        rotations.append(symop.rotation)
        translations.append(symop.translation)
    return np.array(orbit), np.array(rotations), np.array(translations)
