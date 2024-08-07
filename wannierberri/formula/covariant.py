import numpy as np
from ..__utility import alpha_A, beta_A
from . import Formula_ln, Matrix_ln, Matrix_GenDer_ln, FormulaProduct
from ..symmetry import transform_ident, transform_odd


class Identity(Formula_ln):

    def __init__(self,data_K=None):
        self.ndim = 0
        self.transformTR=transform_ident
        self.transformInv=transform_ident

    def nn(self, ik, inn, out):
        return np.eye(len(inn))

    def aa(self, ik, inn, out):
        return np.eye(len(inn)+len(out))
    
    def ln(self, ik, inn, out):
        return np.zeros((len(out), len(inn)))


class Eavln(Matrix_ln):
    """ be careful : this is not a covariant matrix"""

    def __init__(self, data_K):
        super().__init__(0.5 * (data_K.E_K[:, :, None] + data_K.E_K[:, None, :]))
        self.ndim = 0
        self.transformTR=transform_ident
        self.transformInv=transform_ident

class HH_K(Matrix_ln):
    """ be careful : this is not a covariant matrix"""
    def __init__(self, data_K):
        super().__init__(data_K.HH_K)
        self.ndim = 0
        self.transformTR=transform_ident
        self.transformInv=transform_ident

class DEinv_ln(Matrix_ln):
    "DEinv_ln.matrix[ik, m, n] = 1 / (E_mk - E_nk)"

    def __init__(self, data_K):
        super().__init__(data_K.dEig_inv)

    #def nn(self, ik, inn, out):
    #    raise NotImplementedError("dEinv_ln should not be called within inner states")


class Dcov(Matrix_ln):

    def __init__(self, data_K):
        super().__init__(data_K.D_H)

    def nn(self, ik, inn, out):
        raise ValueError("Dln should not be called within inner states")
    
    def aa(self, ik, inn, out):
        return self.matrix[ik]

class DerDcov(Dcov):

    def __init__(self, data_K):
        self.W = data_K.covariant('Ham', commader=2)
        self.V = data_K.covariant('Ham', gender=1)
        self.D = data_K.Dcov
        self.dEinv = DEinv_ln(data_K)

    def ln(self, ik, inn, out):
        summ = self.W.ln(ik, inn, out)
        tmp = np.einsum("lpb,pnd->lnbd", self.V.ll(ik, inn, out), self.D.ln(ik, inn, out))
        summ += tmp + tmp.swapaxes(2, 3)
        tmp = -np.einsum("lmb,mnd->lnbd", self.D.ln(ik, inn, out), self.V.nn(ik, inn, out))
        summ += tmp + tmp.swapaxes(2, 3)
        summ *= -self.dEinv.ln(ik, inn, out)[:, :, None, None]
        return summ


class InvMass(Matrix_GenDer_ln):
    r""" :math:`\overline{V}^{b:d}`"""

    def __init__(self, data_K):
        super().__init__(data_K.covariant('Ham', commader=1), data_K.covariant('Ham', commader=2), data_K.Dcov)
        self.transformTR=transform_ident
        self.transformInv=transform_ident


class DerWln(Matrix_GenDer_ln):
    r""" :math:`\overline{W}^{bc:d}`"""

    def __init__(self, data_K):
        super().__init__(data_K.covariant('Ham', 2), data_K.covariant('Ham', 3), data_K.Dcov)
        self.transformTR=transform_odd
        self.transformInv=transform_odd


#############################
#   Third derivative of  E  #
#############################


class Der3E(Formula_ln):

    def __init__(self, data_K, **parameters):
        super().__init__(data_K, **parameters)
        self.V = data_K.covariant('Ham', commader=1)
        self.D = data_K.Dcov
        self.dV = InvMass(data_K)
        self.dD = DerDcov(data_K)
        self.dW = DerWln(data_K)
        self.ndim = 3
        self.transformTR=transform_odd
        self.transformInv=transform_odd

    def nn(self, ik, inn, out):
        summ = np.zeros((len(inn), len(inn), 3, 3, 3), dtype=complex)
        summ += 1 * self.dW.nn(ik, inn, out)
        summ += 1 * np.einsum("mlac,lnb->mnabc", self.dV.nl(ik, inn, out), self.D.ln(ik, inn, out))
        summ += 1 * np.einsum("mla,lnbc->mnabc", self.V.nl(ik, inn, out), self.dD.ln(ik, inn, out))
        summ += -1 * np.einsum("mlbc,lna->mnabc", self.dD.nl(ik, inn, out), self.V.ln(ik, inn, out))
        summ += -1 * np.einsum("mlb,lnac->mnabc", self.D.nl(ik, inn, out), self.dV.ln(ik, inn, out))

        # TODO: alternatively: add factor 0.5 to first term, remove 4th and 5th, and ad line below.
        # Should give the same, I think, but needs to be tested
        # summ+=summ.swapaxes(0,1).conj()
        return summ

    def ln(self, ik, inn, out):
        raise NotImplementedError()


########################
#   Berry curvature    #
########################


class Omega(Formula_ln):

    def __init__(self, data_K, **parameters):
        super().__init__(data_K, **parameters)
        self.D = data_K.Dcov

        if self.external_terms:
            self.A = data_K.covariant('AA')
            self.O = data_K.covariant('OO')

        self.ndim = 1
        self.transformTR=transform_odd
        self.transformInv=transform_ident

    def nn(self, ik, inn, out):
        summ = np.zeros((len(inn), len(inn), 3), dtype=complex)

        if self.internal_terms:
            summ += -1j * np.einsum(
                "mlc,lnc->mnc",
                self.D.nl(ik, inn, out)[:, :, alpha_A],
                self.D.ln(ik, inn, out)[:, :, beta_A])

        if self.external_terms:
            summ += 0.5 * self.O.nn(ik, inn, out)
            summ += -1 * np.einsum(
                "mlc,lnc->mnc",
                self.D.nl(ik, inn, out)[:, :, alpha_A],
                self.A.ln(ik, inn, out)[:, :, beta_A])
            summ += +1 * np.einsum(
                "mlc,lnc->mnc",
                self.D.nl(ik, inn, out)[:, :, beta_A],
                self.A.ln(ik, inn, out)[:, :, alpha_A])
            summ += -1j * np.einsum(
                "mlc,lnc->mnc",
                self.A.nn(ik, inn, out)[:, :, alpha_A],
                self.A.nn(ik, inn, out)[:, :, beta_A])

        summ += summ.swapaxes(0, 1).conj()
        return summ

    def ln(self, ik, inn, out):
        raise NotImplementedError()


########################
#   derivative of      #
#   Berry curvature    #
########################


class DerOmega(Formula_ln):

    def __init__(self, data_K, **parameters):
        super().__init__(data_K, **parameters)
        self.dD = DerDcov(data_K)
        self.D = data_K.Dcov

        if self.external_terms:
            self.A = data_K.covariant('AA')
            self.dA = data_K.covariant('AA', gender=1)
            self.dO = data_K.covariant('OO', gender=1)
        self.ndim = 2
        self.transformTR=transform_ident
        self.transformInv=transform_odd

    def nn(self, ik, inn, out):
        summ = np.zeros((len(inn), len(inn), 3, 3), dtype=complex)
        if self.external_terms:
            summ += 0.5 * self.dO.nn(ik, inn, out)

        for s, a, b in (+1, alpha_A, beta_A), (-1, beta_A, alpha_A):
            if self.internal_terms:
                summ += -1j * s * np.einsum(
                    "mlc,lncd->mncd",
                    self.D.nl(ik, inn, out)[:, :, a],
                    self.dD.ln(ik, inn, out)[:, :, b])
                pass

            if self.external_terms:
                summ += -1 * s * np.einsum(
                    "mlc,lncd->mncd",
                    self.D.nl(ik, inn, out)[:, :, a],
                    self.dA.ln(ik, inn, out)[:, :, b, :])
                summ += -1 * s * np.einsum(
                    "mlcd,lnc->mncd",
                    self.dD.nl(ik, inn, out)[:, :, a, :],
                    self.A.ln(ik, inn, out)[:, :, b])
                summ += -1j * s * np.einsum(
                    "mlc,lncd->mncd",
                    self.A.nn(ik, inn, out)[:, :, a],
                    self.dA.nn(ik, inn, out)[:, :, b, :])
                pass

        summ += summ.swapaxes(0, 1).conj()
        return summ

    def ln(self, ik, inn, out):
        raise NotImplementedError()

class Hamiltonian(Matrix_ln):

    def __init__(self, data_K):
        v = data_K.covariant('Ham', gender=0)
        self.__dict__.update(v.__dict__)


class Velocity(Matrix_ln):

    def __init__(self, data_K):
        v = data_K.covariant('Ham', gender=1)
        self.__dict__.update(v.__dict__)


class Spin(Matrix_ln):

    def __init__(self, data_K):
        s = data_K.covariant('SS')
        self.__dict__.update(s.__dict__)


class DerSpin(Matrix_GenDer_ln):

    def __init__(self, data_K):
        s = data_K.covariant('SS', gender=1)
        self.__dict__.update(s.__dict__)

########################
#   orbital moment     #
########################

class Morb_H(Formula_ln):

    def __init__(self, data_K, **parameters):
        r"""  :math:`\varepcilon_{abc} \langle \partial_a u | H | \partial_b \rangle` """
        super().__init__(data_K, **parameters)
        if self.external_terms:
            self.A = data_K.covariant('AA')
            self.B = data_K.covariant('BB')
            self.C = data_K.covariant('CC')
        self.D = data_K.Dcov
        self.E = data_K.E_K
        self.ndim = 1
        self.transformTR=transform_odd
        self.transformInv=transform_ident

    def nn(self, ik, inn, out):
        summ = np.zeros((len(inn), len(inn), 3), dtype=complex)

        if self.internal_terms:
            summ += -1j * np.einsum(
                "mlc,lnc->mnc",
                self.D.nl(ik, inn, out)[:, :, alpha_A] * self.E[ik][out][None, :, None],
                self.D.ln(ik, inn, out)[:, :, beta_A])

        if self.external_terms:
            summ += 0.5 * self.C.nn(ik, inn, out)
            summ += -1 * np.einsum(
                "mlc,lnc->mnc",
                self.D.nl(ik, inn, out)[:, :, alpha_A],
                self.B.ln(ik, inn, out)[:, :, beta_A])
            summ += +1 * np.einsum(
                "mlc,lnc->mnc",
                self.D.nl(ik, inn, out)[:, :, beta_A],
                self.B.ln(ik, inn, out)[:, :, alpha_A])
            summ += -1j * np.einsum(
                "mlc,lnc->mnc",
                self.A.nn(ik, inn, out)[:, :, alpha_A] * self.E[ik][inn][None, :, None],
                self.A.nn(ik, inn, out)[:, :, beta_A])
        summ += summ.swapaxes(0, 1).conj()
        return summ

    def ln(self, ik, inn, out):
        raise NotImplementedError()

    @property
    def additive(self):
        return False


class Morb_Hpm(Formula_ln):

    def __init__(self, data_K, sign=+1, **parameters):
        r""" Morb_H  +- (En+Em)/2 * Omega """
        super().__init__(data_K, **parameters)
        self.H = Morb_H(data_K, **parameters)
        self.sign = sign
        if self.sign != 0:
            self.O = Omega(data_K, **parameters)
            self.Eav = Eavln(data_K)
        self.ndim = 1
        self.transformTR=transform_odd
        self.transformInv=transform_ident

    @property
    def additive(self):
        return False

    def nn(self, ik, inn, out):
        res = self.H.nn(ik, inn, out)
        if self.sign != 0:
            res += self.sign * self.Eav.nn(ik, inn, out)[:, :, None] * self.O.nn(ik, inn, out)
        return res

    def ln(self, ik, inn, out):
        raise NotImplementedError()


class morb(Morb_Hpm):

    def __init__(self, data_K, **parameters):
        super().__init__(data_K, sign=-1, **parameters)


########################
#   derivative of      #
#   orbital moment     #
########################


class DerMorb(Formula_ln):

    def __init__(self, data_K, **parameters):
        super().__init__(data_K, **parameters)
        self.dD = DerDcov(data_K)
        self.D = data_K.Dcov
        self.V = data_K.covariant('Ham', commader=1)
        self.E = data_K.E_K
        self.dO = DerOmega(data_K, **parameters)
        self.Omega = Omega(data_K, **parameters)
        if self.external_terms:
            self.A = data_K.covariant('AA')
            self.dA = data_K.covariant('AA', gender=1)
            self.B = data_K.covariant('BB')
            self.dB = data_K.covariant('BB', gender=1)
            self.dH = data_K.covariant('CC', gender=1)
        self.ndim = 2
        self.transformTR=transform_ident
        self.transformInv=transform_odd

    def nn(self, ik, inn, out):
        summ = np.zeros((len(inn), len(inn), 3, 3), dtype=complex)
        if self.internal_terms:
            summ += -2j * np.einsum(
                "mpc,pld,lnc->mncd",
                self.D.nl(ik, inn, out)[:, :, alpha_A], self.V.ll(ik, inn, out),
                self.D.ln(ik, inn, out)[:, :, beta_A])
            for s, a, b in (+1, alpha_A, beta_A), (-1, beta_A, alpha_A):
                summ += -2j * s * np.ei1nsum(
                    "mlc,lncd->mncd",
                    self.D.nl(ik, inn, out)[:, :, a],
                    self.E[ik][out][:, None, None, None] * self.dD.ln(ik, inn, out)[:, :, b])
        if self.external_terms:
            summ += 1 * self.dH.nn(ik, inn, out)
            summ += -2j * np.einsum(
                "mpc,pld,lnc->mncd",
                self.A.nn(ik, inn, out)[:, :, alpha_A], self.V.nn(ik, inn, out),
                self.A.nn(ik, inn, out)[:, :, beta_A])
            for s, a, b in (+1, alpha_A, beta_A), (-1, beta_A, alpha_A):
                summ += -2j * s * np.einsum(
                    "mlc,lncd->mncd",
                    self.A.nn(ik, inn, out)[:, :, a] * self.E[ik][inn][None, :, None],
                    self.dA.nn(ik, inn, out)[:, :, b, :])
                summ += -2 * s * np.einsum(
                    "mlc,lncd->mncd",
                    self.D.nl(ik, inn, out)[:, :, a],
                    self.dB.ln(ik, inn, out)[:, :, b, :])
                summ += -2 * s * np.einsum(
                    "mlc,lncd->mncd", (self.B.ln(ik, inn, out)[:, :, a]).transpose(1, 0, 2).conj(),
                    self.dD.ln(ik, inn, out)[:, :, b, :])

        summ += 1 * np.einsum("mlc,lnd->mncd", self.Omega.nn(ik, inn, out), self.V.nn(ik, inn, out))
        summ += 1 * self.E[ik][inn][:, None, None, None] * self.dO.nn(ik, inn, out)

        # Stepan: Shopuldn't we use the line below?
        # TODO: check this formula
        #summ+=summ.swapaxes(0,1).conj()
        return summ

    def ln(self, ik, inn, out):
        raise NotImplementedError()

    @property
    def additive(self):
        return False


########################
#   spin transport     #
########################


def _spin_velocity_einsum_opt(C, A, B):
    # Optimized version of C += np.einsum('knls,klma->knmas', A, B). Used in shc_B_H.
    nk = C.shape[0]
    nw = C.shape[1]
    for ik in range(nk):
        # Performing C[ik] += np.einsum('nls,lma->nmas', A[ik], B[ik])
        tmp_a = np.swapaxes(A[ik], 1, 2)  # nls -> nsl
        tmp_a = np.reshape(tmp_a, (nw * 3, nw))  # nsl -> (ns)l
        tmp_b = np.reshape(B[ik], (nw, nw * 3))  # lma -> l(ma)
        tmp_c = tmp_a @ tmp_b  # (ns)l, l(ma) -> (ns)(ma)
        tmp_c = np.reshape(tmp_c, (nw, 3, nw, 3))  # (ns)(ma) -> nsma
        C[ik] += np.transpose(tmp_c, (0, 2, 3, 1))  # nsma -> nmas


def _J_H_qiao(data_K,external_terms=True):
    if not external_terms :
        raise NotImplementedError("spin Hall without external terms is not implemented yet")
    # Spin current operator, J. Qiao et al PRB (2019)
    # J_H_qiao[k,m,n,a,s] = <mk| {S^s, v^a} |nk> / 2
    SS_H = data_K.Xbar('SS')
    SH_H = data_K.Xbar("SH")
    shc_K_H = -1j * data_K.Xbar("SR")
    _spin_velocity_einsum_opt(shc_K_H, SS_H, data_K.D_H)
    shc_L_H = -1j * data_K._R_to_k_H(data_K.get_R_mat('SHR'), hermitean=False)
    _spin_velocity_einsum_opt(shc_L_H, SH_H, data_K.D_H)
    J = (
        data_K.delE_K[:, None, :, :, None] * SS_H[:, :, :, None, :]
        + data_K.E_K[:, None, :, None, None] * shc_K_H[:, :, :, :, :] - shc_L_H)
    return (J + J.swapaxes(1, 2).conj()) / 2


def _J_H_ryoo(data_K,external_terms=True):
    if not external_terms :
        raise NotImplementedError("spin Hall without external terms is not implemented yet")
    # Spin current operator, J. H. Ryoo et al PRB (2019)
    # J_H_ryoo[k,m,n,a,s] = <mk| {S^s, v^a} |nk> / 2
    SA_H = data_K.Xbar("SA")
    SHA_H = data_K.Xbar("SHA")
    J = -1j * (data_K.E_K[:, None, :, None, None] * SA_H - SHA_H)
    _spin_velocity_einsum_opt(J, data_K.Xbar('SS'), data_K.Xbar('Ham', 1))
    return (J + J.swapaxes(1, 2).conj()) / 2


class SpinVelocity(Matrix_ln):
    "spin current matrix elements. SpinVelocity.matrix[ik, m, n, a, s] = <u_mk|{v^a S^s}|u_nk> / 2"

    def __init__(self, data_K, spin_current_type,external_terms=True):
        if spin_current_type == "qiao":
            # J. Qiao et al PRB (2018)
            super().__init__(_J_H_qiao(data_K,external_terms=external_terms))
        elif spin_current_type == "ryoo":
            # J. H. Ryoo et al PRB (2019)
            super().__init__(_J_H_ryoo(data_K,external_terms=external_terms))
        else:
            raise ValueError(f"spin_current_type must be qiao or ryoo, not {spin_current_type}")
        self.transformTR=transform_ident
        self.transformInv=transform_odd


class SpinOmega(Formula_ln):
    "spin Berry curvature"

    def __init__(self, data_K, spin_current_type="ryoo", **parameters):
        super().__init__(data_K, **parameters)
        self.A = data_K.covariant('AA')
        self.D = data_K.Dcov
        self.J = SpinVelocity(data_K, spin_current_type)
        self.dEinv = DEinv_ln(data_K)
        self.ndim = 3
        self.transformTR=transform_ident
        self.transformInv=transform_ident

    def nn(self, ik, inn, out):
        summ = np.zeros((len(inn), len(inn), 3, 3, 3), dtype=complex)

        # v_over_de[l,n,b] = v[l,n,b] / (e[n] - e[l]) = D[l,n,b] - 1j * A[l,n,b]
        v_over_de = self.D.ln(ik, inn, out) - 1j * self.A.ln(ik, inn, out)

        # j_over_de[m,l,a,s] = j[m,l,a,s] / (e[m] - e[l])
        j_over_de = self.J.nl(ik, inn, out) * self.dEinv.nl(ik, inn, out)[:, :, None, None]

        summ += -2 * np.einsum("mlas,lnb->mnabs", j_over_de, v_over_de).imag

        return summ

    def ln(self, ik, inn, out):
        raise NotImplementedError()

class Omegakp(Formula_ln):
    def __init__(self, data_K, **parameters):
        super().__init__(data_K, **parameters)    
        self.dEinv = DEinv_ln(data_K)
        self.V = data_K.covariant('Ham', commader=1)
        self.ndim = 1
        self.transformTR=transform_ident
        self.transformInv=transform_ident

    def nn(self, ik, inn, out):
        summ = -2j * np.einsum("mla,ml,nl,lna->mna", self.V.nl(ik,inn,out)[:, :, alpha_A] , 
                self.dEinv.nl(ik,inn,out), 
                self.dEinv.nl(ik,inn,out), 
                self.V.ln(ik,inn,out)[:, :, beta_A] )
        return summ
    
    def ln(self, ik, inn, out):
        raise NotImplementedError()

class BCP_G_kp(Formula_ln):
    def __init__(self, data_K, **parameters):
        super().__init__(data_K, **parameters)    
        self.Omega = Omegakp
        self.dEinv = DEinv_ln(data_K)
        self.V = data_K.covariant('Ham', commader=1)
        self.ndim = 1
        self.transformTR=transform_ident
        self.transformInv=transform_ident

    def nn(self, ik, inn, out):
        summ = 2 * np.einsum("mla,ml,nl,lna,ml->mna", self.V.nl(ik,inn,out)[:, :, alpha_A] , 
                self.dEinv.nl(ik,inn,out), 
                self.dEinv.nl(ik,inn,out), 
                self.V.ln(ik,inn,out)[:, :, beta_A],
                self.dEinv.nl(ik,inn,out) ) 
        return summ
    
    def ln(self, ik, inn, out):
        raise NotImplementedError()

class BCP_G(Formula_ln):

    def __init__(self, data_K, **parameters):
        super().__init__(data_K, **parameters)
        self.D = data_K.Dcov
        self.dEinv = DEinv_ln(data_K)

        if self.external_terms:
            self.A = data_K.covariant('AA')
            self.O = data_K.covariant('OO')

        self.ndim = 1
        self.transformTR=transform_odd
        self.transformInv=transform_ident

    def nn(self, ik, inn, out):
        summ = np.zeros((len(inn), len(inn), 3), dtype=complex)

        if self.internal_terms:
            summ += np.einsum(
                "nl,mlc,lnc->mnc",
                self.dEinv.nl(ik,inn,out), 
                self.D.nl(ik, inn, out)[:, :, alpha_A],
                self.D.ln(ik, inn, out)[:, :, beta_A])

        if self.external_terms:
            summ += -0.5j * self.O.nn(ik, inn, out)
            summ += 1j * np.einsum(
                "ml,mlc,lnc->mnc",
                self.dEinv.nl(ik,inn,out), 
                self.D.nl(ik, inn, out)[:, :, alpha_A],
                self.A.ln(ik, inn, out)[:, :, beta_A])
            summ += -1j * np.einsum(
                "ml,mlc,lnc->mnc",
                self.dEinv.nl(ik,inn,out), 
                self.D.nl(ik, inn, out)[:, :, beta_A],
                self.A.ln(ik, inn, out)[:, :, alpha_A])
            summ +=  np.einsum(
                "ml,mlc,lnc->mnc",
                self.dEinv.nl(ik,inn,out), 
                self.A.nn(ik, inn, out)[:, :, alpha_A],
                self.A.nn(ik, inn, out)[:, :, beta_A])

        summ += summ.swapaxes(0, 1).conj()
        return summ

    def ln(self, ik, inn, out):
        raise NotImplementedError()

class X1(Formula_ln):
    def __init__(self, data_K, **parameters):
        super().__init__(data_K, **parameters)
        self.Ham = data_K.HH_K
        #print(self.Ham)
        numwann = np.shape(self.Ham)[1]
        self.GR = np.linalg.inv(self.Ef*np.eye(numwann)[None,:,:] - self.Ham + self.Gamma*np.eye(numwann)[None,:,:] )
        self.GA = np.linalg.inv(self.Ef*np.eye(numwann)[None,:,:] - self.Ham - self.Gamma*np.eye(numwann)[None,:,:] )
        self.V = data_K.covariant('Ham', gender=1)
        self.ndim = 3
        self.transformTR=transform_ident
        self.transformInv=transform_ident

    def nn(self, ik, inn, out):
        #summ = np.zeros((len(inn), len(inn), 3, 3, 3), dtype=complex)
        #print(np.shape(self.GR) )
        #print(self.V.aa(ik,inn,out),'VVVVVVVVV')
        summ = -0.5*np.einsum("mqa,ql,lk,kxb,xy,yzc,zn->mnabc",
                self.V.aa(ik,inn,out),
                self.GR[ik],
                self.GR[ik],
                self.V.aa(ik,inn,out),
                self.GR[ik],
                self.V.aa(ik,inn,out),
                self.GA[ik]
                ).imag

        return summ + summ.transpose(0,1,2,4,3)

    def ln(self, ik, inn, out):
        raise NotImplementedError()



class X2(Formula_ln):
    def __init__(self, data_K, **parameters):
        super().__init__(data_K, **parameters)    
        self.Ham = data_K.HH_K
        numwann = np.shape(self.Ham)[1]
        self.GR = np.linalg.inv(self.Ef*np.eye(numwann)[None,:,:] - self.Ham + self.Gamma*np.eye(numwann)[None,:,:] )
        self.GA = np.linalg.inv(self.Ef*np.eye(numwann)[None,:,:] - self.Ham - self.Gamma*np.eye(numwann)[None,:,:] )
        self.V = data_K.covariant('Ham', gender=1)
        #self.W = data_K.covariant('Ham', commader=2)
        self.W = InvMass(data_K)
        self.ndim = 3
        self.transformTR=transform_ident
        self.transformInv=transform_ident

    def nn(self, ik, inn, out):
        #summ = np.zeros((len(inn), len(inn), 3, 3, 3), dtype=complex)
        summ = -0.5 * np.einsum("mqa,ql,lk,kxbc,xn->mnabc", 
                self.V.aa(ik,inn,out),
                self.GR[ik],
                self.GR[ik],
                self.W.aa(ik,inn,out),
                self.GA[ik]
            ).imag

        return summ + summ.transpose(0,1,2,4,3)
    
    def ln(self, ik, inn, out):
        raise NotImplementedError()


class X3(Formula_ln):
    def __init__(self, data_K, **parameters):
        super().__init__(data_K, **parameters)    
        self.Ham = data_K.HH_K
        numwann = np.shape(self.Ham)[1]
        self.GR = np.linalg.inv(self.Ef*np.eye(numwann)[None,:,:] - self.Ham + self.Gamma*np.eye(numwann)[None,:,:] )
        self.GA = np.linalg.inv(self.Ef*np.eye(numwann)[None,:,:] - self.Ham - self.Gamma*np.eye(numwann)[None,:,:] )
        self.V = data_K.covariant('Ham', gender=1)
        self.W = data_K.covariant('Ham', commader=2)
        #self.W = InvMass(data_K)
        self.ndim = 3
        self.transformTR=transform_ident
        self.transformInv=transform_ident
        self.transformInv=transform_ident

    def nn(self, ik, inn, out):
        #summ = np.zeros((len(inn), len(inn), 3, 3, 3), dtype=complex)
        summ = -1*np.einsum("mqa,ql,lk,kx,xybc,yn->mnabc", 
                self.V.aa(ik,inn,out),
                self.GR[ik],
                self.GR[ik],
                self.GR[ik],
                self.W.aa(ik,inn,out),
                self.GR[ik]
            ).imag
        
        summ += -2*np.einsum("mqa,ql,lk,kx,xyb,yz,zic,in->mnabc", 
                self.V.aa(ik,inn,out),
                self.GR[ik],
                self.GR[ik],
                self.GR[ik],
                self.V.aa(ik,inn,out),
                self.GR[ik],
                self.V.aa(ik,inn,out),
                self.GR[ik]
            ).imag

        summ += -1*np.einsum("mqa,ql,lk,kxb,xy,yz,zic,in->mnabc", 
                self.V.aa(ik,inn,out),
                self.GR[ik],
                self.GR[ik],
                self.V.aa(ik,inn,out),
                self.GR[ik],
                self.GR[ik],
                self.V.aa(ik,inn,out),
                self.GR[ik]
            ).imag
        return summ + summ.transpose(0,1,2,4,3)
    
    def ln(self, ik, inn, out):
        raise NotImplementedError()


####################################
#                                  #
#    Some Prooducts                #
#                                  #
####################################

class VelOmega(FormulaProduct):

    def __init__(self, data_K, **kwargs_formula):
        super().__init__([data_K.covariant('Ham', commader=1), Omega(data_K, **kwargs_formula)], name='VelOmega')

class VelOmegakp(FormulaProduct):

    def __init__(self, data_K, **kwargs_formula):
        super().__init__([data_K.covariant('Ham', commader=1), Omegakp(data_K, **kwargs_formula)], name='VelOmegakp')

class BCPkp(FormulaProduct):

    def __init__(self, data_K, **kwargs_formula):
        super().__init__([data_K.covariant('Ham', commader=1), BCP_G_kp(data_K, **kwargs_formula)], name='VelOmegakp')

class BCP(FormulaProduct):

    def __init__(self, data_K, **kwargs_formula):
        super().__init__([data_K.covariant('Ham', commader=1), BCP_G(data_K, **kwargs_formula)], name='VelOmegakp')

class VelHplus(FormulaProduct):

    def __init__(self, data_K, **kwargs_formula):
        super().__init__([data_K.covariant('Ham', commader=1), Morb_Hpm(data_K, sign=+1, **kwargs_formula)], name='VelHplus')


class VelSpin(FormulaProduct):

    def __init__(self, data_K, **kwargs_formula):
        super().__init__([data_K.covariant('Ham', commader=1), Spin(data_K)], name='VelSpin')


class VelVel(FormulaProduct):

    def __init__(self, data_K, **kwargs_formula):
        super().__init__([data_K.covariant('Ham', commader=1), data_K.covariant('Ham', commader=1)], name='VelVel')


class VelVelVel(FormulaProduct):

    def __init__(self, data_K, **kwargs_formula):
        super().__init__([data_K.covariant('Ham', commader=1), data_K.covariant('Ham', commader=1), data_K.covariant('Ham', commader=1)], name='VelVelVel')


class MassVel(FormulaProduct):

    def __init__(self, data_K, **kwargs_formula):
        super().__init__([InvMass(data_K), data_K.covariant('Ham', commader=1)], name='MassVel')


class MassMass(FormulaProduct):

    def __init__(self, data_K, **kwargs_formula):
        super().__init__([InvMass(data_K), InvMass(data_K)], name='MassMass')


class VelMassVel(FormulaProduct):

    def __init__(self, data_K, **kwargs_formula):
        super().__init__([data_K.covariant('Ham', commader=1), InvMass(data_K),
            data_K.covariant('Ham', commader=1)], name='VelMassVel')


class OmegaS(FormulaProduct):

    def __init__(self, data_K, **kwargs_formula):
        super().__init__([Omega(data_K, **kwargs_formula), Spin(data_K)], name='SpinOmega')


class OmegaOmega(FormulaProduct):

    def __init__(self, data_K, **kwargs_formula):
        super().__init__([Omega(data_K, **kwargs_formula), Omega(data_K, **kwargs_formula)], name='OmegaOmega')


class OmegaHplus(FormulaProduct):

    def __init__(self, data_K, **kwargs_formula):
        super().__init__([Omega(data_K, **kwargs_formula), Morb_Hpm(data_K, sign=+1, **kwargs_formula)], name='OmegaHplus')




