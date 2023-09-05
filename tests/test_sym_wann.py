"""Test symmetrization of Wannier models"""
import numpy as np
import pytest
from pytest import approx

from wannierberri import calculators as calc

from common_systems import (
    Efermi_GaAs,
    Efermi_Fe,
    Efermi_Mn3Sn,
)


from test_run import (
        #calculators_GaAs,
        calculators_GaAs_internal,
                        )




@pytest.fixture
def check_symmetry(check_run):
    def _inner(system,
        calculators={},
        precision=1e-8,
        **kwargs,
            ):
        kwargs['do_not_compare']=True
        result_irr_k = check_run(system, use_symmetry=True, calculators=calculators,suffix="irr_k", **kwargs)
        result_full_k = check_run(system, use_symmetry=False, calculators=calculators,suffix="full_k", **kwargs)
        print (calculators.keys(),result_irr_k.results.keys(),result_full_k.results.keys())

        for quant in calculators.keys():
            assert result_full_k.results[quant].data == approx(
                    result_irr_k.results[quant].data,
                    rel=abs(precision) if precision<0 else None,
                    abs=precision if precision>0 else None)

    return _inner



def test_shiftcurrent_symmetry(check_symmetry, system_GaAs_sym_tb):
    """Test shift current with and without symmetry is the same for a symmetrized system"""
    param = dict(
        Efermi=Efermi_GaAs,
        omega=np.arange(1.0, 5.1, 0.5),
        smr_fixed_width=0.2,
        smr_type='Gaussian',
        kBT=0.01,
                )
    calculators = dict(
        shift_current=calc.dynamic.ShiftCurrent(sc_eta=0.1, **param),
                        )

    check_symmetry( system=system_GaAs_sym_tb,
                    grid_param=dict(NK=6, NKFFT=3),
                    calculators=calculators,
                    precision=1e-6
                    )





def test_Mn3Sn_sym_tb(check_symmetry, system_Mn3Sn_sym_tb):
    param = {'Efermi': Efermi_Mn3Sn}
    calculators = {}
#    calculators.update({k: v(**param) for k, v in calculators_GaAs.items()})
    calculators.update({k: v(**param) for k, v in calculators_GaAs_internal.items()})
    calculators.update({
        'ahc':calc.static.AHC(Efermi=Efermi_Mn3Sn, kwargs_formula={"external_terms":True}),
        # 'gyrotropic_Korb':calc.static.GME_orb_FermiSea(Efermi=Efermi_GaAs, kwargs_formula={"external_terms":False}),
        # 'gyrotropic_Kspin':calc.static.GME_spin_FermiSea(Efermi=Efermi_GaAs),
        # 'gyrotropic_Kspin_fsurf':calc.static.GME_spin_FermiSurf(Efermi=Efermi_GaAs),
        # 'gyrotropic_Korb_test':calc.static.GME_orb_FermiSea_test(Efermi=Efermi_GaAs),
                        })

    check_symmetry(system=system_Mn3Sn_sym_tb,calculators=calculators)


def test_Fe_sym_W90(check_run, system_Fe_sym_W90, compare_any_result):
    param = {'Efermi': Efermi_Fe}
    cals = {'ahc': calc.static.AHC,
            'Morb': calc.static.Morb,
            'spin': calc.static.Spin}
    calculators = {k: v(**param) for k, v in cals.items()}
    check_run(
        system_Fe_sym_W90,
        calculators,
        fout_name="berry_Fe_sym_W90",
        suffix="-run",
        use_symmetry=False
    )
    cals = {'gyrotropic_Korb': calc.static.GME_orb_FermiSea,
            'berry_dipole': calc.static.BerryDipole_FermiSea,
            'gyrotropic_Kspin': calc.static.GME_spin_FermiSea}
    calculators = {k: v(**param) for k, v in cals.items()}
    check_run(
        system_Fe_sym_W90,
        calculators,
        fout_name="berry_Fe_sym_W90",
        precision=1e-8,
        suffix="-run",
        compare_zero=True,
        use_symmetry=False
    )


def test_Fe_sym_W90_sym(check_run, system_Fe_sym_W90, compare_any_result):
    param = {'Efermi': Efermi_Fe}
    cals = {'ahc': calc.static.AHC,
            'Morb': calc.static.Morb,
            'spin': calc.static.Spin}
    calculators = {k: v(**param) for k, v in cals.items()}
    check_run(
        system_Fe_sym_W90,
        calculators,
        fout_name="berry_Fe_sym_W90",
        suffix="sym-run",
        use_symmetry=True
    )
    cals = {'gyrotropic_Korb': calc.static.GME_orb_FermiSea,
            'berry_dipole': calc.static.BerryDipole_FermiSea,
            'gyrotropic_Kspin': calc.static.GME_spin_FermiSea}
    calculators = {k: v(**param) for k, v in cals.items()}
    check_run(
        system_Fe_sym_W90,
        calculators,
        fout_name="berry_Fe_sym_W90",
        suffix="sym-run",
        precision=1e-8,
        compare_zero=True,
        use_symmetry=True
    )


def test_GaAs_sym_tb(check_symmetry, check_run, system_GaAs_sym_tb, compare_any_result):
    param = {'Efermi': Efermi_GaAs}
    calculators = {}
    #calculators.update({k: v(**param) for k, v in calculators_GaAs.items()})
    #calculators.update({k: v(**param) for k, v in calculators_GaAs_internal.items()})
    calculators.update({
        'berry_dipole':calc.static.BerryDipole_FermiSea(**param, kwargs_formula={"external_terms":True}),
        # 'gyrotropic_Korb':calc.static.GME_orb_FermiSea(Efermi=Efermi_GaAs, kwargs_formula={"external_terms":False}),
        # 'gyrotropic_Kspin':calc.static.GME_spin_FermiSea(Efermi=Efermi_GaAs),
        # 'gyrotropic_Kspin_fsurf':calc.static.GME_spin_FermiSurf(Efermi=Efermi_GaAs),
        # 'gyrotropic_Korb_test':calc.static.GME_orb_FermiSea_test(Efermi=Efermi_GaAs),
                        })

    check_symmetry(system=system_GaAs_sym_tb,calculators=calculators)

    check_run(
        system_GaAs_sym_tb,
        {'ahc': calc.static.AHC(Efermi=Efermi_GaAs)},
        fout_name="berry_GaAs_sym_tb",
        precision=1e-5,
        compare_zero=True,
        suffix="sym-zero",
                )


def test_GaAs_dynamic_sym(check_run, system_GaAs_sym_tb, compare_any_result):
    "Test shift current and injection current"

    param = dict(
        Efermi=Efermi_GaAs,
        omega=np.arange(1.0, 5.1, 0.5),
        smr_fixed_width=0.2,
        smr_type='Gaussian',
        kBT=0.01,
    )
    calculators = dict(
        shift_current=calc.dynamic.ShiftCurrent(sc_eta=0.1, **param),
        injection_current=calc.dynamic.InjectionCurrent(**param),
        opt_conductivity=calc.dynamic.OpticalConductivity(**param)
    )

    result_full_k = check_run(
        system_GaAs_sym_tb,
        calculators,
        fout_name="dynamic_GaAs_sym",
        grid_param={
            'NK': [6, 6, 6],
            'NKFFT': [3, 3, 3]
        },
        use_symmetry=False,
        do_not_compare=True,
            )

    result_irr_k = check_run(
        system_GaAs_sym_tb,
        calculators,
        fout_name="dynamic_GaAs_sym",
        suffix="sym",
        suffix_ref="",
        grid_param={
            'NK': [6, 6, 6],
            'NKFFT': [3, 3, 3]
        },
        use_symmetry=True,
        do_not_compare=True,
    )


    assert result_full_k.results["shift_current"].data == approx(
        result_irr_k.results["shift_current"].data, abs=1e-6)

    assert result_full_k.results["injection_current"].data == approx(
        result_irr_k.results["injection_current"].data, abs=1e-6)

    assert result_full_k.results["opt_conductivity"].data == approx(
        result_irr_k.results["opt_conductivity"].data, abs=1e-7)
