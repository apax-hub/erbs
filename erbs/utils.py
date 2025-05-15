ERBS_PROPERTIES = [
    "forces_bias",
    "energy_bias",
]


def setup_ase():
    """Add uncertainty keys to ASE all properties.
    from https://github.com/zincware/IPSuite/blob/main/ipsuite/utils/helpers.py#L10
    """
    from ase.calculators.calculator import all_properties

    for val in ERBS_PROPERTIES:
        if val not in all_properties:
            all_properties.append(val)
