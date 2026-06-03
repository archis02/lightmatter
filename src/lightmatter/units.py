from __future__ import annotations
from typing import Final

# Units, time = ps => frequency = THz, 
# mass = m_e, q = e, x = A
# => E = 5.686e2 V / m

# UNIT_M: Final[float] = 9.109e-31  # kg per electron mass
# UNIT_Q: Final[float] = 1.602e-19  # Coulomb per e
UNIT_E: Final[float] = 5.686e2    # derived from other units

UNIT_F: Final[float] = 1e12   # Hz per THz
UNIT_T: Final[float] = 1e-12  # s per ps
UNIT_L: Final[float] = 1e-10  # m per angstrom

C0: Final[float] = 2.99792458e6  # in code units
C0_SI: Final[float] = 299_792_458.0

# these are not required normally, recheck if you are uncommenting this!
# mu0 = 4e-7 * np.pi
# eps0 = 1.0 / (mu0 * c0**2)
# Z0 = np.sqrt(mu0 / eps0)