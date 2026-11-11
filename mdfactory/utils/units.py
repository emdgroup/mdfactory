# ABOUTME: Physical constants and unit conversions shared by the build code
# ABOUTME: Leaf module (imports nothing from mdfactory) so models and utils can both use it
"""Physical constants and unit conversions shared by the build code."""

#: Avogadro's number, matching the g <-> Dalton convention used throughout.
N_AVOGADRO = 6.022e23
#: Å³ -> liter (1 Å³ = 1e-27 L), for molar concentrations.
A3_TO_L = 1e-27
#: Å³ -> cm³ (1 Å³ = 1e-24 cm³), for densities in g/cm³.
A3_TO_CM3 = 1e-24
#: cm³ -> Å³.
CM3_TO_A3 = 1e24
