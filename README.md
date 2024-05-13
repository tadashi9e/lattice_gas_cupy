# Lattice gas cellular automata written in cupy

This is a simulation of flow in lattice gas cellular automata.

You can observe the formation of Karman vortices when a cylinder is placed in a uniform flow to the right.

## Requirement

- cupy
- cv2
- PIL

## lattice_gas.py

- Flow upwards is represented in red.
- Flow downwards is represented in green.
- Flow to the left (or stagnation against the flow to the right) is represented in blue.

## lattice_gas2.py

Same as lattice_gas.py, but

- Right rotation flow is represented in green.
- Left rotation flow is represented in red.
