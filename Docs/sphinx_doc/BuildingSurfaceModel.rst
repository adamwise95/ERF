.. _BuildingSurfaceModel:

Building Surface Model (BSM)
=============================

Overview
--------

The Building Surface Model (BSM) provides physically-based thermal boundary conditions
for building walls and roofs in urban simulations. Instead of prescribing fixed surface
temperatures, BSM solves a surface energy balance to compute temperatures that respond
to radiation, atmospheric conditions, and subsurface heat conduction.

Physics
-------

BSM solves the surface energy balance equation:

.. math::

   R_{net} + H + LE + G = 0

where:

* :math:`R_{net}` = Net radiation (shortwave + longwave)
* :math:`H` = Sensible heat flux (MOST with stability corrections)
* :math:`LE` = Latent heat flux (Dudhia scheme)
* :math:`G` = Ground heat conduction (4-layer subsurface model)

The surface temperature :math:`T_{surf}` is iteratively determined using Newton-Raphson
iteration to satisfy this energy balance.

Subsurface Heat Conduction
^^^^^^^^^^^^^^^^^^^^^^^^^^^

BSM includes a 4-layer subsurface thermal diffusion model solving:

.. math::

   \frac{\partial T}{\partial t} = d \frac{\partial^2 T}{\partial z^2}

where :math:`d` is the thermal diffusivity (:math:`d = k/c_p`).

Currently uses explicit time stepping (like MM5 LSM). Implicit tridiagonal solver
can be added later for improved stability with larger time steps.

Model Types
-----------

Simple
^^^^^^

Basic subsurface thermal diffusion with prescribed surface temperature.
Useful for testing infrastructure before adding full energy balance.

**Features:**

* 4-layer subsurface conduction
* Explicit time stepping
* Prescribed surface temperature (BC)
* Independent of atmospheric state

EnergyBalance
^^^^^^^^^^^^^

Full surface energy balance model coupling radiation, atmosphere, and subsurface.

**Features:**

* All Simple model features, plus:
* Radiation coupling (RRTMGP SW/LW fluxes)
* Sensible heat (MOST)
* Latent heat (Dudhia scheme with moisture availability)
* Newton-Raphson solver for surface temperature
* Surface properties (albedo, emissivity, moisture availability)

Input Parameters
----------------

Enable BSM in the input file:

.. code-block:: bash

   # Building surface model type
   erf.building_surface_model = "EnergyBalance"  # or "Simple" or "None"

   # Material properties (concrete defaults)
   erf.bsm_albedo = 0.2              # Surface albedo
   erf.bsm_emissivity = 0.9          # Surface emissivity
   erf.bsm_moisture_avail = 0.0      # Moisture availability (0-1)
   erf.bsm_cp_concrete = 2.0e6       # Specific heat [J/m³K]
   erf.bsm_k_concrete = 1.5          # Thermal conductivity [W/mK]

   # Subsurface layers
   erf.bsm_nz_layers = 4             # Number of layers
   erf.bsm_dz_layer = 0.05           # Layer thickness [m]
   erf.bsm_deep_temp = 300.0         # Deep boundary condition [K]

   # Radiation must be enabled for EnergyBalance model
   erf.radiation_model = "RRTMGP"

Architecture
------------

BSM follows the same design pattern as the Land Surface Model (LSM):

* **BuildingSurface**: Wrapper class managing models by AMR level
* **BuildingSurfBase**: Abstract base class for polymorphism
* **BSM_Simple**: Simple thermal diffusion implementation
* **BSM_EnergyBalance**: Full energy balance implementation

Key Design Decisions
^^^^^^^^^^^^^^^^^^^^

1. **Separation from LSM**: BSM handles building surfaces (k>0), LSM handles ground (k=0)
2. **No additional EB infrastructure**: Uses existing terrain_blank masks
3. **On-the-fly masking**: Surface types (roof/wall) identified via masks in GPU kernels
4. **All velocity components**: xvel, yvel, and zvel passed for different wall orientations

Time Integration
----------------

BSM is advanced after LSM in the time stepping loop::

   advance_lsm(lev, ...)   // Update ground surface
   advance_bsm(lev, ...)   // Update building surfaces

For EnergyBalance model, the sequence is:

1. Solve surface energy balance → update T_surf
2. Advance subsurface diffusion → update layer temperatures
3. Apply T_surf via immersed forcing in make_sources()

Surface Identification
----------------------

Building surfaces are identified using terrain_blank neighbor checks
(same logic as ERF_MakeMomSources.cpp):

.. code-block:: cpp

   // Roof: partial cell with open above, solid below
   Real roof_mask = (t_blank > 0 && t_blank < t_blank_below
                     && t_blank_above == 0) ? 1.0 : 0.0;

   // Walls: partial cell adjacent to solid in x or y direction
   Real south_mask = (t_blank > 0 && t_blank <= t_blank_north
                      && t_blank_south == 0) ? 1.0 : 0.0;
   // Similar for north, east, west walls

This naturally excludes k=0 (where LSM operates) since t_blank_below is zero.

Differences from OpenFOAM
--------------------------

ERF's implementation differs from the OpenFOAM ABLLouisWallT boundary condition:

1. **Boundary conditions**: OpenFOAM applies direct fixedValue BC; ERF uses immersed forcing drag
2. **Grid structure**: OpenFOAM is body-fitted; ERF is Cartesian with embedded boundaries
3. **Surface representation**: OpenFOAM has explicit patches; ERF uses on-the-fly masking
4. **Iteration location**: OpenFOAM solves in updateCoeffs(); ERF solves in advance_bsm()

These are architectural differences that don't affect the physics - the same energy
balance equation is solved, just applied differently to the flow.

Future Enhancements
-------------------

Planned features (not yet implemented):

* **Shading mask**: Ray-tracing or simple geometric shadow for direct SW
* **Variable materials**: Different properties for roof vs walls
* **Window modeling**: Transparent surfaces with different radiation treatment
* **Green roofs**: Vegetation layer on roofs
* **HVAC coupling**: Internal heat gains/losses

References
----------

* OpenFOAM ABLLouisWallT: https://github.com/hgopalan/scout-openfoam/tree/main/ABLBC/ABLLouisWallT
* ERF Land Surface Model: :ref:`LandSurfaceModel`
* MOST (Monin-Obukhov Similarity Theory): :ref:`SurfaceLayer`
