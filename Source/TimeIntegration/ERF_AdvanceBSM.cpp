#include <ERF.H>

using namespace amrex;

/**
 * @brief Advance the building surface model
 *
 * This function advances the building surface model by one time step.
 * It supports two modes:
 * 1. Simple mode: Only subsurface thermal diffusion
 * 2. EnergyBalance mode: Full surface energy balance with radiation coupling
 *
 * @param lev AMR level
 * @param cons_in Conservative variables (density, momentum, energy)
 * @param xvel_in X-velocity field
 * @param yvel_in Y-velocity field
 * @param zvel_in Z-velocity field (needed for vertical walls)
 * @param time Current simulation time
 * @param dt_advance Time step
 */
void ERF::advance_bsm (int lev,
                       MultiFab& cons_in,
                       MultiFab& xvel_in,
                       MultiFab& yvel_in,
                       MultiFab& zvel_in,
                       const double& time,
                       const double& dt_advance)
{
    if (solverChoice.building_surface_type == BuildingSurfaceType::EnergyBalance) {
        // TODO: Get actual sun angles from radiation module
        // For now, use placeholder values (overhead sun from south)
        double sun_azimuth_deg = 180.0;  // South
        double sun_zenith_deg = 30.0;    // 30° from vertical

        // Full energy balance model with state coupling
        bsm.Advance(lev, cons_in, xvel_in, yvel_in, zvel_in,
                    rad_fluxes[lev].get(),
                    terrain_blanking[lev].get(),
                    building_shadow_mask[lev].get(),
                    sun_azimuth_deg,
                    sun_zenith_deg,
                    time, dt_advance);
    } else if (solverChoice.building_surface_type == BuildingSurfaceType::Simple) {
        // Simple thermal diffusion only
        bsm.Advance(lev, dt_advance);
    }
}
