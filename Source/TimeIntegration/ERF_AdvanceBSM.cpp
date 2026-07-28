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
        // Get sun angles for shadow mask
        // If radiation is enabled, extract from radiation module
        double sun_azimuth_deg = 180.0;   // Default: South
        double sun_zenith_deg = 45.0;      // Default: 45° from vertical

        if (solverChoice.rad_type != RadiationType::None && rad[lev]) {
            // TODO: Add Get_Sun_Angles() method to Radiation class
            // For now, use time-of-day approximation
            // This is a simplified calculation - full implementation would use
            // orbital_cos_zenith and azimuth calculation like WRF

            // Simple time-of-day estimate (placeholder)
            // Assumes: solar noon at time=12h local, sun moves 15°/hour
            double hour = fmod(time / 3600.0, 24.0);  // Convert seconds to hours
            double hour_angle = (hour - 12.0) * 15.0;  // -180° to +180°

            // Very rough zenith estimate (minimum at noon)
            sun_zenith_deg = 30.0 + 30.0 * std::abs(hour - 12.0) / 6.0;
            sun_zenith_deg = std::min(sun_zenith_deg, 85.0);  // Cap at grazing

            // Azimuth: 0°=North, 90°=East, 180°=South, 270°=West
            // Sun moves East->South->West
            if (hour < 6.0) {
                sun_azimuth_deg = 90.0;  // East (sunrise)
            } else if (hour < 12.0) {
                sun_azimuth_deg = 90.0 + 90.0 * (hour - 6.0) / 6.0;  // E->S
            } else if (hour < 18.0) {
                sun_azimuth_deg = 180.0 + 90.0 * (hour - 12.0) / 6.0;  // S->W
            } else {
                sun_azimuth_deg = 270.0;  // West (sunset)
            }
        }

        // Grid spacing array for stretched/terrain-following coordinates
        GpuArray<Real,3> dx_arr = {geom[lev].CellSize(0),
                                   geom[lev].CellSize(1),
                                   geom[lev].CellSize(2)};

        // Full energy balance model with state coupling
        bsm.Advance(lev, cons_in, xvel_in, yvel_in, zvel_in,
                    rad_fluxes[lev].get(),
                    terrain_blanking[lev].get(),
                    z_phys_cc[lev].get(),
                    building_shadow_mask[lev].get(),
                    dx_arr,
                    sun_azimuth_deg,
                    sun_zenith_deg,
                    time, dt_advance);
    } else if (solverChoice.building_surface_type == BuildingSurfaceType::Simple) {
        // Simple thermal diffusion only
        bsm.Advance(lev, dt_advance);
    }
}
