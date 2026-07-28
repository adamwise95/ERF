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
        // Check if we need to update shadow mask (only for Raycast mode)
        bool update_shadow = false;

        if (solverChoice.shadow_type == ShadowType::Raycast) {
            // Raycast mode: expensive ray-casting, update periodically
            int shadow_freq = solverChoice.shadow_freq_in_steps;

            if (shadow_freq > 0) {
                // Update shadow mask every shadow_freq steps
                update_shadow = ( (istep[lev] == 0) ||
                                 (istep[lev] % shadow_freq == 0) ||
                                 (shadow_last_updated[lev] < 0) );

                if (update_shadow) {
                    shadow_last_updated[lev] = istep[lev];
                }
            }
        }
        // Geometric mode: no shadow mask needed (computed inline in energy balance)

        // Get sun angles
        double sun_azimuth_deg = 180.0;   // Default: South
        double sun_zenith_deg = 45.0;      // Default: 45° from vertical

        if (solverChoice.rad_type != RadiationType::None && rad[lev]) {
            Real az, zen;
            rad[lev]->Get_Sun_Angles(az, zen);
            sun_azimuth_deg = az;
            sun_zenith_deg = zen;
        }

        // Always print sun angles (helps debug geometric shading)
        if (update_shadow) {
            amrex::Print() << "BSM: Updating shadow mask (Raycast) at step " << istep[lev] << std::endl;
        }
        amrex::Print() << "BSM: Sun angles - Azimuth: " << sun_azimuth_deg
                      << "° (0=N, 90=E, 180=S, 270=W), Zenith: " << sun_zenith_deg
                      << "° (0=overhead, 90=horizon)" << std::endl;

        // Grid spacing array for stretched/terrain-following coordinates
        GpuArray<Real,3> dx_arr = {geom[lev].CellSize(0),
                                   geom[lev].CellSize(1),
                                   geom[lev].CellSize(2)};

        // Full energy balance model with state coupling
        // Pass shadow_mask only if we're updating it this step
        bsm.Advance(lev, cons_in, xvel_in, yvel_in, zvel_in,
                    rad_fluxes[lev].get(),
                    terrain_blanking[lev].get(),
                    z_phys_cc[lev].get(),
                    update_shadow ? building_shadow_mask[lev].get() : nullptr,
                    dx_arr,
                    sun_azimuth_deg,
                    sun_zenith_deg,
                    time, dt_advance);
    } else if (solverChoice.building_surface_type == BuildingSurfaceType::Simple) {
        // Simple thermal diffusion only
        bsm.Advance(lev, dt_advance);
    }
}
