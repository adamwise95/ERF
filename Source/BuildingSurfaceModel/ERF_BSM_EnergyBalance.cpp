#include <ERF_BSM_EnergyBalance.H>
#include <ERF_Constants.H>

using namespace amrex;

/**
 * Advance the energy balance model with atmospheric state coupling
 *
 * @param lev AMR level
 * @param cons_in Conservative variables (rho, rho*theta, etc.)
 * @param xvel_in X-velocity
 * @param yvel_in Y-velocity
 * @param zvel_in Z-velocity (for vertical walls)
 * @param rad_fluxes Radiation fluxes (SW/LW up/down)
 * @param time Current simulation time
 * @param dt_advance Time step
 */
void
BSM_EnergyBalance::Advance_With_State (const int& lev,
                                       const MultiFab& cons_in,
                                       const MultiFab& xvel_in,
                                       const MultiFab& yvel_in,
                                       const MultiFab& zvel_in,
                                       const MultiFab* rad_fluxes,
                                       const MultiFab* terrain_blank,
                                       const Real& time,
                                       const Real& dt_advance)
{
    m_dt = dt_advance;

    // Solve surface energy balance to get updated surface temperature
    Solve_Surface_Energy_Balance(cons_in, xvel_in, yvel_in, zvel_in,
                                  rad_fluxes, terrain_blank, time);

    // Update subsurface temperatures via thermal diffusion
    AdvanceSubsurface();
}

/**
 * Solve the surface energy balance iteratively
 *
 * Energy balance: R_net - H - LE - G = 0
 * where:
 *   R_net = Net radiation (SW + LW)
 *   H = Sensible heat flux (MOST)
 *   LE = Latent heat flux (Dudhia scheme)
 *   G = Ground heat conduction
 *
 * Uses Newton-Raphson iteration to find T_surf that satisfies balance.
 */
void
BSM_EnergyBalance::Solve_Surface_Energy_Balance(
    const MultiFab& cons_in,
    const MultiFab& xvel_in,
    const MultiFab& yvel_in,
    const MultiFab& zvel_in,
    const MultiFab* rad_fluxes,
    const MultiFab* terrain_blank,
    Real time)
{
    // Get temperature arrays
    MultiFab& T_surf = *m_vars[surf_temp_idx];
    MultiFab& T1 = *m_vars[layer1_temp_idx];

    // Physical constants
    const Real sigma = 5.67e-8;        // Stefan-Boltzmann [W/m²K⁴]
    const Real Cp_d_val = Cp_d;        // Specific heat at constant pressure [J/kgK]
    const Real kappa = KAPPA;          // von Karman constant
    const Real z0 = 0.01;              // Roughness length [m] - should be configurable
    const Real max_iter = 5;           // Maximum Newton iterations
    const Real min_t_blank = 1.e-4;    // Minimum terrain_blank threshold

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(T_surf, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.tilebox();

        const Array4<const Real>& cons_arr = cons_in.const_array(mfi);
        const Array4<const Real>& u_arr = xvel_in.const_array(mfi);
        const Array4<const Real>& v_arr = yvel_in.const_array(mfi);
        const Array4<const Real>& w_arr = zvel_in.const_array(mfi);

        const Array4<Real>& T_surf_arr = T_surf.array(mfi);
        const Array4<const Real>& T1_arr = T1.const_array(mfi);

        // Terrain blanking for surface identification
        const Array4<const Real>& t_blank_arr = (terrain_blank) ? terrain_blank->const_array(mfi)
                                                                : Array4<const Real>{};

        // Radiation fluxes (SW_up, SW_dn, LW_up, LW_dn at each k level)
        const Array4<const Real>& rad_arr = (rad_fluxes) ? rad_fluxes->const_array(mfi)
                                                         : Array4<const Real>{};
        const bool has_radiation = (rad_fluxes != nullptr);

        const Real albedo = m_albedo;
        const Real emissivity = m_emissivity;
        const Real moisture_avail = m_moisture_avail;
        const Real k_concrete = m_k;
        const Real dz = m_dz;

        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            // Skip if no terrain_blank data
            if (!t_blank_arr) return;

            // Get terrain_blank for this cell and neighbors
            Real t_blank = t_blank_arr(i,j,k);
            Real t_blank_below = (k > 0) ? t_blank_arr(i,j,k-1) : 0.0;
            Real t_blank_above = t_blank_arr(i,j,k+1);
            Real t_blank_north = t_blank_arr(i,j+1,k);
            Real t_blank_south = t_blank_arr(i,j-1,k);
            Real t_blank_east  = t_blank_arr(i+1,j,k);
            Real t_blank_west  = t_blank_arr(i-1,j,k);

            // Apply minimum threshold
            if (t_blank < min_t_blank) t_blank = 0.0;
            if (t_blank_below < min_t_blank) t_blank_below = 0.0;
            if (t_blank_above < min_t_blank) t_blank_above = 0.0;
            if (t_blank_north < min_t_blank) t_blank_north = 0.0;
            if (t_blank_south < min_t_blank) t_blank_south = 0.0;
            if (t_blank_east < min_t_blank) t_blank_east = 0.0;
            if (t_blank_west < min_t_blank) t_blank_west = 0.0;

            // Round to avoid issues with very small volfracs
            t_blank = std::round(t_blank * 10000.0) / 10000.0;
            t_blank_below = std::round(t_blank_below * 10000.0) / 10000.0;
            t_blank_above = std::round(t_blank_above * 10000.0) / 10000.0;
            t_blank_north = std::round(t_blank_north * 10000.0) / 10000.0;
            t_blank_south = std::round(t_blank_south * 10000.0) / 10000.0;
            t_blank_east = std::round(t_blank_east * 10000.0) / 10000.0;
            t_blank_west = std::round(t_blank_west * 10000.0) / 10000.0;

            // Identify surface type (same logic as ERF_MakeMomSources.cpp)
            Real roof_mask  = (t_blank > 0.0 && t_blank < t_blank_below && t_blank_above == 0.0) ? 1.0 : 0.0;
            Real south_mask = (t_blank > 0.0 && t_blank <= t_blank_north && t_blank_south == 0.0) ? 1.0 : 0.0;
            Real north_mask = (t_blank > 0.0 && t_blank <= t_blank_south && t_blank_north == 0.0) ? 1.0 : 0.0;
            Real east_mask  = (t_blank > 0.0 && t_blank <= t_blank_west  && t_blank_east == 0.0) ? 1.0 : 0.0;
            Real west_mask  = (t_blank > 0.0 && t_blank <= t_blank_east  && t_blank_west == 0.0) ? 1.0 : 0.0;

            // Skip if not a building surface
            if (roof_mask == 0.0 && south_mask == 0.0 && north_mask == 0.0 &&
                east_mask == 0.0 && west_mask == 0.0) return;

            // Get atmospheric state
            Real rho = cons_arr(i,j,k,Rho_comp);
            Real theta = cons_arr(i,j,k,RhoTheta_comp) / rho;

            // Select velocity components based on surface orientation
            // Roof: u and v (horizontal velocities)
            // North/South walls: u and w (tangential to y-normal wall)
            // East/West walls: v and w (tangential to x-normal wall)
            Real u1, u2, u_tang;
            if (roof_mask > 0.0) {
                // Roof: horizontal wind speed from u,v
                Real ux = 0.5 * (u_arr(i,j,k) + u_arr(i+1,j,k));
                Real uy = 0.5 * (v_arr(i,j,k) + v_arr(i,j+1,k));
                u1 = ux;
                u2 = uy;
                u_tang = sqrt(u1*u1 + u2*u2);
            } else if (south_mask > 0.0 || north_mask > 0.0) {
                // North/South walls: u and w
                Real ux = 0.5 * (u_arr(i,j,k) + u_arr(i+1,j,k));
                Real uz = 0.5 * (w_arr(i,j,k) + w_arr(i,j,k+1));
                u1 = ux;
                u2 = uz;
                u_tang = sqrt(u1*u1 + u2*u2);
            } else if (east_mask > 0.0 || west_mask > 0.0) {
                // East/West walls: v and w
                Real uy = 0.5 * (v_arr(i,j,k) + v_arr(i,j+1,k));
                Real uz = 0.5 * (w_arr(i,j,k) + w_arr(i,j,k+1));
                u1 = uy;
                u2 = uz;
                u_tang = sqrt(u1*u1 + u2*u2);
            } else {
                u_tang = 0.1;  // Shouldn't reach here
            }
            u_tang = amrex::max(u_tang, 0.1);  // Minimum wind speed

            // Get radiation at the height of this building surface cell
            // Only read SW_dn and LW_dn (atmospheric/solar inputs)
            // Note: SW_up = albedo * SW_dn (accounted for in R_net calculation)
            //       LW_up = f(T_surf) (recomputed in Newton iteration below)
            Real sw_dn = 0.0;
            Real lw_dn = 0.0;
            if (has_radiation) {
                // Radiation fluxes: component 0=SW_up, 1=SW_dn, 2=LW_up, 3=LW_dn
                // Use radiation at height k (buildings are resolved in height)
                sw_dn = rad_arr(i,j,k,1);
                lw_dn = rad_arr(i,j,k,3);
            }

            // Simple shadow mask: check if there's a building above this cell
            // If shaded, zero out the direct shortwave component
            // This is a simplified approach - assumes vertical walls can be shaded from above
            bool is_shaded = false;
            if (roof_mask == 0.0) {  // Only walls can be shaded (not roofs)
                // Check if there's solid material above (building extending higher)
                // Scan upward to see if we encounter a solid cell
                for (int kk = k+1; kk < k+10; ++kk) {  // Check up to 10 cells above
                    Real t_above = t_blank_arr(i,j,kk);
                    if (t_above > 0.5) {  // Solid cell above
                        is_shaded = true;
                        break;
                    }
                }
            }

            // Apply shadow mask: reduce direct SW to zero if shaded
            // Note: This zeroes ALL SW, not just direct component
            // Could be refined to separate direct/diffuse SW components
            if (is_shaded) {
                sw_dn = 0.0;  // Shaded surface receives no direct SW
            }

            // Initial guess for surface temperature
            Real T_surf_old = T_surf_arr(i,j,k);
            Real T_surf_new = T_surf_old;

            // Newton-Raphson iteration
            for (int iter = 0; iter < max_iter; ++iter) {
                // 1. Net radiation
                Real lw_up = emissivity * sigma * pow(T_surf_new, 4.0);
                Real R_net = sw_dn * (1.0 - albedo) + lw_dn - lw_up;

                // 2. Sensible heat flux (simplified MOST)
                // H = rho * Cp * Ch * U * (T_surf - T_air)
                // where Ch ~ kappa² / (ln(z/z0))²
                Real z_ref = dz;  // Reference height
                Real Ch = (kappa * kappa) / pow(log(z_ref / z0), 2.0);
                Real H = rho * Cp_d_val * Ch * u_tang * (T_surf_new - theta);

                // 3. Latent heat flux (Dudhia scheme, simplified)
                // LE = rho * L_v * Ce * U * q_surf
                // For now, simplified version with moisture availability
                Real L_v = 2.5e6;  // Latent heat of vaporization [J/kg]
                Real Ce = Ch;      // Assume Ce = Ch (neutral conditions)
                Real q_sat = 0.0;  // Simplified - should compute saturation mixing ratio
                Real LE = rho * L_v * Ce * u_tang * moisture_avail * q_sat;

                // 4. Ground heat conduction
                Real G = k_concrete * (T_surf_new - T1_arr(i,j,k)) / dz;

                // 5. Energy balance residual
                Real residual = R_net - H - LE - G;

                // 6. Derivative of residual w.r.t. T_surf
                // d(residual)/dT = -4*emiss*sigma*T³ - rho*Cp*Ch*U - k/dz
                Real d_lw_up_dT = 4.0 * emissivity * sigma * pow(T_surf_new, 3.0);
                Real d_H_dT = rho * Cp_d_val * Ch * u_tang;
                Real d_G_dT = k_concrete / dz;
                Real derivative = -d_lw_up_dT - d_H_dT - d_G_dT;

                // 7. Newton step
                Real delta_T = -residual / derivative;

                // Limit temperature change
                delta_T = amrex::max(amrex::min(delta_T, 5.0), -5.0);

                T_surf_new += delta_T;

                // Check convergence
                if (abs(delta_T) < 0.01) break;
            }

            // Store converged surface temperature
            T_surf_arr(i,j,k) = T_surf_new;
        });
    }
}
