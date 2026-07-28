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
                                       const MultiFab* z_phys_cc,
                                       MultiFab* shadow_mask,
                                       const GpuArray<Real,3>& dx_arr,
                                       const Real& sun_azimuth_deg,
                                       const Real& sun_zenith_deg,
                                       const Real& time,
                                       const Real& dt_advance)
{
    m_dt = dt_advance;

    // Solve surface energy balance to get updated surface temperature
    Solve_Surface_Energy_Balance(cons_in, xvel_in, yvel_in, zvel_in,
                                  rad_fluxes, terrain_blank, z_phys_cc, shadow_mask,
                                  dx_arr, sun_azimuth_deg, sun_zenith_deg, time);

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
    const MultiFab* z_phys_cc,
    MultiFab* shadow_mask,
    const GpuArray<Real,3>& dx_arr,
    Real sun_azimuth_deg,
    Real sun_zenith_deg,
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

        // Physical grid heights (cell-centered)
        const Array4<const Real>& z_cc_arr = (z_phys_cc) ? z_phys_cc->const_array(mfi)
                                                         : Array4<const Real>{};

        // Radiation fluxes (SW_up, SW_dn, LW_up, LW_dn at each k level)
        const Array4<const Real>& rad_arr = (rad_fluxes) ? rad_fluxes->const_array(mfi)
                                                         : Array4<const Real>{};
        const bool has_radiation = (rad_fluxes != nullptr);

        // Shadow mask output
        const Array4<Real>& shadow_arr = (shadow_mask) ? shadow_mask->array(mfi)
                                                       : Array4<Real>{};
        const bool compute_shadow = (shadow_mask != nullptr);

        // Sun angles for shadow calculation
        const Real sun_az = sun_azimuth_deg;
        const Real sun_zen = sun_zenith_deg;

        const Real albedo = m_albedo;
        const Real emissivity = m_emissivity;
        const Real moisture_avail = m_moisture_avail;
        const Real k_concrete = m_k;
        const Real dz_subsurface = m_dz;  // Subsurface layer spacing

        // Domain bounds for shadow mask
        const auto& domain = cons_in.boxArray().minimalBox();
        const int k_max = domain.bigEnd(2);

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

            // Shadow mask via horizon angle method (WRF-style)
            // Scan surrounding cells in sun direction to find max elevation angle
            // If any neighbor blocks the sun path, this cell is shaded
            bool is_shaded = false;
            if (compute_shadow && has_radiation) {
                const Real PI = 3.14159265358979323846;
                Real zen_rad = sun_zen * (PI / 180.0);

                // Skip if sun is too low (below horizon or grazing)
                if (zen_rad > 80.0 * PI / 180.0) {
                    is_shaded = true;  // Sun too low
                } else {
                    // Shadow azimuth (opposite of sun)
                    Real shadow_az = fmod(sun_az + 180.0, 360.0);
                    Real theta = shadow_az * (PI / 180.0);

                    // Max shadow length (like WRF's gpshad)
                    // Use max 25000m / cell_size, or cap at 50 cells for efficiency
                    const Real dx_horiz = sqrt(dx_arr[0]*dx_arr[0] + dx_arr[1]*dx_arr[1]);
                    const int max_shadow_cells = amrex::min(50, int(25000.0 / dx_horiz));

                    // Get this cell's physical height (use z_cc if available)
                    Real cell_height;
                    if (z_cc_arr) {
                        cell_height = z_cc_arr(i,j,k);
                    } else {
                        // Fall back to uniform grid
                        cell_height = (Real(k) + 0.5) * dx_arr[2];
                    }

                    // Scan in shadow direction (opposite of sun)
                    Real tan_zen = tan(zen_rad);
                    for (int step = 1; step <= max_shadow_cells; ++step) {
                        // Step in shadow direction (horizontal)
                        Real scan_i = Real(i) + step * sin(theta);
                        Real scan_j = Real(j) + step * cos(theta);

                        int ii = int(round(scan_i));
                        int jj = int(round(scan_j));

                        // Check horizontal bounds
                        if (!bx.contains(IntVect(ii, jj, k))) break;

                        // Horizontal distance from this cell
                        Real di = Real(ii - i) * dx_arr[0];
                        Real dj = Real(jj - j) * dx_arr[1];
                        Real dist_horiz = sqrt(di*di + dj*dj);

                        // Scan vertically at this (i,j) location
                        // Find highest solid cell (building top)
                        Real neighbor_height = 0.0;
                        for (int kk = 0; kk <= k_max; ++kk) {
                            Real t_neighbor = t_blank_arr(ii, jj, kk);
                            if (t_neighbor > 0.5) {
                                // Get physical height of top of this cell
                                if (z_cc_arr) {
                                    Real dz_cell = (kk > 0) ? (z_cc_arr(ii,jj,kk) - z_cc_arr(ii,jj,kk-1))
                                                            : dx_arr[2];
                                    neighbor_height = z_cc_arr(ii,jj,kk) + 0.5*dz_cell;
                                } else {
                                    neighbor_height = Real(kk+1) * dx_arr[2];
                                }
                            }
                        }

                        // Check if this neighbor blocks sun
                        // Elevation angle from this cell to neighbor top
                        Real elev_angle = atan((neighbor_height - cell_height) / dist_horiz);
                        Real sun_elev = PI/2.0 - zen_rad;  // Sun elevation above horizon

                        if (elev_angle > sun_elev) {
                            is_shaded = true;
                            break;
                        }
                    }
                }
            }

            // Store shadow mask for output/verification
            if (compute_shadow) {
                shadow_arr(i,j,k) = is_shaded ? 1.0 : 0.0;
            }

            // Apply shadow: zero SW_dn if shaded
            // Note: This zeros ALL SW (direct + diffuse)
            // Could be refined to only zero direct component
            if (is_shaded) {
                sw_dn = 0.0;
            }

            // Get cell vertical spacing for MOST reference height
            Real dz_cell;
            if (z_cc_arr && k > 0) {
                dz_cell = z_cc_arr(i,j,k) - z_cc_arr(i,j,k-1);
            } else {
                dz_cell = dx_arr[2];
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
                Real z_ref = dz_cell;  // Reference height (actual cell spacing)
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
                Real d_G_dT = k_concrete / dz_subsurface;
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
