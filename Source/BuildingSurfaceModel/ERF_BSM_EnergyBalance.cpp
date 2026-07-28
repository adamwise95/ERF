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
                                       const Real& time,
                                       const Real& dt_advance)
{
    m_dt = dt_advance;

    // Note: terrain_blank should be passed through the interface
    // For now, we'll note this needs to be added to the function signature
    // or accessed from ERF class

    // Solve surface energy balance to get updated surface temperature
    Solve_Surface_Energy_Balance(cons_in, xvel_in, yvel_in, zvel_in,
                                  rad_fluxes, nullptr, time);

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
    const Real Cp_d = Cp_d;            // Specific heat at constant pressure
    const Real kappa = KAPPA;          // von Karman constant
    const Real z0 = 0.01;              // Roughness length [m] - should be configurable
    const Real max_iter = 5;           // Maximum Newton iterations

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

        // Radiation fluxes at surface (k=0 in rad_fluxes corresponds to surface)
        const Array4<const Real>& rad_arr = (rad_fluxes) ? rad_fluxes->const_array(mfi)
                                                         : Array4<const Real>{};

        const Real albedo = m_albedo;
        const Real emissivity = m_emissivity;
        const Real moisture_avail = m_moisture_avail;
        const Real k_concrete = m_k;
        const Real dz = m_dz;

        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            // Skip if not a building surface cell
            // TODO: Add terrain_blank masking here (roof_mask, wall_masks)
            // For now, process all cells

            // Get atmospheric state
            Real rho = cons_arr(i,j,k,Rho_comp);
            Real theta = cons_arr(i,j,k,RhoTheta_comp) / rho;

            // Get velocity components
            // TODO: Select appropriate velocity pair based on surface orientation
            // Roof: u,v; North/South walls: u,w; East/West walls: v,w
            Real ux = 0.5 * (u_arr(i,j,k) + u_arr(i+1,j,k));
            Real uy = 0.5 * (v_arr(i,j,k) + v_arr(i,j+1,k));
            Real u_horiz = sqrt(ux*ux + uy*uy);
            u_horiz = amrex::max(u_horiz, 0.1);  // Minimum wind speed

            // Get radiation at surface
            Real sw_dn = 0.0;
            Real lw_dn = 0.0;
            if (rad_arr) {
                // Radiation fluxes: component 0=SW_up, 1=SW_dn, 2=LW_up, 3=LW_dn
                // At surface k=0
                sw_dn = rad_arr(i,j,0,1);
                lw_dn = rad_arr(i,j,0,3);
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
                Real H = rho * Cp_d * Ch * u_horiz * (T_surf_new - theta);

                // 3. Latent heat flux (Dudhia scheme, simplified)
                // LE = rho * L_v * Ce * U * q_surf
                // For now, simplified version with moisture availability
                Real L_v = 2.5e6;  // Latent heat of vaporization [J/kg]
                Real Ce = Ch;      // Assume Ce = Ch (neutral conditions)
                Real q_sat = 0.0;  // Simplified - should compute saturation mixing ratio
                Real LE = rho * L_v * Ce * u_horiz * moisture_avail * q_sat;

                // 4. Ground heat conduction
                Real G = k_concrete * (T_surf_new - T1_arr(i,j,k)) / dz;

                // 5. Energy balance residual
                Real residual = R_net - H - LE - G;

                // 6. Derivative of residual w.r.t. T_surf
                // d(residual)/dT = -4*emiss*sigma*T³ - rho*Cp*Ch*U - k/dz
                Real d_lw_up_dT = 4.0 * emissivity * sigma * pow(T_surf_new, 3.0);
                Real d_H_dT = rho * Cp_d * Ch * u_horiz;
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
