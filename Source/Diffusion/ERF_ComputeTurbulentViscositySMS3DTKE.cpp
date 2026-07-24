/*
============================================================================
SMS-3DTKE: Scale-Adaptive 3D Turbulent Kinetic Energy Scheme
============================================================================

Implementation of the SMS-3DTKE (Scale-adaptive Mellor-Sutton-Simpson 3D
Turbulent Kinetic Energy) scheme from Zhang et al. (2018), which provides
a unified framework for representing subgrid turbulent mixing across the
gray zone from LES to mesoscale resolutions.

PRIMARY REFERENCE:
------------------
Zhang, X., J.-W. Bao, B. Chen, and E. D. Grell, 2018: A Three-Dimensional
Scale-Adaptive Turbulent Kinetic Energy Scheme in the WRF-ARW Model.
Monthly Weather Review, 146, 2023-2045.
https://doi.org/10.1175/MWR-D-17-0356.1

WRF REFERENCE IMPLEMENTATION:
-----------------------------
WRF-ARW dyn_em/module_diffusion_em.F (diffusion_driver subroutine)
https://github.com/wrf-model/WRF

CORE METHODOLOGY:
-----------------
1. Prognostic TKE Equation with scale-adaptive mixing length and dissipation
2. Master mixing length: harmonic average of surface, integral, buoyancy lengths
3. Scale-adaptive partition functions: smooth blending of local/nonlocal transport
4. Nonlocal heat flux: piecewise linear profile fitted to LES results
5. Nonlocal momentum flux: countergradient term in vertical
6. Horizontal diffusion: pragmatic blend of Smagorinsky and TKE-based

KEY FEATURES:
-------------
- Seamless transition from LES (Δ << z_i) to mesoscale (Δ >> z_i) regimes
- Unified treatment: same TKE equation across all scales
- No arbitrary switching between schemes
- Resolves gray zone problem for 250m-3km resolutions

SCALE-ADAPTIVE BLENDING:
------------------------
Partition functions P_L, P_NL, P_TKE depend on Δ/z_i (grid size / PBL height):
- At LES limit (Δ << z_i): P_L→0, P_NL→0 → pure local eddy-diffusivity from TKE
- At mesoscale (Δ >> z_i): P_L→1, P_NL→1 → local + nonlocal (traditional PBL)

This allows single parameterization to work across resolutions without
arbitrary scheme switching.

IMPLEMENTATION NOTES:
---------------------
- Extends ERF's Deardorff TKE infrastructure
- Uses existing RhoKE_comp for prognostic TKE
- Leverages ERF_PBLHeight.H for PBL height diagnostics
- GPU-compatible using AMReX parallel constructs
- Supports terrain-fitted coordinates
- Moisture-aware via virtual potential temperature

PARAMETERS:
-----------
See TurbChoice::SMS3DTKEParams in ERF_TurbStruct.H for configurable parameters.
Default values match Zhang et al. (2018) specifications.

COMPARISON TO OTHER ERF SCHEMES:
--------------------------------
- Deardorff LES: SMS-3DTKE reduces to similar form at LES limit, but adds
  mesoscale capability and nonlocal transport
- PBL schemes (MRF, YSU): SMS-3DTKE incorporates nonlocal transport similar
  to these, but unified with LES closure via TKE
- MYNN: Both use TKE and MYNN-style length scales, but SMS-3DTKE adds
  scale-adaptive blending for gray zone

VALIDATION:
-----------
Zhang et al. (2018) validated against:
- Idealized dry CBL LES benchmarks (Section 5)
- Real-case fair-weather simulations (Section 6)
- Comparison with traditional 1D PBL schemes (MYJ, MYNN, YSU)
Showed improved performance in gray zone (500m-3km resolutions)

============================================================================
*/

#include "ERF_SurfaceLayer.H"
#include "ERF_DirectionSelector.H"
#include "ERF_Diffusion.H"
#include "ERF_EddyViscosity.H"
#include "ERF_Constants.H"
#include "ERF_TurbStruct.H"
#include "ERF_PBLModels.H"
#include "ERF_TileNoZ.H"
#include "ERF_MoistUtils.H"
#include "ERF_RichardsonNumber.H"
#include "ERF_PBLHeight.H"
#include "ERF_SMS3DTKE_Utils.H"

using namespace amrex;

/**
 * Compute turbulent viscosity using SMS-3DTKE scale-adaptive TKE scheme
 *
 * @param[in]  Tau_lev        Strain rate tensors (for horizontal diffusion blending)
 * @param[in]  cons_in        Cell-centered conserved quantities (includes RhoKE_comp)
 * @param[out] eddyViscosity  Turbulent diffusivities (output)
 * @param[out] Hfx1           Heat flux in x-direction (output)
 * @param[out] Hfx2           Heat flux in y-direction (output)
 * @param[out] Hfx3           Heat flux in z-direction (output, includes nonlocal)
 * @param[out] Diss           TKE dissipation rate (output)
 * @param[in]  geom           Problem geometry
 * @param[in]  use_terrain_fitted_coords  Terrain-fitted coordinate flag
 * @param[in]  mapfac         Map scale factors
 * @param[in]  z_phys_nd      Physical height at nodes
 * @param[in]  z_phys_cc      Physical height at cell centers
 * @param[in]  turbChoice     Turbulence parameters
 * @param[in]  const_grav     Gravitational acceleration
 * @param[in]  SurfLayer      Surface layer interface (for u*, θ*, w*)
 * @param[in]  moisture_indices  Moisture component indices
 * @param[in]  xvel           X-velocity (for strain rate)
 * @param[in]  yvel           Y-velocity (for strain rate)
 */
void
ComputeTurbulentViscositySMS3DTKE(Vector<std::unique_ptr<MultiFab>>& Tau_lev,
                                   const MultiFab& cons_in,
                                   MultiFab& eddyViscosity,
                                   MultiFab& Hfx1, MultiFab& Hfx2, MultiFab& Hfx3,
                                   MultiFab& Diss,
                                   const Geometry& geom,
                                   bool use_terrain_fitted_coords,
                                   Vector<std::unique_ptr<MultiFab>>& mapfac,
                                   const std::unique_ptr<MultiFab>& z_phys_nd,
                                   const std::unique_ptr<MultiFab>& z_phys_cc,
                                   const TurbChoice& turbChoice,
                                   const Real const_grav,
                                   std::unique_ptr<SurfaceLayer>& SurfLayer,
                                   const MoistureComponentIndices& moisture_indices,
                                   const MultiFab* /*xvel*/,
                                   const MultiFab* /*yvel*/)
{
    // Get geometry information
    const GpuArray<Real, AMREX_SPACEDIM> cellSizeInv = geom.InvCellSizeArray();
    const Box& domain = geom.Domain();

    // Get turbulence parameters
    const Real inv_Pr_t = turbChoice.Pr_t_inv;
    const Real l_abs_g = const_grav;

    // Get SMS-3DTKE parameters
    const SMS3DTKEParams& sms = turbChoice.sms3dtke;

    // Stratification type
    bool use_thetav_grad = (turbChoice.strat_type == StratType::thetav);
    bool use_thetal_grad = (turbChoice.strat_type == StratType::thetal);

    // Isotropic vs anisotropic mixing
    bool isotropic = turbChoice.mix_isotropic;

    // Reference theta
    const bool use_ref_theta = (turbChoice.theta_ref > 0);
    const Real l_inv_theta0 = (use_ref_theta) ? one / turbChoice.theta_ref : one;

    // von Kármán constant
    const Real kappa = KAPPA;

    //==========================================================================
    // Compute PBL height using MYNN hybrid method
    //==========================================================================
    MultiFab pblh_mf(cons_in.boxArray(), cons_in.DistributionMap(), 1, 1);
    pblh_mf.setVal(1000.0); // Default fallback value

    MYNNPBLH pblh_calc;
    pblh_calc.compute_pblh(geom, z_phys_cc.get(), &pblh_mf, cons_in,
                           nullptr, moisture_indices);

    //==========================================================================
    // Compute integral length scale L_T via column integration
    // Zhang et al. (2018), Eq. 30:
    //   L_T = α_2 * ∫[0 to z_i] e^(1/2) dz / ∫[0 to z_i] e^(-1/2) dz
    //==========================================================================
    MultiFab L_T_mf(cons_in.boxArray(), cons_in.DistributionMap(), 2, 1);
    L_T_mf.setVal(0.0); // Will accumulate integrals: component 0 = numerator, 1 = denominator

    for (MFIter mfi(L_T_mf, TileNoZ()); mfi.isValid(); ++mfi)
    {
        const Box& domain_box = geom.Domain();
        Box gtbx = mfi.growntilebox();
        gtbx.setSmall(2, domain_box.smallEnd(2));
        gtbx.setBig(2, domain_box.bigEnd(2));

        const auto cons_arr = cons_in.const_array(mfi);
        const auto pblh_arr = pblh_mf.const_array(mfi);
        auto L_T_arr = L_T_mf.array(mfi);

        if (z_phys_cc) {
            const auto z_cc = z_phys_cc->const_array(mfi);

            // Column integration with terrain
            ParallelFor(gtbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
            {
                Real zi = pblh_arr(i, j, 0);
                Real z_agl = z_cc(i, j, k);

                // Only integrate within PBL
                if (z_agl <= zi) {
                    Real rho = cons_arr(i, j, k, Rho_comp);
                    Real e = amrex::max(cons_arr(i, j, k, RhoKE_comp) / rho, Real(0.0));
                    Real sqrt_e = std::sqrt(e);
                    Real inv_sqrt_e = (e > Real(1.0e-6)) ? (one / sqrt_e) : Real(0.0);

                    Real dz = (k > 0) ? (z_cc(i, j, k) - z_cc(i, j, k-1)) : z_cc(i, j, 0);

                    // Accumulate integrals: numerator = ∫sqrt(e) dz, denominator = ∫1/sqrt(e) dz
                    atomicAdd(&L_T_arr(i, j, 0), sqrt_e * dz);
                    atomicAdd(&L_T_arr(i, j, 1), inv_sqrt_e * dz); // Need 2 components!
                }
            });
        } else {
            // Uniform grid
            const Real dz = geom.CellSize(2);

            ParallelFor(gtbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
            {
                Real zi = pblh_arr(i, j, 0);
                Real z_agl = (k + myhalf) * dz;

                if (z_agl <= zi) {
                    Real rho = cons_arr(i, j, k, Rho_comp);
                    Real e = amrex::max(cons_arr(i, j, k, RhoKE_comp) / rho, Real(0.0));
                    Real sqrt_e = std::sqrt(e);
                    Real inv_sqrt_e = (e > Real(1.0e-6)) ? (one / sqrt_e) : Real(0.0);

                    atomicAdd(&L_T_arr(i, j, 0), sqrt_e * dz);
                    atomicAdd(&L_T_arr(i, j, 1), inv_sqrt_e * dz);
                }
            });
        }
    }

    // Finalize: L_T = α_2 * (numerator / denominator)
    for (MFIter mfi(L_T_mf); mfi.isValid(); ++mfi)
    {
        auto L_T_arr = L_T_mf.array(mfi);
        const auto pblh_arr = pblh_mf.const_array(mfi);
        Box bx = mfi.tilebox();

        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int) noexcept
        {
            Real numerator = L_T_arr(i, j, 0);
            Real denominator = L_T_arr(i, j, 1);
            Real pblh_val = pblh_arr(i, j, 0);

            if (denominator > Real(1.0e-6) && numerator > Real(1.0e-6)) {
                L_T_arr(i, j, 0) = sms.alpha_2 * numerator / denominator;
            } else {
                // Fallback to proportional estimate
                L_T_arr(i, j, 0) = sms.alpha_2 * pblh_val;
            }
            // Clear component 1 (no longer needed)
            L_T_arr(i, j, 1) = zero;
        });
    }

    //==========================================================================
    // Get surface layer parameters if available
    //==========================================================================
    const MultiFab* u_star_mf = nullptr;
    const MultiFab* t_star_mf = nullptr;
    const MultiFab* w_star_mf = nullptr;

    if (SurfLayer) {
        u_star_mf = SurfLayer->get_u_star(0);
        t_star_mf = SurfLayer->get_t_star(0);
        w_star_mf = SurfLayer->get_w_star(0);
    }

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(eddyViscosity, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        Box bxcc = mfi.tilebox();

        // Get arrays
        const Array4<Real>& mu_turb = eddyViscosity.array(mfi);
        const Array4<Real>& hfx_x = Hfx1.array(mfi);
        const Array4<Real>& hfx_y = Hfx2.array(mfi);
        const Array4<Real>& hfx_z = Hfx3.array(mfi);
        const Array4<Real>& diss = Diss.array(mfi);

        const Array4<Real const>& cell_data = cons_in.array(mfi);

        Array4<Real const> mf_u = mapfac[MapFacType::u_x]->const_array(mfi);
        Array4<Real const> mf_v = mapfac[MapFacType::v_y]->const_array(mfi);

        Array4<Real const> z_nd_arr = z_phys_nd->const_array(mfi);
        Array4<Real const> z_cc_arr = z_phys_cc->const_array(mfi);

        // Get strain rate tensors for horizontal diffusion (Zhang et al. Eq. 40)
        Array4<Real const> tau11 = Tau_lev[TauType::tau11]->const_array(mfi);
        Array4<Real const> tau22 = Tau_lev[TauType::tau22]->const_array(mfi);
        Array4<Real const> tau12 = Tau_lev[TauType::tau12]->const_array(mfi);

        // Get PBL height, integral length, and surface parameters
        Array4<Real const> pblh_arr = pblh_mf.const_array(mfi);
        Array4<Real const> L_T_arr = L_T_mf.const_array(mfi);

        Array4<Real const> u_star_arr = (u_star_mf) ? u_star_mf->const_array(mfi) : Array4<Real const>{};
        Array4<Real const> t_star_arr = (t_star_mf) ? t_star_mf->const_array(mfi) : Array4<Real const>{};
        Array4<Real const> w_star_arr = (w_star_mf) ? w_star_mf->const_array(mfi) : Array4<Real const>{};

        bool have_surface_params = (u_star_mf && t_star_mf && w_star_mf);

        ParallelFor(bxcc, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
        {
            //==================================================================
            // STEP 1: Compute grid spacing and filter width
            //==================================================================
            Real dxInv = cellSizeInv[0];
            Real dyInv = cellSizeInv[1];
            Real dzInv = cellSizeInv[2];

            if (use_terrain_fitted_coords) {
                dzInv /= Compute_h_zeta_AtCellCenter(i, j, k, cellSizeInv, z_nd_arr);
            }

            // Horizontal grid spacing
            Real dx = one / (dxInv * mf_u(i, j, 0));
            Real dy = one / (dyInv * mf_v(i, j, 0));
            Real Delta_h = std::sqrt(dx * dy);

            // Filter width for LES
            Real Delta;
            if (isotropic) {
                Real cellVolMsf = one / (dxInv * mf_u(i, j, 0) * dyInv * mf_v(i, j, 0) * dzInv);
                Delta = std::cbrt(cellVolMsf);
            } else {
                Delta = one / dzInv;  // Vertical grid spacing
            }

            //==================================================================
            // STEP 2: Get TKE and basic quantities
            //==================================================================
            Real rho = cell_data(i, j, k, Rho_comp);
            Real E = amrex::max(cell_data(i, j, k, RhoKE_comp) / rho, Real(0.0));
            Real sqrt_E = std::sqrt(E);

            // Get height above surface
            Real z_agl = z_cc_arr(i, j, k); // Assuming z_phys_cc is height AGL
            z_agl = amrex::max(z_agl, Real(1.0)); // Minimum 1m above surface

            //==================================================================
            // STEP 3: Compute stratification (Brunt-Väisälä frequency)
            //==================================================================
            Real dtheta_dz;
            if (use_thetav_grad) {
                dtheta_dz = myhalf * (GetThetav(i, j, k + 1, cell_data, moisture_indices) -
                                      GetThetav(i, j, k - 1, cell_data, moisture_indices)) * dzInv;
            } else if (use_thetal_grad) {
                dtheta_dz = myhalf * (GetThetal(i, j, k + 1, cell_data, moisture_indices) -
                                      GetThetal(i, j, k - 1, cell_data, moisture_indices)) * dzInv;
            } else {
                dtheta_dz = myhalf * (cell_data(i, j, k + 1, RhoTheta_comp) / cell_data(i, j, k + 1, Rho_comp) -
                                      cell_data(i, j, k - 1, RhoTheta_comp) / cell_data(i, j, k - 1, Rho_comp)) * dzInv;
            }

            Real stratification = l_abs_g * dtheta_dz * l_inv_theta0;
            if (!use_ref_theta) {
                stratification *= rho / cell_data(i, j, k, RhoTheta_comp);
            }

            Real N_sq = stratification; // N² = (g/θ₀) * ∂θ/∂z
            Real N = (N_sq > zero) ? std::sqrt(N_sq) : zero;

            //==================================================================
            // STEP 4: Get PBL height and surface parameters
            //==================================================================
            // PBL height from hybrid theta-increase + TKE method (MYNNPBLH)
            Real zi = pblh_arr(i, j, 0);
            zi = amrex::max(zi, Real(10.0)); // Minimum 10m

            // Surface layer parameters from MOST
            Real u_star = Real(0.3);   // Default friction velocity [m/s]
            Real t_star = Real(0.0);   // Default temperature scale [K]
            Real w_star = Real(1.0);   // Default convective velocity [m/s]

            if (have_surface_params) {
                u_star = amrex::max(u_star_arr(i, j, 0), Real(1.0e-3));
                t_star = t_star_arr(i, j, 0);
                w_star = amrex::max(w_star_arr(i, j, 0), Real(1.0e-3));
            }

            // Compute Δ/z_i ratio for partition functions
            Real Delta_over_zi = Delta_h / zi;

            //==================================================================
            // STEP 5: Compute partition functions
            //==================================================================
            Real P_L = SMS3DTKE_Partition_Local(Delta_over_zi);
            Real P_NL = (sms.use_nonlocal_heat) ? SMS3DTKE_Partition_Nonlocal(Delta_over_zi) : zero;
            Real P_TKE = SMS3DTKE_Partition_TKE(Delta_over_zi);

            //==================================================================
            // STEP 6: Compute master mixing length components
            //==================================================================

            // Surface layer length (Zhang et al. 2018, Eq. 29)
            Real L_S = SMS3DTKE_Length_Surface(z_agl, kappa, sms.alpha_s);

            // Integral length (Zhang et al. 2018, Eq. 30)
            // From column integration: L_T = α_2 * ∫√e dz / ∫(1/√e) dz
            Real L_T = L_T_arr(i, j, 0);

            // Buoyancy length (Zhang et al. 2018, Eq. 31)
            Real alpha_k_meso = sms.c_k2; // Mesoscale mixing coefficient
            Real L_B = SMS3DTKE_Length_Buoyancy(E, N, L_T, l_abs_g / l_inv_theta0,
                                                 alpha_k_meso, sms.alpha_1);

            // Minimum length (Zhang et al. 2018, Eq. 32)
            Real L_f = SMS3DTKE_Length_Minimum(E, N, sms.alpha_4);

            // Mesoscale vertical mixing length (from MYNN-style)
            Real l_Meso = L_T; // Simplified

            // Harmonic average (Zhang et al. 2018, Eq. 28)
            Real l_v = SMS3DTKE_Harmonic_Length(l_Meso, L_S, L_T, L_B);

            // LES mixing length (for scale-adaptive blending)
            Real l_LES = Delta;
            if (stratification > zero) {
                // Stratification-dependent (similar to Deardorff)
                Real l_strat = Real(0.76) * sqrt_E / std::sqrt(stratification);
                l_LES = amrex::min(l_strat, Delta);
                l_LES = amrex::max(l_LES, Real(0.001) * Delta);
            }

            // Scale-adaptive mixing length (Zhang et al. 2018, Eq. 37)
            Real L_Delta = P_L * l_Meso + (one - P_L) * l_LES;

            //==================================================================
            // STEP 7: Compute local eddy diffusivities
            //==================================================================

            // Vertical eddy viscosity (Zhang et al. 2018, similar to Eq. 5)
            Real alpha_k = (one - P_L) * sms.c_k1 + P_L * sms.c_k2;
            Real K_m_v = alpha_k * L_Delta * sqrt_E;
            Real K_h_v = K_m_v * inv_Pr_t;

            // Horizontal eddy diffusivity
            Real K_m_h, K_h_h;
            if (sms.use_horizontal_blend) {
                // Zhang et al. 2018, Eq. 40: blend Smagorinsky and TKE-based
                // Extract horizontal strain components (interpolated to cell center)
                Real s11 = tau11(i, j, k);
                Real s22 = tau22(i, j, k);
                Real s12 = fourth * (tau12(i, j, k) + tau12(i, j+1, k) +
                                     tau12(i+1, j, k) + tau12(i+1, j+1, k));

                // Horizontal deformation rate D_h
                Real D_h = SMS3DTKE_Horizontal_Deformation(s11, s22, s12);

                // Smagorinsky component: K_D = (c_h * l_v)² * D_h
                Real K_D = sms.c_h * sms.c_h * l_v * l_v * D_h;

                // TKE component: K_T = c_k * l_v * √e
                Real K_T = alpha_k * l_v * sqrt_E;

                // Blend: K_h = P_L * K_D + (1 - P_L) * K_T
                K_m_h = P_L * K_D + (one - P_L) * K_T;
                K_h_h = K_m_h * inv_Pr_t;
            } else {
                // Simple TKE-based without deformation blending
                Real DeltaH = (isotropic) ? L_Delta : Delta_h;
                K_m_h = alpha_k * DeltaH * sqrt_E;
                K_h_h = K_m_h * inv_Pr_t;
            }

            // Store eddy diffusivities
            mu_turb(i, j, k, EddyDiff::Mom_v) = rho * K_m_v;
            mu_turb(i, j, k, EddyDiff::Mom_h) = rho * K_m_h;
            mu_turb(i, j, k, EddyDiff::Theta_v) = rho * K_h_v;
            mu_turb(i, j, k, EddyDiff::Theta_h) = rho * K_h_h;
            mu_turb(i, j, k, EddyDiff::Turb_lengthscale) = L_Delta;

            //==================================================================
            // STEP 8: Compute nonlocal heat flux
            //==================================================================

            Real hfx_nonlocal = zero;
            if (sms.use_nonlocal_heat && P_NL > Real(1.0e-6)) {
                // Zhang et al. 2018, Eq. 23-24
                // Surface heat flux from MOST: H = -ρ C_p w'θ' ≈ ρ u* θ*
                // For buoyancy flux in TKE equation: (ρ w'θ') / ρ = u* θ*
                Real hfx_sfc = u_star * t_star; // [K m/s]

                Real z_star = z_agl / zi;
                hfx_nonlocal = P_NL * SMS3DTKE_Nonlocal_Heat_Profile(z_star, zi, w_star,
                                                                      hfx_sfc, sms);
            }

            //==================================================================
            // STEP 9: Compute total heat flux
            //==================================================================

            // Local (gradient) part
            Real hfx_local = -mu_turb(i, j, k, EddyDiff::Theta_v) * dtheta_dz;

            // Total vertical heat flux = local + nonlocal
            hfx_x(i, j, k) = zero;
            hfx_y(i, j, k) = zero;
            hfx_z(i, j, k) = hfx_local + rho * hfx_nonlocal;

            //==================================================================
            // STEP 10: Compute scale-adaptive dissipation
            //==================================================================

            // Zhang et al. 2018, Eq. 38-39
            Real c_eps1 = SMS3DTKE_Dissipation_Coefficient_LES(l_LES, Delta,
                                                                sms.c_eps1_base,
                                                                sms.c_eps1_slope);
            Real c_eps2 = sms.c_eps2;

            // Scale-adaptive dissipation length
            Real L_eps_LES = c_eps1 * l_LES;
            Real L_eps_Meso = c_eps2 * l_Meso;
            Real L_eps_Delta = P_TKE * L_eps_Meso + (one - P_TKE) * L_eps_LES;

            // Dissipation rate: ε = e^(3/2) / L_ε,Δ
            Real eps = std::numeric_limits<Real>::epsilon();
            diss(i, j, k) = rho * std::pow(E, Real(1.5)) /
                            amrex::max(L_eps_Delta, eps);
        });
    }
}
