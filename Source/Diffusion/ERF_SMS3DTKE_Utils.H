#ifndef ERF_SMS3DTKE_UTILS_H_
#define ERF_SMS3DTKE_UTILS_H_

#include <AMReX_REAL.H>
#include <AMReX_GpuQualifiers.H>
#include <ERF_TurbStruct.H>
#include <ERF_Constants.H>
#include <cmath>

/**
 * @file ERF_SMS3DTKE_Utils.H
 * @brief GPU-inline helper functions for SMS-3DTKE scale-adaptive TKE scheme
 *
 * Contains utility functions for the SMS-3DTKE (Scale-adaptive Mellor-Sutton
 * -Simpson 3D Turbulent Kinetic Energy) scheme from Zhang et al. (2018).
 *
 * All functions are AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE for GPU compatibility.
 *
 * Primary Reference:
 *   Zhang, X., J.-W. Bao, B. Chen, and E. D. Grell, 2018: A Three-Dimensional
 *   Scale-Adaptive Turbulent Kinetic Energy Scheme in the WRF-ARW Model.
 *   Monthly Weather Review, 146, 2023-2045.
 *   https://doi.org/10.1175/MWR-D-17-0356.1
 */

//==============================================================================
// PARTITION FUNCTIONS (Zhang et al. 2018, Fig. 3)
//==============================================================================

/**
 * @brief Local heat flux partition function P_L(Δ/z_i)
 *
 * Controls the weight of local (eddy-diffusivity) transport in the
 * scale-adaptive blending. Based on Zhang et al. (2018), Fig. 3.
 *
 * Asymptotic behavior:
 *   - P_L → 0 as Δ/z_i → 0 (LES limit: no PBL local diffusivity)
 *   - P_L → 1 as Δ/z_i → ∞ (mesoscale limit: full PBL local diffusivity)
 *
 * @param Delta_over_zi  Ratio of horizontal grid spacing to PBL height
 * @return               Partition value in [0, 1]
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real SMS3DTKE_Partition_Local(amrex::Real Delta_over_zi) noexcept
{
    // Smooth transition using tanh, tuned to match Zhang et al. 2018 Fig. 3
    // Transition centered at Δ/z_i ≈ 1
    amrex::Real x = std::log10(amrex::max(Delta_over_zi, amrex::Real(0.01)));
    amrex::Real x0 = amrex::Real(0.0);  // Transition center (log10(1) = 0)
    amrex::Real width = amrex::Real(1.0); // Transition width
    return amrex::Real(0.5) * (amrex::Real(1.0) + std::tanh((x - x0) / width));
}

/**
 * @brief Nonlocal heat flux partition function P_NL(Δ/z_i)
 *
 * Controls the weight of nonlocal (countergradient) transport in the
 * scale-adaptive blending. Based on Zhang et al. (2018), Fig. 3.
 *
 * Asymptotic behavior:
 *   - P_NL → 0 as Δ/z_i → 0 (LES limit: no nonlocal flux)
 *   - P_NL → 1 as Δ/z_i → ∞ (mesoscale limit: full nonlocal flux)
 *
 * @param Delta_over_zi  Ratio of horizontal grid spacing to PBL height
 * @return               Partition value in [0, 1]
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real SMS3DTKE_Partition_Nonlocal(amrex::Real Delta_over_zi) noexcept
{
    // Similar to P_L, may have slightly different coefficients
    // For now, use same form as P_L (can be tuned based on WRF implementation)
    return SMS3DTKE_Partition_Local(Delta_over_zi);
}

/**
 * @brief TKE dissipation partition function P_TKE(Δ/z_i)
 *
 * Controls the blending of LES and mesoscale dissipation length scales.
 * Based on Zhang et al. (2018), Fig. 3.
 *
 * Used to compute:
 *   L_ε,Δ = P_TKE * L_ε,meso + (1 - P_TKE) * L_ε,LES
 *
 * @param Delta_over_zi  Ratio of horizontal grid spacing to PBL height
 * @return               Partition value in [0, 1]
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real SMS3DTKE_Partition_TKE(amrex::Real Delta_over_zi) noexcept
{
    // Similar sigmoid transition
    return SMS3DTKE_Partition_Local(Delta_over_zi);
}

//==============================================================================
// MASTER MIXING LENGTH COMPONENTS (Zhang et al. 2018, Eq. 28-33)
//==============================================================================

/**
 * @brief Integral length scale L_T via vertical integration
 *
 * Zhang et al. (2018), Eq. 30:
 *   L_T = α_2 * ∫[0 to z_i] e^(1/2) dz / ∫[0 to z_i] e^(-1/2) dz
 *
 * Simplified form for column computation:
 *   L_T ≈ α_2 * Σ[e_k^(1/2) * Δz_k] / Σ[e_k^(-1/2) * Δz_k]
 *
 * @param e         TKE at current level [m²/s²]
 * @param zi        PBL height [m]
 * @param z_agl     Height above ground [m]
 * @param alpha_2   Integral length coefficient (default 0.9)
 * @return          Integral length scale [m]
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real SMS3DTKE_Length_Integral_Simple(amrex::Real e,
                                             amrex::Real zi,
                                             amrex::Real z_agl,
                                             amrex::Real alpha_2) noexcept
{
    // Simplified: use characteristic length ~ PBL height
    // For proper implementation, would need vertical integration
    // over TKE profile from surface to z_i
    return alpha_2 * zi;
}

/**
 * @brief Surface layer mixing length L_S
 *
 * Zhang et al. (2018), Eq. 29:
 *   L_S = α_s * κ * z
 *
 * @param z        Height above surface [m]
 * @param kappa    von Kármán constant (≈ 0.4)
 * @param alpha_s  Surface layer coefficient (default 1.0)
 * @return         Surface layer length scale [m]
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real SMS3DTKE_Length_Surface(amrex::Real z,
                                     amrex::Real kappa,
                                     amrex::Real alpha_s) noexcept
{
    return alpha_s * kappa * z;
}

/**
 * @brief Buoyancy mixing length L_B
 *
 * Zhang et al. (2018), Eq. 31:
 *   L_B = [α_k * e^(1/2) + α_1 * e^(1/2) * (g_s/L_T/N)^(1/2)] / N  for N² > 0
 *   L_B = ∞  for N² ≤ 0 (handled by returning large value)
 *
 * @param e         Turbulent kinetic energy [m²/s²]
 * @param N         Brunt-Väisälä frequency [1/s] (√(N²))
 * @param L_T       Integral length scale [m]
 * @param g_s       Gravitational acceleration / reference temperature [m/s²/K]
 * @param alpha_k   TKE coefficient (c_k1 or c_k2)
 * @param alpha_1   Buoyancy length coefficient (default 1.9)
 * @return          Buoyancy length scale [m]
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real SMS3DTKE_Length_Buoyancy(amrex::Real e,
                                      amrex::Real N,
                                      amrex::Real L_T,
                                      amrex::Real g_s,
                                      amrex::Real alpha_k,
                                      amrex::Real alpha_1) noexcept
{
    amrex::Real eps = std::numeric_limits<amrex::Real>::epsilon();

    // For unstable or neutral stratification, return large value
    if (N <= eps) {
        return amrex::Real(1.0e10);
    }

    amrex::Real sqrt_e = std::sqrt(amrex::max(e, amrex::Real(0.0)));
    amrex::Real term1 = alpha_k * sqrt_e;
    amrex::Real term2 = alpha_1 * sqrt_e * std::sqrt(amrex::max(g_s / (L_T * N), amrex::Real(0.0)));

    return (term1 + term2) / N;
}

/**
 * @brief Minimum mixing length L_f
 *
 * Zhang et al. (2018), Eq. 32:
 *   L_f = α_4 * e^(1/2) / N  for N² > 0
 *   L_f = ∞  for N² ≤ 0
 *
 * @param e        Turbulent kinetic energy [m²/s²]
 * @param N        Brunt-Väisälä frequency [1/s]
 * @param alpha_4  Minimum length coefficient (default 5.0)
 * @return         Minimum length scale [m]
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real SMS3DTKE_Length_Minimum(amrex::Real e,
                                     amrex::Real N,
                                     amrex::Real alpha_4) noexcept
{
    amrex::Real eps = std::numeric_limits<amrex::Real>::epsilon();

    if (N <= eps) {
        return amrex::Real(1.0e10);
    }

    return alpha_4 * std::sqrt(amrex::max(e, amrex::Real(0.0))) / N;
}

//==============================================================================
// NONLOCAL FLUX PROFILES (Zhang et al. 2018, Eq. 23-24)
//==============================================================================

/**
 * @brief Nonlocal heat flux profile w'θ'^NL(z)
 *
 * Piecewise linear profile fitted to LES results.
 * Zhang et al. (2018), Eq. 23-24:
 *
 * Three zones:
 *   1. Surface layer:     z < δ_SBL * z_i
 *   2. Mixed layer:       δ_SBL * z_i < z < (1 - δ_EZ) * z_i
 *   3. Entrainment zone:  (1 - δ_EZ) * z_i < z < z_i
 *
 * @param z_star    Normalized height z/z_i
 * @param zi        PBL height [m]
 * @param w_star    Convective velocity scale [m/s]
 * @param hfx_sfc   Surface heat flux [K m/s]
 * @param params    SMS3DTKE parameters (contains δ_SBL, d1_EZ, d2_EZ)
 * @return          Nonlocal heat flux [K m/s]
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real SMS3DTKE_Nonlocal_Heat_Profile(amrex::Real z_star,
                                            amrex::Real zi,
                                            amrex::Real w_star,
                                            amrex::Real hfx_sfc,
                                            const SMS3DTKEParams& params) noexcept
{
    // Normalized height must be in [0, 1]
    z_star = amrex::max(amrex::Real(0.0), amrex::min(z_star, amrex::Real(1.0)));

    // Zone boundaries (from paper)
    amrex::Real delta_SBL = params.delta_SBL; // ~0.08
    amrex::Real delta_EZ_lo = amrex::Real(1.0) - params.d1_EZ - params.d2_EZ; // ~0.93
    amrex::Real delta_EZ_hi = amrex::Real(1.0) - params.d2_EZ;                // ~0.95

    amrex::Real flux;

    if (z_star <= delta_SBL) {
        // Surface layer: linear from surface flux to mixed layer value
        amrex::Real flux_ML = w_star * hfx_sfc / amrex::max(w_star, amrex::Real(1.0e-6));
        flux = hfx_sfc + (flux_ML - hfx_sfc) * (z_star / delta_SBL);
    } else if (z_star <= delta_EZ_lo) {
        // Mixed layer: approximately constant
        flux = w_star * hfx_sfc / amrex::max(w_star, amrex::Real(1.0e-6));
    } else if (z_star <= delta_EZ_hi) {
        // Lower entrainment zone: linear decrease
        amrex::Real flux_ML = w_star * hfx_sfc / amrex::max(w_star, amrex::Real(1.0e-6));
        amrex::Real frac = (z_star - delta_EZ_lo) / (delta_EZ_hi - delta_EZ_lo);
        flux = flux_ML * (amrex::Real(1.0) - frac);
    } else {
        // Upper entrainment zone: continues to decrease
        amrex::Real frac = (z_star - delta_EZ_hi) / (amrex::Real(1.0) - delta_EZ_hi);
        flux = amrex::Real(0.0); // Simplified: goes to zero at z_i
    }

    return flux;
}

/**
 * @brief Nonlocal momentum flux countergradient term γ_m
 *
 * Zhang et al. (2018), Eq. 27:
 *   γ_m = -S_m * (u*²/w*z_i) * (w*/u*)³
 *
 * Only applied in vertical momentum equation.
 *
 * @param u_star  Friction velocity [m/s]
 * @param w_star  Convective velocity scale [m/s]
 * @param zi      PBL height [m]
 * @param S_m     Momentum countergradient coefficient (default 15.9)
 * @return        Countergradient term [m/s²]
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real SMS3DTKE_Nonlocal_Momentum_Countergradient(amrex::Real u_star,
                                                        amrex::Real w_star,
                                                        amrex::Real zi,
                                                        amrex::Real S_m) noexcept
{
    amrex::Real u_star_sq = u_star * u_star;
    amrex::Real w_over_u = w_star / amrex::max(u_star, amrex::Real(1.0e-6));
    amrex::Real w_over_u_cubed = w_over_u * w_over_u * w_over_u;

    return -S_m * (u_star_sq / amrex::max(w_star * zi, amrex::Real(1.0e-6))) * w_over_u_cubed;
}

//==============================================================================
// SCALE-ADAPTIVE DISSIPATION (Zhang et al. 2018, Eq. 38-39)
//==============================================================================

/**
 * @brief LES dissipation coefficient c_ε1
 *
 * Zhang et al. (2018), Eq. 39:
 *   c_ε1 = 1 / (c_ε1_base + c_ε1_slope * l_LES / Δ)
 *
 * Typical values: c_ε1_base = 0.19, c_ε1_slope = 0.74
 *
 * @param l_LES           LES mixing length [m]
 * @param Delta           Grid spacing [m]
 * @param c_eps1_base     Base coefficient (default 0.19)
 * @param c_eps1_slope    Slope coefficient (default 0.74)
 * @return                Dissipation coefficient c_ε1
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real SMS3DTKE_Dissipation_Coefficient_LES(amrex::Real l_LES,
                                                  amrex::Real Delta,
                                                  amrex::Real c_eps1_base,
                                                  amrex::Real c_eps1_slope) noexcept
{
    amrex::Real ratio = l_LES / amrex::max(Delta, amrex::Real(1.0e-6));
    return amrex::Real(1.0) / (c_eps1_base + c_eps1_slope * ratio);
}

/**
 * @brief Compute horizontal deformation rate D_h
 *
 * Zhang et al. (2018), Eq. 40 uses horizontal deformation for blending:
 *   D_h = √(2 * S_h_ij * S_h_ij)
 * where S_h_ij includes only horizontal strain components (11, 22, 12)
 *
 * @param s11  Horizontal strain ∂u/∂x [1/s]
 * @param s22  Horizontal strain ∂v/∂y [1/s]
 * @param s12  Shear strain 0.5*(∂u/∂y + ∂v/∂x) [1/s]
 * @return     Horizontal deformation rate [1/s]
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real SMS3DTKE_Horizontal_Deformation(amrex::Real s11,
                                             amrex::Real s22,
                                             amrex::Real s12) noexcept
{
    // S_ij : S_ij for horizontal components only
    amrex::Real S_h_ij_squared = s11*s11 + s22*s22 + amrex::Real(2.0)*s12*s12;
    return std::sqrt(amrex::Real(2.0) * S_h_ij_squared);
}

/**
 * @brief Harmonic average of multiple length scales
 *
 * Zhang et al. (2018), Eq. 28:
 *   1/l_v = 1/l_Meso + 1/L_S + 1/L_T + 1/L_B
 *
 * Handles infinite values (unstable stratification) gracefully.
 *
 * @param l_Meso   Mesoscale length [m]
 * @param L_S      Surface layer length [m]
 * @param L_T      Integral length [m]
 * @param L_B      Buoyancy length [m]
 * @return         Harmonic average length [m]
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real SMS3DTKE_Harmonic_Length(amrex::Real l_Meso,
                                      amrex::Real L_S,
                                      amrex::Real L_T,
                                      amrex::Real L_B) noexcept
{
    amrex::Real inv_sum = amrex::Real(0.0);
    amrex::Real large_val = amrex::Real(1.0e10);

    // Only add contributions from finite lengths
    if (l_Meso < large_val) inv_sum += amrex::Real(1.0) / l_Meso;
    if (L_S < large_val)     inv_sum += amrex::Real(1.0) / L_S;
    if (L_T < large_val)     inv_sum += amrex::Real(1.0) / L_T;
    if (L_B < large_val)     inv_sum += amrex::Real(1.0) / L_B;

    // Return harmonic average, or l_Meso if all others are infinite
    if (inv_sum > amrex::Real(0.0)) {
        return amrex::Real(1.0) / inv_sum;
    } else {
        return l_Meso;
    }
}

#endif // ERF_SMS3DTKE_UTILS_H_
