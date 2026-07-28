#ifndef ERF_BSM_ENERGYBALANCE_H
#define ERF_BSM_ENERGYBALANCE_H

#include <AMReX_MultiFab.H>
#include <AMReX_ArrayLim.H>
#include <AMReX_BCRec.H>
#include <AMReX_GpuContainers.H>

#include <ERF_BSM_Simple.H>
#include <ERF_IndexDefines.H>
#include <ERF_DataStruct.H>
#include <ERF_Constants.H>

/**
 * @brief Full energy balance building surface model
 *
 * This model extends BSM_Simple to solve the full surface energy balance:
 * Net_Radiation + Sensible_Heat + Latent_Heat + Ground_Conduction = 0
 *
 * The surface temperature is iteratively determined to satisfy this balance,
 * using:
 * - MOST (Monin-Obukhov Similarity Theory) for sensible heat
 * - Dudhia scheme for latent heat
 * - Subsurface conduction from BSM_Simple
 * - RRTMGP radiation input
 *
 * Operates on building surface cells (k>0) only.
 */
class BSM_EnergyBalance : public BSM_Simple {

public:
    BSM_EnergyBalance () = default;

    virtual ~BSM_EnergyBalance () = default;

    virtual
    void
    Define (SolverChoice& sc) override
    {
        // Call base class Define for material properties
        BSM_Simple::Define(sc);

        // Store additional EnergyBalance parameters
        m_albedo = sc.bsm_albedo;
        m_emissivity = sc.bsm_emissivity;
        m_moisture_avail = sc.bsm_moisture_avail;
    }

    virtual
    void
    Advance_With_State (const int& lev,
                        const amrex::MultiFab& cons_in,
                        const amrex::MultiFab& xvel_in,
                        const amrex::MultiFab& yvel_in,
                        const amrex::MultiFab& zvel_in,
                        const amrex::MultiFab* rad_fluxes,
                        const amrex::MultiFab* terrain_blank,
                        const amrex::Real& time,
                        const amrex::Real& dt_advance) override;

protected:
    // Energy balance solver
    void Solve_Surface_Energy_Balance(
        const amrex::MultiFab& cons_in,
        const amrex::MultiFab& xvel_in,
        const amrex::MultiFab& yvel_in,
        const amrex::MultiFab& zvel_in,
        const amrex::MultiFab* rad_fluxes,
        const amrex::MultiFab* terrain_blank,
        amrex::MultiFab* shadow_mask,
        amrex::Real sun_azimuth_deg,
        amrex::Real sun_zenith_deg,
        amrex::Real time);

    // Surface properties
    amrex::Real m_albedo = 0.2;          // Surface albedo
    amrex::Real m_emissivity = 0.9;      // Surface emissivity
    amrex::Real m_moisture_avail = 0.0;  // Moisture availability (0-1)

};

#endif
