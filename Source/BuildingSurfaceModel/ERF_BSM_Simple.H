#ifndef ERF_BSM_SIMPLE_H
#define ERF_BSM_SIMPLE_H

#include <AMReX_MultiFab.H>
#include <AMReX_ArrayLim.H>
#include <AMReX_BCRec.H>
#include <AMReX_GpuContainers.H>

#include <ERF_BuildingSurfBase.H>
#include <ERF_IndexDefines.H>
#include <ERF_DataStruct.H>
#include <ERF_Constants.H>

/**
 * @brief Simple building surface model with subsurface thermal diffusion
 *
 * This model implements a simple 4-layer subsurface heat conduction model
 * for building walls and roofs. It uses explicit time stepping for the
 * thermal diffusion equation. Surface temperature is prescribed as a
 * boundary condition initially.
 *
 * The model operates only on building surface cells (k>0, identified by
 * terrain_blank masks), leaving the bottom boundary (k=0) for the LSM.
 */
class BSM_Simple : public BuildingSurfBase {

public:
    // Variable indices for internal storage
    enum VarType {
        surf_temp_idx = 0,    // Surface temperature
        layer1_temp_idx,       // Subsurface layer 1
        layer2_temp_idx,       // Subsurface layer 2
        layer3_temp_idx,       // Subsurface layer 3
        layer4_temp_idx,       // Subsurface layer 4
        NumVars
    };

    BSM_Simple () = default;

    virtual ~BSM_Simple () = default;

    virtual
    void
    Define (SolverChoice& sc) override
    {
        // Store material properties from SolverChoice
        m_nz = sc.bsm_nz_layers;
        m_dz = sc.bsm_dz_layer;
        m_cp = sc.bsm_cp_concrete;
        m_k  = sc.bsm_k_concrete;
        m_d  = m_k / m_cp;  // Thermal diffusivity
        m_theta_dir = sc.bsm_deep_temp;
    }

    virtual
    void
    Init (const int& lev,
          const amrex::MultiFab& cons_in,
          const amrex::Geometry& geom,
          const amrex::Real& dt_advance) override;

    virtual
    void
    Advance (const double& dt_advance) override;

    virtual
    amrex::MultiFab*
    Get_Data_Ptr (const int& varIdx) override
    {
        if (varIdx >= 0 && varIdx < NumVars) {
            return m_vars[varIdx].get();
        }
        return nullptr;
    }

    virtual
    amrex::MultiFab*
    Get_Data_Ptr (const std::string& varname) override
    {
        int idx = Get_Data_Index(varname);
        return (idx >= 0) ? Get_Data_Ptr(idx) : nullptr;
    }

    virtual
    int
    Get_Data_Index (const std::string& varname) override
    {
        if (varname == "surf_temp")  return surf_temp_idx;
        if (varname == "layer1_temp") return layer1_temp_idx;
        if (varname == "layer2_temp") return layer2_temp_idx;
        if (varname == "layer3_temp") return layer3_temp_idx;
        if (varname == "layer4_temp") return layer4_temp_idx;
        return -1;
    }

    virtual
    amrex::Geometry
    Get_Geom () override { return m_geom; }

    virtual
    int
    Get_Data_Size () override { return NumVars; }

    virtual
    std::string
    Get_DataName (const int& varIdx) override
    {
        if (varIdx == surf_temp_idx)   return "surf_temp";
        if (varIdx == layer1_temp_idx) return "layer1_temp";
        if (varIdx == layer2_temp_idx) return "layer2_temp";
        if (varIdx == layer3_temp_idx) return "layer3_temp";
        if (varIdx == layer4_temp_idx) return "layer4_temp";
        return "";
    }

protected:
    // Subsurface diffusion methods
    void ComputeFluxes ();
    void AdvanceSubsurface ();
    void ComputeSurfTemp ();

    // Material properties
    int m_nz = 4;                         // Number of subsurface layers
    amrex::Real m_dz = 0.05;              // Layer thickness [m]
    amrex::Real m_cp = 2.0e6;             // Specific heat [J/m³K]
    amrex::Real m_k  = 1.5;               // Thermal conductivity [W/mK]
    amrex::Real m_d  = 7.5e-7;            // Thermal diffusivity [m²/s]
    amrex::Real m_theta_dir = 300.0;      // Deep layer BC [K]

    // Time step
    amrex::Real m_dt = 0.0;

    // Storage for temperature fields
    amrex::Array<std::unique_ptr<amrex::MultiFab>, NumVars> m_vars;

    // Geometry
    amrex::Geometry m_geom;
};

#endif
