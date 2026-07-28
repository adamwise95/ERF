#ifndef ERF_BUILDINGSURFBASE_H
#define ERF_BUILDINGSURFBASE_H

#include "AMReX_BCRec.H"
#include "AMReX_MultiFabUtil.H"
#include "AMReX_Geometry.H"
#include "ERF_DataStruct.H"

/**
 * @brief Abstract base class for building surface models
 *
 * This class provides the interface for different building surface model
 * implementations (e.g., Simple thermal diffusion, Full energy balance).
 * Similar to NullSurf for land surface models, but specifically for
 * building walls and roofs (k>0 cells only).
 */
class BuildingSurfBase {

 public:
    BuildingSurfBase () {}

    virtual ~BuildingSurfBase () = default;

    virtual
    void
    Define (SolverChoice& /*sc*/) { }

    virtual
    void Init (const int& /*lev*/,
               const amrex::MultiFab& /*cons_in*/,
               const amrex::Geometry& /*geom*/,
               const amrex::Real& /*dt_advance*/) { }

    virtual
    void
    Advance (const double& /*dt_advance*/) { }

    virtual
    void
    Advance_With_State (const int& /*lev*/,
                        const amrex::MultiFab& /*cons_in*/,
                        const amrex::MultiFab& /*xvel_in*/,
                        const amrex::MultiFab& /*yvel_in*/,
                        const amrex::MultiFab& /*zvel_in*/,
                        const amrex::MultiFab* /*rad_fluxes*/,
                        const amrex::MultiFab* /*terrain_blank*/,
                        const amrex::Real& /*time*/,
                        const amrex::Real& /*dt_advance*/) { }

    virtual
    amrex::MultiFab*
    Get_Data_Ptr (const int& /*varIdx*/ ) { return nullptr; }

    virtual
    amrex::MultiFab*
    Get_Data_Ptr (const std::string& /*varname*/ ) { return nullptr; }

    virtual
    int
    Get_Data_Index (const std::string& /*varname*/ ) { return -1; }

    virtual
    amrex::Geometry
    Get_Geom ( ) { return m_geom; }

    virtual
    int
    Get_Data_Size () { return 0; }

    virtual
    std::string
    Get_DataName (const int& /*varIdx*/) { return ""; }

 protected:
    amrex::Geometry m_geom;

};

#endif
