#ifndef ERF_BUILDINGSURFACE_H
#define ERF_BUILDINGSURFACE_H

#include <typeinfo>

#include <AMReX_PlotFileUtil.H>
#include <AMReX_MultiFabUtil.H>

#include <ERF_BuildingSurfBase.H>

/**
 * @brief Wrapper class for building surface models
 *
 * This class manages a vector of building surface model implementations,
 * one per AMR level. It provides a polymorphic interface to access
 * different BSM types (Simple, EnergyBalance) through the BuildingSurfBase
 * abstract base class.
 *
 * Similar to the LandSurface wrapper for LSM models.
 */
class BuildingSurface {

public:

    BuildingSurface () { }

    ~BuildingSurface () = default;

    void
    ReSize (const int& nlev) { m_bsm_model.resize(nlev); }

    template<class NewSurfModel>
    void
    SetModel ()
    {
        for (int lev(0); lev<m_bsm_model.size(); ++lev) {
            m_bsm_model[lev] = std::make_unique<NewSurfModel>();
        }
    }

    void
    Define (const int& lev,
            SolverChoice& sc)
    {
        m_bsm_model[lev]->Define(sc);
    }

    void
    Init (const int& lev,
          const amrex::MultiFab& cons_in,
          const amrex::Geometry& geom,
          const amrex::Real& dt_advance)
    {
        m_bsm_model[lev]->Init(lev, cons_in, geom, dt_advance);
    }

    void
    Advance (const int& lev,
             const amrex::MultiFab& cons_in,
             const amrex::MultiFab& xvel_in,
             const amrex::MultiFab& yvel_in,
             const amrex::MultiFab& zvel_in,
             const amrex::MultiFab* rad_fluxes,
             const double& time,
             const double& dt_advance)
    {
        amrex::Real time_r = static_cast<amrex::Real>(time);
        amrex::Real dt_r = static_cast<amrex::Real>(dt_advance);
        m_bsm_model[lev]->Advance_With_State(lev, cons_in, xvel_in, yvel_in, zvel_in,
                                             rad_fluxes, time_r, dt_r);
    }

    void
    Advance (const int& lev, const double& dt_advance)
    {
        amrex::Real dt_r = static_cast<amrex::Real>(dt_advance);
        m_bsm_model[lev]->Advance(dt_r);
    }

    bool
    Has_Model () const { return !m_bsm_model.empty() && m_bsm_model[0] != nullptr; }

    amrex::MultiFab*
    Get_Data_Ptr (const int& lev, const int& varIdx) {
        return m_bsm_model[lev]->Get_Data_Ptr(varIdx);
    }

    amrex::MultiFab*
    Get_Data_Ptr (const int& lev, const std::string& varname)
    {
        std::string name = varname;
        const int index = m_bsm_model[lev]->Get_Data_Index(name);
        return (index >= 0) ? m_bsm_model[lev]->Get_Data_Ptr(index) : nullptr;
    }

    amrex::Geometry
    Get_Geom (const int& lev ) { return m_bsm_model[lev]->Get_Geom(); }

    int
    Get_Data_Size () { return m_bsm_model[0]->Get_Data_Size(); }

    std::string
    Get_DataName (const int& varIdx) { return m_bsm_model[0]->Get_DataName(varIdx); }

    amrex::Vector<std::string>
    Get_DataNames ()
    {
        amrex::Vector<std::string> names;
        if (m_bsm_model.empty() || m_bsm_model[0] == nullptr) {
            return names;
        }
        const int nvars = Get_Data_Size();
        names.reserve(nvars);
        for (int var = 0; var < nvars; ++var) {
            names.push_back(Get_DataName(var));
        }
        return names;
    }

private:
    amrex::Vector<std::unique_ptr<BuildingSurfBase>> m_bsm_model;

};

#endif
