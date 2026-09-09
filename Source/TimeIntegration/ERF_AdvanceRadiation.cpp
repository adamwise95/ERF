#include <ERF.H>

using namespace amrex;

void ERF::advance_radiation (int lev,
                             MultiFab& cons,
                             const double& dt_advance)
{
    if (solverChoice.rad_type != RadiationType::None) {
#ifdef ERF_USE_NETCDF
        MultiFab *lat_ptr = lat_m[lev].get();
        MultiFab *lon_ptr = lon_m[lev].get();
#else
        MultiFab *lat_ptr = nullptr;
        MultiFab *lon_ptr = nullptr;
#endif
        // T surf from SurfaceLayer if we have it
        MultiFab* t_surf = (m_SurfaceLayer) ? m_SurfaceLayer->get_t_surf(lev) : nullptr;

        // RRTMGP inputs names and pointers
        Vector<std::string> lsm_input_names = rad[lev]->get_lsm_input_varnames();
        Vector<MultiFab*> lsm_input_ptrs(lsm_input_names.size(),nullptr);
        for (int i(0); i<lsm_input_ptrs.size(); ++i) {
            int varIdx = lsm.Get_DataIdx(lev,lsm_input_names[i]);
            if (varIdx >= 0) { lsm_input_ptrs[i] = lsm.Get_Data_Ptr(lev,varIdx); }
        }

        // RRTMGP output names and pointers
        Vector<std::string> lsm_output_names = rad[lev]->get_lsm_output_varnames();
        Vector<MultiFab*> lsm_output_ptrs(lsm_output_names.size(),nullptr);
        for (int i(0); i<lsm_output_ptrs.size(); ++i) {
            int varIdx = lsm.Get_DataIdx(lev,lsm_output_names[i]);
            if (varIdx >= 0) { lsm_output_ptrs[i] = lsm.Get_Data_Ptr(lev,varIdx); }
        }

        // Force radiation update to sync with lsm?
        bool lsm_updated = (lev==0 && max_level>0) ? lsm.Get_LSM_Update_Status(lev) : false;

        // Enter radiation class driver
        double time_for_rad = t_old[lev] + start_time;
        rad[lev]->Run(lev, istep[lev], time_for_rad, dt_advance,
                      cons.boxArray(), geom[lev], &(cons),
                      lmask_lev[lev][0].get(), t_surf,
                      lsm_input_ptrs, lsm_output_ptrs,
                      qheating_rates[lev].get(), rad_fluxes[lev].get(),
                      z_phys_nd[lev].get()     , lat_ptr, lon_ptr,
                      lsm_updated);

        // Fill ghost cells after radiation computes (needed for interpolation to finer levels)
        // This should be fast since it only fills this level's own ghost cells
        if (solverChoice.rad_type != RadiationType::None && !rad[lev]->is_nested_patch()) {
            qheating_rates[lev]->FillBoundary(geom[lev].periodicity());
        }

        // For nested patches (fine levels that don't reach model top), radiation
        // was skipped. Interpolate heating rates from parent level.
        if (lev > 0 && rad[lev]->is_nested_patch()) {
            // Ensure parent level's ghost cells are filled before interpolation
            // This is needed even when radiation doesn't run this step, especially
            // with two-way coupling where the grid structure may have changed
            if (!rad[lev-1]->is_nested_patch()) {
                qheating_rates[lev-1]->FillBoundary(geom[lev-1].periodicity());
            }

            InterpFromCoarseLevel(*qheating_rates[lev], qheating_rates[lev]->nGrowVect(),
                                  IntVect(0,0,0),
                                  *qheating_rates[lev-1], 0, 0, 2,
                                  geom[lev-1], geom[lev],
                                  refRatio(lev-1), &cell_cons_interp,
                                  domain_bcs_type, BCVars::cons_bc);

        }
    }
}
