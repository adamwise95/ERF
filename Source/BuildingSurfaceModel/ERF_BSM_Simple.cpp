#include <ERF_BSM_Simple.H>

using namespace amrex;

/**
 * Initialize the building surface model
 *
 * @param lev Level of refinement
 * @param cons_in Conservative variables (for getting grid info)
 * @param geom Geometry at this level
 * @param dt_advance Time step
 */
void
BSM_Simple::Init (const int& lev,
                  const MultiFab& cons_in,
                  const Geometry& geom,
                  const Real& dt_advance)
{
    m_geom = geom;
    m_dt = dt_advance;

    // Get grid info
    const BoxArray& ba = cons_in.boxArray();
    const DistributionMapping& dm = cons_in.DistributionMap();
    int ngrow = 1;  // One ghost cell for gradients

    // Allocate temperature fields
    // surf_temp: surface temperature [K]
    m_vars[surf_temp_idx] = std::make_unique<MultiFab>(ba, dm, 1, ngrow);

    // layer temperatures: subsurface layers 1-4 [K]
    m_vars[layer1_temp_idx] = std::make_unique<MultiFab>(ba, dm, 1, ngrow);
    m_vars[layer2_temp_idx] = std::make_unique<MultiFab>(ba, dm, 1, ngrow);
    m_vars[layer3_temp_idx] = std::make_unique<MultiFab>(ba, dm, 1, ngrow);
    m_vars[layer4_temp_idx] = std::make_unique<MultiFab>(ba, dm, 1, ngrow);

    // Initialize all temperatures to deep layer temperature
    m_vars[surf_temp_idx]->setVal(m_theta_dir);
    m_vars[layer1_temp_idx]->setVal(m_theta_dir);
    m_vars[layer2_temp_idx]->setVal(m_theta_dir);
    m_vars[layer3_temp_idx]->setVal(m_theta_dir);
    m_vars[layer4_temp_idx]->setVal(m_theta_dir);
}

/**
 * Advance the building surface model by one time step
 *
 * Solves the 1D thermal diffusion equation in the subsurface layers
 * using explicit time stepping.
 *
 * @param dt_advance Time step
 */
void
BSM_Simple::Advance (const double& dt_advance)
{
    m_dt = dt_advance;

    // Update subsurface temperatures via thermal diffusion
    AdvanceSubsurface();

    // Update surface temperature (currently no-op, will be computed by energy balance model)
    ComputeSurfTemp();
}

/**
 * Compute diffusive fluxes between layers
 *
 * Currently not needed for explicit scheme, but provided for
 * future implicit solver implementation.
 */
void
BSM_Simple::ComputeFluxes ()
{
    // For explicit time stepping, fluxes are computed inline in AdvanceSubsurface
    // This function is a placeholder for future implicit solver
}

/**
 * Advance subsurface temperatures using explicit thermal diffusion
 *
 * Solves: dT/dt = d * d²T/dz² where d is thermal diffusivity
 *
 * Uses explicit forward Euler with centered differences:
 * T_new = T_old + dt * d * (T[i+1] - 2*T[i] + T[i-1]) / dz²
 */
void
BSM_Simple::AdvanceSubsurface ()
{
    // Diffusion coefficient: dt * d / dz²
    Real coeff = m_dt * m_d / (m_dz * m_dz);

    // Stability criterion for explicit diffusion: coeff <= 0.5
    if (coeff > 0.5) {
        amrex::Print() << "WARNING: BSM_Simple diffusion may be unstable! "
                       << "dt*d/dz^2 = " << coeff << " > 0.5\n";
    }

    // Get temperature arrays
    MultiFab& T_surf = *m_vars[surf_temp_idx];
    MultiFab& T1 = *m_vars[layer1_temp_idx];
    MultiFab& T2 = *m_vars[layer2_temp_idx];
    MultiFab& T3 = *m_vars[layer3_temp_idx];
    MultiFab& T4 = *m_vars[layer4_temp_idx];

    // Create temporary storage for old values
    MultiFab T1_old(T1.boxArray(), T1.DistributionMap(), 1, 0);
    MultiFab T2_old(T2.boxArray(), T2.DistributionMap(), 1, 0);
    MultiFab T3_old(T3.boxArray(), T3.DistributionMap(), 1, 0);
    MultiFab T4_old(T4.boxArray(), T4.DistributionMap(), 1, 0);

    MultiFab::Copy(T1_old, T1, 0, 0, 1, 0);
    MultiFab::Copy(T2_old, T2, 0, 0, 1, 0);
    MultiFab::Copy(T3_old, T3, 0, 0, 1, 0);
    MultiFab::Copy(T4_old, T4, 0, 0, 1, 0);

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(T1, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.tilebox();

        const Array4<const Real>& T_surf_arr = T_surf.const_array(mfi);
        const Array4<const Real>& T1_old_arr = T1_old.const_array(mfi);
        const Array4<const Real>& T2_old_arr = T2_old.const_array(mfi);
        const Array4<const Real>& T3_old_arr = T3_old.const_array(mfi);
        const Array4<const Real>& T4_old_arr = T4_old.const_array(mfi);

        const Array4<Real>& T1_arr = T1.array(mfi);
        const Array4<Real>& T2_arr = T2.array(mfi);
        const Array4<Real>& T3_arr = T3.array(mfi);
        const Array4<Real>& T4_arr = T4.array(mfi);

        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            // Layer 1: flux from surface and layer 2
            T1_arr(i,j,k) = T1_old_arr(i,j,k) + coeff * (
                T_surf_arr(i,j,k) - 2.0*T1_old_arr(i,j,k) + T2_old_arr(i,j,k)
            );

            // Layer 2: flux from layers 1 and 3
            T2_arr(i,j,k) = T2_old_arr(i,j,k) + coeff * (
                T1_old_arr(i,j,k) - 2.0*T2_old_arr(i,j,k) + T3_old_arr(i,j,k)
            );

            // Layer 3: flux from layers 2 and 4
            T3_arr(i,j,k) = T3_old_arr(i,j,k) + coeff * (
                T2_old_arr(i,j,k) - 2.0*T3_old_arr(i,j,k) + T4_old_arr(i,j,k)
            );

            // Layer 4: flux from layer 3 and deep BC
            T4_arr(i,j,k) = T4_old_arr(i,j,k) + coeff * (
                T3_old_arr(i,j,k) - 2.0*T4_old_arr(i,j,k) + m_theta_dir
            );
        });
    }
}

/**
 * Compute surface temperature
 *
 * For the Simple model, surface temperature is currently prescribed.
 * This will be overridden by the EnergyBalance model to solve the
 * full surface energy balance.
 */
void
BSM_Simple::ComputeSurfTemp ()
{
    // Simple model: surface temperature is prescribed (no-op)
    // The EnergyBalance model will override this to solve the energy balance
}
