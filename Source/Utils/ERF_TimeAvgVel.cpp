#include <ERF_Utils.H>
#include <ERF_IndexDefines.H>

using namespace amrex;

/*
 * Accumulate time averaged velocity fields
 */
void
Time_Avg_Vel_atCC (double dt_d,
                   double& t_avg_cnt,
                   MultiFab* vel_t_avg,
                   MultiFab& xvel,
                   MultiFab& yvel,
                   MultiFab& zvel,
                   MultiFab& cons,
                   MultiFab* tau11,
                   MultiFab* tau22,
                   MultiFab* tau33,
                   MultiFab* tau12,
                   MultiFab* tau13,
                   MultiFab* tau23)
{
    // Augment the counter
    t_avg_cnt += dt_d;

    Real dt = static_cast<Real>(dt_d);

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for ( MFIter mfi(*(vel_t_avg),TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        // CC tilebox
        Box tbx = mfi.tilebox();

        // Velocity on faces
        const Array4<Real>& velx = xvel.array(mfi);
        const Array4<Real>& vely = yvel.array(mfi);
        const Array4<Real>& velz = zvel.array(mfi);

        // Conservative variables at CC
        const Array4<Real>& cons_arr = cons.array(mfi);

        // Turbulent stress components (diagonal are already CC, off-diagonal need interpolation)
        const Array4<Real>& tau11_arr = (tau11) ? tau11->array(mfi) : Array4<Real>{};
        const Array4<Real>& tau22_arr = (tau22) ? tau22->array(mfi) : Array4<Real>{};
        const Array4<Real>& tau33_arr = (tau33) ? tau33->array(mfi) : Array4<Real>{};
        const Array4<Real>& tau12_arr = (tau12) ? tau12->array(mfi) : Array4<Real>{};
        const Array4<Real>& tau13_arr = (tau13) ? tau13->array(mfi) : Array4<Real>{};
        const Array4<Real>& tau23_arr = (tau23) ? tau23->array(mfi) : Array4<Real>{};

        // Time average at CC
        Array4<Real> vel_t_avg_arr = vel_t_avg->array(mfi);

        ParallelFor(tbx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            Real u_cc = myhalf * ( velx(i,j,k) + velx(i+1,j  ,k  ) );
            Real v_cc = myhalf * ( vely(i,j,k) + vely(i  ,j+1,k  ) );
            Real w_cc = myhalf * ( velz(i,j,k) + velz(i  ,j  ,k+1) );
            Real umag_cc = std::sqrt(u_cc*u_cc + v_cc*v_cc + w_cc*w_cc);

            Real u_mean = vel_t_avg_arr(i,j,k,0) / t_avg_cnt;
            Real v_mean = vel_t_avg_arr(i,j,k,1) / t_avg_cnt;
            Real w_mean = vel_t_avg_arr(i,j,k,2) / t_avg_cnt;

            Real up = u_cc - u_mean;
            Real vp = v_cc - v_mean;
            Real wp = w_cc - w_mean;
            Real tke = myhalf * (up*up + vp*vp + wp*wp);

            // Scalar = RhoScalar / rho
            Real scalar_cc = cons_arr(i,j,k,RhoScalar_comp) / cons_arr(i,j,k,Rho_comp);

            // Turbulent stress components interpolated to cell center
            // tau11, tau22, tau33 are already at cell centers
            // tau12 is at (i+1/2, j+1/2, k) - average to cell center
            // tau13 is at (i+1/2, j, k+1/2) - average to cell center
            // tau23 is at (i, j+1/2, k+1/2) - average to cell center
            Real tau11_cc = (tau11_arr) ? tau11_arr(i,j,k) : Real(0.0);
            Real tau22_cc = (tau22_arr) ? tau22_arr(i,j,k) : Real(0.0);
            Real tau33_cc = (tau33_arr) ? tau33_arr(i,j,k) : Real(0.0);
            Real tau12_cc = (tau12_arr) ? Real(0.25) * (tau12_arr(i,j,k) + tau12_arr(i+1,j,k) +
                                                         tau12_arr(i,j+1,k) + tau12_arr(i+1,j+1,k)) : Real(0.0);
            Real tau13_cc = (tau13_arr) ? Real(0.25) * (tau13_arr(i,j,k) + tau13_arr(i+1,j,k) +
                                                         tau13_arr(i,j,k+1) + tau13_arr(i+1,j,k+1)) : Real(0.0);
            Real tau23_cc = (tau23_arr) ? Real(0.25) * (tau23_arr(i,j,k) + tau23_arr(i,j+1,k) +
                                                         tau23_arr(i,j,k+1) + tau23_arr(i,j+1,k+1)) : Real(0.0);

            vel_t_avg_arr(i,j,k,0)  += u_cc * dt;
            vel_t_avg_arr(i,j,k,1)  += v_cc * dt;
            vel_t_avg_arr(i,j,k,2)  += w_cc * dt;
            vel_t_avg_arr(i,j,k,3)  += umag_cc * dt;
            vel_t_avg_arr(i,j,k,4)  += tke * dt;
            vel_t_avg_arr(i,j,k,5)   = tke;
            vel_t_avg_arr(i,j,k,6)  += scalar_cc * dt;
            vel_t_avg_arr(i,j,k,7)  += up * dt;
            vel_t_avg_arr(i,j,k,8)  += vp * dt;
            vel_t_avg_arr(i,j,k,9)  += wp * dt;
            vel_t_avg_arr(i,j,k,10)  = up;
            vel_t_avg_arr(i,j,k,11)  = vp;
            vel_t_avg_arr(i,j,k,12)  = wp;
            vel_t_avg_arr(i,j,k,13) += tau11_cc * dt;
            vel_t_avg_arr(i,j,k,14) += tau22_cc * dt;
            vel_t_avg_arr(i,j,k,15) += tau33_cc * dt;
            vel_t_avg_arr(i,j,k,16) += tau12_cc * dt;
            vel_t_avg_arr(i,j,k,17) += tau13_cc * dt;
            vel_t_avg_arr(i,j,k,18) += tau23_cc * dt;
            vel_t_avg_arr(i,j,k,19)  = tau11_cc;
            vel_t_avg_arr(i,j,k,20)  = tau22_cc;
            vel_t_avg_arr(i,j,k,21)  = tau33_cc;
            vel_t_avg_arr(i,j,k,22)  = tau12_cc;
            vel_t_avg_arr(i,j,k,23)  = tau13_cc;
            vel_t_avg_arr(i,j,k,24)  = tau23_cc;
        });
    }
}

void
Accumulate_Interval_Means (double dt_d,
                           double& t_mean_cnt,
                           MultiFab* interval_means,
                           MultiFab& xvel,
                           MultiFab& yvel,
                           MultiFab& zvel,
                           MultiFab& cons)
{
    AMREX_ALWAYS_ASSERT(interval_means != nullptr);

    t_mean_cnt += dt_d;
    const Real dt = static_cast<Real>(dt_d);

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(*interval_means, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.tilebox();
        const Array4<const Real>& u = xvel.const_array(mfi);
        const Array4<const Real>& v = yvel.const_array(mfi);
        const Array4<const Real>& w = zvel.const_array(mfi);
        const Array4<const Real>& state = cons.const_array(mfi);
        const Array4<Real>& mean = interval_means->array(mfi);

        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            const Real u_cc = Real(0.5) * (u(i,j,k) + u(i+1,j,k));
            const Real v_cc = Real(0.5) * (v(i,j,k) + v(i,j+1,k));
            const Real w_cc = Real(0.5) * (w(i,j,k) + w(i,j,k+1));
            const Real theta = state(i,j,k,RhoTheta_comp) / state(i,j,k,Rho_comp);

            mean(i,j,k,0) += u_cc * dt;
            mean(i,j,k,1) += v_cc * dt;
            mean(i,j,k,2) += w_cc * dt;
            mean(i,j,k,3) += theta * dt;
            mean(i,j,k,4) += u_cc * u_cc * dt;
            mean(i,j,k,5) += v_cc * v_cc * dt;
            mean(i,j,k,6) += w_cc * w_cc * dt;
            mean(i,j,k,7) += u_cc * w_cc * dt;
            mean(i,j,k,8) += v_cc * w_cc * dt;
            mean(i,j,k,9) += w_cc * theta * dt;
        });
    }
}
