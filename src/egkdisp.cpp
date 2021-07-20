//
// Created by mjschnie on 7/13/21.
//

#include "egkdisp.h"
#include "stdio.h"
#include "potent.h"
#include "mdpq.h"
#include "mdcalc.h"
#include "glob.energi.h"
#include "tool/host_zero.h"
#include "mdegv.h"
#include <tinker/detail/atomid.hh>
#include <tinker/detail/kvdws.hh>
#include <tinker/detail/nonpol.hh>

namespace tinker {

real* wcaeps;
real* wcarad;
real* wcacdisp;

count_buffer newca;
energy_buffer ewca;
virial_buffer vir_ewca;
grad_prec* dewcax;
grad_prec* dewcay;
grad_prec* dewcaz;
energy_prec energy_ewca;
virial_prec virial_ewca[9];

void egkdisp_data(rc_op op) {
   if (not use_potent(solv_term))
      return;

   bool rc_a = rc_flag & calc::analyz;

   if (op & rc_dealloc) {
      if (rc_a) {
         buffer_deallocate(rc_flag, newca);
         buffer_deallocate(rc_flag, ewca, vir_ewca, dewcax, dewcay, dewcaz);
      }
      newca = nullptr;
      ewca = nullptr;
      vir_ewca = nullptr;
      dewcax = nullptr;
      dewcay = nullptr;
      dewcaz = nullptr;
      darray::deallocate(wcaeps, wcarad, wcacdisp);
   }

   if (op & rc_alloc) {
      newca = nullptr;
      ewca = eng_buf_vdw;
      vir_ewca = vir_buf_vdw;
      dewcax = gx_vdw;
      dewcay = gy_vdw;
      dewcaz = gz_vdw;
      if (rc_a) {
         buffer_allocate(rc_flag, &newca);
         buffer_allocate(rc_flag, &ewca, &vir_ewca, &dewcax, &dewcay, &dewcaz);
      }
      darray::allocate(n, &wcaeps, &wcarad, &wcacdisp);
   }

   if (op & rc_init) {
      std::vector<real> vrad(n), veps(n), vcdisp(n);
      for (int i=0; i<n; i++) {
         int jj = atomid::class_[i] - 1;
         vrad[i] = kvdws::rad[jj];
         veps[i] = kvdws::eps[jj];
         vcdisp[i] = nonpol::cdisp[i];
      }
      darray::copyin(g::q0, n, wcarad, vrad.data());
      darray::copyin(g::q0, n, wcaeps, veps.data());
      darray::copyin(g::q0, n, wcacdisp, vcdisp.data());
      wait_for(g::q0);
   }

}

extern void egkdisp_acc(int vers);
extern void egkdisp_cu(int vers);

void egkdisp(int vers)
{
   bool rc_a = rc_flag & calc::analyz;
   bool do_a = vers & calc::analyz;
   bool do_e = vers & calc::energy;
   bool do_v = vers & calc::virial;
   bool do_g = vers & calc::grad;


   host_zero(energy_ewca, virial_ewca);
   size_t bsize = buffer_size();
   if (rc_a) {
      if (do_a)
         darray::zero(g::q0, bsize, newca);
      if (do_e)
         darray::zero(g::q0, bsize, ewca);
      if (do_v)
         darray::zero(g::q0, bsize, vir_ewca);
      if (do_g)
         darray::zero(g::q0, n, dewcax, dewcay, dewcaz);
   }


   egkdisp_acc(vers);


   if (rc_a) {
      if (do_e) {
         energy_prec e = energy_reduce(ewca);
         energy_ewca += e;
         energy_vdw += e;
      }
      if (do_v) {
         virial_buffer u = vir_ewca;
         virial_prec v[9];
         virial_reduce(v, u);
         for (int iv = 0; iv < 9; ++iv) {
            virial_ewca[iv] += v[iv];
            virial_vdw[iv] += v[iv];
         }
      }
      if (do_g)
         sum_gradient(gx_vdw, gy_vdw, gz_vdw, dewcax, dewcay, dewcaz);
   }
}



}