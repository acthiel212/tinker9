#include "add.h"
#include "egkdisp.h"
#include "mdcalc.h"
#include "mdpq.h"
#include "tool/gpu_card.h"
#include <tinker/detail/nonpol.hh>

namespace tinker {

template <class Ver>
void egkdisp_acc1()
{
   constexpr bool do_e = Ver::e;
   // constexpr bool do_a = Ver::a;
   constexpr bool do_g = Ver::g;
   // constexpr bool do_v = Ver::v;

   auto bufsize = buffer_size();

   constexpr real epso = nonpol::epso;
   constexpr real epsh = nonpol::epsh;
   constexpr real rmino = nonpol::rmino;
   constexpr real rminh = nonpol::rminh;
   constexpr real awater = nonpol::awater;
   constexpr real slevy = nonpol::slevy;
   constexpr real dispoff = 1.056;
   constexpr real shctd = 0.75;

   MAYBE_UNUSED int GRID_DIM = get_grid_size(BLOCK_DIM);
#pragma acc parallel async num_gangs(GRID_DIM) vector_length(BLOCK_DIM)        \
   deviceptr(ewca, dewcax, dewcay, dewcaz, wcaeps, wcarad, wcacdisp, x, y, z)
#pragma acc loop gang independent
   for (int i = 0; i < n; ++i) {
      real epsi = wcaeps[i];
      real rmini = wcarad[i];

      real emixo =
         4 * epso * epsi / REAL_POW((REAL_SQRT(epso) + REAL_SQRT(epsi)), 2);
      real rmixo = 2 * (rmino * rmino * rmino + rmini * rmini * rmini) /
         (rmino * rmino + rmini * rmini);
      real rmixo7 = REAL_POW(rmixo, 7);
      real ao = emixo * rmixo7;
      real emixh =
         4 * epsh * epsi / REAL_POW((REAL_SQRT(epsh) + REAL_SQRT(epsi)), 2);
      real rmixh = 2 * (rminh * rminh * rminh + rmini * rmini * rmini) /
         (rminh * rminh + rmini * rmini);
      real rmixh7 = REAL_POW(rmixh, 7);
      real ah = emixh * rmixh7;
      real xi = x[i];
      real yi = y[i];
      real zi = z[i];
      real rio = rmixo / 2 + dispoff;
      real rih = rmixh / 2 + dispoff;

// remove contribution due to solvent displaced by solute atoms
#pragma acc loop vector independent
      for (int k = 0; k < n; ++k) {
         if (i != k) {
            int offset = (k + i * n) & (bufsize - 1);
            real sum = 0.0;
            real xr = x[k] - xi;
            real yr = y[k] - yi;
            real zr = z[k] - zi;
            real r2 = xr * xr + yr * yr + zr * zr;
            real r = REAL_SQRT(r2);
            real r3 = r * r * r;
            real rk = wcarad[k];
            real sk = rk * shctd;
            real sk2 = sk * sk;
            real de = 0;
            // Atom i with water oxygen
            if (rio < r + sk) {
               real rmax = REAL_MAX(rio, r - sk);
               real lik = rmax;
               if (lik < rmixo) {
                  real lik2 = lik * lik;
                  real lik3 = lik2 * lik;
                  real lik4 = lik3 * lik;
                  real uik = REAL_MIN(r + sk, rmixo);
                  real uik2 = uik * uik;
                  real uik3 = uik2 * uik;
                  real uik4 = uik3 * uik;
                  real term = 4 * M_PI / (48 * r) *
                     (3 * (lik4 - uik4) - 8 * r * (lik3 - uik3) +
                      6 * (r2 - sk2) * (lik2 - uik2));
                  if CONSTEXPR (do_g) {
                     real dl;
                     real du;
                     if (rio > r - sk) {
                        dl = -lik2 + 2 * r2 + 2 * sk2;
                        dl = dl * lik2;
                     } else {
                        real dl = -lik3 + 4 * lik2 * r - 6 * lik * r2 +
                           2 * lik * sk2 + 4 * r3 - 4 * r * sk2;
                        dl = dl * lik;
                     }
                     if (r + sk > rmixo) {
                        du = -uik2 + 2 * r2 + 2 * sk2;
                        du = -du * uik2;
                     } else {
                        du = -uik3 + 4 * uik2 * r - 6 * uik * r2 +
                           2 * uik * sk2 + 4 * r3 - 4 * r * sk2;
                        du = -du * uik;
                     }
                     de = de - emixo * M_PI * (dl + du) / (4 * r2);
                  }

                  real iwca = -emixo * term;
                  sum = sum + iwca;
               }
               real uik = r + sk;
               if (uik > rmixo) {
                  real uik2 = uik * uik;
                  real uik3 = uik2 * uik;
                  real uik4 = uik3 * uik;
                  real uik5 = uik4 * uik;
                  real uik6 = uik5 * uik;
                  real uik10 = uik5 * uik5;
                  real uik11 = uik10 * uik;
                  real uik12 = uik11 * uik;
                  real uik13 = uik12 * uik;
                  real lik = REAL_MAX(rmax, rmixo);
                  real lik2 = lik * lik;
                  real lik3 = lik2 * lik;
                  real lik4 = lik3 * lik;
                  real lik5 = lik4 * lik;
                  real lik6 = lik5 * lik;
                  real lik10 = lik5 * lik5;
                  real lik11 = lik10 * lik;
                  real lik12 = lik11 * lik;
                  real lik13 = lik12 * lik;
                  real term = 4 * M_PI / (120 * r * lik5 * uik5) *
                     (15 * uik * lik * r * (uik4 - lik4) -
                      10 * uik2 * lik2 * (uik3 - lik3) +
                      6 * (sk2 - r2) * (uik5 - lik5));
                  if CONSTEXPR (do_g) {
                     real dl;
                     real du;
                     if (rio > r - sk || rmax < rmixo) {
                        dl = -5 * lik2 + 3 * r2 + 3 * sk2;
                        dl = -dl / lik5;
                     } else {
                        dl = 5 * lik3 - 33 * lik * r2 - 3 * lik * sk2 +
                           15 * (lik2 * r + r3 - r * sk2);
                        dl = dl / lik6;
                     }
                     du = 5 * uik3 - 33 * uik * r2 - 3 * uik * sk2 +
                        15 * (uik2 * r + r3 - r * sk2);
                     du = -du / uik6;
                     de = de - 2 * ao * M_PI * (dl + du) / (15 * r2);
                  }

                  real idisp = -2 * ao * term;
                  term = 4 * M_PI / (2640 * r * lik12 * uik12) *
                     (120 * uik * lik * r * (uik11 - lik11) -
                      66 * uik2 * lik2 * (uik10 - lik10) +
                      55 * (sk2 - r2) * (uik12 - lik12));

                  if CONSTEXPR (do_g) {
                     real dl;
                     real du;
                     if (rio > r - sk || rmax < rmixo) {
                        dl = -6 * lik2 + 5 * r2 + 5 * sk2;
                        dl = -dl / lik12;
                     } else {
                        dl = 6 * lik3 - 125 * lik * r2 - 5 * lik * sk2 +
                           60 * (lik2 * r + r3 - r * sk2);
                        dl = dl / lik13;
                     }
                     du = 6 * uik3 - 125 * uik * r2 - 5 * uik * sk2 +
                        60 * (uik2 * r + r3 - r * sk2);
                     du = -du / uik13;
                     de = de + ao * rmixo7 * M_PI * (dl + du) / (60 * r2);
                  }

                  real irep = ao * rmixo7 * term;
                  sum = sum + irep + idisp;
               }
            }
            //  Atom i with water hydrogen
            if (rih < r + sk) {
               real rmax = REAL_MAX(rih, r - sk);
               real lik = rmax;
               if (lik < rmixh) {
                  real lik2 = lik * lik;
                  real lik3 = lik2 * lik;
                  real lik4 = lik3 * lik;
                  real uik = REAL_MIN(r + sk, rmixh);
                  real uik2 = uik * uik;
                  real uik3 = uik2 * uik;
                  real uik4 = uik3 * uik;
                  real term = 4 * M_PI / (48 * r) *
                     (3 * (lik4 - uik4) - 8 * r * (lik3 - uik3) +
                      6 * (r2 - sk2) * (lik2 - uik2));
                  if CONSTEXPR (do_g) {
                     real dl;
                     real du;
                     if (rih > r - sk) {
                        dl = -lik2 + 2 * r2 + 2 * sk2;
                        dl = dl * lik2;
                     } else {
                        dl = -lik3 + 4 * lik2 * r - 6 * lik * r2 +
                           2 * lik * sk2 + 4 * r3 - 4 * r * sk2;
                        dl = dl * lik;
                     }
                     if (r + sk > rmixh) {
                        du = -uik2 + 2 * r2 + 2 * sk2;
                        du = -du * uik2;
                     } else {
                        du = -uik3 + 4 * uik2 * r - 6 * uik * r2 +
                           2 * uik * sk2 + 4 * r3 - 4 * r * sk2;
                        du = -du * uik;
                     }
                     de = de - 2 * emixh * M_PI * (dl+du) / (4 * r2);
                  }
                  real iwca = -2 * emixh * term;
                  sum = sum + iwca;
               }
               real uik = r + sk;
               if (uik > rmixh) {
                  real uik2 = uik * uik;
                  real uik3 = uik2 * uik;
                  real uik4 = uik3 * uik;
                  real uik5 = uik4 * uik;
                  real uik6 = uik5 * uik;
                  real uik10 = uik5 * uik5;
                  real uik11 = uik10 * uik;
                  real uik12 = uik11 * uik;
                  real uik13 = uik12 * uik;
                  real lik = REAL_MAX(rmax, rmixh);
                  real lik2 = lik * lik;
                  real lik3 = lik2 * lik;
                  real lik4 = lik3 * lik;
                  real lik5 = lik4 * lik;
                  real lik6 = lik5 * lik;
                  real lik10 = lik5 * lik5;
                  real lik11 = lik10 * lik;
                  real lik12 = lik11 * lik;
                  real lik13 = lik12 * lik;
                  real term = 4 * M_PI / (120 * r * lik5 * uik5) *
                     (15 * uik * lik * r * (uik4 - lik4) -
                      10 * uik2 * lik2 * (uik3 - lik3) +
                      6 * (sk2 - r2) * (uik5 - lik5));
                  if CONSTEXPR(do_g){
                     real dl;
                     real du;
                     if (rih > r-sk || rmax < rmixh) {
                        dl = -5 * lik2 + 3 * r2 + 3 * sk2;
                        dl = -dl / lik5;
                     }
                     else {
                        dl = 5 * lik3 - 33 * lik * r2 - 3 * lik * sk2 + 15 * (lik2 * r + r3 - r * sk2);
                        dl = dl / lik6;
                     }
                     du = 5 * uik3 - 33 * uik * r2 - 3 * uik * sk2 + 15 * (uik2 * r + r3 - r * sk2);
                     du = -du / uik6;
                     de = de - 4 * ah * M_PI * (dl + du) / (15 * r2);
                  }
                  real idisp = -4 * ah * term;
                  term = 4 * M_PI / (2640 * r * lik12 * uik12) *
                     (120 * uik * lik * r * (uik11 - lik11) -
                      66 * uik2 * lik2 * (uik10 - lik10) +
                      55 * (sk2 - r2) * (uik12 - lik12));
                  if CONSTEXPR(do_g){
                     real dl;
                     real du;
                     if (rih > r-sk || rmax < rmixh) {
                        dl = -6 * lik2 + 5 * r2 + 5 * sk2;
                        dl = -dl / lik12;
                     }
                     else {
                        dl = 6 * lik3 - 125 * lik * r2 - 5 * lik * sk2 + 60 * (lik2 * r + r3 - r * sk2);
                        dl = dl / lik13;
                     }
                     du = 6 * uik3 - 125 * uik * r2 -5 * uik * sk2 + 60 * (uik2 * r + r3 - r * sk2);
                     du = -du / uik13;
                     de = de + ah * rmixh7 * M_PI * (dl+du) / (30 * r2);
                  }
                  real irep = 2 * ah * rmixh7 * term;
                  sum = sum + irep + idisp;
               }
            }
            if CONSTEXPR (do_e)
               atomic_add(-slevy * awater * sum, ewca, offset);
         }
      }
      // Add the isolated atom dispersion energy.
      if CONSTEXPR (do_e)
         atomic_add(wcacdisp[i], ewca, i & (bufsize - 1));
   }
}


void egkdisp_acc(int vers)
{
   if (vers == calc::v0)
      egkdisp_acc1<calc::V0>();
   else if (vers == calc::v1)
      egkdisp_acc1<calc::V1>();
   else if (vers == calc::v3)
      egkdisp_acc1<calc::V3>();
   else if (vers == calc::v4)
      egkdisp_acc1<calc::V4>();
   else if (vers == calc::v5)
      egkdisp_acc1<calc::V5>();
   else if (vers == calc::v6)
      egkdisp_acc1<calc::V6>();
}

}
