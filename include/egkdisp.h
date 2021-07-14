#pragma once
#include "tool/rc_man.h"
#include "tool/energy_buffer.h"

namespace tinker {

extern real* wcaeps;
extern real* wcarad;
extern real* wcacdisp;

extern count_buffer newca;
extern energy_buffer ewca;
extern virial_buffer vir_ewca;
extern grad_prec* dewcax;
extern grad_prec* dewcay;
extern grad_prec* dewcaz;
extern energy_prec energy_ewca;
extern virial_prec virial_ewca[9];

void egkdisp_data(rc_op op);

void egkdisp(int vers);
}
