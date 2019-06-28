#ifndef TINKER_GPU_E_BOND_H_
#define TINKER_GPU_E_BOND_H_

#include "decl_basic.h"

TINKER_NAMESPACE_BEGIN
namespace gpu {
enum { bond_harmonic = 0x001, bond_morse = 0x002 };
extern int bndtyp;
extern std::string bndtyp_str;

extern real cbnd, qbnd, bndunit;
extern int nbond;
extern int (*ibnd)[2];
extern real *bl, *bk;

extern real* eb;
extern real* vir_eb;

void ebond_data(rc_t rc);

void ebond_acc_impl__(int vers);
void ebond(int vers);
}
TINKER_NAMESPACE_END

#endif
