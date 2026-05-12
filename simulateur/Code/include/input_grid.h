#ifndef SPLITMERGE_H
#define SPLITMERGE_H

#ifndef CPU_VERSION
#define CPU_VERSION 1
#endif


#ifdef __cplusplus
extern "C" {
#endif

#include "preesm.h"

void input_grid_setup(int GRID_SIZE, IN Config *config, OUT float2 *output);

#ifdef __cplusplus
}
#endif

#endif
