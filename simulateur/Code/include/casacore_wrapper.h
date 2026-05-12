//
// Created by orenaud on 4/14/25.
//

#ifndef CASACORE_WRAPPER_H
#define CASACORE_WRAPPER_H



#ifdef __cplusplus
extern "C" {
#endif

#include "common.h"


void load_visibilities_from_ms(const char* ms_path, int num_vis,
                                   Config* config, float3* uvw_coords);

#ifdef __cplusplus
}
#endif

#endif //CASACORE_WRAPPER_H
