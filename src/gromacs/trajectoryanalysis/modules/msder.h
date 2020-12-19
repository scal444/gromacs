//
// Created by kevin on 12/18/20.
//

#ifndef GROMACS_MSDER_H
#define GROMACS_MSDER_H

#include <memory>

#include "gromacs/math/vectypes.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace gmx
{


class BufferManager {
public:
    BufferManager(int nAtoms);
    ~BufferManager();
    void AddFrame(float3* frame_start);
    real GetMsd(int frame_1, int frame_2);
private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};


}  // namespace gmx


#endif // GROMACS_MSDER_H
