//
// Created by kevin on 12/18/20.
//

#include "msder.h"

#include <thrust/device_vector.h>
#include <thrust/transform.h>

#include "gromacs/math/vectypes.h"

namespace gmx {

struct MSD_functor{
    __host__ __device__ float operator()(const float3& f1, const float3& f2) const {
        float displacement =  (f1.x - f2.x) + (f1.y - f2.y) + (f1.z - f2.z);
        return displacement * displacement;
    }
};

real MeanSquaredDisplacement(const thrust::device_vector<float3>& c1, const thrust::device_vector<float3>& c2, thrust::device_vector<float>* accumulator ) {
    thrust::transform(c1.begin(), c1.end(), c2.begin(), accumulator->begin(), MSD_functor());
    return thrust::reduce(accumulator->begin(), accumulator->end(), 0.0) / c1.size();
}

class BufferManager::Impl {
public:
    Impl(int nAtoms)
    : nAtoms_(nAtoms)
    {
        accumulator_.reserve(nAtoms);
    }
    ~Impl() = default;
    void AddFrame(float3* frame_start) {
        frames_.push_back(thrust::device_vector<float3>(frame_start, frame_start + nAtoms_));
    }

    real GetMsd(int frame_1, int frame_2) {
        return MeanSquaredDisplacement(frames_[frame_1], frames_[frame_2], &accumulator_);
    }
private:
    int nAtoms_;
    thrust::host_vector<thrust::device_vector<float3>> frames_;
    thrust::device_vector<float> accumulator_;

};



BufferManager::BufferManager(int nAtoms) : impl_(new Impl(nAtoms)) {}

BufferManager::~BufferManager() = default;

void BufferManager::AddFrame(float3* frame_start) {
    impl_->AddFrame(frame_start);
}

real BufferManager::GetMsd(int frame_1, int frame_2) {
    return impl_->GetMsd(frame_1, frame_2);
}





}  // namespace gmx