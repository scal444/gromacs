/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2026- The GROMACS Authors
 * and the project initiators Erik Lindahl, Berk Hess and David van der Spoel.
 * Consult the AUTHORS/COPYING files and https://www.gromacs.org for details.
 *
 * GROMACS is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation; either version 2.1
 * of the License, or (at your option) any later version.
 *
 * GROMACS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with GROMACS; if not, see
 * https://www.gnu.org/licenses, or write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
 *
 * If you want to redistribute modifications to GROMACS, please
 * consider that scientific software is very special. Version
 * control is crucial - bugs must be traceable. We will be happy to
 * consider code for inclusion in the official distribution, but
 * derived work must not be called official GROMACS. Details are found
 * in the README & COPYING files - if they are missing, get the
 * official version at https://www.gromacs.org.
 *
 * To help us fund GROMACS development, we humbly ask that you cite
 * the research papers on the package. Check out https://www.gromacs.org.
 */
/*! \internal \file
 * \brief Implements the stochastic dynamics (SD) integrator update kernels.
 *
 * \ingroup module_mdlib
 */
#include "gmxpre.h"

#include "update_sd.h"

#include <cmath>

#include <algorithm>

#include "gromacs/random/seed.h"
#include "gromacs/random/tabulatednormaldistribution.h"
#include "gromacs/random/threefry.h"
#include "gromacs/simd/simd.h"
#include "gromacs/topology/topology_enums.h"
#include "gromacs/utility/gmxassert.h"

namespace gmx
{

namespace
{

/*! \brief Scalar reference SD integrator update, supports everything
 *
 * Three instantiations of this templated function will be made,
 * two with only one contribution, and one with both contributions.
 *
 * To maximize the ability of the compiler to optimize, all the arrays
 * of RVec should be annotated with gmx_restrict, so the compiler knows
 * there is no aliasing, and for the same reason we do not use
 * ArrayRef<RVec> for them. */
template<SDUpdate updateType>
void doSDUpdateGeneral(ArrayRef<const gmx_sd_const_t> sdc,
                       ArrayRef<const gmx_sd_sigma_t> sdsig,
                       int                            start,
                       int                            nrend,
                       real                           dt,
                       ArrayRef<const ivec>           nFreeze,
                       ArrayRef<const real>           invmass,
                       ArrayRef<const ParticleType>   ptype,
                       ArrayRef<const unsigned short> cFREEZE,
                       ArrayRef<const unsigned short> cTC,
                       ArrayRef<const unsigned short> cAcceleration,
                       const RVec* gmx_restrict       acceleration,
                       const RVec* gmx_restrict       x,
                       RVec* gmx_restrict             xprime,
                       RVec* gmx_restrict             v,
                       const RVec* gmx_restrict       f,
                       int64_t                        step,
                       int                            seed,
                       const int*                     gatindex,
                       real                           dtPressureCouple,
                       const Matrix3x3&               parrinelloRahmanM)
{
    // Even 0 bits internal counter gives 2x64 ints (more than enough for three table lookups)
    ThreeFry2x64<0>                       rng(seed, RandomDomain::UpdateCoordinates);
    TabulatedNormalDistribution<real, 14> dist;

    for (int n = start; n < nrend; n++)
    {
        const int globalAtomIndex = gatindex ? gatindex[n] : n;
        rng.restart(step, globalAtomIndex);
        dist.reset();

        const real inverseMass = invmass[n];
        const real invsqrtMass = std::sqrt(inverseMass);

        const int freezeGroup       = !cFREEZE.empty() ? cFREEZE[n] : 0;
        const int accelerationGroup = !cAcceleration.empty() ? cAcceleration[n] : 0;
        const int temperatureGroup  = !cTC.empty() ? cTC[n] : 0;

        RVec parrinelloRahmanScaledVelocity;
        if (updateType != SDUpdate::FrictionAndNoiseOnly)
        {
            parrinelloRahmanScaledVelocity = dtPressureCouple * (parrinelloRahmanM * v[n]);
        }
        for (int d = 0; d < DIM; d++)
        {
            if ((ptype[n] != ParticleType::Shell) && !nFreeze[freezeGroup][d])
            {
                if (updateType == SDUpdate::ForcesOnly)
                {
                    real vn = v[n][d] + (inverseMass * f[n][d] + acceleration[accelerationGroup][d]) * dt
                              - parrinelloRahmanScaledVelocity[d];
                    v[n][d] = vn;
                    // Simple position update.
                    xprime[n][d] = x[n][d] + v[n][d] * dt;
                }
                else if (updateType == SDUpdate::FrictionAndNoiseOnly)
                {
                    real vn = v[n][d];
                    v[n][d] = (vn * sdc[temperatureGroup].em
                               + invsqrtMass * sdsig[temperatureGroup].V * dist(rng));
                    // The previous phase already updated the
                    // positions with a full v*dt term that must
                    // now be half removed.
                    xprime[n][d] = xprime[n][d] + 0.5 * (v[n][d] - vn) * dt;
                }
                else
                {
                    real vn = v[n][d] + (inverseMass * f[n][d] + acceleration[accelerationGroup][d]) * dt
                              - parrinelloRahmanScaledVelocity[d];
                    v[n][d] = (vn * sdc[temperatureGroup].em
                               + invsqrtMass * sdsig[temperatureGroup].V * dist(rng));
                    // Here we include half of the friction+noise
                    // update of v into the position update.
                    xprime[n][d] = x[n][d] + 0.5 * (vn + v[n][d]) * dt;
                }
            }
            else
            {
                // When using constraints, the update is split into
                // two phases, but we only need to zero the update of
                // virtual, shell or frozen particles in at most one
                // of the phases.
                if (updateType != SDUpdate::FrictionAndNoiseOnly)
                {
                    v[n][d]      = 0.0;
                    xprime[n][d] = x[n][d];
                }
            }
        }
    }
}

#if GMX_SIMD && GMX_SIMD_HAVE_REAL

//! The SIMD width used for the SD update
constexpr int c_sdSimdWidth = GMX_SIMD_REAL_WIDTH;
//! The number of atoms per chunk for which the random numbers are pre-generated
constexpr int c_sdChunkSize = std::max(64, GMX_SIMD_REAL_WIDTH);
static_assert(c_sdChunkSize % c_sdSimdWidth == 0,
              "The chunk size should be a multiple of the SIMD width");

//! Unaligned load of c_sdSimdWidth RVecs starting at \p r into 3 SIMD registers, flat order
inline void loadRvecsU(const RVec* r, SimdReal* r0, SimdReal* r1, SimdReal* r2)
{
    const real* p = r[0];
    *r0           = loadU<SimdReal>(p);
    *r1           = loadU<SimdReal>(p + c_sdSimdWidth);
    *r2           = loadU<SimdReal>(p + 2 * c_sdSimdWidth);
}

//! Unaligned store of 3 SIMD registers to c_sdSimdWidth RVecs starting at \p r, flat order
inline void storeRvecsU(RVec* r, SimdReal r0, SimdReal r1, SimdReal r2)
{
    real* p = r[0];
    storeU(p, r0);
    storeU(p + c_sdSimdWidth, r1);
    storeU(p + 2 * c_sdSimdWidth, r2);
}

#    if !GMX_DOUBLE && GMX_SIMD_HAVE_DOUBLE && (GMX_SIMD_FLOAT_WIDTH == 2 * GMX_SIMD_DOUBLE_WIDTH)
//! Whether the friction is applied in double precision, as the scalar kernel does
#        define GMX_SD_SIMD_DOUBLE_FRICTION 1
//! The type for the friction factor
using SDFrictionType = double;
#    else
#        define GMX_SD_SIMD_DOUBLE_FRICTION 0
//! The type for the friction factor
using SDFrictionType = real;
#    endif

/*! \brief Returns \p vn * \p em + \p noise, with \p em loaded from \p emFlat
 *
 * In mixed precision, i.e. with float as real, the scalar kernel computes
 * this expression in double precision, since the friction factor em
 * is stored in double precision. When the SIMD width allows conversion
 * between float and double, we do the same here. This gives bitwise
 * identical results when the compiler contracts the scalar expression
 * into an FMA, as GCC does by default.
 */
inline SimdReal frictionAndNoise(SimdReal vn, const SDFrictionType* emFlat, SimdReal noise)
{
#    if GMX_SD_SIMD_DOUBLE_FRICTION
    SimdDouble vnLo, vnHi, noiseLo, noiseHi;
    cvtF2DD(vn, &vnLo, &vnHi);
    cvtF2DD(noise, &noiseLo, &noiseHi);
    const SimdDouble emLo = load<SimdDouble>(emFlat);
    const SimdDouble emHi = load<SimdDouble>(emFlat + GMX_SIMD_DOUBLE_WIDTH);
    return cvtDD2F(fma(vnLo, emLo, noiseLo), fma(vnHi, emHi, noiseHi));
#    else
    return fma(vn, load<SimdReal>(emFlat), noise);
#    endif
}

/*! \brief SIMD SD integrator update for the atoms in whole SIMD blocks starting at \p start
 *
 * Requires that sdUpdateSimdIsSupported() returns true. Each atom is either
 * fully updated or fully skipped (shells and fully frozen atoms).
 *
 * The normal random numbers are generated per chunk of atoms with exactly
 * the same scalar per-atom generator calls as doSDUpdateGeneral(), and
 * pre-multiplied by the same per-atom noise magnitude in the same order,
 * so the noise term is bitwise identical to the scalar kernel. The rest of
 * the arithmetic is done in SIMD with FMA, with the friction term in double
 * precision when possible (see frictionAndNoise()). Any differences with
 * the scalar kernel are at rounding level and depend on how the compiler
 * contracts the scalar expressions into FMAs.
 *
 * \returns the index of the first atom not updated, at most \p nrend.
 */
template<SDUpdate updateType>
int doSDUpdateSimd(ArrayRef<const gmx_sd_const_t> sdc,
                   ArrayRef<const gmx_sd_sigma_t> sdsig,
                   int                            start,
                   int                            nrend,
                   real                           dt,
                   ArrayRef<const ivec>           nFreeze,
                   ArrayRef<const real>           invmass,
                   ArrayRef<const ParticleType>   ptype,
                   ArrayRef<const unsigned short> cFREEZE,
                   ArrayRef<const unsigned short> cTC,
                   const RVec&                    acceleration,
                   const RVec* gmx_restrict       x,
                   RVec* gmx_restrict             xprime,
                   RVec* gmx_restrict             v,
                   const RVec* gmx_restrict       f,
                   int64_t                        step,
                   int                            seed,
                   const int*                     gatindex)
{
    constexpr bool haveForces = (updateType != SDUpdate::FrictionAndNoiseOnly);
    constexpr bool haveNoise  = (updateType != SDUpdate::ForcesOnly);

    const int simdEnd = start + ((nrend - start) / c_sdSimdWidth) * c_sdSimdWidth;

    ThreeFry2x64<0>                       rng(seed, RandomDomain::UpdateCoordinates);
    TabulatedNormalDistribution<real, 14> dist;

    // Per-chunk buffers: 1 or 0 for updated/skipped atoms, the friction factor
    // per atom and dimension and the noise term per atom and dimension, both in RVec layout.
    AlignedArray<real, c_sdChunkSize>                 active;
    AlignedArray<SDFrictionType, DIM * c_sdChunkSize> emFlat;
    AlignedArray<real, DIM * c_sdChunkSize>           noise;
    AlignedArray<real, DIM * c_sdSimdWidth>           accelerationFlat;
    for (int i = 0; i < DIM * c_sdSimdWidth; i++)
    {
        accelerationFlat[i] = acceleration[i % DIM];
    }
    const SimdReal acc0 = load<SimdReal>(accelerationFlat.data());
    const SimdReal acc1 = load<SimdReal>(accelerationFlat.data() + c_sdSimdWidth);
    const SimdReal acc2 = load<SimdReal>(accelerationFlat.data() + 2 * c_sdSimdWidth);

    const SimdReal timestep(dt);
    const SimdReal halfTimestep(real(0.5) * dt);
    const SimdReal zero = setZero();

    for (int chunkStart = start; chunkStart < simdEnd; chunkStart += c_sdChunkSize)
    {
        const int chunkEnd = std::min(chunkStart + c_sdChunkSize, simdEnd);

        // Scalar part: per-atom parameters and random numbers
        for (int n = chunkStart; n < chunkEnd; n++)
        {
            const int  i           = n - chunkStart;
            const int  freezeGroup = !cFREEZE.empty() ? cFREEZE[n] : 0;
            const bool isActive    = (ptype[n] != ParticleType::Shell && !nFreeze[freezeGroup][XX]);
            active[i]              = isActive ? 1 : 0;
            if constexpr (haveNoise)
            {
                const int temperatureGroup = !cTC.empty() ? cTC[n] : 0;
                for (int d = 0; d < DIM; d++)
                {
                    emFlat[DIM * i + d] = static_cast<SDFrictionType>(sdc[temperatureGroup].em);
                }
                if (isActive)
                {
                    const int globalAtomIndex = gatindex ? gatindex[n] : n;
                    rng.restart(step, globalAtomIndex);
                    dist.reset();

                    const real invsqrtMass = std::sqrt(invmass[n]);
                    for (int d = 0; d < DIM; d++)
                    {
                        noise[DIM * i + d] = invsqrtMass * sdsig[temperatureGroup].V * dist(rng);
                    }
                }
                else
                {
                    for (int d = 0; d < DIM; d++)
                    {
                        noise[DIM * i + d] = 0;
                    }
                }
            }
        }

        // SIMD part
        for (int n = chunkStart; n < chunkEnd; n += c_sdSimdWidth)
        {
            const int i = n - chunkStart;

            SimdReal active0, active1, active2;
            expandScalarsToTriplets(load<SimdReal>(active.data() + i), &active0, &active1, &active2);
            const SimdBool mask0 = (active0 != zero);
            const SimdBool mask1 = (active1 != zero);
            const SimdBool mask2 = (active2 != zero);

            SimdReal v0, v1, v2;
            loadRvecsU(v + n, &v0, &v1, &v2);

            SimdReal vn0 = v0;
            SimdReal vn1 = v1;
            SimdReal vn2 = v2;
            if constexpr (haveForces)
            {
                SimdReal invMass0, invMass1, invMass2;
                expandScalarsToTriplets(loadU<SimdReal>(invmass.data() + n), &invMass0, &invMass1, &invMass2);
                SimdReal f0, f1, f2;
                loadRvecsU(f + n, &f0, &f1, &f2);
                vn0 = fma(fma(invMass0, f0, acc0), timestep, v0);
                vn1 = fma(fma(invMass1, f1, acc1), timestep, v1);
                vn2 = fma(fma(invMass2, f2, acc2), timestep, v2);
            }

            SimdReal vNew0 = vn0;
            SimdReal vNew1 = vn1;
            SimdReal vNew2 = vn2;
            if constexpr (haveNoise)
            {
                const real*           noisePtr = noise.data() + DIM * i;
                const SDFrictionType* emPtr    = emFlat.data() + DIM * i;
                vNew0 = frictionAndNoise(vn0, emPtr, load<SimdReal>(noisePtr));
                vNew1 = frictionAndNoise(
                        vn1, emPtr + c_sdSimdWidth, load<SimdReal>(noisePtr + c_sdSimdWidth));
                vNew2 = frictionAndNoise(
                        vn2, emPtr + 2 * c_sdSimdWidth, load<SimdReal>(noisePtr + 2 * c_sdSimdWidth));
            }

            if constexpr (updateType == SDUpdate::FrictionAndNoiseOnly)
            {
                // Skipped atoms are left untouched in this phase
                SimdReal xp0, xp1, xp2;
                loadRvecsU(xprime + n, &xp0, &xp1, &xp2);
                xp0 = blend(xp0, fma(vNew0 - vn0, halfTimestep, xp0), mask0);
                xp1 = blend(xp1, fma(vNew1 - vn1, halfTimestep, xp1), mask1);
                xp2 = blend(xp2, fma(vNew2 - vn2, halfTimestep, xp2), mask2);
                storeRvecsU(v + n, blend(v0, vNew0, mask0), blend(v1, vNew1, mask1), blend(v2, vNew2, mask2));
                storeRvecsU(xprime + n, xp0, xp1, xp2);
            }
            else
            {
                // Skipped atoms get zero velocity and unchanged position
                SimdReal x0, x1, x2;
                loadRvecsU(x + n, &x0, &x1, &x2);
                SimdReal xp0, xp1, xp2;
                if constexpr (updateType == SDUpdate::ForcesOnly)
                {
                    xp0 = fma(vNew0, timestep, x0);
                    xp1 = fma(vNew1, timestep, x1);
                    xp2 = fma(vNew2, timestep, x2);
                }
                else
                {
                    xp0 = fma(vn0 + vNew0, halfTimestep, x0);
                    xp1 = fma(vn1 + vNew1, halfTimestep, x1);
                    xp2 = fma(vn2 + vNew2, halfTimestep, x2);
                }
                storeRvecsU(v + n,
                            selectByMask(vNew0, mask0),
                            selectByMask(vNew1, mask1),
                            selectByMask(vNew2, mask2));
                storeRvecsU(xprime + n, blend(x0, xp0, mask0), blend(x1, xp1, mask1), blend(x2, xp2, mask2));
            }
        }
    }

    return simdEnd;
}

#endif // GMX_SIMD && GMX_SIMD_HAVE_REAL

} // namespace

#if GMX_SIMD && GMX_SIMD_HAVE_REAL
const bool c_haveSimdSDUpdate = true;
#else
const bool c_haveSimdSDUpdate = false;
#endif

bool sdUpdateSimdIsSupported(ArrayRef<const ivec> nFreeze, bool haveAccelerationGroups, real dtPressureCouple)
{
    if (!c_haveSimdSDUpdate || haveAccelerationGroups || dtPressureCouple != 0)
    {
        return false;
    }
    // The SIMD kernel handles atoms that are either fully frozen or not frozen at all
    return std::all_of(nFreeze.begin(),
                       nFreeze.end(),
                       [](const ivec& freeze) {
                           return (freeze[XX] != 0) == (freeze[YY] != 0)
                                  && (freeze[XX] != 0) == (freeze[ZZ] != 0);
                       });
}

void doSDUpdate(SDUpdate                       updateType,
                SDUpdateKernel                 kernel,
                ArrayRef<const gmx_sd_const_t> sdc,
                ArrayRef<const gmx_sd_sigma_t> sdsig,
                int                            start,
                int                            nrend,
                real                           dt,
                ArrayRef<const ivec>           nFreeze,
                ArrayRef<const real>           invmass,
                ArrayRef<const ParticleType>   ptype,
                ArrayRef<const unsigned short> cFREEZE,
                ArrayRef<const unsigned short> cTC,
                ArrayRef<const unsigned short> cAcceleration,
                const RVec*                    acceleration,
                const RVec*                    x,
                RVec*                          xprime,
                RVec*                          v,
                const RVec*                    f,
                int64_t                        step,
                int                            seed,
                const int*                     gatindex,
                real                           dtPressureCouple,
                const Matrix3x3&               parrinelloRahmanM)
{
    // cTC, cACC and cFREEZE can be empty any time, but various
    // update types do not make sense with particular pointer values.
    if (updateType == SDUpdate::ForcesOnly)
    {
        GMX_ASSERT(f != nullptr, "SD update with only forces requires forces");
        GMX_ASSERT(cTC.empty(), "SD update with only forces cannot handle temperature groups");
    }
    if (updateType == SDUpdate::FrictionAndNoiseOnly)
    {
        GMX_ASSERT(f == nullptr, "SD update with only noise cannot handle forces");
        GMX_ASSERT(cAcceleration.empty(),
                   "SD update with only noise cannot handle acceleration groups");
    }
    if (updateType == SDUpdate::Combined)
    {
        GMX_ASSERT(f != nullptr, "SD update with forces and noise requires forces");
    }

    int scalarStart = start;

#if GMX_SIMD && GMX_SIMD_HAVE_REAL
    if (kernel == SDUpdateKernel::SimdWhenPossible
        && sdUpdateSimdIsSupported(nFreeze, !cAcceleration.empty(), dtPressureCouple))
    {
        // Without acceleration groups, all atoms use acceleration group 0
        // (only accessed when forces are applied).
        const RVec accelerationGroup0 =
                (updateType != SDUpdate::FrictionAndNoiseOnly) ? acceleration[0] : RVec{ 0, 0, 0 };
        switch (updateType)
        {
            case SDUpdate::ForcesOnly:
                scalarStart = doSDUpdateSimd<SDUpdate::ForcesOnly>(sdc,
                                                                   sdsig,
                                                                   start,
                                                                   nrend,
                                                                   dt,
                                                                   nFreeze,
                                                                   invmass,
                                                                   ptype,
                                                                   cFREEZE,
                                                                   cTC,
                                                                   accelerationGroup0,
                                                                   x,
                                                                   xprime,
                                                                   v,
                                                                   f,
                                                                   step,
                                                                   seed,
                                                                   gatindex);
                break;
            case SDUpdate::FrictionAndNoiseOnly:
                scalarStart = doSDUpdateSimd<SDUpdate::FrictionAndNoiseOnly>(sdc,
                                                                             sdsig,
                                                                             start,
                                                                             nrend,
                                                                             dt,
                                                                             nFreeze,
                                                                             invmass,
                                                                             ptype,
                                                                             cFREEZE,
                                                                             cTC,
                                                                             accelerationGroup0,
                                                                             x,
                                                                             xprime,
                                                                             v,
                                                                             f,
                                                                             step,
                                                                             seed,
                                                                             gatindex);
                break;
            case SDUpdate::Combined:
                scalarStart = doSDUpdateSimd<SDUpdate::Combined>(sdc,
                                                                 sdsig,
                                                                 start,
                                                                 nrend,
                                                                 dt,
                                                                 nFreeze,
                                                                 invmass,
                                                                 ptype,
                                                                 cFREEZE,
                                                                 cTC,
                                                                 accelerationGroup0,
                                                                 x,
                                                                 xprime,
                                                                 v,
                                                                 f,
                                                                 step,
                                                                 seed,
                                                                 gatindex);
                break;
            default: GMX_RELEASE_ASSERT(false, "Invalid SD update type");
        }
    }
#else
    GMX_UNUSED_VALUE(kernel);
#endif

    // The scalar kernel handles everything not done by SIMD,
    // which can be the remainder of atoms not filling a SIMD register.
    switch (updateType)
    {
        case SDUpdate::ForcesOnly:
            doSDUpdateGeneral<SDUpdate::ForcesOnly>(sdc,
                                                    sdsig,
                                                    scalarStart,
                                                    nrend,
                                                    dt,
                                                    nFreeze,
                                                    invmass,
                                                    ptype,
                                                    cFREEZE,
                                                    cTC,
                                                    cAcceleration,
                                                    acceleration,
                                                    x,
                                                    xprime,
                                                    v,
                                                    f,
                                                    step,
                                                    seed,
                                                    gatindex,
                                                    dtPressureCouple,
                                                    parrinelloRahmanM);
            break;
        case SDUpdate::FrictionAndNoiseOnly:
            doSDUpdateGeneral<SDUpdate::FrictionAndNoiseOnly>(sdc,
                                                              sdsig,
                                                              scalarStart,
                                                              nrend,
                                                              dt,
                                                              nFreeze,
                                                              invmass,
                                                              ptype,
                                                              cFREEZE,
                                                              cTC,
                                                              cAcceleration,
                                                              acceleration,
                                                              x,
                                                              xprime,
                                                              v,
                                                              f,
                                                              step,
                                                              seed,
                                                              gatindex,
                                                              dtPressureCouple,
                                                              parrinelloRahmanM);
            break;
        case SDUpdate::Combined:
            doSDUpdateGeneral<SDUpdate::Combined>(sdc,
                                                  sdsig,
                                                  scalarStart,
                                                  nrend,
                                                  dt,
                                                  nFreeze,
                                                  invmass,
                                                  ptype,
                                                  cFREEZE,
                                                  cTC,
                                                  cAcceleration,
                                                  acceleration,
                                                  x,
                                                  xprime,
                                                  v,
                                                  f,
                                                  step,
                                                  seed,
                                                  gatindex,
                                                  dtPressureCouple,
                                                  parrinelloRahmanM);
            break;
        default: GMX_RELEASE_ASSERT(false, "Invalid SD update type");
    }
}

} // namespace gmx
