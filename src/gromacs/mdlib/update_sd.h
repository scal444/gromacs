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
 * \brief Declares the stochastic dynamics (SD) integrator update kernels.
 *
 * The kernels come in a scalar reference version, which supports
 * everything, and a SIMD version for the common case. The SIMD
 * version generates the per-atom normal random numbers with exactly
 * the same scalar random stream as the reference, so trajectories
 * remain independent of the SIMD width and thread count.
 *
 * \ingroup module_mdlib
 */
#ifndef GMX_MDLIB_UPDATE_SD_H
#define GMX_MDLIB_UPDATE_SD_H

#include <cstdint>

#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/matrix.h"
#include "gromacs/utility/real.h"
#include "gromacs/utility/vectypes.h"

enum class ParticleType : int;

namespace gmx
{

//! SD friction constant per temperature-coupling group
struct gmx_sd_const_t
{
    //! exp(-dt/tau_t)
    double em = 0;
};

//! SD noise magnitude per temperature-coupling group
struct gmx_sd_sigma_t
{
    //! sqrt(kT (1 - em^2)), the mass is accounted for per atom
    real V = 0;
};

/*! \brief Sets the SD update type */
enum class SDUpdate : int
{
    ForcesOnly,
    FrictionAndNoiseOnly,
    Combined,
    Count
};

//! Which kernel implementation to use for the SD update
enum class SDUpdateKernel : int
{
    //! Scalar reference implementation, supports everything
    Scalar,
    //! SIMD implementation when supported for this input, otherwise falls back to scalar
    SimdWhenPossible
};

//! Whether a SIMD SD update kernel is compiled in
extern const bool c_haveSimdSDUpdate;

/*! \brief Returns whether the SIMD SD kernel supports the given input
 *
 * Conditions: SIMD support is compiled in, no freeze group has only
 * part of its dimensions frozen, no acceleration groups are used
 * (\p haveAccelerationGroups false; the constant acceleration of group 0 is
 * supported) and there is no Parrinello-Rahman velocity scaling at this
 * step (\p dtPressureCouple == 0).
 */
bool sdUpdateSimdIsSupported(ArrayRef<const ivec> nFreeze, bool haveAccelerationGroups, real dtPressureCouple);

/*! \brief SD integrator update for atoms \p start to \p nrend - 1
 *
 * Two phases are required in the general case of a constrained
 * update, the first phase from the contribution of forces, before
 * applying constraints, and then a second phase applying the friction
 * and noise, and then further constraining. For details, see
 * Goga2012.
 *
 * Without constraints, the two phases can be combined, for
 * efficiency.
 *
 * cTC, cAcceleration and cFREEZE can be empty, which implies group 0
 * for all atoms. The arrays of RVec must not alias.
 *
 * With \p kernel == SimdWhenPossible and sdUpdateSimdIsSupported()
 * returning true, a SIMD kernel is used that produces bitwise
 * identical random numbers to the scalar kernel and the same
 * results up to rounding differences in the arithmetic.
 */
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
                const Matrix3x3&               parrinelloRahmanM);

} // namespace gmx

#endif
