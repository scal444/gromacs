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
 *
 * \brief Helper for accumulating shift forces in SIMD bonded kernels
 *
 * \ingroup module_listed_forces
 */
#ifndef GMX_LISTED_FORCES_SHIFT_FORCES_SIMD_H
#define GMX_LISTED_FORCES_SHIFT_FORCES_SIMD_H

#include "config.h"

#include "gromacs/pbcutil/ishift.h"
#include "gromacs/simd/simd.h"
#include "gromacs/utility/real.h"
#include "gromacs/utility/vectypes.h"

#if GMX_SIMD_HAVE_REAL

namespace gmx
{

/*! \internal
 * \brief Accumulates shift-force contributions in SIMD registers
 *
 * The scalar bonded kernels add the force on each atom to fshift[s], with s the
 * periodic shift of the image of that atom. Only the virial contribution
 * -0.5 sum_s shift_vec[s] x fshift[s] matters. The SIMD PBC correction
 * pbc_correct_dx_simd() with shifts returns, for each difference vector
 * dx = x_a - x_b, the number of box vectors sh[d] that were subtracted.
 * The image of atom a relative to atom b is then shifted by -sum_d sh[d]*box[d],
 * so the shift-force contribution of force f acting on atom a is -sh[d]*f
 * on the shift slot of unit box vector d. This class accumulates
 * G[d] = sum sh[d]*f and adds -G[d] to the unit shift slots (and +sum_d G[d]
 * to the central slot, so the sum over all shift forces is unchanged).
 *
 * This gives the same virial as the scalar kernels, but the distribution
 * over the individual shift indices differs when images are shifted
 * along more than one box vector.
 */
class SimdShiftForceAccumulator
{
public:
    SimdShiftForceAccumulator()
    {
        for (int d = 0; d < DIM; d++)
        {
            for (int m = 0; m < DIM; m++)
            {
                g_[d][m] = setZero();
            }
        }
    }

    /*! \brief Adds the shift contribution of force (fx,fy,fz) acting on atom a
     *
     * \param[in] sh  The PBC shifts returned by pbc_correct_dx_simd() for dx = x_a - x_b
     * \param[in] fx  Force x-component acting on atom a
     * \param[in] fy  Force y-component acting on atom a
     * \param[in] fz  Force z-component acting on atom a
     */
    inline void gmx_simdcall add(const SimdReal sh[DIM], SimdReal fx, SimdReal fy, SimdReal fz)
    {
        for (int d = 0; d < DIM; d++)
        {
            g_[d][XX] = fma(sh[d], fx, g_[d][XX]);
            g_[d][YY] = fma(sh[d], fy, g_[d][YY]);
            g_[d][ZZ] = fma(sh[d], fz, g_[d][ZZ]);
        }
    }

    //! As add(), but for the force acting on atom b, i.e. -f on atom a
    inline void gmx_simdcall sub(const SimdReal sh[DIM], SimdReal fx, SimdReal fy, SimdReal fz)
    {
        for (int d = 0; d < DIM; d++)
        {
            g_[d][XX] = fnma(sh[d], fx, g_[d][XX]);
            g_[d][YY] = fnma(sh[d], fy, g_[d][YY]);
            g_[d][ZZ] = fnma(sh[d], fz, g_[d][ZZ]);
        }
    }

    //! Reduces the SIMD accumulators and adds the result to the shift-force buffer
    void addToShiftForces(rvec* fshift) const
    {
        for (int d = 0; d < DIM; d++)
        {
            const int shiftIndex = xyzToShiftIndex(d == XX ? 1 : 0, d == YY ? 1 : 0, d == ZZ ? 1 : 0);
            for (int m = 0; m < DIM; m++)
            {
                const real g = reduce(g_[d][m]);
                fshift[shiftIndex][m] -= g;
                fshift[c_centralShiftIndex][m] += g;
            }
        }
    }

private:
    //! Accumulated shift counts times forces, indexed by box vector and force component
    SimdReal g_[DIM][DIM];
};

} // namespace gmx

#endif // GMX_SIMD_HAVE_REAL

#endif
