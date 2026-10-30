/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 1991- The GROMACS Authors
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
 * \brief
 * Implements the CPU PME spline setup, scalar and SIMD over atoms.
 *
 * \ingroup module_ewald
 */
#include "gmxpre.h"

#include "pme_spline_setup.h"

#include "config.h"

#include <cassert>
#include <cstdint>

#include <algorithm>

#include "gromacs/simd/simd.h"
#include "gromacs/utility/basedefinitions.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/gmxassert.h"

#include "pme_grid.h"
#include "pme_internal.h"

namespace gmx
{

namespace
{

/*! \brief Stores the grid index and fraction for atom \p i given the
 * truncated scaled fractional coordinates \p ti and the fractional
 * parts \p frac = t - ti, and assigns the atom to a thread when requested.
 */
inline void storeInterpolationIndex(const PmeInterpolationIndexLookup& lookup,
                                    const int                          i,
                                    const int                          tix,
                                    const int                          tiy,
                                    const int                          tiz,
                                    const real                         fracX,
                                    const real                         fracY,
                                    const real                         fracZ,
                                    RVec*                              fractx,
                                    IVec*                              idx,
                                    const PmeAtomThreadAssignment*     threadAssignment)
{
#ifdef DEBUG
    range_check(tix, 0, c_pmeNeighborUnitcellCount * lookup.numGridPoints[XX]);
    range_check(tiy, 0, c_pmeNeighborUnitcellCount * lookup.numGridPoints[YY]);
    range_check(tiz, 0, c_pmeNeighborUnitcellCount * lookup.numGridPoints[ZZ]);
#endif
    /* Because decomposition only occurs in x and y,
     * we never have a fraction correction in z.
     */
    fractx[i][XX] = fracX + lookup.fractionShift[XX][tix];
    fractx[i][YY] = fracY + lookup.fractionShift[YY][tiy];
    fractx[i][ZZ] = fracZ;

    const int idxX = lookup.gridIndexToLocalIndex[XX][tix];
    const int idxY = lookup.gridIndexToLocalIndex[YY][tiy];
    const int idxZ = lookup.gridIndexToLocalIndex[ZZ][tiz];
    idx[i][XX]     = idxX;
    idx[i][YY]     = idxY;
    idx[i][ZZ]     = idxZ;

    if (threadAssignment)
    {
        const int thread = threadAssignment->gridToThread[XX][idxX]
                           + threadAssignment->gridToThread[YY][idxY]
                           + threadAssignment->gridToThread[ZZ][idxZ];
        threadAssignment->threadIndex[i] = thread;
        threadAssignment->threadCounts[thread]++;
    }
}

} // namespace

void pmeCalcInterpolationIndices(const PmeInterpolationIndexLookup& lookup,
                                 ArrayRef<const RVec>               x,
                                 const int                          start,
                                 const int                          end,
                                 ArrayRef<RVec>                     fractx,
                                 ArrayRef<IVec>                     idx,
                                 const PmeAtomThreadAssignment*     threadAssignment,
                                 const PmeSplineSetupKernel         kernel)
{
    GMX_ASSERT(end <= x.ssize() && end <= fractx.ssize() && end <= idx.ssize(),
               "Buffers should cover the atom range");

    const real nx = lookup.numGridPoints[XX];
    const real ny = lookup.numGridPoints[YY];
    const real nz = lookup.numGridPoints[ZZ];

    const real rxx = lookup.recipbox[XX][XX];
    const real ryx = lookup.recipbox[YY][XX];
    const real ryy = lookup.recipbox[YY][YY];
    const real rzx = lookup.recipbox[ZZ][XX];
    const real rzy = lookup.recipbox[ZZ][YY];
    const real rzz = lookup.recipbox[ZZ][ZZ];

    const real shift = c_pmeMaxUnitcellShift;

    int i = start;

#if GMX_SIMD_HAVE_REAL
    if (kernel == PmeSplineSetupKernel::Simd)
    {
        constexpr int c_simdWidth = GMX_SIMD_REAL_WIDTH;

        alignas(GMX_SIMD_ALIGNMENT) std::int32_t offsets[c_simdWidth];
        for (int k = 0; k < c_simdWidth; k++)
        {
            offsets[k] = k;
        }
        alignas(GMX_SIMD_ALIGNMENT) std::int32_t tixBuf[c_simdWidth];
        alignas(GMX_SIMD_ALIGNMENT) std::int32_t tiyBuf[c_simdWidth];
        alignas(GMX_SIMD_ALIGNMENT) std::int32_t tizBuf[c_simdWidth];
        alignas(GMX_SIMD_ALIGNMENT) real         fracXBuf[c_simdWidth];
        alignas(GMX_SIMD_ALIGNMENT) real         fracYBuf[c_simdWidth];
        alignas(GMX_SIMD_ALIGNMENT) real         fracZBuf[c_simdWidth];

        const SimdReal nxS(nx);
        const SimdReal nyS(ny);
        const SimdReal nzS(nz);
        const SimdReal rxxS(rxx);
        const SimdReal ryxS(ryx);
        const SimdReal ryyS(ryy);
        const SimdReal rzxS(rzx);
        const SimdReal rzyS(rzy);
        const SimdReal rzzS(rzz);
        const SimdReal shiftS(shift);

        /* gatherLoadUTranspose<3> may load a full SIMD width of reals starting
         * at the last atom, so we only use it when this stays within x.
         * The remaining atoms are handled by the scalar loop below.
         */
        const int numPaddingAtoms = (c_simdWidth + DIM - 1) / DIM;
        const int simdEnd = std::min(end, static_cast<int>(x.ssize()) - numPaddingAtoms + 1);

        for (; i + c_simdWidth <= simdEnd; i += c_simdWidth)
        {
            SimdReal xS, yS, zS;
            gatherLoadUTranspose<DIM>(x[i].as_vec(), offsets, &xS, &yS, &zS);

            /* Fractional coordinates along box vectors, add a positive shift to ensure
             * tx/ty/tz are positive for triclinic boxes
             */
            const SimdReal tx = nxS * (fma(zS, rzxS, fma(yS, ryxS, xS * rxxS)) + shiftS);
            const SimdReal ty = nyS * (fma(zS, rzyS, yS * ryyS) + shiftS);
            const SimdReal tz = nzS * (fma(zS, rzzS, shiftS));

            const SimdInt32 tix = cvttR2I(tx);
            const SimdInt32 tiy = cvttR2I(ty);
            const SimdInt32 tiz = cvttR2I(tz);

            store(tixBuf, tix);
            store(tiyBuf, tiy);
            store(tizBuf, tiz);
            store(fracXBuf, tx - cvtI2R(tix));
            store(fracYBuf, ty - cvtI2R(tiy));
            store(fracZBuf, tz - cvtI2R(tiz));

            for (int k = 0; k < c_simdWidth; k++)
            {
                storeInterpolationIndex(lookup,
                                        i + k,
                                        tixBuf[k],
                                        tiyBuf[k],
                                        tizBuf[k],
                                        fracXBuf[k],
                                        fracYBuf[k],
                                        fracZBuf[k],
                                        fractx.data(),
                                        idx.data(),
                                        threadAssignment);
            }
        }
    }
#else
    GMX_UNUSED_VALUE(kernel);
#endif

    for (; i < end; i++)
    {
        const real* xptr = x[i].as_vec();

        /* Fractional coordinates along box vectors, add a positive shift to ensure tx/ty/tz are positive for triclinic boxes */
        const real tx = nx * (xptr[XX] * rxx + xptr[YY] * ryx + xptr[ZZ] * rzx + shift);
        const real ty = ny * (xptr[YY] * ryy + xptr[ZZ] * rzy + shift);
        const real tz = nz * (xptr[ZZ] * rzz + shift);

        const int tix = static_cast<int>(tx);
        const int tiy = static_cast<int>(ty);
        const int tiz = static_cast<int>(tz);

        storeInterpolationIndex(
                lookup, i, tix, tiy, tiz, tx - tix, ty - tiy, tz - tiz, fractx.data(), idx.data(), threadAssignment);
    }
}

// At run time, the values of order used and asserted upon mean that
// indexing out of bounds does not occur. However compilers don't
// always understand that, so we suppress this warning for this code
// region.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"

/* Macro to force loop unrolling by fixing order.
 * This gives a significant performance gain.
 */
#define CALC_SPLINE(order)                                                                       \
    {                                                                                            \
        for (int j = 0; (j < DIM); j++)                                                          \
        {                                                                                        \
            real dr, div;                                                                        \
            real data[PME_ORDER_MAX];                                                            \
                                                                                                 \
            dr = xptr[j];                                                                        \
                                                                                                 \
            /* dr is relative offset from lower cell limit */                                    \
            data[(order) - 1] = 0;                                                               \
            data[1]           = dr;                                                              \
            data[0]           = 1 - dr;                                                          \
                                                                                                 \
            for (int k = 3; (k < (order)); k++)                                                  \
            {                                                                                    \
                div         = 1.0 / (k - 1.0);                                                   \
                data[k - 1] = div * dr * data[k - 2];                                            \
                for (int l = 1; (l < (k - 1)); l++)                                              \
                {                                                                                \
                    data[k - l - 1] =                                                            \
                            div * ((dr + l) * data[k - l - 2] + (k - l - dr) * data[k - l - 1]); \
                }                                                                                \
                data[0] = div * (1 - dr) * data[0];                                              \
            }                                                                                    \
            /* differentiate */                                                                  \
            dtheta[j][i * (order) + 0] = -data[0];                                               \
            for (int k = 1; (k < (order)); k++)                                                  \
            {                                                                                    \
                dtheta[j][i * (order) + k] = data[k - 1] - data[k];                              \
            }                                                                                    \
                                                                                                 \
            div               = 1.0 / ((order) - 1);                                             \
            data[(order) - 1] = div * dr * data[(order) - 2];                                    \
            for (int l = 1; (l < ((order) - 1)); l++)                                            \
            {                                                                                    \
                data[(order) - l - 1] = div                                                      \
                                        * ((dr + l) * data[(order) - l - 2]                      \
                                           + ((order) - l - dr) * data[(order) - l - 1]);        \
            }                                                                                    \
            data[0] = div * (1 - dr) * data[0];                                                  \
                                                                                                 \
            for (int k = 0; k < (order); k++)                                                    \
            {                                                                                    \
                theta[j][i * (order) + k] = data[k];                                             \
            }                                                                                    \
        }                                                                                        \
    }

namespace
{

//! Scalar computation of the splines for the atom with fraction \p xptr, stored at index \p i
inline void makeBsplineScalar(ArrayRef<real*> theta,
                              ArrayRef<real*> dtheta,
                              const int       order,
                              const real*     xptr,
                              const int       i)
{
    assert(order >= 3 && order <= PME_ORDER_MAX);
    switch (order)
    {
        case 4: CALC_SPLINE(4) break;
        case 5: CALC_SPLINE(5) break;
        default: CALC_SPLINE(order) break;
    }
}

#if GMX_SIMD_HAVE_REAL

/*! \brief SIMD computation of the splines for GMX_SIMD_REAL_WIDTH atoms
 *
 * Uses the same recursion as CALC_SPLINE, with atoms in SIMD lanes.
 * The results, per dimension \p order SIMD variables with one atom per lane,
 * are transposed through a stack buffer into the [atom*order] layout.
 *
 * \tparam c_order  The compile time PME order, or 0 for runtime \p runtimeOrder
 * \param  atomIndices   Aligned list of atom indices into \p fractx
 * \param  outputIndices The spline indices i to store the coefficients at
 */
template<int c_order>
inline void makeBsplinesSimd(ArrayRef<real*>     theta,
                             ArrayRef<real*>     dtheta,
                             const int           runtimeOrder,
                             const real*         fractx,
                             const std::int32_t* atomIndices,
                             const int*          outputIndices)
{
    constexpr int c_simdWidth = GMX_SIMD_REAL_WIDTH;

    const int order = (c_order > 0 ? c_order : runtimeOrder);

    SimdReal fracS[DIM];
    gatherLoadUTranspose<DIM>(fractx, atomIndices, &fracS[XX], &fracS[YY], &fracS[ZZ]);

    alignas(GMX_SIMD_ALIGNMENT) real thetaBuf[PME_ORDER_MAX * c_simdWidth];
    alignas(GMX_SIMD_ALIGNMENT) real dthetaBuf[PME_ORDER_MAX * c_simdWidth];

    const SimdReal one(static_cast<real>(1.0));

    for (int j = 0; j < DIM; j++)
    {
        const SimdReal dr = fracS[j];
        SimdReal       data[PME_ORDER_MAX];

        /* dr is relative offset from lower cell limit */
        data[order - 1] = setZero();
        data[1]         = dr;
        data[0]         = one - dr;

        for (int k = 3; k < order; k++)
        {
            const SimdReal div(static_cast<real>(1.0 / (k - 1.0)));
            data[k - 1] = div * dr * data[k - 2];
            for (int l = 1; l < k - 1; l++)
            {
                data[k - l - 1] = div
                                  * fma(dr + SimdReal(static_cast<real>(l)),
                                        data[k - l - 2],
                                        (SimdReal(static_cast<real>(k - l)) - dr) * data[k - l - 1]);
            }
            data[0] = div * (one - dr) * data[0];
        }
        /* differentiate */
        store(dthetaBuf, -data[0]);
        for (int k = 1; k < order; k++)
        {
            store(dthetaBuf + k * c_simdWidth, data[k - 1] - data[k]);
        }

        const SimdReal div(static_cast<real>(1.0 / (order - 1)));
        data[order - 1] = div * dr * data[order - 2];
        for (int l = 1; l < order - 1; l++)
        {
            data[order - l - 1] =
                    div
                    * fma(dr + SimdReal(static_cast<real>(l)),
                          data[order - l - 2],
                          (SimdReal(static_cast<real>(order - l)) - dr) * data[order - l - 1]);
        }
        data[0] = div * (one - dr) * data[0];

        for (int k = 0; k < order; k++)
        {
            store(thetaBuf + k * c_simdWidth, data[k]);
        }

        /* Transpose from [order][lane] to [atom*order] */
        real* gmx_restrict thetaJ  = theta[j];
        real* gmx_restrict dthetaJ = dtheta[j];
        for (int lane = 0; lane < c_simdWidth; lane++)
        {
            const int offset = outputIndices[lane] * order;
            for (int k = 0; k < order; k++)
            {
                thetaJ[offset + k]  = thetaBuf[k * c_simdWidth + lane];
                dthetaJ[offset + k] = dthetaBuf[k * c_simdWidth + lane];
            }
        }
    }
}

//! Dispatches to the SIMD spline kernel specialized for \p order when available
inline void makeBsplinesSimdDispatch(ArrayRef<real*>     theta,
                                     ArrayRef<real*>     dtheta,
                                     const int           order,
                                     const real*         fractx,
                                     const std::int32_t* atomIndices,
                                     const int*          outputIndices)
{
    switch (order)
    {
        case 4:
            makeBsplinesSimd<4>(theta, dtheta, order, fractx, atomIndices, outputIndices);
            break;
        case 5:
            makeBsplinesSimd<5>(theta, dtheta, order, fractx, atomIndices, outputIndices);
            break;
        default:
            makeBsplinesSimd<0>(theta, dtheta, order, fractx, atomIndices, outputIndices);
            break;
    }
}

#endif // GMX_SIMD_HAVE_REAL

} // namespace

void pmeMakeBsplines(ArrayRef<real*>            theta,
                     ArrayRef<real*>            dtheta,
                     const int                  order,
                     ArrayRef<const RVec>       fractx,
                     const int                  numAtoms,
                     const int*                 ind,
                     const real*                coefficient,
                     const bool                 computeAllSplineCoefficients,
                     const PmeSplineSetupKernel kernel)
{
    GMX_ASSERT(order >= 3 && order <= PME_ORDER_MAX, "The PME order should be in range");

    /* With free energy we do not use the coefficient check.
     * In most cases this will be more efficient than calling make_bsplines
     * twice, since usually more than half the particles have non-zero coefficients.
     */

#if GMX_SIMD_HAVE_REAL
    if (kernel == PmeSplineSetupKernel::Simd)
    {
        constexpr int c_simdWidth = GMX_SIMD_REAL_WIDTH;

        /* gatherLoadUTranspose<3> may load a full SIMD width of reals starting
         * at the last atom, atoms beyond this index are handled with scalar code.
         */
        const int maxSimdAtomIndex = static_cast<int>(fractx.ssize()) - (c_simdWidth + DIM - 1) / DIM;

        /* Collect atoms with non-zero coefficients (or all atoms) in lists
         * of SIMD width, so the theta/dtheta output layout and which entries
         * are written are identical to the scalar code.
         */
        alignas(GMX_SIMD_ALIGNMENT) std::int32_t atomIndices[c_simdWidth];
        int                                      outputIndices[c_simdWidth];
        int                                      numInList   = 0;
        int                                      maxAtomList = 0;

        for (int i = 0; i < numAtoms; i++)
        {
            const int ii = ind[i];
            if (computeAllSplineCoefficients || coefficient[ii] != 0.0)
            {
                atomIndices[numInList]   = ii;
                outputIndices[numInList] = i;
                maxAtomList              = std::max(maxAtomList, ii);
                numInList++;

                if (numInList == c_simdWidth)
                {
                    if (maxAtomList <= maxSimdAtomIndex)
                    {
                        makeBsplinesSimdDispatch(
                                theta, dtheta, order, fractx.data()[0].as_vec(), atomIndices, outputIndices);
                    }
                    else
                    {
                        for (int k = 0; k < c_simdWidth; k++)
                        {
                            makeBsplineScalar(
                                    theta, dtheta, order, fractx[atomIndices[k]].as_vec(), outputIndices[k]);
                        }
                    }
                    numInList   = 0;
                    maxAtomList = 0;
                }
            }
        }
        for (int k = 0; k < numInList; k++)
        {
            makeBsplineScalar(theta, dtheta, order, fractx[atomIndices[k]].as_vec(), outputIndices[k]);
        }

        return;
    }
#else
    GMX_UNUSED_VALUE(kernel);
#endif

    for (int i = 0; i < numAtoms; i++)
    {
        const int ii = ind[i];
        if (computeAllSplineCoefficients || coefficient[ii] != 0.0)
        {
            makeBsplineScalar(theta, dtheta, order, fractx[ii].as_vec(), i);
        }
    }
}

#pragma GCC diagnostic pop

} // namespace gmx
