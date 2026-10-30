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
 * \brief
 * Declares the CPU PME spline setup: computation of the interpolation
 * grid indices and fractions, and of the B-spline coefficients.
 *
 * Both routines have a SIMD implementation, vectorized over atoms, and
 * a scalar reference implementation which is selectable for testing.
 *
 * \ingroup module_ewald
 */
#ifndef GMX_EWALD_PME_SPLINE_SETUP_H
#define GMX_EWALD_PME_SPLINE_SETUP_H

#include <array>

#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/real.h"
#include "gromacs/utility/vectypes.h"

namespace gmx
{

//! Which implementation of the spline setup routines to use
enum class PmeSplineSetupKernel
{
    Scalar, //!< The scalar reference implementation
    Simd    //!< SIMD over atoms, falls back to Scalar when SIMD is not available
};

//! The kernel used by default in the PME code
static constexpr PmeSplineSetupKernel c_pmeSplineSetupKernelDefault = PmeSplineSetupKernel::Simd;

/*! \internal
 * \brief Grid parameters and lookup tables for computing interpolation indices
 */
struct PmeInterpolationIndexLookup
{
    //! The number of grid points along each dimension
    IVec numGridPoints;
    //! The reciprocal box, lower triangular
    matrix recipbox;
    //! Global to local grid line index tables, size c_pmeNeighborUnitcellCount * numGridPoints[d]
    std::array<ArrayRef<const int>, DIM> gridIndexToLocalIndex;
    //! Fraction shift tables for x and y (z is never decomposed), same size as the index tables
    std::array<ArrayRef<const real>, DIM - 1> fractionShift;
};

/*! \internal
 * \brief Output buffers for the assignment of atoms to spreading threads
 *
 * When used, for each atom the thread index is computed from the local
 * grid index using the grid-to-thread tables and \p threadCounts[thread]
 * is incremented.
 */
struct PmeAtomThreadAssignment
{
    //! Grid line index to thread index contribution, per dimension
    std::array<const int*, DIM> gridToThread = { { nullptr, nullptr, nullptr } };
    //! Thread index per atom, indexed by atom index
    int* threadIndex = nullptr;
    //! Atom counts per thread, incremented, not reset here
    int* threadCounts = nullptr;
};

/*! \brief Compute the local grid index and fraction for atoms \p start to \p end
 *
 * \param[in]  lookup            Grid parameters and lookup tables
 * \param[in]  x                 Atom coordinates, may extend beyond \p end
 * \param[in]  start             First atom to process
 * \param[in]  end               End of the atom range to process
 * \param[out] fractx            Fraction within the grid cell, indexed with the atom index
 * \param[out] idx               Local grid cell index, indexed with the atom index
 * \param[in,out] threadAssignment  When not nullptr, the atoms are assigned to threads
 * \param[in]  kernel            The implementation to use
 */
void pmeCalcInterpolationIndices(const PmeInterpolationIndexLookup& lookup,
                                 ArrayRef<const RVec>               x,
                                 int                                start,
                                 int                                end,
                                 ArrayRef<RVec>                     fractx,
                                 ArrayRef<IVec>                     idx,
                                 const PmeAtomThreadAssignment*     threadAssignment,
                                 PmeSplineSetupKernel               kernel);

/*! \brief Compute the B-spline coefficients and their derivatives
 *
 * For each i in 0..numAtoms-1, with atom index ind[i], the coefficients are
 * stored at theta[d][i*order..i*order+order-1], but only when
 * \p computeAllSplineCoefficients is true or coefficient[ind[i]] != 0.
 * Other entries are left untouched.
 *
 * \param[out] theta        The spline coefficients, per dimension
 * \param[out] dtheta       The spline coefficient derivatives, per dimension
 * \param[in]  order        The PME interpolation order, 3 <= order <= PME_ORDER_MAX
 * \param[in]  fractx       Fractions within the grid cells, indexed with the atom index
 * \param[in]  numAtoms     The number of atoms in \p ind to process
 * \param[in]  ind          Atom indices
 * \param[in]  coefficient  Atom charges or C6 coefficients, indexed with the atom index
 * \param[in]  computeAllSplineCoefficients  When false, skip atoms with zero coefficient
 * \param[in]  kernel       The implementation to use
 */
void pmeMakeBsplines(ArrayRef<real*>      theta,
                     ArrayRef<real*>      dtheta,
                     int                  order,
                     ArrayRef<const RVec> fractx,
                     int                  numAtoms,
                     const int*           ind,
                     const real*          coefficient,
                     bool                 computeAllSplineCoefficients,
                     PmeSplineSetupKernel kernel);

} // namespace gmx

#endif
