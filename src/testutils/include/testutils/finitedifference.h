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
/*! \libinternal \file
 * \brief
 * Helpers for testing that analytical derivatives, such as forces,
 * are consistent with the energy they are derived from.
 *
 * Reference-data tests only detect that a kernel changed, not that
 * the force it computes is the derivative of the energy it computes.
 * Comparing against a numerical derivative of the energy checks that
 * directly.
 *
 * \author Kevin Boyd <kevin44boyd@gmail.com>
 * \inlibraryapi
 * \ingroup module_testutils
 */
#ifndef GMX_TESTUTILS_FINITEDIFFERENCE_H
#define GMX_TESTUTILS_FINITEDIFFERENCE_H

#include <cmath>

#include <algorithm>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/real.h"
#include "gromacs/utility/vec.h"
#include "gromacs/utility/vectypes.h"


namespace gmx
{
namespace test
{

/*! \brief Whether numerical derivatives of energies are accurate enough to test against
 *
 * Energies computed in single precision carry a rounding error that is
 * amplified by the division with the step size to a level comparable
 * to the errors we want to detect, so such tests only run in double
 * precision.
 */
constexpr bool c_canTestWithFiniteDifferences = (GMX_DOUBLE != 0);

/*! \brief Returns a numerical estimate of the derivative of \c function at zero
 *
 * Uses a five-point central-difference stencil, which has a truncation
 * error of order \c stepSize^4.
 *
 * \tparam Function  Callable taking a displacement (double) of the
 *                   differentiation variable from the point of interest
 *                   and returning the function value there.
 * \param[in] function  The function to differentiate
 * \param[in] stepSize  The spacing of the stencil points
 */
template<typename Function>
double finiteDifferenceDerivative(Function&& function, const double stepSize)
{
    const double oneStep  = function(stepSize) - function(-stepSize);
    const double twoSteps = function(2.0 * stepSize) - function(-2.0 * stepSize);
    return (8.0 * oneStep - twoSteps) / (12.0 * stepSize);
}

//! Parameters for comparing analytical with numerical derivatives of an energy
struct FiniteDifferenceSettings
{
    //! Stencil spacing for derivatives with respect to coordinates, in nm
    double coordinateStep = 1e-5;
    /*! \brief Permitted error, relative to the largest of the derivatives that are compared
     *
     * This needs to be increased when the energy is not a smooth function,
     * e.g. because it is interpolated from a table. */
    double relativeTolerance = 1e-7;
};

/*! \brief Returns the largest norm of the vectors in \c vectors, zero when empty
 *
 * Used as the scale relative to which deviations are judged.
 */
inline double largestNorm(ArrayRef<const RVec> vectors)
{
    const auto largest =
            std::max_element(vectors.begin(),
                             vectors.end(),
                             [](const RVec& a, const RVec& b) { return a.norm2() < b.norm2(); });
    return largest != vectors.end() ? largest->norm() : 0.0;
}

/*! \brief Returns the error to permit in a numerical derivative of an energy
 *
 * This is the sum of two contributions:
 * - \c relativeTolerance times \c scale, which covers the truncation error of
 *   the stencil and inaccuracies of the analytical derivative,
 * - the rounding error of the energy divided by the step size, which
 *   matters when the energy is large compared to its derivative.
 *
 * \param[in] scale              Magnitude of the largest of the derivatives that are compared
 * \param[in] relativeTolerance  Permitted error relative to \c scale
 * \param[in] energyMagnitude    Magnitude of the energy that is differentiated
 * \param[in] stepSize           Stencil spacing used for the numerical derivative
 */
inline double finiteDifferenceTolerance(const double scale,
                                        const double relativeTolerance,
                                        const double energyMagnitude,
                                        const double stepSize)
{
    const double roundingError = 100 * std::numeric_limits<double>::epsilon() * energyMagnitude / stepSize;
    return relativeTolerance * scale + roundingError;
}

/*! \brief Checks that \c forces are minus the gradient of the energy
 *
 * The force on each atom is compared, as a vector, with the numerical
 * derivative of the energy with respect to the position of that atom.
 * The norm of the difference should not exceed a tolerance that is
 * relative to the largest force on any atom, so atoms with small
 * forces are checked with the same absolute accuracy as the others.
 *
 * \tparam EnergyFunction  Callable taking coordinates as ArrayRef<const RVec>
 *                         and returning the energy
 * \param[in] x         Coordinates at which \c forces were computed
 * \param[in] forces    Forces to check
 * \param[in] energy    Function returning the energy for given coordinates
 * \param[in] settings  Parameters for the comparison
 */
template<typename EnergyFunction>
void checkForcesMatchEnergy(ArrayRef<const RVec>            x,
                            ArrayRef<const RVec>            forces,
                            EnergyFunction&&                energy,
                            const FiniteDifferenceSettings& settings = {})
{
    ASSERT_EQ(x.size(), forces.size());

    const double tolerance = finiteDifferenceTolerance(
            largestNorm(forces), settings.relativeTolerance, std::abs(energy(x)), settings.coordinateStep);

    // Find the atoms for which the force deviates too much and keep the worst one,
    // so a failure produces a single message instead of one per atom.
    std::vector<RVec> xTrial(x.begin(), x.end());
    int               numDeviatingAtoms = 0;
    Index             worstAtom         = -1;
    double            worstDeviation    = 0;
    DVec              worstForce        = { 0, 0, 0 };
    DVec              worstNumerical    = { 0, 0, 0 };
    for (Index atom = 0; atom < x.ssize(); atom++)
    {
        // Minus the derivative of the energy with respect to each coordinate of this atom
        DVec numericalForce;
        for (int d = 0; d < DIM; d++)
        {
            numericalForce[d] = 0.0
                                - finiteDifferenceDerivative(
                                        [&](const double displacement)
                                        {
                                            xTrial[atom][d]     = x[atom][d] + displacement;
                                            const double result = energy(xTrial);
                                            xTrial[atom][d]     = x[atom][d];
                                            return result;
                                        },
                                        settings.coordinateStep);
        }

        const DVec   force(forces[atom][XX], forces[atom][YY], forces[atom][ZZ]);
        const double deviation = (force - numericalForce).norm();
        if (deviation > tolerance)
        {
            numDeviatingAtoms++;
        }
        if (deviation > worstDeviation)
        {
            worstAtom      = atom;
            worstDeviation = deviation;
            worstForce     = force;
            worstNumerical = numericalForce;
        }
    }

    EXPECT_LE(worstDeviation, tolerance)
            << "The forces on " << numDeviatingAtoms << " of " << x.size()
            << " atoms do not match minus the numerical derivative of the energy. The largest "
               "deviation is for atom "
            << worstAtom << ", with force (" << worstForce[XX] << ", " << worstForce[YY] << ", "
            << worstForce[ZZ] << ") and numerical derivative (" << worstNumerical[XX] << ", "
            << worstNumerical[YY] << ", " << worstNumerical[ZZ] << ")";
}

} // namespace test
} // namespace gmx

#endif
