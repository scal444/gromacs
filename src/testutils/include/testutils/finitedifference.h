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

#include "testutils/testasserts.h"

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
    /*! \brief Permitted error, relative to the largest of the quantities compared
     *
     * This needs to be increased when the energy is not a smooth function,
     * e.g. because it is interpolated from a table. */
    double relativeTolerance = 1e-7;
};

/*! \brief Returns the tolerance for comparing with a numerical derivative of an energy
 *
 * The rounding error of the energy limits the accuracy of the numerical
 * derivative, which matters when the energy is large compared to its
 * derivative, so that is permitted in addition to the relative tolerance.
 *
 * \param[in] magnitude          Magnitude of the quantity compared, should not be less than one
 * \param[in] relativeTolerance  Permitted error relative to \c magnitude
 * \param[in] energyMagnitude    Magnitude of the energy that is differentiated
 * \param[in] stepSize           Stencil spacing used for the numerical derivative
 */
inline FloatingPointTolerance finiteDifferenceTolerance(const double magnitude,
                                                        const double relativeTolerance,
                                                        const double energyMagnitude,
                                                        const double stepSize)
{
    const double roundingError = 100 * std::numeric_limits<double>::epsilon() * energyMagnitude / stepSize;
    return absoluteTolerance(relativeTolerance * magnitude + roundingError);
}

/*! \brief Checks that \c forces are minus the gradient of the energy
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

    double maxForce = 1.0;
    for (const RVec& force : forces)
    {
        for (int d = 0; d < DIM; d++)
        {
            maxForce = std::max(maxForce, std::abs(static_cast<double>(force[d])));
        }
    }
    const FloatingPointTolerance tolerance = finiteDifferenceTolerance(
            maxForce, settings.relativeTolerance, std::abs(energy(x)), settings.coordinateStep);

    std::vector<RVec> xTrial(x.begin(), x.end());
    for (Index atom = 0; atom < x.ssize(); atom++)
    {
        for (int d = 0; d < DIM; d++)
        {
            const real numericalForce = -finiteDifferenceDerivative(
                    [&](const double displacement)
                    {
                        xTrial[atom][d]     = x[atom][d] + displacement;
                        const double result = energy(xTrial);
                        xTrial[atom][d]     = x[atom][d];
                        return result;
                    },
                    settings.coordinateStep);
            EXPECT_REAL_EQ_TOL(forces[atom][d], numericalForce, tolerance)
                    << "Force on atom " << atom << " in dimension " << d
                    << " does not match the derivative of the energy";
        }
    }
}

} // namespace test
} // namespace gmx

#endif
