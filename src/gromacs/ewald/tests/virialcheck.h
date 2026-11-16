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
 * Helper for testing that a virial tensor is consistent with the energy
 *
 * \author Kevin Boyd <kevin44boyd@gmail.com>
 * \ingroup module_ewald
 */
#ifndef GMX_EWALD_TESTS_VIRIALCHECK_H
#define GMX_EWALD_TESTS_VIRIALCHECK_H

#include <cmath>

#include <algorithm>
#include <vector>

#include <gtest/gtest.h>

#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/real.h"
#include "gromacs/utility/vec.h"
#include "gromacs/utility/vectypes.h"

#include "testutils/finitedifference.h"
#include "testutils/testasserts.h"

namespace gmx
{
namespace test
{

/*! \brief Checks that \c virial is half the derivative of the energy with respect to strain
 *
 * A homogeneous strain of coordinates and box changes the energy by
 * twice the corresponding element of the virial -0.5 sum_i x_i (x) f_i.
 * This is checked for all strains that keep the box lower-triangular;
 * the other elements are checked through the symmetry of the virial.
 *
 * \tparam EnergyFunction  Callable taking coordinates as ArrayRef<const RVec>
 *                         and a box as \c matrix and returning the energy
 * \param[in] x         Coordinates at which the virial was computed
 * \param[in] box       Box with which the virial was computed
 * \param[in] virial    The virial to check
 * \param[in] energy    Function returning the energy for given coordinates and box
 * \param[in] settings  Parameters for the comparison
 */
template<typename EnergyFunction>
void checkVirialTensorMatchesEnergy(ArrayRef<const RVec>            x,
                                    const matrix                    box,
                                    const matrix                    virial,
                                    EnergyFunction&&                energy,
                                    const FiniteDifferenceSettings& settings = {})
{
    double maxElement = 1.0;
    for (int d1 = 0; d1 < DIM; d1++)
    {
        for (int d2 = 0; d2 < DIM; d2++)
        {
            maxElement = std::max(maxElement, std::abs(static_cast<double>(virial[d1][d2])));
        }
    }
    const FloatingPointTolerance tolerance = finiteDifferenceTolerance(
            maxElement, settings.relativeTolerance, std::abs(energy(x, box)), settings.strainStep);

    std::vector<RVec> xTrial(x.begin(), x.end());
    for (int d1 = 0; d1 < DIM; d1++)
    {
        const int endDim = (settings.onlyDiagonalStrain ? d1 + 1 : DIM);
        for (int d2 = d1; d2 < endDim; d2++)
        {
            const real numericalElement =
                    0.5
                    * finiteDifferenceDerivative(
                            [&](const double strain)
                            {
                                for (Index atom = 0; atom < x.ssize(); atom++)
                                {
                                    xTrial[atom]     = x[atom];
                                    xTrial[atom][d1] = x[atom][d1] + strain * x[atom][d2];
                                }
                                matrix boxTrial;
                                copy_mat(box, boxTrial);
                                for (int boxVector = 0; boxVector < DIM; boxVector++)
                                {
                                    boxTrial[boxVector][d1] += strain * box[boxVector][d2];
                                }
                                return energy(xTrial, boxTrial);
                            },
                            settings.strainStep);
            EXPECT_REAL_EQ_TOL(virial[d1][d2], numericalElement, tolerance)
                    << "Virial does not match the derivative of the energy with respect to strain "
                       "element "
                    << d1 << "," << d2;
            EXPECT_REAL_EQ_TOL(virial[d2][d1], virial[d1][d2], tolerance)
                    << "Virial is not symmetric for element " << d1 << "," << d2;
        }
    }
}

} // namespace test
} // namespace gmx

#endif
