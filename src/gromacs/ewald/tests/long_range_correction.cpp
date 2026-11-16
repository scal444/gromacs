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
 * Tests that forces, virial and dV/dlambda of the Ewald surface and
 * geometry corrections are consistent with the energy
 *
 * \author Kevin Boyd <kevin44boyd@gmail.com>
 * \ingroup module_ewald
 */
#include "gmxpre.h"

#include "gromacs/ewald/long_range_correction.h"

#include <cctype>
#include <cmath>

#include <algorithm>
#include <array>
#include <numeric>
#include <string>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "gromacs/math/units.h"
#include "gromacs/mdtypes/forceoutput.h"
#include "gromacs/mdtypes/md_enums.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/enumerationhelpers.h"
#include "gromacs/utility/mpicomm.h"
#include "gromacs/utility/real.h"
#include "gromacs/utility/stringutil.h"
#include "gromacs/utility/vec.h"
#include "gromacs/utility/vectypes.h"

#include "testutils/finitedifference.h"
#include "testutils/testasserts.h"

namespace gmx
{
namespace test
{
namespace
{

/*! \brief Test parameters
 *
 * The Ewald geometry, whether the system has net charge, whether charges
 * are perturbed, whether to use two walls with a scaled Ewald box and lambda
 */
using LongRangeCorrectionTestParameters = std::tuple<EwaldGeometry, bool, bool, bool, real>;

//! Test fixture for the Ewald surface and geometry corrections
class LongRangeCorrectionTest : public ::testing::TestWithParam<LongRangeCorrectionTestParameters>
{
public:
    LongRangeCorrectionTest()
    {
        bool haveNetCharge;
        std::tie(ewaldGeometry_, haveNetCharge, havePerturbedCharges_, haveWalls_, lambda_) = GetParam();

        if (!haveNetCharge)
        {
            // Make both states neutral by changing the charge of the last atom
            chargeA_.back() -= std::accumulate(chargeA_.begin(), chargeA_.end(), 0.0);
            chargeB_.back() -= std::accumulate(chargeB_.begin(), chargeB_.end(), 0.0);
        }
        if (!havePerturbedCharges_)
        {
            chargeB_ = chargeA_;
        }

        clear_mat(box_);
        box_[XX][XX] = 2.0;
        box_[YY][YY] = 2.5;
        box_[ZZ][ZZ] = 3.0;
    }

    //! Output of the correction
    struct Output
    {
        //! The energy
        real energy = 0;
        //! The derivative of the energy with respect to lambda
        real dvdlambda = 0;
        //! The forces
        std::vector<RVec> forces;
        //! The diagonal of the virial
        RVec virial;
    };

    //! Computes the correction for coordinates \c x with \c box and \c lambda
    Output computeCorrection(ArrayRef<const RVec> x, const matrix box, const real lambda)
    {
        // The total dipole in Debye and total charge for states A and B
        std::array<RVec, 2>   dipole    = { RVec{ 0, 0, 0 }, RVec{ 0, 0, 0 } };
        std::array<double, 2> chargeSum = { 0, 0 };
        for (Index i = 0; i < x.ssize(); i++)
        {
            dipole[0] += x[i] * static_cast<real>(chargeA_[i] * c_enm2Debye);
            dipole[1] += x[i] * static_cast<real>(chargeB_[i] * c_enm2Debye);
            chargeSum[0] += chargeA_[i];
            chargeSum[1] += chargeB_[i];
        }

        Output output;
        output.forces.resize(x.size(), { 0.0_real, 0.0_real, 0.0_real });
        ForceWithVirial forceWithVirial(output.forces, true);

        ewald_LRcorrection(x.size(),
                           MpiComm(MpiComm::SingleRank{}),
                           1,
                           0,
                           c_epsilonR,
                           chargeSum,
                           ewaldGeometry_,
                           c_epsilonSurface,
                           haveWalls_,
                           c_wallEwaldZFactor,
                           chargeA_,
                           chargeB_,
                           havePerturbedCharges_,
                           x,
                           box,
                           dipole,
                           &forceWithVirial,
                           &output.energy,
                           lambda,
                           &output.dvdlambda);

        for (int d = 0; d < DIM; d++)
        {
            output.virial[d] = forceWithVirial.getVirial()[d][d];
        }

        return output;
    }

protected:
    //! The relative dielectric constant
    static constexpr real c_epsilonR = 1.5;
    //! The dielectric constant of the surrounding medium, only used with 3D geometry
    static constexpr real c_epsilonSurface = 3;
    //! The scaling factor for the box along z with walls
    static constexpr real c_wallEwaldZFactor = 3;

    EwaldGeometry ewaldGeometry_;
    bool          havePerturbedCharges_;
    bool          haveWalls_;
    real          lambda_;
    matrix        box_;

    //! Coordinates
    std::vector<RVec> x_ = { { 0.5, 1.0, 0.2 }, { 1.7, 0.3, 1.4 }, { 1.0, 2.0, 0.5 }, { 1.5, 1.5, 2.1 } };
    //! Charges in state A
    std::vector<real> chargeA_ = { 0.5, -1.0, 0.7, 0.4 };
    //! Charges in state B
    std::vector<real> chargeB_ = { -0.3, 0.8, 0.2, 0.5 };
};

TEST_P(LongRangeCorrectionTest, DerivativesMatchEnergy)
{
    if (!c_canTestWithFiniteDifferences)
    {
        GTEST_SKIP() << "Numerical derivatives of the energy are only accurate enough in double "
                        "precision";
    }

    const Output output = computeCorrection(x_, box_, lambda_);

    checkForcesMatchEnergy(x_,
                           output.forces,
                           [&](ArrayRef<const RVec> xTrial)
                           { return computeCorrection(xTrial, box_, lambda_).energy; });

    checkDvdlambdaMatchesEnergy(output.dvdlambda,
                                [&](const double lambdaChange) {
                                    return computeCorrection(x_, box_, lambda_ + lambdaChange).energy;
                                });

    // Scaling the coordinates and box along a dimension changes the energy
    // by twice the corresponding diagonal element of the virial.
    const FiniteDifferenceSettings settings;
    for (int d = 0; d < DIM; d++)
    {
        const real numericalVirial = 0.5
                                     * finiteDifferenceDerivative(
                                             [&](const double strain)
                                             {
                                                 std::vector<RVec> xTrial = x_;
                                                 for (RVec& xAtom : xTrial)
                                                 {
                                                     xAtom[d] *= 1 + strain;
                                                 }
                                                 matrix boxTrial;
                                                 copy_mat(box_, boxTrial);
                                                 boxTrial[d][d] *= 1 + strain;
                                                 return computeCorrection(xTrial, boxTrial, lambda_).energy;
                                             },
                                             settings.strainStep);
        EXPECT_REAL_EQ_TOL(output.virial[d],
                           numericalVirial,
                           finiteDifferenceTolerance(
                                   std::max(1.0, std::abs(static_cast<double>(output.virial[d]))),
                                   settings.relativeTolerance,
                                   std::abs(output.energy),
                                   settings.strainStep))
                << "Virial does not match the derivative of the energy with respect to scaling "
                   "dimension "
                << d;
    }
}

//! Returns a name for a test with parameters \c info
std::string nameOfTest(const ::testing::TestParamInfo<LongRangeCorrectionTestParameters>& info)
{
    std::string name = formatString("%s_%s_%s_%s_Lambda%g",
                                    enumValueToString(std::get<0>(info.param)),
                                    std::get<1>(info.param) ? "Charged" : "Neutral",
                                    std::get<2>(info.param) ? "Perturbed" : "Unperturbed",
                                    std::get<3>(info.param) ? "Walls" : "NoWalls",
                                    std::get<4>(info.param));
    std::replace_if(name.begin(), name.end(), [](const char c) { return !std::isalnum(c); }, '_');
    return name;
}

INSTANTIATE_TEST_SUITE_P(Corrections,
                         LongRangeCorrectionTest,
                         ::testing::Combine(::testing::Values(EwaldGeometry::ThreeD, EwaldGeometry::ThreeDC),
                                            ::testing::Bool(),
                                            ::testing::Bool(),
                                            ::testing::Bool(),
                                            ::testing::Values(0.0, 0.4, 1.0)),
                         nameOfTest);

} // namespace
} // namespace test
} // namespace gmx
