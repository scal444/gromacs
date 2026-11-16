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
        //! The virial
        matrix virial = { { 0 } };
    };

    /*! \brief Computes the correction for a rank with local atoms \c localAtoms
     *
     * \param[in] x           The coordinates of all atoms
     * \param[in] box         The box
     * \param[in] lambda      The value of lambda
     * \param[in] localAtoms  The indices of the atoms that are local to the rank
     * \param[in] isMainRank  Whether the rank is the main rank
     *
     * \returns the output, with the forces for the local atoms
     */
    Output computeCorrectionForRank(ArrayRef<const RVec> x,
                                    const matrix         box,
                                    const real           lambda,
                                    ArrayRef<const int>  localAtoms,
                                    const bool           isMainRank)
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

        std::vector<RVec> xLocal;
        std::vector<real> chargeALocal;
        std::vector<real> chargeBLocal;
        for (const int atom : localAtoms)
        {
            xLocal.push_back(x[atom]);
            chargeALocal.push_back(chargeA_[atom]);
            chargeBLocal.push_back(chargeB_[atom]);
        }

        Output output;
        output.forces.resize(xLocal.size(), { 0.0_real, 0.0_real, 0.0_real });
        ForceWithVirial forceWithVirial(output.forces, true);

        ewald_LRcorrection(xLocal.size(),
                           isMainRank,
                           1,
                           0,
                           c_epsilonR,
                           chargeSum,
                           ewaldGeometry_,
                           c_epsilonSurface,
                           haveWalls_,
                           c_wallEwaldZFactor,
                           chargeALocal,
                           chargeBLocal,
                           havePerturbedCharges_,
                           xLocal,
                           box,
                           dipole,
                           &forceWithVirial,
                           &output.energy,
                           lambda,
                           &output.dvdlambda);

        copy_mat(forceWithVirial.getVirial(), output.virial);

        return output;
    }

    //! Computes the correction for coordinates \c x with \c box and \c lambda
    Output computeCorrection(ArrayRef<const RVec> x, const matrix box, const real lambda)
    {
        std::vector<int> allAtoms(x.size());
        std::iota(allAtoms.begin(), allAtoms.end(), 0);
        return computeCorrectionForRank(x, box, lambda, allAtoms, true);
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
        EXPECT_REAL_EQ_TOL(output.virial[d][d],
                           numericalVirial,
                           finiteDifferenceTolerance(
                                   std::max(1.0, std::abs(static_cast<double>(output.virial[d][d]))),
                                   settings.relativeTolerance,
                                   std::abs(output.energy),
                                   settings.strainStep))
                << "Virial does not match the derivative of the energy with respect to scaling "
                   "dimension "
                << d;
    }
}

/*! \brief Checks the off-diagonal elements of the virial
 *
 * As the volume does not change with shear, these should be equal to
 * the single sum virial -0.5 sum_i x_i (x) f_i.
 */
TEST_P(LongRangeCorrectionTest, OffDiagonalVirialMatchesForces)
{
    const Output output = computeCorrection(x_, box_, lambda_);

    const FloatingPointTolerance tolerance =
            relativeToleranceAsFloatingPoint(10.0, GMX_DOUBLE ? 1e-12 : 1e-5);
    for (int d1 = 0; d1 < DIM; d1++)
    {
        for (int d2 = 0; d2 < DIM; d2++)
        {
            if (d1 == d2)
            {
                continue;
            }
            real expectedVirial = 0;
            for (Index i = 0; i < gmx::ssize(x_); i++)
            {
                expectedVirial -= 0.5_real * x_[i][d1] * output.forces[i][d2];
            }
            EXPECT_REAL_EQ_TOL(output.virial[d1][d2], expectedVirial, tolerance)
                    << "for virial element " << d1 << "," << d2;
        }
    }
}

/*! \brief Checks that the contributions of two ranks add up to those of a single rank
 *
 * With domain decomposition each rank only has access to its local atoms.
 */
TEST_P(LongRangeCorrectionTest, ContributionsOfRanksAddUp)
{
    const Output output = computeCorrection(x_, box_, lambda_);

    const std::vector<int> atomsMainRank  = { 0, 2 };
    const std::vector<int> atomsOtherRank = { 1, 3 };
    const Output outputMainRank = computeCorrectionForRank(x_, box_, lambda_, atomsMainRank, true);
    const Output outputOtherRank = computeCorrectionForRank(x_, box_, lambda_, atomsOtherRank, false);

    const FloatingPointTolerance tolerance =
            relativeToleranceAsFloatingPoint(10.0, GMX_DOUBLE ? 1e-12 : 1e-5);
    EXPECT_REAL_EQ_TOL(output.energy, outputMainRank.energy + outputOtherRank.energy, tolerance);
    EXPECT_REAL_EQ_TOL(output.dvdlambda, outputMainRank.dvdlambda + outputOtherRank.dvdlambda, tolerance);
    for (int d1 = 0; d1 < DIM; d1++)
    {
        for (int d2 = 0; d2 < DIM; d2++)
        {
            EXPECT_REAL_EQ_TOL(output.virial[d1][d2],
                               outputMainRank.virial[d1][d2] + outputOtherRank.virial[d1][d2],
                               tolerance)
                    << "for virial element " << d1 << "," << d2;
        }
    }
    for (Index i = 0; i < gmx::ssize(atomsMainRank); i++)
    {
        EXPECT_REAL_EQ_TOL(norm(output.forces[atomsMainRank[i]] - outputMainRank.forces[i]), 0, tolerance);
        EXPECT_REAL_EQ_TOL(norm(output.forces[atomsOtherRank[i]] - outputOtherRank.forces[i]), 0, tolerance);
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
