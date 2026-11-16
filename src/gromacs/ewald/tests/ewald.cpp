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
 * Tests that forces, virial and dV/dlambda of the plain Ewald reciprocal
 * sum and the net-charge correction are consistent with the energy
 * and that the total Ewald energy does not depend on the Ewald coefficient
 *
 * \author Kevin Boyd <kevin44boyd@gmail.com>
 * \ingroup module_ewald
 */
#include "gmxpre.h"

#include "gromacs/ewald/ewald.h"

#include <cctype>
#include <cmath>

#include <algorithm>
#include <array>
#include <string>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "gromacs/math/units.h"
#include "gromacs/math/utilities.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/mdtypes/md_enums.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/real.h"
#include "gromacs/utility/stringutil.h"
#include "gromacs/utility/vec.h"
#include "gromacs/utility/vectypes.h"

#include "testutils/finitedifference.h"
#include "testutils/testasserts.h"

#include "virialcheck.h"

namespace gmx
{
namespace test
{
namespace
{

//! The relative dielectric constant
constexpr real c_epsilonR = 1.5;
//! The Ewald splitting parameter
constexpr real c_ewaldCoefficient = 2.5;
//! The scaling factor for the box along z with walls
constexpr real c_wallEwaldZFactor = 3;

//! Test parameters: whether charges are perturbed, whether to use walls and lambda
using EwaldTestParameters = std::tuple<bool, bool, real>;

//! Test fixture for the plain Ewald reciprocal sum
class EwaldTest : public ::testing::TestWithParam<EwaldTestParameters>
{
public:
    EwaldTest()
    {
        std::tie(havePerturbedCharges_, haveWalls_, lambda_) = GetParam();

        ir_.nkx = 5;
        ir_.nky = 6;
        ir_.nkz = 7;

        clear_mat(box_);
        box_[XX][XX] = 2.0;
        box_[YY][YY] = 2.5;
        box_[ZZ][ZZ] = 3.0;
    }

    //! Output of the reciprocal sum
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

    //! Calls do_ewald() for coordinates \c x with \c box and \c lambda
    Output computeEwald(ArrayRef<const RVec> x, const matrix box, const real lambda)
    {
        gmx_ewald_tab_t ewaldTable(ir_, nullptr);

        Output output;
        output.forces.resize(x.size(), { 0.0_real, 0.0_real, 0.0_real });

        output.energy = do_ewald(haveWalls_,
                                 c_wallEwaldZFactor,
                                 c_epsilonR,
                                 havePerturbedCharges_ ? FreeEnergyPerturbationType::Yes
                                                       : FreeEnergyPerturbationType::No,
                                 x,
                                 output.forces,
                                 chargeA_,
                                 chargeB_,
                                 box,
                                 nullptr,
                                 x.size(),
                                 output.virial,
                                 c_ewaldCoefficient,
                                 lambda,
                                 &output.dvdlambda,
                                 &ewaldTable);

        return output;
    }

    /*! \brief Returns the reciprocal energy for general, triclinic boxes for one set of charges
     *
     * Sums over the same wave vectors as do_ewald(), but using both
     * half spaces.
     */
    double directSum(ArrayRef<const RVec> x, const matrix box, ArrayRef<const real> charges) const
    {
        // Double precision box with the unit-cell elongated along z with walls
        std::array<std::array<double, DIM>, DIM> scaledBox;
        for (int d1 = 0; d1 < DIM; d1++)
        {
            for (int d2 = 0; d2 < DIM; d2++)
            {
                scaledBox[d1][d2] = box[d1][d2] * (d1 == ZZ && haveWalls_ ? c_wallEwaldZFactor : 1);
            }
        }
        const double volume = scaledBox[XX][XX] * scaledBox[YY][YY] * scaledBox[ZZ][ZZ];

        // The reciprocal vectors, without the factor 2 pi, of a lower-triangular box
        std::array<std::array<double, DIM>, DIM> reciprocalBox = { { { 0 } } };
        reciprocalBox[XX][XX]                                  = 1 / scaledBox[XX][XX];
        reciprocalBox[YY][YY]                                  = 1 / scaledBox[YY][YY];
        reciprocalBox[ZZ][ZZ]                                  = 1 / scaledBox[ZZ][ZZ];
        reciprocalBox[XX][YY] = -scaledBox[YY][XX] * reciprocalBox[XX][XX] * reciprocalBox[YY][YY];
        reciprocalBox[YY][ZZ] = -scaledBox[ZZ][YY] * reciprocalBox[YY][YY] * reciprocalBox[ZZ][ZZ];
        reciprocalBox[XX][ZZ] =
                -(scaledBox[ZZ][XX] * reciprocalBox[XX][XX] + scaledBox[ZZ][YY] * reciprocalBox[XX][YY])
                * reciprocalBox[ZZ][ZZ];

        double sum = 0;
        for (int ix = -ir_.nkx; ix <= ir_.nkx; ix++)
        {
            for (int iy = -ir_.nky; iy <= ir_.nky; iy++)
            {
                for (int iz = -ir_.nkz; iz <= ir_.nkz; iz++)
                {
                    if (ix == 0 && iy == 0 && iz == 0)
                    {
                        continue;
                    }
                    std::array<double, DIM> k;
                    double                  k2 = 0;
                    for (int d = 0; d < DIM; d++)
                    {
                        k[d] = 2 * M_PI
                               * (ix * reciprocalBox[XX][d] + iy * reciprocalBox[YY][d]
                                  + iz * reciprocalBox[ZZ][d]);
                        k2 += k[d] * k[d];
                    }
                    double structureFactorCos = 0;
                    double structureFactorSin = 0;
                    for (Index i = 0; i < x.ssize(); i++)
                    {
                        const double phase = k[XX] * x[i][XX] + k[YY] * x[i][YY] + k[ZZ] * x[i][ZZ];
                        structureFactorCos += charges[i] * std::cos(phase);
                        structureFactorSin += charges[i] * std::sin(phase);
                    }
                    sum += std::exp(-k2 / (4 * c_ewaldCoefficient * c_ewaldCoefficient)) / k2
                           * (structureFactorCos * structureFactorCos
                              + structureFactorSin * structureFactorSin);
                }
            }
        }

        return 2 * M_PI * c_one4PiEps0 / (c_epsilonR * volume) * sum;
    }

    //! Returns the reciprocal energy for general, triclinic boxes at \c lambda
    double directSum(ArrayRef<const RVec> x, const matrix box, const double lambda) const
    {
        if (havePerturbedCharges_)
        {
            return (1 - lambda) * directSum(x, box, chargeA_) + lambda * directSum(x, box, chargeB_);
        }
        else
        {
            return directSum(x, box, chargeA_);
        }
    }

protected:
    t_inputrec ir_;
    bool       havePerturbedCharges_;
    bool       haveWalls_;
    real       lambda_;
    matrix     box_;

    //! Coordinates
    std::vector<RVec> x_ = { { 0.5, 1.0, 0.2 }, { 1.7, 0.3, 1.4 }, { 1.0, 2.0, 0.5 },
                             { 1.5, 1.5, 2.1 }, { 0.3, 2.2, 2.7 }, { 1.2, 0.8, 1.1 } };
    //! Charges in state A, with net charge
    std::vector<real> chargeA_ = { 0.5, -1.0, 0.7, 0.4, -0.8, 0.3 };
    //! Charges in state B
    std::vector<real> chargeB_ = { -0.3, 0.8, 0.2, 0.5, -0.6, -0.9 };
};

TEST_P(EwaldTest, EnergyMatchesDirectSum)
{
    const double reference = directSum(x_, box_, lambda_);

    EXPECT_REAL_EQ_TOL(computeEwald(x_, box_, lambda_).energy,
                       real(reference),
                       relativeToleranceAsFloatingPoint(reference, GMX_DOUBLE ? 1e-12 : 1e-4));
}

TEST_P(EwaldTest, DerivativesMatchEnergy)
{
    if (!c_canTestWithFiniteDifferences)
    {
        GTEST_SKIP() << "Numerical derivatives of the energy are only accurate enough in double "
                        "precision";
    }

    const Output output = computeEwald(x_, box_, lambda_);

    checkForcesMatchEnergy(x_,
                           output.forces,
                           [&](ArrayRef<const RVec> xTrial)
                           { return computeEwald(xTrial, box_, lambda_).energy; });

    if (havePerturbedCharges_)
    {
        checkDvdlambdaMatchesEnergy(output.dvdlambda,
                                    [&](const double lambdaChange)
                                    { return computeEwald(x_, box_, lambda_ + lambdaChange).energy; });
    }
    else
    {
        EXPECT_EQ(output.dvdlambda, 0);
    }

    // do_ewald() only supports rectangular boxes, so off-diagonal elements of the virial
    // can only be compared with the strain derivative of the general direct sum
    {
        SCOPED_TRACE("Comparing the diagonal of the virial with the energy from do_ewald()");
        FiniteDifferenceSettings settings;
        settings.onlyDiagonalStrain = true;
        checkVirialTensorMatchesEnergy(
                x_,
                box_,
                output.virial,
                [&](ArrayRef<const RVec> xTrial, const matrix boxTrial)
                { return computeEwald(xTrial, boxTrial, lambda_).energy; },
                settings);
    }
    {
        SCOPED_TRACE("Comparing the virial with the energy from the direct sum");
        checkVirialTensorMatchesEnergy(x_,
                                       box_,
                                       output.virial,
                                       [&](ArrayRef<const RVec> xTrial, const matrix boxTrial)
                                       { return directSum(xTrial, boxTrial, lambda_); });
    }
}

//! Returns a name for a test with parameters \c info
std::string nameOfTest(const ::testing::TestParamInfo<EwaldTestParameters>& info)
{
    std::string name = formatString("%s_%s_Lambda%g",
                                    std::get<0>(info.param) ? "Perturbed" : "Unperturbed",
                                    std::get<1>(info.param) ? "Walls" : "NoWalls",
                                    std::get<2>(info.param));
    std::replace_if(name.begin(), name.end(), [](const char c) { return !std::isalnum(c); }, '_');
    return name;
}

INSTANTIATE_TEST_SUITE_P(Unperturbed,
                         EwaldTest,
                         ::testing::Combine(::testing::Values(false),
                                            ::testing::Bool(),
                                            ::testing::Values(0.0, 0.4)),
                         nameOfTest);

INSTANTIATE_TEST_SUITE_P(Perturbed,
                         EwaldTest,
                         ::testing::Combine(::testing::Values(true),
                                            ::testing::Bool(),
                                            ::testing::Values(0.0, 0.4, 1.0)),
                         nameOfTest);

//! Test parameters for the net-charge correction: whether to use walls and lambda
using EwaldChargeCorrectionTestParameters = std::tuple<bool, real>;

//! Test fixture for the Ewald correction for systems with net charge
class EwaldChargeCorrectionTest : public ::testing::TestWithParam<EwaldChargeCorrectionTestParameters>
{
public:
    //! Output of the correction
    struct Output
    {
        //! The energy
        real energy = 0;
        //! The derivative of the energy with respect to lambda
        real dvdlambda = 0;
        //! The virial
        matrix virial = { { 0 } };
    };

    //! Computes the correction for \c box and \c lambda
    static Output computeCorrection(const bool haveWalls, const matrix box, const real lambda)
    {
        // The net charge in states A and B
        const std::array<double, 2> chargeSum = { 3.0, -2.0 };

        Output output;
        output.energy = ewald_charge_correction(nullptr,
                                                haveWalls,
                                                c_wallEwaldZFactor,
                                                c_epsilonR,
                                                c_ewaldCoefficient,
                                                chargeSum,
                                                lambda,
                                                box,
                                                &output.dvdlambda,
                                                output.virial);

        return output;
    }
};

TEST_P(EwaldChargeCorrectionTest, DerivativesMatchEnergy)
{
    if (!c_canTestWithFiniteDifferences)
    {
        GTEST_SKIP() << "Numerical derivatives of the energy are only accurate enough in double "
                        "precision";
    }

    const auto [haveWalls, lambda] = GetParam();

    matrix box  = { { 0 } };
    box[XX][XX] = 2.0;
    box[YY][XX] = 0.4;
    box[YY][YY] = 2.5;
    box[ZZ][XX] = -0.3;
    box[ZZ][YY] = 0.5;
    box[ZZ][ZZ] = 3.0;

    const Output output = computeCorrection(haveWalls, box, lambda);

    checkDvdlambdaMatchesEnergy(
            output.dvdlambda,
            [&](const double lambdaChange)
            { return computeCorrection(haveWalls, box, lambda + lambdaChange).energy; });

    // The correction does not depend on the coordinates
    const std::vector<RVec> x;
    checkVirialTensorMatchesEnergy(x,
                                   box,
                                   output.virial,
                                   [&](ArrayRef<const RVec> /* x */, const matrix boxTrial)
                                   { return computeCorrection(haveWalls, boxTrial, lambda).energy; });
}

INSTANTIATE_TEST_SUITE_P(Lambda,
                         EwaldChargeCorrectionTest,
                         ::testing::Combine(::testing::Bool(), ::testing::Values(0.0, 0.4, 1.0)));

//! Test fixture for the total Ewald energy of a system with net charge, parametrized on using walls
class EwaldNetChargeTest : public ::testing::TestWithParam<bool>
{
public:
    /*! \brief Returns the total electrostatic energy using Ewald coefficient \c ewaldCoefficient
     *
     * Sums the reciprocal part, the direct-space part, the self
     * interaction and the correction for the net charge, all converged
     * to high accuracy.
     */
    double totalEnergy(const real ewaldCoefficient) const
    {
        const bool haveWalls = GetParam();

        matrix box  = { { 0 } };
        box[XX][XX] = 2.0;
        box[YY][YY] = 2.5;
        box[ZZ][ZZ] = 3.0;

        t_inputrec ir;
        ir.nkx = 14;
        ir.nky = 16;
        ir.nkz = (haveWalls ? 60 : 20);
        gmx_ewald_tab_t   ewaldTable(ir, nullptr);
        std::vector<RVec> forces(x_.size(), { 0.0_real, 0.0_real, 0.0_real });
        matrix            virial;
        real              dvdlambda        = 0;
        const double      reciprocalEnergy = do_ewald(haveWalls,
                                                 c_wallEwaldZFactor,
                                                 c_epsilonR,
                                                 FreeEnergyPerturbationType::No,
                                                 x_,
                                                 forces,
                                                 charges_,
                                                 charges_,
                                                 box,
                                                 nullptr,
                                                 x_.size(),
                                                 virial,
                                                 ewaldCoefficient,
                                                 0,
                                                 &dvdlambda,
                                                 &ewaldTable);

        // The Ewald unit cell is elongated along z with walls
        const double unitCellZ = box[ZZ][ZZ] * (haveWalls ? c_wallEwaldZFactor : 1);

        const double coulombFactor = c_one4PiEps0 / c_epsilonR;
        double       directEnergy  = 0;
        double       chargeSum     = 0;
        double       selfEnergy    = 0;
        for (size_t i = 0; i < x_.size(); i++)
        {
            chargeSum += charges_[i];
            selfEnergy -= coulombFactor * ewaldCoefficient / std::sqrt(M_PI) * charges_[i] * charges_[i];
            for (size_t j = 0; j < x_.size(); j++)
            {
                for (int imageX = -4; imageX <= 4; imageX++)
                {
                    for (int imageY = -4; imageY <= 4; imageY++)
                    {
                        for (int imageZ = -3; imageZ <= 3; imageZ++)
                        {
                            if (i == j && imageX == 0 && imageY == 0 && imageZ == 0)
                            {
                                continue;
                            }
                            const double dx = x_[i][XX] - x_[j][XX] + imageX * box[XX][XX];
                            const double dy = x_[i][YY] - x_[j][YY] + imageY * box[YY][YY];
                            const double dz = x_[i][ZZ] - x_[j][ZZ] + imageZ * unitCellZ;
                            const double r  = std::sqrt(dx * dx + dy * dy + dz * dz);
                            directEnergy += 0.5 * coulombFactor * charges_[i] * charges_[j]
                                            * std::erfc(ewaldCoefficient * r) / r;
                        }
                    }
                }
            }
        }

        matrix                      virialCorrection = { { 0 } };
        const std::array<double, 2> chargeSums       = { chargeSum, chargeSum };
        const double                chargeCorrection = ewald_charge_correction(
                nullptr, haveWalls, c_wallEwaldZFactor, c_epsilonR, ewaldCoefficient, chargeSums, 0, box, &dvdlambda, virialCorrection);

        return reciprocalEnergy + directEnergy + selfEnergy + chargeCorrection;
    }

protected:
    //! Coordinates
    std::vector<RVec> x_ = { { 0.5, 1.0, 0.2 }, { 1.7, 0.3, 1.4 }, { 1.0, 2.0, 0.5 },
                             { 1.5, 1.5, 2.1 }, { 0.3, 2.2, 2.7 }, { 1.2, 0.8, 1.1 } };
    //! Charges, with net charge
    std::vector<real> charges_ = { 0.5, -1.0, 0.7, 0.4, -0.8, 0.3 };
};

TEST_P(EwaldNetChargeTest, TotalEnergyDoesNotDependOnEwaldCoefficient)
{
    if (!GMX_DOUBLE)
    {
        GTEST_SKIP() << "The sum of the Ewald contributions is only accurate enough in double "
                        "precision";
    }

    // The splitting of the interactions over direct and reciprocal space should
    // not affect the total energy. With net charge this only holds when
    // the correction for the uniform background charge is included.
    const double energy = totalEnergy(2.0);
    EXPECT_REAL_EQ_TOL(totalEnergy(2.5), energy, absoluteTolerance(1e-6));
    EXPECT_REAL_EQ_TOL(totalEnergy(3.0), energy, absoluteTolerance(1e-6));
}

INSTANTIATE_TEST_SUITE_P(Walls, EwaldNetChargeTest, ::testing::Bool());

} // namespace
} // namespace test
} // namespace gmx
