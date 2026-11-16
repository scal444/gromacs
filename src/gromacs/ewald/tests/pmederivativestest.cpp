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
 * Tests that forces, virial and dV/dlambda of the PME mesh part for
 * Coulomb and Lennard-Jones on the CPU are consistent with the mesh energy
 *
 * Smooth PME obtains the forces by analytical differentiation of the
 * B-splines used for spreading, so the forces are the exact gradient
 * of the mesh energy, even though both differ from the exact Ewald
 * reciprocal sum by the interpolation error.
 *
 * \author Kevin Boyd <kevin44boyd@gmail.com>
 * \ingroup module_ewald
 */
#include "gmxpre.h"

#include <cctype>
#include <cmath>

#include <algorithm>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "gromacs/domdec/domdec.h"
#include "gromacs/ewald/pme.h"
#include "gromacs/ewald/pme_internal.h"
#include "gromacs/gmxlib/nrnb.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/mdtypes/md_enums.h"
#include "gromacs/pbcutil/pbc.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/enumerationhelpers.h"
#include "gromacs/utility/logger.h"
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

//! The interactions for which to use a mesh
struct MeshInteractions
{
    CoulombInteractionType coulombType;
    VanDerWaalsType        vdwType;
    LongRangeVdW           ljPmeCombinationRule;
};

//! Which parameters differ between states A and B
enum class Perturbation
{
    None,
    Charges,
    LennardJones,
    Both
};

//! The values of lambda for Coulomb and Lennard-Jones
struct Lambdas
{
    real coulomb;
    real lennardJones;
};

//! The unit cell and the interpolation order to use with it
struct GridSetup
{
    bool useTriclinicBox;
    int  pmeOrder;
    //! Whether to use two walls with the unit cell elongated along z
    bool useWalls;
};

//! Test parameters
using PmeDerivativesTestParameters = std::tuple<MeshInteractions, Perturbation, Lambdas, GridSetup>;

//! Test fixture for derivatives of the PME mesh energy
class PmeDerivativesTest : public ::testing::TestWithParam<PmeDerivativesTestParameters>
{
public:
    PmeDerivativesTest()
    {
        MeshInteractions meshInteractions;
        Perturbation     perturbation;
        GridSetup        gridSetup;
        std::tie(meshInteractions, perturbation, lambdas_, gridSetup) = GetParam();

        const bool perturbCharges =
                (perturbation == Perturbation::Charges || perturbation == Perturbation::Both);
        const bool perturbLennardJones =
                (perturbation == Perturbation::LennardJones || perturbation == Perturbation::Both);

        ir_.pbcType                = (gridSetup.useWalls ? PbcType::XY : PbcType::Xyz);
        ir_.nwall                  = (gridSetup.useWalls ? 2 : 0);
        ir_.wall_ewald_zfac        = 3;
        ir_.coulombtype            = meshInteractions.coulombType;
        ir_.vdwtype                = meshInteractions.vdwType;
        ir_.ljpme_combination_rule = meshInteractions.ljPmeCombinationRule;
        ir_.efep      = (perturbation != Perturbation::None ? FreeEnergyPerturbationType::Yes
                                                            : FreeEnergyPerturbationType::No);
        ir_.pme_order = gridSetup.pmeOrder;
        ir_.nkx       = 12;
        ir_.nky       = 14;
        ir_.nkz       = 16;
        ir_.epsilon_r = 1.5;

        clear_mat(box_);
        box_[XX][XX] = 2.0;
        box_[YY][YY] = 2.4;
        box_[ZZ][ZZ] = 2.8;
        if (gridSetup.useTriclinicBox)
        {
            box_[YY][XX] = 0.5;
            box_[ZZ][XX] = -0.4;
            box_[ZZ][YY] = 0.7;
        }

        // Place the atoms using fractional coordinates
        const std::vector<RVec> fractionalCoordinates = {
            { 0.11, 0.43, 0.07 }, { 0.83, 0.12, 0.51 }, { 0.52, 0.81, 0.19 }, { 0.74, 0.61, 0.73 },
            { 0.16, 0.89, 0.92 }, { 0.61, 0.33, 0.38 }, { 0.37, 0.24, 0.66 }, { 0.93, 0.56, 0.27 }
        };
        for (const RVec& fractional : fractionalCoordinates)
        {
            RVec x = { 0.0_real, 0.0_real, 0.0_real };
            for (int d = 0; d < DIM; d++)
            {
                x += fractional[d] * RVec(box_[d]);
            }
            x_.push_back(x);
        }

        if (!perturbCharges)
        {
            chargeB_ = chargeA_;
        }
        if (!perturbLennardJones)
        {
            sqrtC6B_ = sqrtC6A_;
            sigmaB_  = sigmaA_;
        }

        const real ewaldCoefficientCoulomb      = 3.0;
        const real ewaldCoefficientLennardJones = 2.5;
        const bool reproducible                 = true;
        pme_                                    = gmx_pme_init(nullptr,
                            NumPmeDomains{ 1, 1 },
                            &ir_,
                            box_,
                            1.0_real,
                            perturbCharges,
                            perturbLennardJones,
                            reproducible,
                            ewaldCoefficientCoulomb,
                            ewaldCoefficientLennardJones,
                            1,
                            PmeRunMode::CPU,
                            nullptr,
                            nullptr,
                            nullptr,
                            MDLogger());
        gmx_pme_reinit_atoms(pme_.get(), x_.size(), {}, {});
    }

    //! Output of the PME mesh calculation
    struct Output
    {
        //! The Coulomb mesh energy
        real energyCoulomb = 0;
        //! The Lennard-Jones mesh energy
        real energyLennardJones = 0;
        //! The derivative of the energy with respect to lambda for Coulomb
        real dvdlambdaCoulomb = 0;
        //! The derivative of the energy with respect to lambda for Lennard-Jones
        real dvdlambdaLennardJones = 0;
        //! The forces
        std::vector<RVec> forces;
        //! The Coulomb virial
        matrix virialCoulomb = { { 0 } };
        //! The Lennard-Jones virial
        matrix virialLennardJones = { { 0 } };

        //! Returns the total mesh energy
        double energy() const { return energyCoulomb + energyLennardJones; }
    };

    //! Calls gmx_pme_do() for coordinates \c x with \c box and \c lambdas
    Output computeMesh(ArrayRef<const RVec> x,
                       const matrix         box,
                       const Lambdas&       lambdas,
                       const bool           computeEnergyAndVirial = true)
    {
        Output output;
        output.forces.resize(x.size(), { 0.0_real, 0.0_real, 0.0_real });

        PmeStepWorkload stepWork;
        stepWork.computeForces          = true;
        stepWork.computeEnergyAndVirial = computeEnergyAndVirial;

        t_nrnb nrnb;

        gmx_pme_do(pme_.get(),
                   x,
                   output.forces,
                   chargeA_,
                   chargeB_,
                   sqrtC6A_,
                   sqrtC6B_,
                   sigmaA_,
                   sigmaB_,
                   box,
                   0,
                   0,
                   &nrnb,
                   nullptr,
                   output.virialCoulomb,
                   output.virialLennardJones,
                   &output.energyCoulomb,
                   &output.energyLennardJones,
                   lambdas.coulomb,
                   lambdas.lennardJones,
                   &output.dvdlambdaCoulomb,
                   &output.dvdlambdaLennardJones,
                   stepWork);

        return output;
    }

protected:
    t_inputrec                 ir_;
    Lambdas                    lambdas_;
    matrix                     box_;
    std::unique_ptr<gmx_pme_t> pme_;

    //! Coordinates
    std::vector<RVec> x_;
    //! Charges in state A, with net charge
    std::vector<real> chargeA_ = { 0.5, -1.0, 0.7, 0.4, -0.8, 0.3, -0.2, 0.6 };
    //! Charges in state B
    std::vector<real> chargeB_ = { -0.3, 0.8, 0.2, 0.5, -0.6, -0.9, 0.4, 0.1 };
    //! Square root of C6 in state A
    std::vector<real> sqrtC6A_ = { 0.05, 0.07, 0.04, 0.06, 0.08, 0.03, 0.05, 0.09 };
    //! Square root of C6 in state B
    std::vector<real> sqrtC6B_ = { 0.06, 0.02, 0.07, 0.04, 0.05, 0.08, 0.03, 0.06 };
    //! Sigma in state A, only used with Lorentz-Berthelot
    std::vector<real> sigmaA_ = { 0.31, 0.35, 0.28, 0.33, 0.37, 0.25, 0.30, 0.34 };
    //! Sigma in state B, only used with Lorentz-Berthelot
    std::vector<real> sigmaB_ = { 0.27, 0.32, 0.36, 0.29, 0.26, 0.34, 0.31, 0.38 };
};

TEST_P(PmeDerivativesTest, DerivativesMatchEnergy)
{
    if (!c_canTestWithFiniteDifferences)
    {
        GTEST_SKIP() << "Numerical derivatives of the energy are only accurate enough in double "
                        "precision";
    }

    const Output output = computeMesh(x_, box_, lambdas_);

    checkForcesMatchEnergy(x_,
                           output.forces,
                           [&](ArrayRef<const RVec> xTrial)
                           { return computeMesh(xTrial, box_, lambdas_).energy(); });

    {
        SCOPED_TRACE("Coulomb dV/dlambda");
        checkDvdlambdaMatchesEnergy(
                output.dvdlambdaCoulomb,
                [&](const double lambdaChange)
                {
                    return computeMesh(x_, box_, { real(lambdas_.coulomb + lambdaChange), lambdas_.lennardJones })
                            .energy();
                });
    }
    {
        SCOPED_TRACE("Lennard-Jones dV/dlambda");
        checkDvdlambdaMatchesEnergy(
                output.dvdlambdaLennardJones,
                [&](const double lambdaChange)
                {
                    return computeMesh(x_, box_, { lambdas_.coulomb, real(lambdas_.lennardJones + lambdaChange) })
                            .energy();
                });
    }

    {
        SCOPED_TRACE("Coulomb virial");
        checkVirialTensorMatchesEnergy(x_,
                                       box_,
                                       output.virialCoulomb,
                                       [&](ArrayRef<const RVec> xTrial, const matrix boxTrial) {
                                           return computeMesh(xTrial, boxTrial, lambdas_).energyCoulomb;
                                       });
    }
    {
        SCOPED_TRACE("Lennard-Jones virial");
        checkVirialTensorMatchesEnergy(
                x_,
                box_,
                output.virialLennardJones,
                [&](ArrayRef<const RVec> xTrial, const matrix boxTrial)
                { return computeMesh(xTrial, boxTrial, lambdas_).energyLennardJones; });
    }
}

TEST_P(PmeDerivativesTest, ForcesDoNotDependOnEnergyCalculation)
{
    const Output outputWithEnergy    = computeMesh(x_, box_, lambdas_, true);
    const Output outputWithoutEnergy = computeMesh(x_, box_, lambdas_, false);

    real maxForce = 1;
    for (const RVec& force : outputWithEnergy.forces)
    {
        maxForce = std::max(maxForce, norm(force));
    }
    const FloatingPointTolerance tolerance = absoluteTolerance(maxForce * (GMX_DOUBLE ? 1e-12 : 1e-5));
    for (size_t i = 0; i < x_.size(); i++)
    {
        for (int d = 0; d < DIM; d++)
        {
            EXPECT_REAL_EQ_TOL(outputWithEnergy.forces[i][d], outputWithoutEnergy.forces[i][d], tolerance)
                    << "for atom " << i << " dimension " << d;
        }
    }
}

//! Returns a name for a test with parameters \c info
std::string nameOfTest(const ::testing::TestParamInfo<PmeDerivativesTestParameters>& info)
{
    const MeshInteractions& meshInteractions = std::get<0>(info.param);
    const Lambdas&          lambdas          = std::get<2>(info.param);
    const GridSetup&        gridSetup        = std::get<3>(info.param);

    const char* const perturbationNames[] = {
        "Unperturbed", "PerturbedCharges", "PerturbedLJ", "PerturbedChargesAndLJ"
    };

    std::string name = formatString(
            "Coulomb%s_Vdw%s%s_%s_Lambda%g_%g_%s_Order%d",
            enumValueToString(meshInteractions.coulombType),
            enumValueToString(meshInteractions.vdwType),
            usingLJPme(meshInteractions.vdwType) ? enumValueToString(meshInteractions.ljPmeCombinationRule)
                                                 : "",
            perturbationNames[static_cast<int>(std::get<1>(info.param))],
            lambdas.coulomb,
            lambdas.lennardJones,
            gridSetup.useWalls ? "Walls" : (gridSetup.useTriclinicBox ? "Triclinic" : "Rectangular"),
            gridSetup.pmeOrder);
    std::replace_if(name.begin(), name.end(), [](const char c) { return !std::isalnum(c); }, '_');
    return name;
}

//! The combinations of mesh interactions to test
const MeshInteractions c_meshInteractions[] = {
    { CoulombInteractionType::Pme, VanDerWaalsType::Cut, LongRangeVdW::Geom },
    { CoulombInteractionType::P3mAD, VanDerWaalsType::Cut, LongRangeVdW::Geom },
    { CoulombInteractionType::Cut, VanDerWaalsType::Pme, LongRangeVdW::Geom },
    { CoulombInteractionType::Cut, VanDerWaalsType::Pme, LongRangeVdW::LB },
    { CoulombInteractionType::Pme, VanDerWaalsType::Pme, LongRangeVdW::Geom },
    { CoulombInteractionType::Pme, VanDerWaalsType::Pme, LongRangeVdW::LB }
};

//! The unit cells and interpolation orders to test
const GridSetup c_gridSetups[] = { { false, 4, false }, { true, 5, false }, { false, 4, true } };

INSTANTIATE_TEST_SUITE_P(Unperturbed,
                         PmeDerivativesTest,
                         ::testing::Combine(::testing::ValuesIn(c_meshInteractions),
                                            ::testing::Values(Perturbation::None),
                                            ::testing::Values(Lambdas{ 0.0, 0.0 }),
                                            ::testing::ValuesIn(c_gridSetups)),
                         nameOfTest);

// Includes end states, as states with zero weight get special treatment
INSTANTIATE_TEST_SUITE_P(Perturbed,
                         PmeDerivativesTest,
                         ::testing::Combine(::testing::ValuesIn(c_meshInteractions),
                                            ::testing::Values(Perturbation::Charges,
                                                              Perturbation::LennardJones,
                                                              Perturbation::Both),
                                            ::testing::Values(Lambdas{ 0.4, 0.7 }, Lambdas{ 1.0, 0.0 }),
                                            ::testing::ValuesIn(c_gridSetups)),
                         nameOfTest);

} // namespace
} // namespace test
} // namespace gmx
