/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2021- The GROMACS Authors
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
 * \brief Implements position restraint tests.
 *
 * \author Kevin Boyd <kevin44boyd@gmail.com>
 * \ingroup module_listed_forces
 */
#include "gmxpre.h"

#include "gromacs/listed_forces/position_restraints.h"

#include <cmath>
#include <cstddef>

#include <algorithm>
#include <array>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "gromacs/gpu_utils/hostallocator.h"
#include "gromacs/listed_forces/listed_forces.h"
#include "gromacs/mdtypes/enerdata.h"
#include "gromacs/mdtypes/forceoutput.h"
#include "gromacs/mdtypes/forcerec.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/mdtypes/interaction_const.h"
#include "gromacs/mdtypes/md_enums.h"
#include "gromacs/mdtypes/simulation_workload.h"
#include "gromacs/pbcutil/pbc.h"
#include "gromacs/tables/forcetable.h"
#include "gromacs/topology/forcefieldparameters.h"
#include "gromacs/topology/idef.h"
#include "gromacs/topology/ifunc.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/enumerationhelpers.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/real.h"
#include "gromacs/utility/stringstream.h"
#include "gromacs/utility/stringutil.h"
#include "gromacs/utility/vec.h"
#include "gromacs/utility/vectypes.h"

#include "testutils/finitedifference.h"
#include "testutils/refdata.h"
#include "testutils/testasserts.h"

namespace gmx
{
namespace test
{
namespace
{

//! Tolerance for float evaluation
constexpr float c_precisionTolerance = 1e-6;


class PositionRestraintsTest : public ::testing::TestWithParam<std::tuple<RefCoordScaling, PbcType>>
{
protected:
    std::vector<RVec> x_;
    std::vector<RVec> f_;

    matrix  box_;
    t_pbc   pbc_;
    PbcType pbcType_;

    t_forcerec                       fr_{ HostAllocationPolicy{} };
    InteractionDefinitions           idef_;
    gmx_enerdata_t                   enerd_;
    std::unique_ptr<ForceWithVirial> forceWithVirial_;
    RefCoordScaling                  refCoordScaling_;

    TestReferenceData refData_;
    //! Created on first use, so tests that do not compare with reference data need no file
    std::optional<TestReferenceChecker> checker_;

    PositionRestraintsTest() : idef_({}), enerd_(1, nullptr)
    {
        refCoordScaling_ = std::get<0>(GetParam());
        pbcType_         = std::get<1>(GetParam());

        clear_mat(box_);
        box_[0][0] = 0.9;
        box_[1][1] = 1.0;
        box_[2][2] = 1.1;
        set_pbc(&pbc_, pbcType_, box_);

        fr_.rc_scaling = refCoordScaling_;
        fr_.pbcType    = pbcType_;
        fr_.posresCom.emplace_back(0.0_real, 0.5_real, 0.0_real);
        fr_.posresComB.emplace_back(0.0_real, 0.5_real, 0.0_real);
    }

    //! Returns the checker for reference data
    TestReferenceChecker& checker()
    {
        if (!checker_.has_value())
        {
            checker_ = refData_.rootChecker();
            checker_->setDefaultTolerance(relativeToleranceAsFloatingPoint(1.0, c_precisionTolerance));
        }
        return *checker_;
    }

    //! Output of a position-restraint kernel
    struct Output
    {
        //! The potential energy
        real energy = 0;
        //! The derivative of the energy with respect to lambda
        real dvdlambda = 0;
        //! The forces
        std::vector<RVec> forces;
        //! The diagonal of the virial
        RVec virial = { 0.0_real, 0.0_real, 0.0_real };
    };

    /*! \brief Calls the kernel for (flat-bottomed) position restraints in \c idef_
     *
     * \param[in] functionType  Selects which kind of position restraints to compute
     * \param[in] x             Coordinates
     * \param[in] box           The box
     * \param[in] lambda        Lambda for the restraints
     */
    Output runKernel(const InteractionFunction functionType, ArrayRef<const RVec> x, const matrix box, const real lambda)
    {
        t_pbc pbc;
        set_pbc(&pbc, pbcType_, box);

        std::vector<RVec> centersOfMassScaledBuffer  = { { 0.0, 0.0, 0.0 } };
        std::vector<RVec> centersOfMassBScaledBuffer = { { 0.0, 0.0, 0.0 } };

        // We cannot store C-style array rvec4 in an std::vector, so we use real and reinterpret
        std::vector<real> forcesStorage(x.size() * 4, 0.0_real);
        ArrayRef<rvec4>   forces =
                arrayRefFromArray(reinterpret_cast<rvec4*>(forcesStorage.data()), x.size());

        Output output;
        if (functionType == InteractionFunction::PositionRestraints)
        {
            std::array<real, static_cast<size_t>(FreeEnergyPerturbationCouplingType::Count)> lambdas;
            lambdas.fill(lambda);
            output.energy = posres_wrapper(idef_.il[InteractionFunction::PositionRestraints].iatoms,
                                           idef_.iparams_posres,
                                           pbc,
                                           as_rvec_array(x.data()),
                                           lambdas,
                                           fr_,
                                           {},
                                           centersOfMassScaledBuffer,
                                           centersOfMassBScaledBuffer,
                                           forces,
                                           &output.virial,
                                           &output.dvdlambda);
        }
        else
        {
            output.energy = fbposres_wrapper(
                    idef_.il[InteractionFunction::FlatBottomedPositionRestraints].iatoms,
                    idef_.iparams_fbposres,
                    pbc,
                    as_rvec_array(x.data()),
                    fr_,
                    {},
                    centersOfMassScaledBuffer,
                    forces,
                    &output.virial);
        }
        for (const rvec4& force : forces)
        {
            output.forces.emplace_back(force[XX], force[YY], force[ZZ]);
        }
        return output;
    }

    /*! \brief Checks that forces, virial and dV/dlambda are derivatives of the energy
     *
     * \param[in] functionType  Selects which kind of position restraints to check
     * \param[in] x             Coordinates
     * \param[in] lambda        Lambda for the restraints
     */
    void testDerivativesMatchEnergy(const InteractionFunction functionType,
                                    ArrayRef<const RVec>      x,
                                    const real                lambda)
    {
        const Output output = runKernel(functionType, x, box_, lambda);

        checkForcesMatchEnergy(x,
                               output.forces,
                               [&](ArrayRef<const RVec> xTrial)
                               { return runKernel(functionType, xTrial, box_, lambda).energy; });

        checkDvdlambdaMatchesEnergy(
                output.dvdlambda,
                [&](const double lambdaChange)
                { return runKernel(functionType, x, box_, lambda + lambdaChange).energy; });

        // Scaling the coordinates and box along a periodic dimension changes
        // the energy by twice the corresponding diagonal virial element.
        // The kernel takes care of how the reference positions scale.
        const FiniteDifferenceSettings settings;
        for (int d = 0; d < numPbcDimensions(pbcType_); d++)
        {
            const real numericalVirial =
                    0.5
                    * finiteDifferenceDerivative(
                            [&](const double strain)
                            {
                                std::vector<RVec> xTrial(x.begin(), x.end());
                                for (RVec& xAtom : xTrial)
                                {
                                    xAtom[d] *= 1 + strain;
                                }
                                matrix boxTrial;
                                copy_mat(box_, boxTrial);
                                boxTrial[d][d] *= 1 + strain;
                                return runKernel(functionType, xTrial, boxTrial, lambda).energy;
                            },
                            settings.strainStep);
            EXPECT_REAL_EQ_TOL(output.virial[d],
                               numericalVirial,
                               finiteDifferenceTolerance(
                                       std::max(1.0, std::abs(static_cast<double>(output.virial[d]))),
                                       settings.relativeTolerance,
                                       std::abs(output.energy),
                                       settings.strainStep))
                    << "Virial does not match the derivative of the energy with respect to "
                       "scaling dimension "
                    << d;
        }
    }

    //! Prepares the test with the coordinate and force constant input.
    void setValues(gmx::ArrayRef<const RVec> positions,
                   gmx::ArrayRef<const RVec> referencePositions,
                   gmx::ArrayRef<const RVec> forceConstants)
    {
        x_.resize(positions.size());
        std::copy(positions.begin(), positions.end(), x_.begin());

        for (size_t i = 0; i < positions.size(); i++)
        {
            // First item is "type" - each atom will have a different forceparam type
            // Second item is index - we'll just go from 0.
            idef_.il[InteractionFunction::PositionRestraints].iatoms.push_back(i);
            idef_.il[InteractionFunction::PositionRestraints].iatoms.push_back(i);

            auto& entry = idef_.iparams_posres.emplace_back();
            copy_rvec(referencePositions[i], entry.posres.pos0A);
            copy_rvec(forceConstants[i], entry.posres.fcA);
            clear_rvec(entry.posres.pos0B);
            clear_rvec(entry.posres.fcB);
        }
        f_.resize(x_.size(), { 0, 0, 0 });
        forceWithVirial_ = std::make_unique<ForceWithVirial>(f_, /*computeVirial=*/true);
    }
};

std::array<real, static_cast<size_t>(FreeEnergyPerturbationCouplingType::Count)> c_emptyLambdas = { { 0 } };

TEST_P(PositionRestraintsTest, BasicPosResNoFreeEnergy)
{
    SCOPED_TRACE(formatString("Testing PBC type: %s, refcoord type: %s",
                              c_pbcTypeNames[pbcType_].c_str(),
                              enumValueToString(refCoordScaling_)));
    const std::vector<RVec> positions          = { { 0.0, 0.0, 0.0 }, { 0.4, 0.5, 0.6 } };
    const std::vector<RVec> referencePositions = { { 0.0, 0.0, 0.0 }, { 0.5, 0.6, 0.0 } };
    const std::vector<RVec> forceConstants     = { { 1000, 500, 250 }, { 0, 200, 400 } };
    setValues(positions, referencePositions, forceConstants);
    std::vector<RVec> centersOfMassScaledBuffer  = { { 0.0, 0.0, 0.0 } };
    std::vector<RVec> centersOfMassBScaledBuffer = { { 0.0, 0.0, 0.0 } };
    // We cannot store C-style array rvec4 in an std::vector, so we use real and reinterpret
    std::vector<real> forcesStorage(positions.size() * 4);
    ArrayRef<rvec4>   forces =
            arrayRefFromArray(reinterpret_cast<rvec4*>(forcesStorage.data()), positions.size());
    for (auto& f : forces)
    {
        for (int d = 0; d < 4; d++)
        {
            f[d] = 0;
        }
    }
    real dvdl   = 0;
    RVec virial = { 0.0_real, 0.0_real, 0.0_real };

    const real v = posres_wrapper(idef_.il[InteractionFunction::PositionRestraints].iatoms,
                                  idef_.iparams_posres,
                                  pbc_,
                                  as_rvec_array(x_.data()),
                                  c_emptyLambdas,
                                  fr_,
                                  {},
                                  centersOfMassScaledBuffer,
                                  centersOfMassBScaledBuffer,
                                  forces,
                                  &virial,
                                  &dvdl);
    // Copy from 4-component vector buffer to forceWithVirial
    for (Index i = 0; i < gmx::ssize(forces); i++)
    {
        for (int d = 0; d < DIM; d++)
        {
            forceWithVirial_->force_[i][d] = forces[i][d];
        }
    }
    forceWithVirial_->addVirialContribution(virial);

    checker().checkSequence(
            std::begin(forceWithVirial_->force_), std::end(forceWithVirial_->force_), "Forces");
    checker().checkSequenceArray(3, forceWithVirial_->getVirial(), "Virial contribution");
    checker().checkReal(v, "Potential energy");
}

TEST_P(PositionRestraintsTest, DerivativesMatchEnergy)
{
    if (!c_canTestWithFiniteDifferences)
    {
        GTEST_SKIP() << "Numerical derivatives of the energy are only accurate enough in double "
                        "precision";
    }
    // Positions, reference positions and force constants in states A and B.
    // The second atom is restrained to a different periodic image.
    const std::vector<RVec> positions           = { { 0.1, 0.2, 0.3 }, { 0.4, 0.5, 0.6 } };
    const std::vector<RVec> referencePositionsA = { { 0.15, 0.1, 0.35 }, { 0.5, 0.6, 1.0 } };
    const std::vector<RVec> referencePositionsB = { { 0.05, 0.25, 0.3 }, { 0.45, 0.65, 0.9 } };
    const std::vector<RVec> forceConstantsA     = { { 1000, 500, 250 }, { 0, 200, 400 } };
    const std::vector<RVec> forceConstantsB     = { { 800, 600, 0 }, { 100, 300, 350 } };
    for (size_t i = 0; i < positions.size(); i++)
    {
        idef_.il[InteractionFunction::PositionRestraints].iatoms.push_back(i);
        idef_.il[InteractionFunction::PositionRestraints].iatoms.push_back(i);
        auto& entry = idef_.iparams_posres.emplace_back();
        copy_rvec(referencePositionsA[i], entry.posres.pos0A);
        copy_rvec(forceConstantsA[i], entry.posres.fcA);
        copy_rvec(referencePositionsB[i], entry.posres.pos0B);
        copy_rvec(forceConstantsB[i], entry.posres.fcB);
    }
    // Use a center of mass in state B that differs from that in state A
    fr_.posresComB[0] = { 0.1_real, 0.4_real, 0.2_real };

    for (const real lambda : { 0.0, 0.4, 1.0 })
    {
        SCOPED_TRACE(formatString("Testing PBC type: %s, refcoord type: %s, lambda %g",
                                  c_pbcTypeNames[pbcType_].c_str(),
                                  enumValueToString(refCoordScaling_),
                                  lambda));
        testDerivativesMatchEnergy(InteractionFunction::PositionRestraints, positions, lambda);
    }
}

TEST_P(PositionRestraintsTest, FlatBottomedDerivativesMatchEnergy)
{
    if (!c_canTestWithFiniteDifferences)
    {
        GTEST_SKIP() << "Numerical derivatives of the energy are only accurate enough in double "
                        "precision";
    }
    // The first atom is close to and the second far from its reference position.
    // The second atom is restrained to a different periodic image.
    const std::vector<RVec> positions          = { { 0.1, 0.2, 0.3 }, { 0.4, 0.5, 0.6 } };
    const std::vector<RVec> referencePositions = { { 0.12, 0.17, 0.31 }, { 0.6, 0.8, 1.3 } };
    idef_.iparams_fbposres.resize(positions.size());
    for (size_t i = 0; i < positions.size(); i++)
    {
        idef_.il[InteractionFunction::FlatBottomedPositionRestraints].iatoms.push_back(i);
        idef_.il[InteractionFunction::FlatBottomedPositionRestraints].iatoms.push_back(i);
    }

    for (const int geometry :
         { efbposresSPHERE, efbposresCYLINDERX, efbposresCYLINDERY, efbposresCYLINDERZ, efbposresX, efbposresY, efbposresZ })
    {
        // A negative radius inverts the restraint
        for (const real radius : { 0.15, -0.15 })
        {
            for (size_t i = 0; i < positions.size(); i++)
            {
                auto& entry = idef_.iparams_fbposres[i];
                copy_rvec(referencePositions[i], entry.fbposres.pos0);
                entry.fbposres.geom = geometry;
                entry.fbposres.r    = radius;
                entry.fbposres.k    = 300 + 100 * i;
            }
            SCOPED_TRACE(
                    formatString("Testing PBC type: %s, refcoord type: %s, geometry %d, radius %g",
                                 c_pbcTypeNames[pbcType_].c_str(),
                                 enumValueToString(refCoordScaling_),
                                 geometry,
                                 radius));
            testDerivativesMatchEnergy(InteractionFunction::FlatBottomedPositionRestraints, positions, 0.0);
        }
    }
}

//! PBC values for testing
std::vector<PbcType> c_pbcForTests = { PbcType::No, PbcType::XY, PbcType::Xyz };
//! Reference Coordinate Scaling values for testing
std::vector<RefCoordScaling> c_refCoordScalingForTests = { RefCoordScaling::No,
                                                           RefCoordScaling::Com,
                                                           RefCoordScaling::All };

INSTANTIATE_TEST_SUITE_P(PosResBasicTest,
                         PositionRestraintsTest,
                         ::testing::Combine(::testing::ValuesIn(c_refCoordScalingForTests),
                                            ::testing::ValuesIn(c_pbcForTests)));

} // namespace

} // namespace test

} // namespace gmx
