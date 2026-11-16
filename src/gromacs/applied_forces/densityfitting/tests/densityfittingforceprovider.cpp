/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2020- The GROMACS Authors
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
 * Tests amplitude lookup for density fitting
 *
 * \author Christian Blau <blau@kth.se>
 * \ingroup module_applied_forces
 */
#include "gmxpre.h"

#include "gromacs/applied_forces/densityfitting/densityfittingforceprovider.h"

#include <cctype>
#include <cmath>

#include <algorithm>
#include <string>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "gromacs/applied_forces/densityfitting/densityfittingamplitudelookup.h"
#include "gromacs/applied_forces/densityfitting/densityfittingparameters.h"
#include "gromacs/domdec/localatomset.h"
#include "gromacs/domdec/localatomsetmanager.h"
#include "gromacs/math/coordinatetransformation.h"
#include "gromacs/math/densityfit.h"
#include "gromacs/math/exponentialmovingaverage.h"
#include "gromacs/math/multidimarray.h"
#include "gromacs/mdspan/extensions.h"
#include "gromacs/mdtypes/enerdata.h"
#include "gromacs/mdtypes/forceoutput.h"
#include "gromacs/mdtypes/iforceprovider.h"
#include "gromacs/pbcutil/pbc.h"
#include "gromacs/topology/ifunc.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/keyvaluetree.h"
#include "gromacs/utility/keyvaluetreebuilder.h"
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

TEST(DensityFittingForceProviderState, RoundTripSaving)
{
    DensityFittingForceProviderState state;
    // set-up state
    state.adaptiveForceConstantScale_                   = 1.0;
    state.stepsSinceLastCalculation_                    = 0;
    state.exponentialMovingAverageState_.increasing_    = false;
    state.exponentialMovingAverageState_.weightedCount_ = 0;
    state.exponentialMovingAverageState_.weightedSum_   = 0;

    KeyValueTreeBuilder kvtBuilder;
    const std::string   identifier = "test-module";
    state.writeState(kvtBuilder.rootObject(), identifier);
    KeyValueTreeObject stateStoredInKvt = kvtBuilder.build();

    // invalidate state
    state.adaptiveForceConstantScale_                   = -1;
    state.stepsSinceLastCalculation_                    = -1;
    state.exponentialMovingAverageState_.increasing_    = true;
    state.exponentialMovingAverageState_.weightedCount_ = -1;
    state.exponentialMovingAverageState_.weightedSum_   = -1;

    // read back the original state
    state.readState(stateStoredInKvt, identifier);

    EXPECT_EQ(state.adaptiveForceConstantScale_, 1.0);
    EXPECT_EQ(state.stepsSinceLastCalculation_, 0);

    EXPECT_EQ(state.exponentialMovingAverageState_.increasing_, false);
    EXPECT_EQ(state.exponentialMovingAverageState_.weightedCount_, 0);
    EXPECT_EQ(state.exponentialMovingAverageState_.weightedSum_, 0);
}

/*! \brief Test parameters: similarity measure, amplitude method, whether densities are normalized,
 * whether the structure is transformed and whether adaptive force scaling is used */
using DensityFittingForceTestParameters =
        std::tuple<DensitySimilarityMeasureMethod, DensityFittingAmplitudeMethod, bool, bool, bool>;

/*! \brief Test fixture for checking density fitting forces against the energy
 *
 * Fits a subset of the atoms of the system to a density on a lattice
 * that differs in extent and spacing along the three dimensions and
 * does not have its origin at the origin of the coordinate system.
 */
class DensityFittingForceTest : public ::testing::TestWithParam<DensityFittingForceTestParameters>
{
public:
    DensityFittingForceTest() : localAtomSet_(atomSets_.add(c_fitAtoms))
    {
        bool useTransformation;
        std::tie(parameters_.similarityMeasureMethod_,
                 parameters_.amplitudeLookupMethod_,
                 parameters_.normalizeDensities_,
                 useTransformation,
                 parameters_.adaptiveForceScaling_) = GetParam();

        parameters_.active_                          = true;
        parameters_.indices_                         = c_fitAtoms;
        parameters_.forceConstant_                   = 1e5;
        parameters_.gaussianTransformSpreadingWidth_ = 0.2;
        // The density of an atom is cut off at this range, which makes the
        // energy discontinuous. Here we spread over the whole lattice
        // to avoid that. This also ensures that there are no voxels without
        // density, which do not contribute to the relative entropy.
        parameters_.gaussianTransformSpreadingRangeInMultiplesOfWidth_ = 9.0;
        parameters_.calculationIntervalInSteps_                        = 1;
        if (useTransformation)
        {
            parameters_.translationString_ = "0.02 -0.03 0.01";
            parameters_.transformationMatrixString_ =
                    "0.95 0.1 0.03 -0.05 1.02 0.04 0.02 -0.06 0.98";
        }

        // A smooth, positive reference density. Note that x is the last index.
        for (int z = 0; z < c_latticeExtent[ZZ]; z++)
        {
            for (int y = 0; y < c_latticeExtent[YY]; y++)
            {
                for (int x = 0; x < c_latticeExtent[XX]; x++)
                {
                    const RVec fromFirstPeak   = RVec(x, y, z) - RVec(4.3, 5.1, 3.2);
                    const RVec fromSecondPeak  = RVec(x, y, z) - RVec(6.6, 3.4, 5.7);
                    referenceDensity_(z, y, x) = 0.1 + std::exp(-fromFirstPeak.norm2() / 8)
                                                 + 0.7 * std::exp(-fromSecondPeak.norm2() / 5);
                }
            }
        }

        clear_mat(box_);
        box_[XX][XX] = 4;
        box_[YY][YY] = 4.5;
        box_[ZZ][ZZ] = 5;
    }

    //! Output of the force provider
    struct Output
    {
        //! The energy
        double energy = 0;
        //! The forces
        std::vector<RVec> forces;
    };

    //! Computes the energy and forces for coordinates \c x
    Output compute(ArrayRef<const RVec> x)
    {
        // The force provider changes its state when adaptive force scaling
        // is used, so we need a new one for every evaluation.
        DensityFittingForceProviderState state;
        state.adaptiveForceConstantScale_ = 1.7;
        const RVec latticeScale           = { 1 / c_latticeSpacing[XX],
                                              1 / c_latticeSpacing[YY],
                                              1 / c_latticeSpacing[ZZ] };
        const RVec latticeTranslation = { -c_latticeOrigin[XX], -c_latticeOrigin[YY], -c_latticeOrigin[ZZ] };
        DensityFittingForceProvider forceProvider(parameters_,
                                                  referenceDensity_.asConstView(),
                                                  TranslateAndScale(latticeScale, latticeTranslation),
                                                  localAtomSet_,
                                                  PbcType::Xyz,
                                                  0.002,
                                                  state);

        ForceProviderInput forceProviderInput(
                x, gmx::ssize(x), c_charges, c_masses, 0.0, 0, box_, mpiComm_, nullptr);

        Output output;
        output.forces.resize(x.size(), { 0.0_real, 0.0_real, 0.0_real });
        ForceWithVirial     forceWithVirial(output.forces, false);
        gmx_enerdata_t      enerd(1, nullptr);
        ForceProviderOutput forceProviderOutput(&forceWithVirial, &enerd);

        forceProvider.calculateForces(forceProviderInput, &forceProviderOutput);

        output.energy = enerd.term[InteractionFunction::DensityFitting];
        return output;
    }

protected:
    //! The number of lattice points of the density along x, y and z
    const IVec c_latticeExtent = { 11, 9, 10 };
    //! The lattice spacing in nm
    const RVec c_latticeSpacing = { 0.15, 0.18, 0.16 };
    //! The location of the first lattice point in nm
    const RVec c_latticeOrigin = { 0.6, 0.5, 0.4 };

    //! Positions of all atoms in the system, the second atom is in another periodic image
    const std::vector<RVec> c_x = { { 1.12, 1.31, 0.93 }, { 5.55, 1.03, 1.27 },
                                    { 0.3, 0.2, 0.1 },    { 1.21, 1.42, 1.38 },
                                    { 1.67, 1.18, 1.04 }, { 1.38, 0.97, 1.52 },
                                    { 2.5, 2.7, 3.1 },    { 1.02, 1.24, 1.45 } };
    //! Masses of all atoms in the system
    const std::vector<real> c_masses = { 12, 1, 16, 14, 32, 16, 12, 1 };
    //! Charges of all atoms in the system, positive to avoid voxels with negative density
    const std::vector<real> c_charges = { 0.3, 0.7, 0.4, 1.1, 0.2, 0.5, 0.6, 0.9 };
    //! The atoms that are fitted to the density
    const std::vector<Index> c_fitAtoms = { 0, 1, 3, 4, 5, 7 };

    MpiComm                                             mpiComm_{ MpiComm::SingleRank{} };
    LocalAtomSetManager                                 atomSets_;
    LocalAtomSet                                        localAtomSet_;
    DensityFittingParameters                            parameters_;
    MultiDimArray<std::vector<float>, dynamicExtents3D> referenceDensity_{ c_latticeExtent[ZZ],
                                                                           c_latticeExtent[YY],
                                                                           c_latticeExtent[XX] };
    matrix                                              box_;
};

TEST_P(DensityFittingForceTest, ForcesMatchEnergy)
{
    if (!c_canTestWithFiniteDifferences)
    {
        GTEST_SKIP() << "Numerical derivatives of the energy are only accurate enough in double "
                        "precision";
    }

    const Output output = compute(c_x);

    /* The density and the similarity are computed in single precision.
     * To limit the effect of the rounding errors on the numerical
     * derivative, we use a step size that is not much smaller than
     * the width of the Gaussians and a larger tolerance.
     */
    FiniteDifferenceSettings settings;
    settings.coordinateStep    = 0.01;
    settings.relativeTolerance = 1e-3;
    checkForcesMatchEnergy(
            c_x, output.forces, [&](ArrayRef<const RVec> xTrial) { return compute(xTrial).energy; }, settings);
}

/*! \brief Checks that atoms are put in the periodic image closest to the center of the density
 *
 * Uses a density lattice that is much longer along x than along y and z,
 * with an atom close to the far end along x. With PBC that atom should
 * not be moved to another periodic image, so the result should be the same
 * as without PBC.
 */
TEST(DensityFittingForceProviderTest, UsesCenterOfNonCubicDensityForPbc)
{
    const IVec latticeExtent  = { 40, 8, 8 };
    const real latticeSpacing = 0.1;
    // Note that x is the last index
    MultiDimArray<std::vector<float>, dynamicExtents3D> referenceDensity(
            latticeExtent[ZZ], latticeExtent[YY], latticeExtent[XX]);
    for (int z = 0; z < latticeExtent[ZZ]; z++)
    {
        for (int y = 0; y < latticeExtent[YY]; y++)
        {
            for (int x = 0; x < latticeExtent[XX]; x++)
            {
                referenceDensity(z, y, x) = 1 + 0.1 * x + 0.2 * y + 0.3 * z;
            }
        }
    }

    DensityFittingParameters parameters;
    parameters.active_                          = true;
    parameters.indices_                         = { 0 };
    parameters.similarityMeasureMethod_         = DensitySimilarityMeasureMethod::innerProduct;
    parameters.amplitudeLookupMethod_           = DensityFittingAmplitudeMethod::Unity;
    parameters.normalizeDensities_              = false;
    parameters.adaptiveForceScaling_            = false;
    parameters.forceConstant_                   = 1e3;
    parameters.gaussianTransformSpreadingWidth_ = 0.2;
    parameters.gaussianTransformSpreadingRangeInMultiplesOfWidth_ = 4.0;
    parameters.calculationIntervalInSteps_                        = 1;

    // The atom is 1.6 nm from the center of the density along x
    const std::vector<RVec> x       = { { 3.6, 0.4, 0.4 } };
    const std::vector<real> charges = { 0 };
    const std::vector<real> masses  = { 1 };
    matrix                  box     = { { 4.5, 0, 0 }, { 0, 4.5, 0 }, { 0, 0, 4.5 } };

    MpiComm             mpiComm{ MpiComm::SingleRank{} };
    LocalAtomSetManager atomSets;
    LocalAtomSet        localAtomSet = atomSets.add(parameters.indices_);

    auto compute = [&](const PbcType pbcType)
    {
        DensityFittingForceProviderState state;
        const RVec latticeScale = { 1 / latticeSpacing, 1 / latticeSpacing, 1 / latticeSpacing };
        DensityFittingForceProvider forceProvider(parameters,
                                                  referenceDensity.asConstView(),
                                                  TranslateAndScale(latticeScale, { 0, 0, 0 }),
                                                  localAtomSet,
                                                  pbcType,
                                                  0.002,
                                                  state);

        ForceProviderInput forceProviderInput(
                x, gmx::ssize(x), charges, masses, 0.0, 0, box, mpiComm, nullptr);

        std::vector<RVec>   forces(x.size(), { 0.0_real, 0.0_real, 0.0_real });
        ForceWithVirial     forceWithVirial(forces, false);
        gmx_enerdata_t      enerd(1, nullptr);
        ForceProviderOutput forceProviderOutput(&forceWithVirial, &enerd);

        forceProvider.calculateForces(forceProviderInput, &forceProviderOutput);

        return std::make_pair(enerd.term[InteractionFunction::DensityFitting], forces[0]);
    };

    const auto [energyWithoutPbc, forceWithoutPbc] = compute(PbcType::No);
    const auto [energyWithPbc, forceWithPbc]       = compute(PbcType::Xyz);

    // Check that the atom actually overlaps with the density
    EXPECT_LT(energyWithoutPbc, -1);
    EXPECT_REAL_EQ(energyWithoutPbc, energyWithPbc);
    for (int d = 0; d < DIM; d++)
    {
        EXPECT_REAL_EQ(forceWithoutPbc[d], forceWithPbc[d]);
    }
}

//! Returns a name for a test with parameters \c info
std::string nameOfTest(const ::testing::TestParamInfo<DensityFittingForceTestParameters>& info)
{
    const char* const c_measureNames[] = { "InnerProduct", "RelativeEntropy", "CrossCorrelation" };
    const char* const c_amplitudeNames[] = { "Unity", "Mass", "Charge" };
    std::string       name               = formatString("%s_Amplitude%s_%s_%s_%s",
                                    c_measureNames[static_cast<int>(std::get<0>(info.param))],
                                    c_amplitudeNames[static_cast<int>(std::get<1>(info.param))],
                                    std::get<2>(info.param) ? "Normalized" : "NotNormalized",
                                    std::get<3>(info.param) ? "Transformed" : "NotTransformed",
                                    std::get<4>(info.param) ? "AdaptiveScaling" : "ConstantScaling");
    std::replace_if(name.begin(), name.end(), [](const char c) { return !std::isalnum(c); }, '_');
    return name;
}

INSTANTIATE_TEST_SUITE_P(AllMeasures,
                         DensityFittingForceTest,
                         ::testing::Combine(::testing::Values(DensitySimilarityMeasureMethod::innerProduct,
                                                              DensitySimilarityMeasureMethod::relativeEntropy,
                                                              DensitySimilarityMeasureMethod::crossCorrelation),
                                            ::testing::Values(DensityFittingAmplitudeMethod::Unity,
                                                              DensityFittingAmplitudeMethod::Mass,
                                                              DensityFittingAmplitudeMethod::Charge),
                                            ::testing::Bool(),
                                            ::testing::Bool(),
                                            ::testing::Bool()),
                         nameOfTest);


} // namespace
} // namespace test
} // namespace gmx
