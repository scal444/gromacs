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
 * \brief Tests comparing the SIMD SD update kernel against the scalar reference.
 *
 * \ingroup module_mdlib
 */
#include "gmxpre.h"

#include <cmath>

#include <algorithm>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "gromacs/mdlib/update_sd.h"
#include "gromacs/topology/topology_enums.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/real.h"
#include "gromacs/utility/stringutil.h"
#include "gromacs/utility/vectypes.h"

#include "testutils/testasserts.h"

namespace gmx
{
namespace test
{
namespace
{

//! Parameters for the SD update comparison
struct SDUpdateTestParameters
{
    //! Number of atoms, including the first \p start ones that are not updated
    int numAtoms;
    //! First atom to update
    int start;
    //! Whether there is a partially frozen freeze group in use (forces the scalar fallback)
    bool partialFreeze;
    //! Whether to use a global atom index (as with domain decomposition)
    bool useGlobalAtomIndex;
    //! Whether the temperature-coupling and freeze group index arrays are empty
    bool emptyGroupArrays;
};

//! Printer for the test parameters
std::string sdTestName(const ::testing::TestParamInfo<SDUpdateTestParameters>& info)
{
    const auto& p = info.param;
    return formatString("N%d_start%d_%s_%s_%s",
                        p.numAtoms,
                        p.start,
                        p.partialFreeze ? "partialFreeze" : "allOrNoneFreeze",
                        p.useGlobalAtomIndex ? "gatindex" : "local",
                        p.emptyGroupArrays ? "noGroups" : "groups");
}

//! Input and output of an SD update
struct SDSystem
{
    std::vector<RVec>           x, xprime, v, f;
    std::vector<real>           invmass;
    std::vector<ParticleType>   ptype;
    std::vector<unsigned short> cFREEZE, cTC;
    std::vector<int>            gatindex;
};

//! Test fixture
class SDUpdateSimdTest : public ::testing::TestWithParam<SDUpdateTestParameters>
{
public:
    SDUpdateSimdTest()
    {
        const auto& p = GetParam();

        // Four temperature-coupling groups, the last without friction and noise (tau_t = 0)
        sdc_ = { { std::exp(-0.002 / 1.0) }, { std::exp(-0.002 / 0.1) }, { std::exp(-0.002 / 5.0) }, { 1.0 } };
        // kT values per group, the noise magnitude is sqrt(kT (1 - em^2))
        const double kT[] = { 1.1, 0.7, 2.3, 1.0 };
        sdsig_.resize(sdc_.size());
        for (size_t g = 0; g < sdc_.size(); g++)
        {
            sdsig_[g].V = std::sqrt(kT[g] * (1 - sdc_[g].em * sdc_[g].em));
        }
        // Freeze groups: none, all, and, optionally, partial
        numFreezeGroups_ = p.partialFreeze ? 3 : 2;
        acceleration_    = { RVec{ 0.1, -0.2, 0.3 } };

        std::mt19937                     gen(1234 + p.numAtoms);
        std::uniform_real_distribution<> uniform(-1.0, 1.0);
        std::uniform_int_distribution<>  typeDist(0, 9);

        const int n = p.numAtoms;
        s_.x.resize(n);
        s_.xprime.resize(n);
        s_.v.resize(n);
        s_.f.resize(n);
        s_.invmass.resize(n);
        s_.ptype.resize(n);
        for (int a = 0; a < n; a++)
        {
            for (int d = 0; d < DIM; d++)
            {
                s_.x[a][d]      = 5 * uniform(gen);
                s_.xprime[a][d] = 5 * uniform(gen);
                s_.v[a][d]      = 2 * uniform(gen);
                s_.f[a][d]      = 500 * uniform(gen);
            }
            const int type = typeDist(gen);
            s_.ptype[a]    = (type == 0) ? ParticleType::Shell
                                         : (type == 1 ? ParticleType::VSite : ParticleType::Atom);
            s_.invmass[a] = (s_.ptype[a] == ParticleType::VSite) ? 0 : 0.05 + 0.5 * (1 + uniform(gen));
        }
        if (!p.emptyGroupArrays)
        {
            std::uniform_int_distribution<> tcDist(0, int(sdc_.size()) - 1);
            std::uniform_int_distribution<> freezeDist(0, 3 * numFreezeGroups_ - 1);
            for (int a = 0; a < n; a++)
            {
                s_.cTC.push_back(tcDist(gen));
                // Mostly unfrozen
                const int freeze = freezeDist(gen);
                s_.cFREEZE.push_back(freeze < numFreezeGroups_ ? freeze : 0);
            }
            if (p.partialFreeze && n > p.start)
            {
                // Make sure the partially frozen group is used
                s_.cFREEZE[p.start] = 2;
            }
        }
        if (p.useGlobalAtomIndex)
        {
            for (int a = 0; a < n; a++)
            {
                s_.gatindex.push_back(100000 + 7 * a);
            }
        }
    }

    //! Run an SD update on a copy of the input system
    SDSystem run(SDUpdate updateType, SDUpdateKernel kernel, const SDSystem& input) const
    {
        const auto& p = GetParam();
        SDSystem    s = input;
        Matrix3x3   parrinelloRahmanM;
        doSDUpdate(updateType,
                   kernel,
                   sdc_,
                   sdsig_,
                   p.start,
                   p.numAtoms,
                   0.002,
                   nFreeze(),
                   s.invmass,
                   s.ptype,
                   s.cFREEZE,
                   updateType == SDUpdate::ForcesOnly ? ArrayRef<const unsigned short>{} : s.cTC,
                   {},
                   acceleration_.data(),
                   s.x.data(),
                   s.xprime.data(),
                   s.v.data(),
                   updateType == SDUpdate::FrictionAndNoiseOnly ? nullptr : s.f.data(),
                   987654321,
                   1993,
                   s.gatindex.empty() ? nullptr : s.gatindex.data(),
                   0,
                   parrinelloRahmanM);
        return s;
    }

    //! Returns whether atom \p a is updated (not a shell and not frozen)
    bool isUpdated(int a) const
    {
        const int freezeGroup = s_.cFREEZE.empty() ? 0 : s_.cFREEZE[a];
        return s_.ptype[a] != ParticleType::Shell && !nFreeze_[freezeGroup][XX];
    }

    //! Compares the SIMD and scalar results, rounding-level differences allowed
    void compare(const SDSystem& ref, const SDSystem& test, const char* updateName)
    {
        // Values are of order 10, the update has a few operations
        const FloatingPointTolerance tolerance = relativeToleranceAsUlp(10.0, 8);
        double                       maxDiffV = 0, maxDiffX = 0;
        for (int a = 0; a < GetParam().numAtoms; a++)
        {
            for (int d = 0; d < DIM; d++)
            {
                SCOPED_TRACE(formatString("%s, atom %d, dim %d", updateName, a, d));
                if (a < GetParam().start || !isUpdated(a))
                {
                    // Not updated or reset exactly, must be bitwise identical
                    EXPECT_EQ(ref.v[a][d], test.v[a][d]);
                    EXPECT_EQ(ref.xprime[a][d], test.xprime[a][d]);
                }
                else
                {
                    EXPECT_REAL_EQ_TOL(ref.v[a][d], test.v[a][d], tolerance);
                    EXPECT_REAL_EQ_TOL(ref.xprime[a][d], test.xprime[a][d], tolerance);
                }
                maxDiffV = std::max(maxDiffV, double(std::abs(ref.v[a][d] - test.v[a][d])));
                maxDiffX = std::max(maxDiffX, double(std::abs(ref.xprime[a][d] - test.xprime[a][d])));
                EXPECT_EQ(ref.x[a][d], test.x[a][d]);
            }
        }
        RecordProperty(formatString("%sMaxAbsDiffV", updateName), formatString("%g", maxDiffV));
        RecordProperty(formatString("%sMaxAbsDiffXprime", updateName), formatString("%g", maxDiffX));
    }

    //! The test system
    SDSystem s_;
    //! SD friction constants per T-coupling group
    std::vector<gmx_sd_const_t> sdc_;
    //! SD noise magnitudes per T-coupling group
    std::vector<gmx_sd_sigma_t> sdsig_;
    //! Freeze flags per freeze group: none, all and partial
    ivec nFreeze_[3] = { { 0, 0, 0 }, { 1, 1, 1 }, { 0, 1, 0 } };
    //! The number of freeze groups in use
    int numFreezeGroups_;
    //! Constant acceleration
    std::vector<RVec> acceleration_;

    //! Returns the freeze groups as ivec array
    ArrayRef<const ivec> nFreeze() const { return arrayRefFromArray(nFreeze_, numFreezeGroups_); }
};

TEST_P(SDUpdateSimdTest, SimdSupportIsDetected)
{
    const bool expectSimd = c_haveSimdSDUpdate && !GetParam().partialFreeze;
    EXPECT_EQ(expectSimd, sdUpdateSimdIsSupported(nFreeze(), false, 0));
    EXPECT_FALSE(sdUpdateSimdIsSupported(nFreeze(), true, 0));
    EXPECT_FALSE(sdUpdateSimdIsSupported(nFreeze(), false, 0.01));
}

TEST_P(SDUpdateSimdTest, AllUpdateTypesMatchScalar)
{
    for (SDUpdate updateType : { SDUpdate::ForcesOnly, SDUpdate::FrictionAndNoiseOnly, SDUpdate::Combined })
    {
        const char* name =
                (updateType == SDUpdate::ForcesOnly)
                        ? "ForcesOnly"
                        : (updateType == SDUpdate::FrictionAndNoiseOnly ? "FrictionAndNoiseOnly"
                                                                        : "Combined");
        const SDSystem ref  = run(updateType, SDUpdateKernel::Scalar, s_);
        const SDSystem simd = run(updateType, SDUpdateKernel::SimdWhenPossible, s_);
        compare(ref, simd, name);
    }
}

TEST_P(SDUpdateSimdTest, SplitUpdateMatchesScalar)
{
    // The constrained SD path: forces first, then friction and noise
    const SDSystem ref  = run(SDUpdate::FrictionAndNoiseOnly,
                             SDUpdateKernel::Scalar,
                             run(SDUpdate::ForcesOnly, SDUpdateKernel::Scalar, s_));
    const SDSystem simd = run(SDUpdate::FrictionAndNoiseOnly,
                              SDUpdateKernel::SimdWhenPossible,
                              run(SDUpdate::ForcesOnly, SDUpdateKernel::SimdWhenPossible, s_));
    compare(ref, simd, "Split");
}

TEST_P(SDUpdateSimdTest, RandomNumbersAreBitwiseIdentical)
{
    // With zero velocities, the new velocity is exactly the noise term,
    // which the SIMD kernel computes with the same scalar operations.
    SDSystem input = s_;
    std::fill(input.v.begin(), input.v.end(), RVec{ 0, 0, 0 });
    const SDSystem ref = run(SDUpdate::FrictionAndNoiseOnly, SDUpdateKernel::Scalar, input);
    const SDSystem simd = run(SDUpdate::FrictionAndNoiseOnly, SDUpdateKernel::SimdWhenPossible, input);
    int numNonZero = 0;
    for (int a = 0; a < GetParam().numAtoms; a++)
    {
        for (int d = 0; d < DIM; d++)
        {
            EXPECT_EQ(ref.v[a][d], simd.v[a][d]) << "atom " << a << " dim " << d;
            numNonZero += (ref.v[a][d] != 0);
        }
    }
    if (GetParam().numAtoms - GetParam().start > 10)
    {
        EXPECT_GT(numNonZero, 0);
    }
}

const SDUpdateTestParameters c_sdUpdateParameters[] = {
    { 1, 0, false, false, false },   { 7, 0, false, false, false },
    { 37, 0, false, false, false },  { 37, 3, false, true, false },
    { 100, 16, false, false, true }, { 131, 0, false, true, false },
    { 131, 5, true, false, false },  { 131, 0, true, true, false },
    { 200, 64, false, true, false }, { 257, 0, false, false, false },
    { 64, 0, false, false, true },
};

INSTANTIATE_TEST_SUITE_P(Kernels, SDUpdateSimdTest, ::testing::ValuesIn(c_sdUpdateParameters), sdTestName);

} // namespace
} // namespace test
} // namespace gmx
