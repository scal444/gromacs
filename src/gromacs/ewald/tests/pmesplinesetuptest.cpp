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
 * Tests the SIMD PME spline setup (interpolation indices and B-splines)
 * against the scalar reference implementation.
 *
 * \ingroup module_ewald
 */
#include "gmxpre.h"

#include <cmath>

#include <algorithm>
#include <numeric>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "gromacs/ewald/pme_grid.h"
#include "gromacs/ewald/pme_internal.h"
#include "gromacs/ewald/pme_spline_setup.h"
#include "gromacs/math/boxmatrix.h"
#include "gromacs/utility/real.h"
#include "gromacs/utility/stringutil.h"
#include "gromacs/utility/vec.h"
#include "gromacs/utility/vectypes.h"

namespace gmx
{
namespace test
{
namespace
{

//! Box types to test
enum class BoxType
{
    Rectangular,
    Triclinic,
    Dodecahedron
};

//! Returns a box of type \p boxType
void makeBox(BoxType boxType, matrix box)
{
    clear_mat(box);
    switch (boxType)
    {
        case BoxType::Rectangular:
            box[XX][XX] = 3.1;
            box[YY][YY] = 2.7;
            box[ZZ][ZZ] = 4.3;
            break;
        case BoxType::Triclinic:
            box[XX][XX] = 3.1;
            box[YY][XX] = 0.9;
            box[YY][YY] = 2.7;
            box[ZZ][XX] = -1.2;
            box[ZZ][YY] = 1.1;
            box[ZZ][ZZ] = 4.3;
            break;
        case BoxType::Dodecahedron:
            // Rhombic dodecahedron, xy-square
            box[XX][XX] = 4.0;
            box[YY][YY] = 4.0;
            box[ZZ][XX] = 2.0;
            box[ZZ][YY] = 2.0;
            box[ZZ][ZZ] = 4.0 * std::sqrt(0.5);
            break;
    }
}

//! Parameters: box type, number of atoms, PME order, whether the grid is decomposed
using SplineSetupParameters = std::tuple<BoxType, int, int, bool>;

class PmeSplineSetupTest : public ::testing::TestWithParam<SplineSetupParameters>
{
};

TEST_P(PmeSplineSetupTest, SimdMatchesScalar)
{
    const auto [boxType, numAtoms, order, decomposed] = GetParam();

    std::mt19937                         rng(1234 + numAtoms * 7 + order);
    std::uniform_real_distribution<real> uniform(0, 1);

    matrix box;
    makeBox(boxType, box);
    matrix recipbox;
    invertBoxMatrix(box, recipbox);

    PmeInterpolationIndexLookup lookup;
    lookup.numGridPoints = { 28, 25, 40 };
    copy_mat(recipbox, lookup.recipbox);

    std::array<std::vector<int>, DIM>  gridIndexToLocal;
    std::array<std::vector<real>, DIM> fractionShift;
    std::array<int, DIM>               localGridSize;
    for (int d = 0; d < DIM; d++)
    {
        const int n = lookup.numGridPoints[d];
        // With decomposition, use a local range in x and y, which leads
        // to non-zero fraction shifts at the boundaries
        const bool decomposeDim = decomposed && d != ZZ;
        const int  localStart   = decomposeDim ? n / 3 : 0;
        const int  localRange   = decomposeDim ? n / 2 : n;
        std::tie(gridIndexToLocal[d], fractionShift[d]) =
                make_gridindex_to_localindex(n, localStart, localRange, true);
        lookup.gridIndexToLocalIndex[d] = gridIndexToLocal[d];
        if (d < DIM - 1)
        {
            lookup.fractionShift[d] = fractionShift[d];
        }
        localGridSize[d] = *std::max_element(gridIndexToLocal[d].begin(), gridIndexToLocal[d].end()) + 1;
    }

    // Random coordinates, in fractional coordinates within [-1.5, 2.5),
    // so also outside the unit cell, but within the shift range of the lookup tables.
    // Some atoms have z exactly on a grid line.
    std::vector<RVec> x(numAtoms);
    for (int i = 0; i < numAtoms; i++)
    {
        RVec s;
        for (int d = 0; d < DIM; d++)
        {
            s[d] = -1.5 + 4 * uniform(rng);
        }
        if (i % 11 == 3)
        {
            s[ZZ] = 0;
        }
        for (int d = 0; d < DIM; d++)
        {
            x[i][d] = s[XX] * box[XX][d] + s[YY] * box[YY][d] + s[ZZ] * box[ZZ][d];
        }
    }

    // Thread assignment tables
    const int                         numThreads = 6;
    std::array<std::vector<int>, DIM> gridToThread;
    for (int d = 0; d < DIM; d++)
    {
        gridToThread[d].resize(localGridSize[d]);
        for (int g = 0; g < localGridSize[d]; g++)
        {
            gridToThread[d][g] = (d == XX ? (g * 3) / localGridSize[d] : 0)
                                 + (d == YY ? ((g * 2) / localGridSize[d]) * 3 : 0);
        }
    }

    const PmeSplineSetupKernel kernels[2] = { PmeSplineSetupKernel::Scalar, PmeSplineSetupKernel::Simd };

    std::array<std::vector<RVec>, 2> fractx;
    std::array<std::vector<IVec>, 2> idx;
    std::array<std::vector<int>, 2>  threadIndex;
    std::array<std::vector<int>, 2>  threadCounts;
    for (int k = 0; k < 2; k++)
    {
        fractx[k].assign(numAtoms, { -1, -1, -1 });
        idx[k].assign(numAtoms, { -1, -1, -1 });
        threadIndex[k].assign(numAtoms, -1);
        threadCounts[k].assign(numThreads, 0);

        PmeAtomThreadAssignment threadAssignment;
        for (int d = 0; d < DIM; d++)
        {
            threadAssignment.gridToThread[d] = gridToThread[d].data();
        }
        threadAssignment.threadIndex  = threadIndex[k].data();
        threadAssignment.threadCounts = threadCounts[k].data();

        // Split in two ranges, as done with OpenMP threads
        const int split = numAtoms / 3;
        pmeCalcInterpolationIndices(lookup, x, 0, split, fractx[k], idx[k], &threadAssignment, kernels[k]);
        pmeCalcInterpolationIndices(
                lookup, x, split, numAtoms, fractx[k], idx[k], &threadAssignment, kernels[k]);
    }

    // The tolerance for the fractions is determined by the magnitude of the scaled coordinates
    const real fractionTolerance =
            16 * GMX_REAL_EPS * c_pmeNeighborUnitcellCount
            * std::max({ lookup.numGridPoints[XX], lookup.numGridPoints[YY], lookup.numGridPoints[ZZ] });
    auto distanceToInteger = [](real f) { return std::abs(f - std::round(f)); };
    bool allIndicesEqual   = true;
    for (int i = 0; i < numAtoms; i++)
    {
        for (int d = 0; d < DIM; d++)
        {
            if (idx[0][i][d] == idx[1][i][d])
            {
                EXPECT_NEAR(fractx[0][i][d], fractx[1][i][d], fractionTolerance)
                        << "atom " << i << " dim " << d;
            }
            else
            {
                // Due to different rounding, an atom on a grid line can end
                // up in a neighboring cell, this is only allowed when the
                // fraction is close to 0 or 1.
                allIndicesEqual = false;
                EXPECT_LT(distanceToInteger(fractx[0][i][d]), fractionTolerance)
                        << "index mismatch for atom " << i << " dim " << d;
                EXPECT_LT(distanceToInteger(fractx[1][i][d]), fractionTolerance)
                        << "index mismatch for atom " << i << " dim " << d;
            }
        }
        if (idx[0][i][XX] == idx[1][i][XX] && idx[0][i][YY] == idx[1][i][YY]
            && idx[0][i][ZZ] == idx[1][i][ZZ])
        {
            EXPECT_EQ(threadIndex[0][i], threadIndex[1][i]) << "atom " << i;
        }
    }
    if (allIndicesEqual)
    {
        EXPECT_EQ(threadCounts[0], threadCounts[1]);
    }

    // B-splines: use the scalar fractions as input for both kernels.
    // Use a shuffled subset of the atoms as index, as with thread-local atom lists.
    std::vector<real> coefficient(numAtoms);
    for (int i = 0; i < numAtoms; i++)
    {
        coefficient[i] = (i % 3 == 1 || i % 7 == 2) ? 0 : uniform(rng) - 0.5;
    }
    std::vector<int> ind(numAtoms);
    std::iota(ind.begin(), ind.end(), 0);
    std::shuffle(ind.begin(), ind.end(), rng);
    // Keep some contiguous stretches, as is common in practice
    std::sort(ind.begin(), ind.begin() + numAtoms / 2);
    const int numInd = numAtoms - numAtoms / 5;

    const real sentinel = -1234.5;
    for (const bool computeAll : { false, true })
    {
        std::array<std::array<std::vector<real>, DIM>, 2> theta;
        std::array<std::array<std::vector<real>, DIM>, 2> dtheta;
        for (int k = 0; k < 2; k++)
        {
            std::array<real*, DIM> thetaPtr;
            std::array<real*, DIM> dthetaPtr;
            for (int d = 0; d < DIM; d++)
            {
                theta[k][d].assign(numAtoms * order, sentinel);
                dtheta[k][d].assign(numAtoms * order, sentinel);
                thetaPtr[d]  = theta[k][d].data();
                dthetaPtr[d] = dtheta[k][d].data();
            }
            pmeMakeBsplines(
                    thetaPtr, dthetaPtr, order, fractx[0], numInd, ind.data(), coefficient.data(), computeAll, kernels[k]);
        }

        const real splineTolerance = 64 * GMX_REAL_EPS;
        int        numFilled       = 0;
        for (int d = 0; d < DIM; d++)
        {
            for (int i = 0; i < numAtoms; i++)
            {
                const bool shouldBeFilled = (i < numInd && (computeAll || coefficient[ind[i]] != 0));
                for (int o = 0; o < order; o++)
                {
                    const int  j     = i * order + o;
                    const real ref   = theta[0][d][j];
                    const real dref  = dtheta[0][d][j];
                    const real test  = theta[1][d][j];
                    const real dtest = dtheta[1][d][j];
                    if (shouldBeFilled)
                    {
                        EXPECT_NE(ref, sentinel);
                        EXPECT_NEAR(ref, test, splineTolerance)
                                << "theta i " << i << " dim " << d << " o " << o;
                        EXPECT_NEAR(dref, dtest, splineTolerance)
                                << "dtheta i " << i << " dim " << d << " o " << o;
                        numFilled++;
                    }
                    else
                    {
                        EXPECT_EQ(ref, sentinel);
                        EXPECT_EQ(test, sentinel) << "theta i " << i << " should not be set";
                        EXPECT_EQ(dtest, sentinel) << "dtheta i " << i << " should not be set";
                    }
                }
            }
        }
        if (numAtoms >= 10)
        {
            EXPECT_GT(numFilled, 0);
        }

        // Sanity check of the reference: the splines sum to one and the derivatives to zero
        for (int i = 0; i < numInd; i++)
        {
            if (computeAll || coefficient[ind[i]] != 0)
            {
                for (int d = 0; d < DIM; d++)
                {
                    real sum  = 0;
                    real dsum = 0;
                    for (int o = 0; o < order; o++)
                    {
                        sum += theta[1][d][i * order + o];
                        dsum += dtheta[1][d][i * order + o];
                    }
                    EXPECT_NEAR(sum, 1, splineTolerance);
                    EXPECT_NEAR(dsum, 0, splineTolerance);
                }
            }
        }
    }
}

//! Returns a readable test name
std::string nameOfTest(const ::testing::TestParamInfo<SplineSetupParameters>& info)
{
    const char* boxNames[] = { "Rectangular", "Triclinic", "Dodecahedron" };
    return formatString("%s_%dAtoms_Order%d_%s",
                        boxNames[static_cast<int>(std::get<0>(info.param))],
                        std::get<1>(info.param),
                        std::get<2>(info.param),
                        std::get<3>(info.param) ? "Decomposed" : "Full");
}

INSTANTIATE_TEST_SUITE_P(SimdVsScalar,
                         PmeSplineSetupTest,
                         ::testing::Combine(::testing::Values(BoxType::Rectangular,
                                                              BoxType::Triclinic,
                                                              BoxType::Dodecahedron),
                                            ::testing::Values(1, 7, 16, 17, 33, 100, 1001),
                                            ::testing::Values(3, 4, 5, 8),
                                            ::testing::Values(false, true)),
                         nameOfTest);

} // namespace
} // namespace test
} // namespace gmx
