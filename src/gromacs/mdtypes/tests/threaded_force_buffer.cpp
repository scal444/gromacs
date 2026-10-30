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
 * Tests for the \p ThreadForceBuffer class
 *
 * \ingroup module_mdtypes
 */
#include "gmxpre.h"

#include "gromacs/mdtypes/threaded_force_buffer.h"

#include <gtest/gtest.h>

#include "gromacs/pbcutil/ishift.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/vectypes.h"

namespace gmx
{
namespace test
{
namespace
{

template<typename T>
class ThreadForceBufferTest : public ::testing::Test
{
};

using ForceBufferElementTypes = ::testing::Types<RVec, rvec4>;
TYPED_TEST_SUITE(ThreadForceBufferTest, ForceBufferElementTypes);

TYPED_TEST(ThreadForceBufferTest, ShiftForcesHaveOneElementPerShiftVector)
{
    ThreadForceBuffer<TypeParam> buffer(0, 1);

    EXPECT_EQ(buffer.shiftForces().ssize(), c_numShiftVectors);
    EXPECT_EQ(buffer.shiftForcesWithPadding().unpaddedArrayRef().ssize(), c_numShiftVectors);
    EXPECT_EQ(buffer.shiftForcesWithPadding().unpaddedArrayRef().data(), buffer.shiftForces().data());
}

// The free-energy kernel accumulates shift forces with transposeScatterIncrU<3>(), which may
// do a 4-wide load and store starting at any shift vector, including the last one. As for the
// normal force buffer, at least one element of padding is therefore needed.
TYPED_TEST(ThreadForceBufferTest, ShiftForcesArePaddedForSimdScatter)
{
    ThreadForceBuffer<TypeParam> buffer(0, 1);

    auto shiftForces = buffer.shiftForcesWithPadding();
    EXPECT_GT(shiftForces.size(), shiftForces.unpaddedArrayRef().ssize())
            << "Need padding of at least one element for SIMD scatter operations";
}

TYPED_TEST(ThreadForceBufferTest, ClearZeroesShiftForcesIncludingPadding)
{
    ThreadForceBuffer<TypeParam> buffer(0, 1);

    for (RVec& v : buffer.shiftForcesWithPadding().paddedArrayRef())
    {
        v = { 1.0_real, 2.0_real, 3.0_real };
    }
    buffer.clearForcesAndEnergies();
    for (const RVec& v : buffer.shiftForcesWithPadding().paddedArrayRef())
    {
        EXPECT_EQ(v[XX], 0.0_real);
        EXPECT_EQ(v[YY], 0.0_real);
        EXPECT_EQ(v[ZZ], 0.0_real);
    }
}

} // namespace
} // namespace test
} // namespace gmx
