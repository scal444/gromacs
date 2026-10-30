/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2014- The GROMACS Authors
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
#include "gmxpre.h"

#include <cmath>

#include <string>

#include <gtest/gtest.h>

#include "gromacs/simd/simd.h"
#include "gromacs/simd/vector_operations.h"

#include "data.h"
#include "simd4.h"

#if GMX_SIMD

namespace gmx
{
namespace test
{
namespace
{

/*! \cond internal */
/*! \addtogroup module_simd */
/*! \{ */

#    if GMX_SIMD4_HAVE_REAL

/*! \brief Test fixture for SIMD4 vector operations (identical to the SIMD4 \ref Simd4Test) */
typedef Simd4Test Simd4VectorOperationsTest;

TEST_F(Simd4VectorOperationsTest, norm2)
{
    Simd4Real simdX  = rSimd4_c0c1c2;
    Simd4Real simdY  = rSimd4_c3c4c5;
    Simd4Real simdZ  = rSimd4_c6c7c8;
    Simd4Real simdR2 = setSimd4RealFrom3R(
            c0 * c0 + c3 * c3 + c6 * c6, c1 * c1 + c4 * c4 + c7 * c7, c2 * c2 + c5 * c5 + c8 * c8);

    setUlpTol(2);
    GMX_EXPECT_SIMD4_REAL_NEAR(simdR2, norm2(simdX, simdY, simdZ));
}

#    endif // GMX_SIMD4_HAVE_REAL

/* norm2() is defined as a fixed chain of fma() operations, so with hardware FMA it must
 * give bitwise-identical results with every compiler, independent of its floating-point
 * contraction setting. See the SimdVectorOperationsFmaTest tests for details; the inputs
 * are the same.
 */
#    if GMX_SIMD4_HAVE_FLOAT
TEST(Simd4VectorOperationsFmaTest, norm2FloatMatchesFmaChain)
{
    if (!GMX_SIMD_HAVE_FMA)
    {
        GTEST_SKIP() << "No hardware FMA in this SIMD implementation";
    }
    const float          s    = 1.0F / 4096.0F; // 2^-12
    volatile const float in[] = { 1 - 19 * s, 1 + 15 * s, 1 + 25 * s };
    const float          x = in[0], y = in[1], z = in[2];

    const float       ref = std::fma(z, z, std::fma(y, y, x * x));
    alignas(64) float buf[GMX_SIMD4_WIDTH];
    store4(buf, norm2(Simd4Float(x), Simd4Float(y), Simd4Float(z)));
    for (int i = 0; i < GMX_SIMD4_WIDTH; i++)
    {
        EXPECT_EQ(ref, buf[i]) << "element " << i;
    }
}
#    endif // GMX_SIMD4_HAVE_FLOAT

#    if GMX_SIMD4_HAVE_DOUBLE
TEST(Simd4VectorOperationsFmaTest, norm2DoubleMatchesFmaChain)
{
    if (!GMX_SIMD_HAVE_FMA)
    {
        GTEST_SKIP() << "No hardware FMA in this SIMD implementation";
    }
    const double          s    = 1.0 / 268435456.0; // 2^-28
    volatile const double in[] = { 1 + 64 * s, 1 - 67 * s, 1 - 63 * s };
    const double          x = in[0], y = in[1], z = in[2];

    const double       ref = std::fma(z, z, std::fma(y, y, x * x));
    alignas(64) double buf[GMX_SIMD4_WIDTH];
    store4(buf, norm2(Simd4Double(x), Simd4Double(y), Simd4Double(z)));
    for (int i = 0; i < GMX_SIMD4_WIDTH; i++)
    {
        EXPECT_EQ(ref, buf[i]) << "element " << i;
    }
}
#    endif // GMX_SIMD4_HAVE_DOUBLE

/*! \} */
/*! \endcond */

} // namespace
} // namespace test
} // namespace gmx

#endif // GMX_SIMD
