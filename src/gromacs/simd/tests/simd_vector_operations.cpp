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

#include <ios>
#include <string>

#include <gtest/gtest.h>

#include "gromacs/simd/simd.h"
#include "gromacs/simd/vector_operations.h"

#include "data.h"
#include "simd.h"

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

#    if GMX_SIMD_HAVE_REAL

/*! \internal \brief Test fixture for vector operations tests (identical to the generic \ref SimdTest) */
typedef SimdTest SimdVectorOperationsTest;

TEST_F(SimdVectorOperationsTest, iprod)
{
    SimdReal aX       = rSimd_c0c1c2;
    SimdReal aY       = rSimd_c3c4c5;
    SimdReal aZ       = rSimd_c6c7c8;
    SimdReal bX       = rSimd_c3c0c4;
    SimdReal bY       = rSimd_c4c6c8;
    SimdReal bZ       = rSimd_c7c2c3;
    SimdReal iprodRef = setSimdRealFrom3R(
            c0 * c3 + c3 * c4 + c6 * c7, c1 * c0 + c4 * c6 + c7 * c2, c2 * c4 + c5 * c8 + c8 * c3);

    setUlpTol(2);
    GMX_EXPECT_SIMD_REAL_NEAR(iprodRef, iprod(aX, aY, aZ, bX, bY, bZ));
}

TEST_F(SimdVectorOperationsTest, norm2)
{
    SimdReal simdX    = rSimd_c0c1c2;
    SimdReal simdY    = rSimd_c3c4c5;
    SimdReal simdZ    = rSimd_c6c7c8;
    SimdReal norm2Ref = setSimdRealFrom3R(
            c0 * c0 + c3 * c3 + c6 * c6, c1 * c1 + c4 * c4 + c7 * c7, c2 * c2 + c5 * c5 + c8 * c8);

    setUlpTol(2);
    GMX_EXPECT_SIMD_REAL_NEAR(norm2Ref, norm2(simdX, simdY, simdZ));
}

TEST_F(SimdVectorOperationsTest, cprod)
{
    SimdReal aX = rSimd_c0c1c2;
    SimdReal aY = rSimd_c3c4c5;
    SimdReal aZ = rSimd_c6c7c8;
    SimdReal bX = rSimd_c3c0c4;
    SimdReal bY = rSimd_c4c6c8;
    SimdReal bZ = rSimd_c7c2c3;
    // The SIMD version might use FMA. If we don't force FMA for the reference value, the compiler is free to use FMA
    // for either product. If the compiler uses FMA for one product and the SIMD version uses FMA for the other, the
    // rounding error of each product adds up and the total possible ulp-error is 12.
    SimdReal refcX = setSimdRealFrom3R(
            std::fma(-c6, c4, c3 * c7), std::fma(-c7, c6, c4 * c2), std::fma(-c8, c8, c5 * c3));
    SimdReal refcY = setSimdRealFrom3R(
            std::fma(-c0, c7, c6 * c3), std::fma(-c1, c2, c7 * c0), std::fma(-c2, c3, c8 * c4));
    SimdReal refcZ = setSimdRealFrom3R(
            std::fma(-c3, c3, c0 * c4), std::fma(-c4, c0, c1 * c6), std::fma(-c5, c4, c2 * c8));
    SimdReal cX, cY, cZ;

    // The test assumes that cprod uses FMA on architectures which have FMA so that the compiler
    // can't choose which product is computed with FMA.
    cprod(aX, aY, aZ, bX, bY, bZ, &cX, &cY, &cZ);

    // The test values cannot be computed without FMA for the case that SIMD has no FMA. Even if no
    // explicit FMA were used, the compiler could choose to use FMA. This causes up to 6upl error
    // because of the product is up to 6 times larger than the final result after the difference.
    setUlpTol(GMX_SIMD_HAVE_FMA ? ulpTol_ : 6);

    GMX_EXPECT_SIMD_REAL_NEAR(refcX, cX);
    GMX_EXPECT_SIMD_REAL_NEAR(refcY, cY);
    GMX_EXPECT_SIMD_REAL_NEAR(refcZ, cZ);
}

#    endif // GMX_SIMD_HAVE_REAL

/* iprod(), norm2() and cprod() are defined as a fixed chain of fma()/fnma() operations.
 * With hardware FMA they must give bitwise-identical results with every compiler,
 * independent of its floating-point contraction setting (GCC defaults to
 * -ffp-contract=fast, Clang to -ffp-contract=on, and GROMACS sets neither). The inputs
 * below are exactly representable and were chosen so that every other order of
 * fusing and summing the products changes the result. Without hardware FMA, fma() is a
 * separately rounded multiply and add, and whether the compiler fuses those is outside
 * our control, so these tests only run with GMX_SIMD_HAVE_FMA.
 */

//! Check that all elements of \p v are bitwise equal to \p ref
template<typename T, typename SimdT, int width>
void expectAllElementsEqual(T ref, SimdT v)
{
    alignas(64) T buf[width];
    store(buf, v);
    for (int i = 0; i < width; i++)
    {
        EXPECT_EQ(ref, buf[i]) << "element " << i << ", ref " << std::hexfloat << ref << ", SIMD "
                               << buf[i] << std::defaultfloat;
    }
}

#    if GMX_SIMD_HAVE_FLOAT
TEST(SimdVectorOperationsFmaTest, iprodFloatMatchesFmaChain)
{
    if (!GMX_SIMD_HAVE_FMA)
    {
        GTEST_SKIP() << "No hardware FMA in this SIMD implementation";
    }
    const float          s    = 1.0F / 4096.0F; // 2^-12
    volatile const float in[] = { 1 + 2 * s,  -(1 - 19 * s), 1 + 21 * s,
                                  1 + 28 * s, 1 + 31 * s,    1 + 5 * s };
    const float          ax = in[0], ay = in[1], az = in[2], bx = in[3], by = in[4], bz = in[5];

    const float ref = std::fma(az, bz, std::fma(ay, by, ax * bx));
    expectAllElementsEqual<float, SimdFloat, GMX_SIMD_FLOAT_WIDTH>(
            ref,
            iprod(SimdFloat(ax), SimdFloat(ay), SimdFloat(az), SimdFloat(bx), SimdFloat(by), SimdFloat(bz)));
}

TEST(SimdVectorOperationsFmaTest, norm2FloatMatchesFmaChain)
{
    if (!GMX_SIMD_HAVE_FMA)
    {
        GTEST_SKIP() << "No hardware FMA in this SIMD implementation";
    }
    const float          s    = 1.0F / 4096.0F; // 2^-12
    volatile const float in[] = { 1 - 19 * s, 1 + 15 * s, 1 + 25 * s };
    const float          x = in[0], y = in[1], z = in[2];

    const float ref = std::fma(z, z, std::fma(y, y, x * x));
    expectAllElementsEqual<float, SimdFloat, GMX_SIMD_FLOAT_WIDTH>(
            ref, norm2(SimdFloat(x), SimdFloat(y), SimdFloat(z)));
}

TEST(SimdVectorOperationsFmaTest, cprodFloatMatchesFmaChain)
{
    if (!GMX_SIMD_HAVE_FMA)
    {
        GTEST_SKIP() << "No hardware FMA in this SIMD implementation";
    }
    const float          s    = 1.0F / 4096.0F; // 2^-12
    volatile const float in[] = { 1 + 3 * s, 1 - 28 * s, 1 + 26 * s,
                                  1 - 9 * s, 1 + 15 * s, 1 + 6 * s };
    const float          ax = in[0], ay = in[1], az = in[2], bx = in[3], by = in[4], bz = in[5];
    SimdFloat            cx, cy, cz;

    cprod(SimdFloat(ax), SimdFloat(ay), SimdFloat(az), SimdFloat(bx), SimdFloat(by), SimdFloat(bz), &cx, &cy, &cz);
    expectAllElementsEqual<float, SimdFloat, GMX_SIMD_FLOAT_WIDTH>(std::fma(-az, by, ay * bz), cx);
    expectAllElementsEqual<float, SimdFloat, GMX_SIMD_FLOAT_WIDTH>(std::fma(-ax, bz, az * bx), cy);
    expectAllElementsEqual<float, SimdFloat, GMX_SIMD_FLOAT_WIDTH>(std::fma(-ay, bx, ax * by), cz);
}
#    endif // GMX_SIMD_HAVE_FLOAT

#    if GMX_SIMD_HAVE_DOUBLE
TEST(SimdVectorOperationsFmaTest, iprodDoubleMatchesFmaChain)
{
    if (!GMX_SIMD_HAVE_FMA)
    {
        GTEST_SKIP() << "No hardware FMA in this SIMD implementation";
    }
    const double s = 1.0 / 134217728.0; // 2^-27
    volatile const double in[] = { 1 + 8 * s, -(1 + 6 * s), 1 + 14 * s, 1 + s, 1 + 19 * s, 1 - 7 * s };
    const double ax = in[0], ay = in[1], az = in[2], bx = in[3], by = in[4], bz = in[5];

    const double ref = std::fma(az, bz, std::fma(ay, by, ax * bx));
    expectAllElementsEqual<double, SimdDouble, GMX_SIMD_DOUBLE_WIDTH>(
            ref,
            iprod(SimdDouble(ax), SimdDouble(ay), SimdDouble(az), SimdDouble(bx), SimdDouble(by), SimdDouble(bz)));
}

TEST(SimdVectorOperationsFmaTest, norm2DoubleMatchesFmaChain)
{
    if (!GMX_SIMD_HAVE_FMA)
    {
        GTEST_SKIP() << "No hardware FMA in this SIMD implementation";
    }
    const double          s    = 1.0 / 268435456.0; // 2^-28
    volatile const double in[] = { 1 + 64 * s, 1 - 67 * s, 1 - 63 * s };
    const double          x = in[0], y = in[1], z = in[2];

    const double ref = std::fma(z, z, std::fma(y, y, x * x));
    expectAllElementsEqual<double, SimdDouble, GMX_SIMD_DOUBLE_WIDTH>(
            ref, norm2(SimdDouble(x), SimdDouble(y), SimdDouble(z)));
}

TEST(SimdVectorOperationsFmaTest, cprodDoubleMatchesFmaChain)
{
    if (!GMX_SIMD_HAVE_FMA)
    {
        GTEST_SKIP() << "No hardware FMA in this SIMD implementation";
    }
    const double          s    = 1.0 / 134217728.0; // 2^-27
    volatile const double in[] = { 1 + 3 * s, 1 - 28 * s, 1 + 26 * s,
                                   1 - 9 * s, 1 + 15 * s, 1 + 6 * s };
    const double          ax = in[0], ay = in[1], az = in[2], bx = in[3], by = in[4], bz = in[5];
    SimdDouble            cx, cy, cz;

    cprod(SimdDouble(ax), SimdDouble(ay), SimdDouble(az), SimdDouble(bx), SimdDouble(by), SimdDouble(bz), &cx, &cy, &cz);
    expectAllElementsEqual<double, SimdDouble, GMX_SIMD_DOUBLE_WIDTH>(std::fma(-az, by, ay * bz), cx);
    expectAllElementsEqual<double, SimdDouble, GMX_SIMD_DOUBLE_WIDTH>(std::fma(-ax, bz, az * bx), cy);
    expectAllElementsEqual<double, SimdDouble, GMX_SIMD_DOUBLE_WIDTH>(std::fma(-ay, bx, ax * by), cz);
}
#    endif // GMX_SIMD_HAVE_DOUBLE

/*! \} */
/*! \endcond */

} // namespace
} // namespace test
} // namespace gmx

#endif // GMX_SIMD
