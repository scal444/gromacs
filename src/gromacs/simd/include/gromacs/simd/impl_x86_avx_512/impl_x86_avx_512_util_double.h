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

#ifndef GMX_SIMD_IMPL_X86_AVX_512_UTIL_DOUBLE_H
#define GMX_SIMD_IMPL_X86_AVX_512_UTIL_DOUBLE_H

#include "config.h"

#include <immintrin.h>

#include <cassert>
#include <cstdint>

#include "gromacs/simd/impl_x86_avx_512/impl_x86_avx_512_definitions.h"
#include "gromacs/simd/impl_x86_avx_512/impl_x86_avx_512_general.h"
#include "gromacs/simd/impl_x86_avx_512/impl_x86_avx_512_simd_double.h"
#include "gromacs/utility/basedefinitions.h"

namespace gmx
{

static const int c_simdBestPairAlignmentDouble = 2;

namespace
{
// Multiply function optimized for powers of 2, for which it is done by
// shifting. Currently up to 8 is accelerated. Could be accelerated for any
// number with a constexpr log2 function.
template<int n>
static inline SimdDInt32 fastMultiply(SimdDInt32 x)
{
    if (n == 2)
    {
        return _mm256_slli_epi32(x.simdInternal_, 1);
    }
    else if (n == 4)
    {
        return _mm256_slli_epi32(x.simdInternal_, 2);
    }
    else if (n == 8)
    {
        return _mm256_slli_epi32(x.simdInternal_, 3);
    }
    else
    {
        return x * n;
    }
}

template<int align>
static inline void gmx_simdcall avx512HwGatherTranspose(const double*, SimdDInt32)
{
    // Nothing to do. Termination of recursion.
}

// Hardware-gather implementation: one 8-element gather per output.
template<int align, typename... Targs>
static inline void gmx_simdcall
avx512HwGatherTranspose(const double* base, SimdDInt32 offset, SimdDouble* v, Targs... Fargs)
{
    if (align > 1)
    {
        offset = fastMultiply<align>(offset);
    }
    constexpr size_t scale = sizeof(double);
    v->simdInternal_       = _mm512_i32gather_pd(offset.simdInternal_, base, scale);
    avx512HwGatherTranspose<1>(base + 1, offset, Fargs...);
}

#if !GMX_SIMD_X86_AVX_512_USE_HW_GATHER
/* Load-and-permute replacements for the hardware gathers, see the float version.
 * With three outputs four doubles are loaded per point, which is only done when the
 * fourth element lies within the same record (align >= 4) or for the "U" functions.
 */
template<int nOut, int align, bool unalignedVariant>
static constexpr bool c_avx512UsePermuteTransposeDouble =
        (nOut == 2 || nOut == 4 || (nOut == 3 && (align >= 4 || unalignedVariant)));

// The pairs (2 doubles) at base + align*o[0..3], in order, as x0 y0 x1 y1 x2 y2 x3 y3.
template<int align>
static inline __m512d gmx_simdcall avx512LoadPairs4(const double* base, const std::int32_t* o)
{
    __m512 r = _mm512_castps128_ps512(_mm_castpd_ps(_mm_loadu_pd(base + align * o[0])));
    r        = _mm512_insertf32x4(r, _mm_castpd_ps(_mm_loadu_pd(base + align * o[1])), 1);
    r        = _mm512_insertf32x4(r, _mm_castpd_ps(_mm_loadu_pd(base + align * o[2])), 2);
    r        = _mm512_insertf32x4(r, _mm_castpd_ps(_mm_loadu_pd(base + align * o[3])), 3);
    return _mm512_castps_pd(r);
}

static inline void gmx_simdcall avx512DeinterleavePairs(__m512d a, __m512d b, __m512d* v0, __m512d* v1)
{
    *v0 = _mm512_permutex2var_pd(a, _mm512_setr_epi64(0, 2, 4, 6, 8, 10, 12, 14), b);
    *v1 = _mm512_permutex2var_pd(a, _mm512_setr_epi64(1, 3, 5, 7, 9, 11, 13, 15), b);
}

template<int align>
static inline void gmx_simdcall
avx512LoadTranspose(const double* base, const std::int32_t* o, SimdDouble* v0, SimdDouble* v1)
{
    avx512DeinterleavePairs(avx512LoadPairs4<align>(base, o),
                            avx512LoadPairs4<align>(base, o + 4),
                            &v0->simdInternal_,
                            &v1->simdInternal_);
}

/* Loads 4 doubles for each of the 8 points into r[j] = { point j | point j+2 } for j = 0,1
 * and r[j] = { point j+2 | point j+4 } for j = 2,3, i.e. r0 = {p0|p2}, r1 = {p1|p3},
 * r2 = {p4|p6}, r3 = {p5|p7}. After unpacking within 128-bit lanes, a single
 * vshuff64x2 per output puts the 128-bit pieces in point order.
 */
template<int align, int nOut>
static inline void gmx_simdcall avx512LoadTransposeQuads(const double* base, const std::int32_t* o, __m512d* v)
{
    auto load2 = [&](int a, int b) {
        return _mm512_insertf64x4(
                _mm512_castpd256_pd512(_mm256_loadu_pd(base + align * o[a])), _mm256_loadu_pd(base + align * o[b]), 1);
    };
    const __m512d r0 = load2(0, 2);
    const __m512d r1 = load2(1, 3);
    const __m512d r2 = load2(4, 6);
    const __m512d r3 = load2(5, 7);
    const __m512d t0 = _mm512_unpacklo_pd(r0, r1); // x0 x1 z0 z1 | x2 x3 z2 z3
    const __m512d t1 = _mm512_unpackhi_pd(r0, r1); // y0 y1 w0 w1 | y2 y3 w2 w3
    const __m512d t2 = _mm512_unpacklo_pd(r2, r3); // x4 x5 z4 z5 | x6 x7 z6 z7
    const __m512d t3 = _mm512_unpackhi_pd(r2, r3); // y4 y5 w4 w5 | y6 y7 w6 w7
    v[0]             = _mm512_shuffle_f64x2(t0, t2, _MM_SHUFFLE(2, 0, 2, 0));
    v[1]             = _mm512_shuffle_f64x2(t1, t3, _MM_SHUFFLE(2, 0, 2, 0));
    v[2]             = _mm512_shuffle_f64x2(t0, t2, _MM_SHUFFLE(3, 1, 3, 1));
    if constexpr (nOut == 4)
    {
        v[3] = _mm512_shuffle_f64x2(t1, t3, _MM_SHUFFLE(3, 1, 3, 1));
    }
}

template<int align>
static inline void gmx_simdcall
avx512LoadTranspose(const double* base, const std::int32_t* o, SimdDouble* v0, SimdDouble* v1, SimdDouble* v2)
{
    __m512d v[3];
    avx512LoadTransposeQuads<align, 3>(base, o, v);
    v0->simdInternal_ = v[0];
    v1->simdInternal_ = v[1];
    v2->simdInternal_ = v[2];
}

template<int align>
static inline void gmx_simdcall avx512LoadTranspose(const double*       base,
                                                    const std::int32_t* o,
                                                    SimdDouble*         v0,
                                                    SimdDouble*         v1,
                                                    SimdDouble*         v2,
                                                    SimdDouble*         v3)
{
    __m512d v[4];
    avx512LoadTransposeQuads<align, 4>(base, o, v);
    v0->simdInternal_ = v[0];
    v1->simdInternal_ = v[1];
    v2->simdInternal_ = v[2];
    v3->simdInternal_ = v[3];
}

// Store the first 3 doubles of 256-bit half 'half' of a to p, leaving p[3] untouched.
template<int half>
static inline void gmx_simdcall avx512StoreHalf3(double* p, __m512d a)
{
#    if defined(__AVX512VL__)
    _mm256_mask_storeu_pd(p, 0x7, half == 0 ? _mm512_castpd512_pd256(a) : _mm512_extractf64x4_pd(a, 1));
#    else
    _mm512_mask_storeu_pd(p, 0x7, half == 0 ? a : _mm512_castpd256_pd512(_mm512_extractf64x4_pd(a, 1)));
#    endif
}
#endif // !GMX_SIMD_X86_AVX_512_USE_HW_GATHER

/* This is an internal helper function used by decr3Hsimd(...).
 */
inline void gmx_simdcall decrHsimd(double* m, SimdDouble a)
{
    __m256d t;

    assert(std::size_t(m) % 32 == 0);

    a.simdInternal_ = _mm512_add_pd(a.simdInternal_,
                                    _mm512_shuffle_f64x2(a.simdInternal_, a.simdInternal_, 0xEE));
    t               = _mm256_load_pd(m);
    t               = _mm256_sub_pd(t, _mm512_castpd512_pd256(a.simdInternal_));
    _mm256_store_pd(m, t);
}
} // namespace


namespace
{
template<int align, bool unalignedVariant, typename... Targs>
static inline void gmx_simdcall
avx512GatherLoadBySimdIntTranspose(const double* base, SimdDInt32 offset, SimdDouble* v, Targs... Fargs)
{
#if !GMX_SIMD_X86_AVX_512_USE_HW_GATHER
    if constexpr (c_avx512UsePermuteTransposeDouble<1 + sizeof...(Targs), align, unalignedVariant>)
    {
        alignas(GMX_SIMD_ALIGNMENT) std::int32_t o[GMX_SIMD_DINT32_WIDTH];
        _mm256_store_si256(reinterpret_cast<__m256i*>(o), offset.simdInternal_);
        avx512LoadTranspose<align>(base, o, v, Fargs...);
    }
    else
#endif
    {
        avx512HwGatherTranspose<align>(base, offset, v, Fargs...);
    }
}

template<int align, bool unalignedVariant, typename... Targs>
static inline void gmx_simdcall
avx512GatherLoadTranspose(const double* base, const std::int32_t offset[], SimdDouble* v, Targs... Fargs)
{
#if !GMX_SIMD_X86_AVX_512_USE_HW_GATHER
    if constexpr (c_avx512UsePermuteTransposeDouble<1 + sizeof...(Targs), align, unalignedVariant>)
    {
        avx512LoadTranspose<align>(base, offset, v, Fargs...);
    }
    else
#endif
    {
        avx512HwGatherTranspose<align>(base, simdLoad(offset, SimdDInt32Tag()), v, Fargs...);
    }
}
} // namespace

template<int align, typename... Targs>
static inline void gmx_simdcall
gatherLoadBySimdIntTranspose(const double* base, SimdDInt32 offset, SimdDouble* v, Targs... Fargs)
{
    avx512GatherLoadBySimdIntTranspose<align, false>(base, offset, v, Fargs...);
}

template<int align, typename... Targs>
static inline void gmx_simdcall
gatherLoadUBySimdIntTranspose(const double* base, SimdDInt32 offset, SimdDouble* v, Targs... Fargs)
{
    avx512GatherLoadBySimdIntTranspose<align, true>(base, offset, v, Fargs...);
}

template<int align, typename... Targs>
static inline void gmx_simdcall
gatherLoadTranspose(const double* base, const std::int32_t offset[], SimdDouble* v, Targs... Fargs)
{
    avx512GatherLoadTranspose<align, false>(base, offset, v, Fargs...);
}

template<int align, typename... Targs>
static inline void gmx_simdcall
gatherLoadUTranspose(const double* base, const std::int32_t offset[], SimdDouble* v, Targs... Fargs)
{
    avx512GatherLoadTranspose<align, true>(base, offset, v, Fargs...);
}

template<int align>
static inline void gmx_simdcall transposeScatterStoreU(double*            base,
                                                       const std::int32_t offset[],
                                                       SimdDouble         v0,
                                                       SimdDouble         v1,
                                                       SimdDouble         v2)
{
#if GMX_SIMD_X86_AVX_512_USE_HW_GATHER
    SimdDInt32 simdoffset = simdLoad(offset, SimdDInt32Tag());

    if (align > 1)
    {
        simdoffset = fastMultiply<align>(simdoffset);
        ;
    }
    constexpr size_t scale = sizeof(double);
    _mm512_i32scatter_pd(base, simdoffset.simdInternal_, v0.simdInternal_, scale);
    _mm512_i32scatter_pd(&(base[1]), simdoffset.simdInternal_, v1.simdInternal_, scale);
    _mm512_i32scatter_pd(&(base[2]), simdoffset.simdInternal_, v2.simdInternal_, scale);
#else
    // q[j] = { x_j y_j z_j z_j | x_{j+4} y_{j+4} z_{j+4} z_{j+4} }, then masked 3-element stores
    // in index order (duplicate offsets: last one wins, as with the scatter).
    const __m512d t0 = _mm512_unpacklo_pd(v0.simdInternal_, v1.simdInternal_); // x0 y0 x2 y2 x4 y4 x6 y6
    const __m512d t1 = _mm512_unpackhi_pd(v0.simdInternal_, v1.simdInternal_); // x1 y1 x3 y3 x5 y5 x7 y7
    const __m512d q0 =
            _mm512_permutex2var_pd(t0, _mm512_setr_epi64(0, 1, 8, 8, 4, 5, 12, 12), v2.simdInternal_);
    const __m512d q1 =
            _mm512_permutex2var_pd(t1, _mm512_setr_epi64(0, 1, 9, 9, 4, 5, 13, 13), v2.simdInternal_);
    const __m512d q2 =
            _mm512_permutex2var_pd(t0, _mm512_setr_epi64(2, 3, 10, 10, 6, 7, 14, 14), v2.simdInternal_);
    const __m512d q3 =
            _mm512_permutex2var_pd(t1, _mm512_setr_epi64(2, 3, 11, 11, 6, 7, 15, 15), v2.simdInternal_);
    avx512StoreHalf3<0>(base + align * offset[0], q0);
    avx512StoreHalf3<0>(base + align * offset[1], q1);
    avx512StoreHalf3<0>(base + align * offset[2], q2);
    avx512StoreHalf3<0>(base + align * offset[3], q3);
    avx512StoreHalf3<1>(base + align * offset[4], q0);
    avx512StoreHalf3<1>(base + align * offset[5], q1);
    avx512StoreHalf3<1>(base + align * offset[6], q2);
    avx512StoreHalf3<1>(base + align * offset[7], q3);
#endif
}

template<int align>
static inline void gmx_simdcall
transposeScatterIncrU(double* base, const std::int32_t offset[], SimdDouble v0, SimdDouble v1, SimdDouble v2)
{
    __m512d                                  t[4], t5, t6, t7, t8;
    alignas(GMX_SIMD_ALIGNMENT) std::int64_t o[8];
    // TODO: should use fastMultiply
    _mm512_store_epi64(o,
                       _mm512_cvtepi32_epi64(_mm256_mullo_epi32(
                               _mm256_load_si256(reinterpret_cast<const __m256i*>(offset)),
                               _mm256_set1_epi32(align))));
    t5   = _mm512_unpacklo_pd(v0.simdInternal_, v1.simdInternal_);
    t6   = _mm512_unpackhi_pd(v0.simdInternal_, v1.simdInternal_);
    t7   = _mm512_unpacklo_pd(v2.simdInternal_, _mm512_setzero_pd());
    t8   = _mm512_unpackhi_pd(v2.simdInternal_, _mm512_setzero_pd());
    t[0] = _mm512_mask_permutex_pd(t5, avx512Int2Mask(0xCC), t7, 0x4E);
    t[1] = _mm512_mask_permutex_pd(t6, avx512Int2Mask(0xCC), t8, 0x4E);
    t[2] = _mm512_mask_permutex_pd(t7, avx512Int2Mask(0x33), t5, 0x4E);
    t[3] = _mm512_mask_permutex_pd(t8, avx512Int2Mask(0x33), t6, 0x4E);
    if (align < 4)
    {
        for (int i = 0; i < 4; i++)
        {
            _mm512_mask_storeu_pd(base + o[0 + i],
                                  avx512Int2Mask(7),
                                  _mm512_castpd256_pd512(_mm256_add_pd(_mm256_loadu_pd(base + o[0 + i]),
                                                                       _mm512_castpd512_pd256(t[i]))));
            _mm512_mask_storeu_pd(
                    base + o[4 + i],
                    avx512Int2Mask(7),
                    _mm512_castpd256_pd512(_mm256_add_pd(_mm256_loadu_pd(base + o[4 + i]),
                                                         _mm512_extractf64x4_pd(t[i], 1))));
        }
    }
    else
    {
        if (align % 4 == 0)
        {
            for (int i = 0; i < 4; i++)
            {
                _mm256_store_pd(
                        base + o[0 + i],
                        _mm256_add_pd(_mm256_load_pd(base + o[0 + i]), _mm512_castpd512_pd256(t[i])));
                _mm256_store_pd(base + o[4 + i],
                                _mm256_add_pd(_mm256_load_pd(base + o[4 + i]),
                                              _mm512_extractf64x4_pd(t[i], 1)));
            }
        }
        else
        {
            for (int i = 0; i < 4; i++)
            {
                _mm256_storeu_pd(
                        base + o[0 + i],
                        _mm256_add_pd(_mm256_loadu_pd(base + o[0 + i]), _mm512_castpd512_pd256(t[i])));
                _mm256_storeu_pd(base + o[4 + i],
                                 _mm256_add_pd(_mm256_loadu_pd(base + o[4 + i]),
                                               _mm512_extractf64x4_pd(t[i], 1)));
            }
        }
    }
}

template<int align>
static inline void gmx_simdcall
transposeScatterDecrU(double* base, const std::int32_t offset[], SimdDouble v0, SimdDouble v1, SimdDouble v2)
{
    __m512d                                  t[4], t5, t6, t7, t8;
    alignas(GMX_SIMD_ALIGNMENT) std::int64_t o[8];
    // TODO: should use fastMultiply
    _mm512_store_epi64(o,
                       _mm512_cvtepi32_epi64(_mm256_mullo_epi32(
                               _mm256_load_si256(reinterpret_cast<const __m256i*>(offset)),
                               _mm256_set1_epi32(align))));
    t5   = _mm512_unpacklo_pd(v0.simdInternal_, v1.simdInternal_);
    t6   = _mm512_unpackhi_pd(v0.simdInternal_, v1.simdInternal_);
    t7   = _mm512_unpacklo_pd(v2.simdInternal_, _mm512_setzero_pd());
    t8   = _mm512_unpackhi_pd(v2.simdInternal_, _mm512_setzero_pd());
    t[0] = _mm512_mask_permutex_pd(t5, avx512Int2Mask(0xCC), t7, 0x4E);
    t[2] = _mm512_mask_permutex_pd(t7, avx512Int2Mask(0x33), t5, 0x4E);
    t[1] = _mm512_mask_permutex_pd(t6, avx512Int2Mask(0xCC), t8, 0x4E);
    t[3] = _mm512_mask_permutex_pd(t8, avx512Int2Mask(0x33), t6, 0x4E);
    if (align < 4)
    {
        for (int i = 0; i < 4; i++)
        {
            _mm512_mask_storeu_pd(base + o[0 + i],
                                  avx512Int2Mask(7),
                                  _mm512_castpd256_pd512(_mm256_sub_pd(_mm256_loadu_pd(base + o[0 + i]),
                                                                       _mm512_castpd512_pd256(t[i]))));
            _mm512_mask_storeu_pd(
                    base + o[4 + i],
                    avx512Int2Mask(7),
                    _mm512_castpd256_pd512(_mm256_sub_pd(_mm256_loadu_pd(base + o[4 + i]),
                                                         _mm512_extractf64x4_pd(t[i], 1))));
        }
    }
    else
    {
        if (align % 4 == 0)
        {
            for (int i = 0; i < 4; i++)
            {
                _mm256_store_pd(
                        base + o[0 + i],
                        _mm256_sub_pd(_mm256_load_pd(base + o[0 + i]), _mm512_castpd512_pd256(t[i])));
                _mm256_store_pd(base + o[4 + i],
                                _mm256_sub_pd(_mm256_load_pd(base + o[4 + i]),
                                              _mm512_extractf64x4_pd(t[i], 1)));
            }
        }
        else
        {
            for (int i = 0; i < 4; i++)
            {
                _mm256_storeu_pd(
                        base + o[0 + i],
                        _mm256_sub_pd(_mm256_loadu_pd(base + o[0 + i]), _mm512_castpd512_pd256(t[i])));
                _mm256_storeu_pd(base + o[4 + i],
                                 _mm256_sub_pd(_mm256_loadu_pd(base + o[4 + i]),
                                               _mm512_extractf64x4_pd(t[i], 1)));
            }
        }
    }
}

static inline void gmx_simdcall expandScalarsToTriplets(SimdDouble  scalar,
                                                        SimdDouble* triplets0,
                                                        SimdDouble* triplets1,
                                                        SimdDouble* triplets2)
{
    triplets0->simdInternal_ = _mm512_castsi512_pd(_mm512_permutexvar_epi32(
            _mm512_set_epi32(5, 4, 5, 4, 3, 2, 3, 2, 3, 2, 1, 0, 1, 0, 1, 0),
            _mm512_castpd_si512(scalar.simdInternal_)));
    triplets1->simdInternal_ = _mm512_castsi512_pd(_mm512_permutexvar_epi32(
            _mm512_set_epi32(11, 10, 9, 8, 9, 8, 9, 8, 7, 6, 7, 6, 7, 6, 5, 4),
            _mm512_castpd_si512(scalar.simdInternal_)));
    triplets2->simdInternal_ = _mm512_castsi512_pd(_mm512_permutexvar_epi32(
            _mm512_set_epi32(15, 14, 15, 14, 15, 14, 13, 12, 13, 12, 13, 12, 11, 10, 11, 10),
            _mm512_castpd_si512(scalar.simdInternal_)));
}


static inline double gmx_simdcall
reduceIncr4ReturnSum(double* m, SimdDouble v0, SimdDouble v1, SimdDouble v2, SimdDouble v3)
{
    __m512d t0, t2;
    __m256d t3, t4;

    assert(std::size_t(m) % 32 == 0);

    t0 = _mm512_add_pd(v0.simdInternal_, _mm512_permute_pd(v0.simdInternal_, 0x55));
    t2 = _mm512_add_pd(v2.simdInternal_, _mm512_permute_pd(v2.simdInternal_, 0x55));
    t0 = _mm512_mask_add_pd(
            t0, avx512Int2Mask(0xAA), v1.simdInternal_, _mm512_permute_pd(v1.simdInternal_, 0x55));
    t2 = _mm512_mask_add_pd(
            t2, avx512Int2Mask(0xAA), v3.simdInternal_, _mm512_permute_pd(v3.simdInternal_, 0x55));
    t0 = _mm512_add_pd(t0, _mm512_shuffle_f64x2(t0, t0, 0x4E));
    t0 = _mm512_mask_add_pd(t0, avx512Int2Mask(0xF0), t2, _mm512_shuffle_f64x2(t2, t2, 0x4E));
    t0 = _mm512_add_pd(t0, _mm512_shuffle_f64x2(t0, t0, 0xB1));
    t0 = _mm512_mask_shuffle_f64x2(t0, avx512Int2Mask(0x0C), t0, t0, 0xEE);

    t3 = _mm512_castpd512_pd256(t0);
    t4 = _mm256_load_pd(m);
    t4 = _mm256_add_pd(t4, t3);
    _mm256_store_pd(m, t4);

    t0 = _mm512_add_pd(t0, _mm512_permutex_pd(t0, 0x4E));
    t0 = _mm512_add_pd(t0, _mm512_permutex_pd(t0, 0xB1));

    return _mm_cvtsd_f64(_mm512_castpd512_pd128(t0));
}

static inline SimdDouble gmx_simdcall loadDualHsimd(const double* m0, const double* m1)
{
    assert(std::size_t(m0) % 32 == 0);
    assert(std::size_t(m1) % 32 == 0);

    return { _mm512_insertf64x4(_mm512_castpd256_pd512(_mm256_load_pd(m0)), _mm256_load_pd(m1), 1) };
}

static inline SimdDouble gmx_simdcall loadDuplicateHsimd(const double* m)
{
    assert(std::size_t(m) % 32 == 0);

    return { _mm512_broadcast_f64x4(_mm256_load_pd(m)) };
}

static inline SimdDouble gmx_simdcall loadU1DualHsimd(const double* m)
{
    return { _mm512_insertf64x4(
            _mm512_broadcastsd_pd(_mm_load_sd(m)), _mm256_broadcastsd_pd(_mm_load_sd(m + 1)), 1) };
}


static inline void gmx_simdcall storeDualHsimd(double* m0, double* m1, SimdDouble a)
{
    assert(std::size_t(m0) % 32 == 0);
    assert(std::size_t(m1) % 32 == 0);

    _mm256_store_pd(m0, _mm512_castpd512_pd256(a.simdInternal_));
    _mm256_store_pd(m1, _mm512_extractf64x4_pd(a.simdInternal_, 1));
}

static inline void gmx_simdcall incrDualHsimd(double* m0, double* m1, SimdDouble a)
{
    assert(std::size_t(m0) % 32 == 0);
    assert(std::size_t(m1) % 32 == 0);

    __m256d x;

    // Lower half
    x = _mm256_load_pd(m0);
    x = _mm256_add_pd(x, _mm512_castpd512_pd256(a.simdInternal_));
    _mm256_store_pd(m0, x);

    // Upper half
    x = _mm256_load_pd(m1);
    x = _mm256_add_pd(x, _mm512_extractf64x4_pd(a.simdInternal_, 1));
    _mm256_store_pd(m1, x);
}

static inline void gmx_simdcall decr3Hsimd(double* m, SimdDouble a0, SimdDouble a1, SimdDouble a2)
{
    decrHsimd(m, a0);
    decrHsimd(m + GMX_SIMD_DOUBLE_WIDTH / 2, a1);
    decrHsimd(m + GMX_SIMD_DOUBLE_WIDTH, a2);
}

template<int align>
static inline void gmx_simdcall gatherLoadTransposeHsimd(const double*      base0,
                                                         const double*      base1,
                                                         const std::int32_t offset[],
                                                         SimdDouble*        v0,
                                                         SimdDouble*        v1)
{
    assert(std::size_t(offset) % 16 == 0);
    assert(std::size_t(base0) % 16 == 0);
    assert(std::size_t(base1) % 16 == 0);

    static_assert(align == 2 || align == 4, "If more are needed use fastMultiply");

#if !GMX_SIMD_X86_AVX_512_USE_HW_GATHER
    // 4 pairs from each base with 128-bit loads, then one two-source permute per output:
    // v0 = { base0[align*offset[0..3]], base1[align*offset[0..3]] }, v1 likewise for +1.
    avx512DeinterleavePairs(avx512LoadPairs4<align>(base0, offset),
                            avx512LoadPairs4<align>(base1, offset),
                            &v0->simdInternal_,
                            &v1->simdInternal_);
#else
    __m128i idx0, idx1;
    __m256i idx;
    __m512d tmp1, tmp2;

    idx0 = _mm_load_si128(reinterpret_cast<const __m128i*>(offset));

    idx0 = _mm_slli_epi32(idx0, align == 2 ? 1 : 2);

    idx1 = _mm_add_epi32(idx0, _mm_set1_epi32(1));

    idx = _mm256_inserti128_si256(_mm256_castsi128_si256(idx0), idx1, 1);

    constexpr size_t scale = sizeof(double);
    tmp1 = _mm512_i32gather_pd(idx, base0, scale); // TODO: Might be faster to use invidual loads
    tmp2 = _mm512_i32gather_pd(idx, base1, scale);

    v0->simdInternal_ = _mm512_shuffle_f64x2(tmp1, tmp2, 0x44);
    v1->simdInternal_ = _mm512_shuffle_f64x2(tmp1, tmp2, 0xEE);
#endif
}

static inline double gmx_simdcall reduceIncr4ReturnSumHsimd(double* m, SimdDouble v0, SimdDouble v1)
{
    __m512d t0;
    __m256d t2, t3;

    assert(std::size_t(m) % 32 == 0);

    t0 = _mm512_add_pd(v0.simdInternal_, _mm512_permutex_pd(v0.simdInternal_, 0x4E));
    t0 = _mm512_mask_add_pd(
            t0, avx512Int2Mask(0xCC), v1.simdInternal_, _mm512_permutex_pd(v1.simdInternal_, 0x4E));
    t0 = _mm512_add_pd(t0, _mm512_permutex_pd(t0, 0xB1));
    t0 = _mm512_mask_shuffle_f64x2(t0, avx512Int2Mask(0xAA), t0, t0, 0xEE);

    t2 = _mm512_castpd512_pd256(t0);
    t3 = _mm256_load_pd(m);
    t3 = _mm256_add_pd(t3, t2);
    _mm256_store_pd(m, t3);

    t0 = _mm512_add_pd(t0, _mm512_permutex_pd(t0, 0x4E));
    t0 = _mm512_add_pd(t0, _mm512_permutex_pd(t0, 0xB1));

    return _mm_cvtsd_f64(_mm512_castpd512_pd128(t0));
}

static inline SimdDouble gmx_simdcall loadU4NOffset(const double* m, int offset)
{
    return { _mm512_insertf64x4(
            _mm512_castpd256_pd512(_mm256_loadu_pd(m)), _mm256_loadu_pd(m + offset), 1) };
}

} // namespace gmx

#endif // GMX_SIMD_IMPL_X86_AVX_512_UTIL_DOUBLE_H
