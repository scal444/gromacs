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

#ifndef GMX_SIMD_IMPL_X86_AVX_512_UTIL_FLOAT_H
#define GMX_SIMD_IMPL_X86_AVX_512_UTIL_FLOAT_H

#include "config.h"

#include <immintrin.h>

#include <cassert>
#include <cstdint>

#include "gromacs/simd/impl_x86_avx_512/impl_x86_avx_512_definitions.h"
#include "gromacs/simd/impl_x86_avx_512/impl_x86_avx_512_general.h"
#include "gromacs/simd/impl_x86_avx_512/impl_x86_avx_512_simd_float.h"
#include "gromacs/utility/basedefinitions.h"

namespace gmx
{

static const int c_simdBestPairAlignmentFloat = 2;

namespace
{
// Multiply function optimized for powers of 2, for which it is done by
// shifting. Currently up to 8 is accelerated. Could be accelerated for any
// number with a constexpr log2 function.
template<int n>
static inline SimdFInt32 fastMultiply(SimdFInt32 x)
{
    if (n == 2)
    {
        return _mm512_slli_epi32(x.simdInternal_, 1);
    }
    else if (n == 4)
    {
        return _mm512_slli_epi32(x.simdInternal_, 2);
    }
    else if (n == 8)
    {
        return _mm512_slli_epi32(x.simdInternal_, 3);
    }
    else
    {
        return x * n;
    }
}

template<int align>
static inline void gmx_simdcall avx512HwGatherTranspose(const float*, SimdFInt32)
{
    // Nothing to do. Termination of recursion.
}

// Hardware-gather implementation: one 16-element gather per output.
template<int align, typename... Targs>
static inline void gmx_simdcall
avx512HwGatherTranspose(const float* base, SimdFInt32 offset, SimdFloat* v, Targs... Fargs)
{
    // For align 1 or 2: No multiplication of offset is needed
    if (align > 2)
    {
        offset = fastMultiply<align>(offset);
    }
    // For align 2: Scale of 2*sizeof(float) is used (maximum supported scale)
    constexpr int align_ = (align > 2) ? 1 : align;
    v->simdInternal_     = _mm512_i32gather_ps(offset.simdInternal_, base, sizeof(float) * align_);
    // Gather remaining elements. Avoid extra multiplication (new align is 1 or 2).
    avx512HwGatherTranspose<align_>(base + 1, offset, Fargs...);
}

#if !GMX_SIMD_X86_AVX_512_USE_HW_GATHER
/* Load-and-permute replacements for the hardware gathers.
 *
 * The permute versions load exactly the elements a gather would, except for three
 * outputs, where four floats are loaded per point. That is only done when the fourth
 * element lies within the same record (align >= 4) or for the "U" functions, whose
 * documentation requires padding after the last point (like AVX_256 does).
 */
template<int nOut, int align, bool unalignedVariant>
static constexpr bool c_avx512UsePermuteTranspose =
        (nOut == 2 || nOut == 4 || (nOut == 3 && (align >= 4 || unalignedVariant)));

// 128-bit lane k of the result holds the 4 floats at p[k].
static inline __m512 gmx_simdcall avx512LoadQuads(const float* p0, const float* p1, const float* p2, const float* p3)
{
    __m512 t = _mm512_castps128_ps512(_mm_loadu_ps(p0));
    t        = _mm512_insertf32x4(t, _mm_loadu_ps(p1), 1);
    t        = _mm512_insertf32x4(t, _mm_loadu_ps(p2), 2);
    return _mm512_insertf32x4(t, _mm_loadu_ps(p3), 3);
}

// The pairs (2 floats) at base + align*o[0..7], in order, as x0 y0 x1 y1 ... x7 y7.
template<int align>
static inline __m512 gmx_simdcall avx512LoadPairs8(const float* base, const std::int32_t* o)
{
    __m128 t[4];
    for (int i = 0; i < 4; i++)
    {
        t[i] = _mm_loadl_pi(_mm_setzero_ps(), reinterpret_cast<const __m64*>(base + align * o[2 * i]));
        t[i] = _mm_loadh_pi(t[i], reinterpret_cast<const __m64*>(base + align * o[2 * i + 1]));
    }
    __m512 r = _mm512_castps128_ps512(t[0]);
    r        = _mm512_insertf32x4(r, t[1], 1);
    r        = _mm512_insertf32x4(r, t[2], 2);
    return _mm512_insertf32x4(r, t[3], 3);
}

// Deinterleave two zmm of 8 pairs each into the 16 first and 16 second elements.
static inline void gmx_simdcall avx512DeinterleavePairs(__m512 a, __m512 b, __m512* v0, __m512* v1)
{
    const __m512i evenIdx =
            _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
    const __m512i oddIdx =
            _mm512_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31);
    *v0 = _mm512_permutex2var_ps(a, evenIdx, b);
    *v1 = _mm512_permutex2var_ps(a, oddIdx, b);
}

template<int align>
static inline void gmx_simdcall avx512LoadTranspose(const float* base, const std::int32_t* o, SimdFloat* v0, SimdFloat* v1)
{
    avx512DeinterleavePairs(avx512LoadPairs8<align>(base, o),
                            avx512LoadPairs8<align>(base, o + 8),
                            &v0->simdInternal_,
                            &v1->simdInternal_);
}

/* Loads 4 floats for each of the 16 points and transposes within 128-bit lanes:
 * lane k of r[j] holds point 4k+j, so after the 4x4 in-lane transpose the outputs
 * are in point order. Writes nOut (3 or 4) outputs.
 */
template<int align, int nOut>
static inline void gmx_simdcall avx512LoadTransposeQuads(const float* base, const std::int32_t* o, __m512* v)
{
    __m512 r[4];
    for (int j = 0; j < 4; j++)
    {
        r[j] = avx512LoadQuads(base + align * o[j],
                               base + align * o[4 + j],
                               base + align * o[8 + j],
                               base + align * o[12 + j]);
    }
    const __m512 t0 = _mm512_unpacklo_ps(r[0], r[1]); // x0 x1 y0 y1
    const __m512 t1 = _mm512_unpacklo_ps(r[2], r[3]); // x2 x3 y2 y3
    const __m512 t2 = _mm512_unpackhi_ps(r[0], r[1]); // z0 z1 w0 w1
    const __m512 t3 = _mm512_unpackhi_ps(r[2], r[3]); // z2 z3 w2 w3
    v[0]            = _mm512_shuffle_ps(t0, t1, _MM_SHUFFLE(1, 0, 1, 0));
    v[1]            = _mm512_shuffle_ps(t0, t1, _MM_SHUFFLE(3, 2, 3, 2));
    v[2]            = _mm512_shuffle_ps(t2, t3, _MM_SHUFFLE(1, 0, 1, 0));
    if constexpr (nOut == 4)
    {
        v[3] = _mm512_shuffle_ps(t2, t3, _MM_SHUFFLE(3, 2, 3, 2));
    }
}

template<int align>
static inline void gmx_simdcall
avx512LoadTranspose(const float* base, const std::int32_t* o, SimdFloat* v0, SimdFloat* v1, SimdFloat* v2)
{
    __m512 v[3];
    avx512LoadTransposeQuads<align, 3>(base, o, v);
    v0->simdInternal_ = v[0];
    v1->simdInternal_ = v[1];
    v2->simdInternal_ = v[2];
}

template<int align>
static inline void gmx_simdcall avx512LoadTranspose(const float*        base,
                                                    const std::int32_t* o,
                                                    SimdFloat*          v0,
                                                    SimdFloat*          v1,
                                                    SimdFloat*          v2,
                                                    SimdFloat*          v3)
{
    __m512 v[4];
    avx512LoadTransposeQuads<align, 4>(base, o, v);
    v0->simdInternal_ = v[0];
    v1->simdInternal_ = v[1];
    v2->simdInternal_ = v[2];
    v3->simdInternal_ = v[3];
}

// Store the first 3 floats of 128-bit lane 'lane' of a to p, leaving p[3] untouched.
template<int lane>
static inline void gmx_simdcall avx512StoreLane3(float* p, __m512 a)
{
#    if defined(__AVX512VL__)
    _mm_mask_storeu_ps(p, 0x7, lane == 0 ? _mm512_castps512_ps128(a) : _mm512_extractf32x4_ps(a, lane));
#    else
    _mm512_mask_storeu_ps(p,
                          avx512Int2Mask(0x7),
                          lane == 0 ? a : _mm512_castps128_ps512(_mm512_extractf32x4_ps(a, lane)));
#    endif
}
#endif // !GMX_SIMD_X86_AVX_512_USE_HW_GATHER

/* This is an internal helper function used by decr3Hsimd(...).
 */
inline void gmx_simdcall decrHsimd(float* m, SimdFloat a)
{
    __m256 t;

    assert(std::size_t(m) % 32 == 0);

    a.simdInternal_ = _mm512_add_ps(a.simdInternal_,
                                    _mm512_shuffle_f32x4(a.simdInternal_, a.simdInternal_, 0xEE));
    t               = _mm256_load_ps(m);
    t               = _mm256_sub_ps(t, _mm512_castps512_ps256(a.simdInternal_));
    _mm256_store_ps(m, t);
}
} // namespace

namespace
{
template<int align, bool unalignedVariant, typename... Targs>
static inline void gmx_simdcall
avx512GatherLoadBySimdIntTranspose(const float* base, SimdFInt32 offset, SimdFloat* v, Targs... Fargs)
{
#if !GMX_SIMD_X86_AVX_512_USE_HW_GATHER
    if constexpr (c_avx512UsePermuteTranspose<1 + sizeof...(Targs), align, unalignedVariant>)
    {
        alignas(GMX_SIMD_ALIGNMENT) std::int32_t o[GMX_SIMD_FINT32_WIDTH];
        _mm512_store_si512(o, offset.simdInternal_);
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
avx512GatherLoadTranspose(const float* base, const std::int32_t offset[], SimdFloat* v, Targs... Fargs)
{
#if !GMX_SIMD_X86_AVX_512_USE_HW_GATHER
    if constexpr (c_avx512UsePermuteTranspose<1 + sizeof...(Targs), align, unalignedVariant>)
    {
        avx512LoadTranspose<align>(base, offset, v, Fargs...);
    }
    else
#endif
    {
        avx512HwGatherTranspose<align>(base, simdLoad(offset, SimdFInt32Tag()), v, Fargs...);
    }
}
} // namespace

template<int align, typename... Targs>
static inline void gmx_simdcall
gatherLoadBySimdIntTranspose(const float* base, SimdFInt32 offset, SimdFloat* v, Targs... Fargs)
{
    avx512GatherLoadBySimdIntTranspose<align, false>(base, offset, v, Fargs...);
}

template<int align, typename... Targs>
static inline void gmx_simdcall
gatherLoadUBySimdIntTranspose(const float* base, SimdFInt32 offset, SimdFloat* v, Targs... Fargs)
{
    avx512GatherLoadBySimdIntTranspose<align, true>(base, offset, v, Fargs...);
}

template<int align, typename... Targs>
static inline void gmx_simdcall
gatherLoadTranspose(const float* base, const std::int32_t offset[], SimdFloat* v, Targs... Fargs)
{
    avx512GatherLoadTranspose<align, false>(base, offset, v, Fargs...);
}

template<int align, typename... Targs>
static inline void gmx_simdcall
gatherLoadUTranspose(const float* base, const std::int32_t offset[], SimdFloat* v, Targs... Fargs)
{
    avx512GatherLoadTranspose<align, true>(base, offset, v, Fargs...);
}

template<int align>
static inline void gmx_simdcall
transposeScatterStoreU(float* base, const std::int32_t offset[], SimdFloat v0, SimdFloat v1, SimdFloat v2)
{
#if GMX_SIMD_X86_AVX_512_USE_HW_GATHER
    SimdFInt32 simdoffset = simdLoad(offset, SimdFInt32Tag());
    if (align > 2)
    {
        simdoffset = fastMultiply<align>(simdoffset);
    }
    constexpr size_t scale = (align > 2) ? sizeof(float) : sizeof(float) * align;

    _mm512_i32scatter_ps(base, simdoffset.simdInternal_, v0.simdInternal_, scale);
    _mm512_i32scatter_ps(&(base[1]), simdoffset.simdInternal_, v1.simdInternal_, scale);
    _mm512_i32scatter_ps(&(base[2]), simdoffset.simdInternal_, v2.simdInternal_, scale);
#else
    // Transpose within 128-bit lanes so that lane k of t[j] is point 4k+j as x y z z,
    // then store 3 floats per point with a masked store. Points are stored in index
    // order, so duplicate offsets give the same result as the scatter (last one wins).
    const __m512 t5 = _mm512_unpacklo_ps(v0.simdInternal_, v1.simdInternal_);
    const __m512 t6 = _mm512_unpackhi_ps(v0.simdInternal_, v1.simdInternal_);
    const __m512 t0 = _mm512_shuffle_ps(t5, v2.simdInternal_, _MM_SHUFFLE(0, 0, 1, 0));
    const __m512 t1 = _mm512_shuffle_ps(t5, v2.simdInternal_, _MM_SHUFFLE(1, 1, 3, 2));
    const __m512 t2 = _mm512_shuffle_ps(t6, v2.simdInternal_, _MM_SHUFFLE(2, 2, 1, 0));
    const __m512 t3 = _mm512_shuffle_ps(t6, v2.simdInternal_, _MM_SHUFFLE(3, 3, 3, 2));
    avx512StoreLane3<0>(base + align * offset[0], t0);
    avx512StoreLane3<0>(base + align * offset[1], t1);
    avx512StoreLane3<0>(base + align * offset[2], t2);
    avx512StoreLane3<0>(base + align * offset[3], t3);
    avx512StoreLane3<1>(base + align * offset[4], t0);
    avx512StoreLane3<1>(base + align * offset[5], t1);
    avx512StoreLane3<1>(base + align * offset[6], t2);
    avx512StoreLane3<1>(base + align * offset[7], t3);
    avx512StoreLane3<2>(base + align * offset[8], t0);
    avx512StoreLane3<2>(base + align * offset[9], t1);
    avx512StoreLane3<2>(base + align * offset[10], t2);
    avx512StoreLane3<2>(base + align * offset[11], t3);
    avx512StoreLane3<3>(base + align * offset[12], t0);
    avx512StoreLane3<3>(base + align * offset[13], t1);
    avx512StoreLane3<3>(base + align * offset[14], t2);
    avx512StoreLane3<3>(base + align * offset[15], t3);
#endif
}

template<int align>
static inline void gmx_simdcall
transposeScatterIncrU(float* base, const std::int32_t offset[], SimdFloat v0, SimdFloat v1, SimdFloat v2)
{
    __m512                                   t[4], t5, t6, t7, t8;
    int                                      i;
    alignas(GMX_SIMD_ALIGNMENT) std::int32_t o[16];
    store(o, fastMultiply<align>(simdLoad(offset, SimdFInt32Tag())));
    if (align < 4)
    {
        t5   = _mm512_unpacklo_ps(v0.simdInternal_, v1.simdInternal_);
        t6   = _mm512_unpackhi_ps(v0.simdInternal_, v1.simdInternal_);
        t[0] = _mm512_shuffle_ps(t5, v2.simdInternal_, _MM_SHUFFLE(0, 0, 1, 0));
        t[1] = _mm512_shuffle_ps(t5, v2.simdInternal_, _MM_SHUFFLE(1, 1, 3, 2));
        t[2] = _mm512_shuffle_ps(t6, v2.simdInternal_, _MM_SHUFFLE(2, 2, 1, 0));
        t[3] = _mm512_shuffle_ps(t6, v2.simdInternal_, _MM_SHUFFLE(3, 3, 3, 2));
        for (i = 0; i < 4; i++)
        {
            _mm512_mask_storeu_ps(base + o[i],
                                  avx512Int2Mask(7),
                                  _mm512_castps128_ps512(_mm_add_ps(_mm_loadu_ps(base + o[i]),
                                                                    _mm512_castps512_ps128(t[i]))));
            _mm512_mask_storeu_ps(base + o[4 + i],
                                  avx512Int2Mask(7),
                                  _mm512_castps128_ps512(_mm_add_ps(_mm_loadu_ps(base + o[4 + i]),
                                                                    _mm512_extractf32x4_ps(t[i], 1))));
            _mm512_mask_storeu_ps(base + o[8 + i],
                                  avx512Int2Mask(7),
                                  _mm512_castps128_ps512(_mm_add_ps(_mm_loadu_ps(base + o[8 + i]),
                                                                    _mm512_extractf32x4_ps(t[i], 2))));
            _mm512_mask_storeu_ps(base + o[12 + i],
                                  avx512Int2Mask(7),
                                  _mm512_castps128_ps512(_mm_add_ps(_mm_loadu_ps(base + o[12 + i]),
                                                                    _mm512_extractf32x4_ps(t[i], 3))));
        }
    }
    else
    {
        // One could use shuffle here too if it is OK to overwrite the padded elements for alignment
        t5   = _mm512_unpacklo_ps(v0.simdInternal_, v2.simdInternal_);
        t6   = _mm512_unpackhi_ps(v0.simdInternal_, v2.simdInternal_);
        t7   = _mm512_unpacklo_ps(v1.simdInternal_, _mm512_setzero_ps());
        t8   = _mm512_unpackhi_ps(v1.simdInternal_, _mm512_setzero_ps());
        t[0] = _mm512_unpacklo_ps(t5, t7); // x0 y0 z0  0 | x4 y4 z4 0
        t[1] = _mm512_unpackhi_ps(t5, t7); // x1 y1 z1  0 | x5 y5 z5 0
        t[2] = _mm512_unpacklo_ps(t6, t8); // x2 y2 z2  0 | x6 y6 z6 0
        t[3] = _mm512_unpackhi_ps(t6, t8); // x3 y3 z3  0 | x7 y7 z7 0
        if (align % 4 == 0)
        {
            for (i = 0; i < 4; i++)
            {
                _mm_store_ps(base + o[i],
                             _mm_add_ps(_mm_load_ps(base + o[i]), _mm512_castps512_ps128(t[i])));
                _mm_store_ps(base + o[4 + i],
                             _mm_add_ps(_mm_load_ps(base + o[4 + i]), _mm512_extractf32x4_ps(t[i], 1)));
                _mm_store_ps(base + o[8 + i],
                             _mm_add_ps(_mm_load_ps(base + o[8 + i]), _mm512_extractf32x4_ps(t[i], 2)));
                _mm_store_ps(base + o[12 + i],
                             _mm_add_ps(_mm_load_ps(base + o[12 + i]), _mm512_extractf32x4_ps(t[i], 3)));
            }
        }
        else
        {
            for (i = 0; i < 4; i++)
            {
                _mm_storeu_ps(base + o[i],
                              _mm_add_ps(_mm_loadu_ps(base + o[i]), _mm512_castps512_ps128(t[i])));
                _mm_storeu_ps(base + o[4 + i],
                              _mm_add_ps(_mm_loadu_ps(base + o[4 + i]), _mm512_extractf32x4_ps(t[i], 1)));
                _mm_storeu_ps(base + o[8 + i],
                              _mm_add_ps(_mm_loadu_ps(base + o[8 + i]), _mm512_extractf32x4_ps(t[i], 2)));
                _mm_storeu_ps(base + o[12 + i],
                              _mm_add_ps(_mm_loadu_ps(base + o[12 + i]), _mm512_extractf32x4_ps(t[i], 3)));
            }
        }
    }
}

template<int align>
static inline void gmx_simdcall
transposeScatterDecrU(float* base, const std::int32_t offset[], SimdFloat v0, SimdFloat v1, SimdFloat v2)
{
    __m512                                   t[4], t5, t6, t7, t8;
    int                                      i;
    alignas(GMX_SIMD_ALIGNMENT) std::int32_t o[16];
    store(o, fastMultiply<align>(simdLoad(offset, SimdFInt32Tag())));
    if (align < 4)
    {
        t5   = _mm512_unpacklo_ps(v0.simdInternal_, v1.simdInternal_);
        t6   = _mm512_unpackhi_ps(v0.simdInternal_, v1.simdInternal_);
        t[0] = _mm512_shuffle_ps(t5, v2.simdInternal_, _MM_SHUFFLE(0, 0, 1, 0));
        t[1] = _mm512_shuffle_ps(t5, v2.simdInternal_, _MM_SHUFFLE(1, 1, 3, 2));
        t[2] = _mm512_shuffle_ps(t6, v2.simdInternal_, _MM_SHUFFLE(2, 2, 1, 0));
        t[3] = _mm512_shuffle_ps(t6, v2.simdInternal_, _MM_SHUFFLE(3, 3, 3, 2));
        for (i = 0; i < 4; i++)
        {
            _mm512_mask_storeu_ps(base + o[i],
                                  avx512Int2Mask(7),
                                  _mm512_castps128_ps512(_mm_sub_ps(_mm_loadu_ps(base + o[i]),
                                                                    _mm512_castps512_ps128(t[i]))));
            _mm512_mask_storeu_ps(base + o[4 + i],
                                  avx512Int2Mask(7),
                                  _mm512_castps128_ps512(_mm_sub_ps(_mm_loadu_ps(base + o[4 + i]),
                                                                    _mm512_extractf32x4_ps(t[i], 1))));
            _mm512_mask_storeu_ps(base + o[8 + i],
                                  avx512Int2Mask(7),
                                  _mm512_castps128_ps512(_mm_sub_ps(_mm_loadu_ps(base + o[8 + i]),
                                                                    _mm512_extractf32x4_ps(t[i], 2))));
            _mm512_mask_storeu_ps(base + o[12 + i],
                                  avx512Int2Mask(7),
                                  _mm512_castps128_ps512(_mm_sub_ps(_mm_loadu_ps(base + o[12 + i]),
                                                                    _mm512_extractf32x4_ps(t[i], 3))));
        }
    }
    else
    {
        // One could use shuffle here too if it is OK to overwrite the padded elements for alignment
        t5   = _mm512_unpacklo_ps(v0.simdInternal_, v2.simdInternal_);
        t6   = _mm512_unpackhi_ps(v0.simdInternal_, v2.simdInternal_);
        t7   = _mm512_unpacklo_ps(v1.simdInternal_, _mm512_setzero_ps());
        t8   = _mm512_unpackhi_ps(v1.simdInternal_, _mm512_setzero_ps());
        t[0] = _mm512_unpacklo_ps(t5, t7); // x0 y0 z0  0 | x4 y4 z4 0
        t[1] = _mm512_unpackhi_ps(t5, t7); // x1 y1 z1  0 | x5 y5 z5 0
        t[2] = _mm512_unpacklo_ps(t6, t8); // x2 y2 z2  0 | x6 y6 z6 0
        t[3] = _mm512_unpackhi_ps(t6, t8); // x3 y3 z3  0 | x7 y7 z7 0
        if (align % 4 == 0)
        {
            for (i = 0; i < 4; i++)
            {
                _mm_store_ps(base + o[i],
                             _mm_sub_ps(_mm_load_ps(base + o[i]), _mm512_castps512_ps128(t[i])));
                _mm_store_ps(base + o[4 + i],
                             _mm_sub_ps(_mm_load_ps(base + o[4 + i]), _mm512_extractf32x4_ps(t[i], 1)));
                _mm_store_ps(base + o[8 + i],
                             _mm_sub_ps(_mm_load_ps(base + o[8 + i]), _mm512_extractf32x4_ps(t[i], 2)));
                _mm_store_ps(base + o[12 + i],
                             _mm_sub_ps(_mm_load_ps(base + o[12 + i]), _mm512_extractf32x4_ps(t[i], 3)));
            }
        }
        else
        {
            for (i = 0; i < 4; i++)
            {
                _mm_storeu_ps(base + o[i],
                              _mm_sub_ps(_mm_loadu_ps(base + o[i]), _mm512_castps512_ps128(t[i])));
                _mm_storeu_ps(base + o[4 + i],
                              _mm_sub_ps(_mm_loadu_ps(base + o[4 + i]), _mm512_extractf32x4_ps(t[i], 1)));
                _mm_storeu_ps(base + o[8 + i],
                              _mm_sub_ps(_mm_loadu_ps(base + o[8 + i]), _mm512_extractf32x4_ps(t[i], 2)));
                _mm_storeu_ps(base + o[12 + i],
                              _mm_sub_ps(_mm_loadu_ps(base + o[12 + i]), _mm512_extractf32x4_ps(t[i], 3)));
            }
        }
    }
}

static inline void gmx_simdcall expandScalarsToTriplets(SimdFloat  scalar,
                                                        SimdFloat* triplets0,
                                                        SimdFloat* triplets1,
                                                        SimdFloat* triplets2)
{
    triplets0->simdInternal_ = _mm512_permutexvar_ps(
            _mm512_set_epi32(5, 4, 4, 4, 3, 3, 3, 2, 2, 2, 1, 1, 1, 0, 0, 0), scalar.simdInternal_);
    triplets1->simdInternal_ = _mm512_permutexvar_ps(
            _mm512_set_epi32(10, 10, 9, 9, 9, 8, 8, 8, 7, 7, 7, 6, 6, 6, 5, 5), scalar.simdInternal_);
    triplets2->simdInternal_ = _mm512_permutexvar_ps(
            _mm512_set_epi32(15, 15, 15, 14, 14, 14, 13, 13, 13, 12, 12, 12, 11, 11, 11, 10),
            scalar.simdInternal_);
}


static inline float gmx_simdcall reduceIncr4ReturnSum(float* m, SimdFloat v0, SimdFloat v1, SimdFloat v2, SimdFloat v3)
{
    __m512 t0, t1, t2;
    __m128 t3, t4;

    assert(std::size_t(m) % 16 == 0);

    t0 = _mm512_add_ps(v0.simdInternal_, _mm512_permute_ps(v0.simdInternal_, 0x4E));
    t0 = _mm512_mask_add_ps(
            t0, avx512Int2Mask(0xCCCC), v2.simdInternal_, _mm512_permute_ps(v2.simdInternal_, 0x4E));
    t1 = _mm512_add_ps(v1.simdInternal_, _mm512_permute_ps(v1.simdInternal_, 0x4E));
    t1 = _mm512_mask_add_ps(
            t1, avx512Int2Mask(0xCCCC), v3.simdInternal_, _mm512_permute_ps(v3.simdInternal_, 0x4E));
    t2 = _mm512_add_ps(t0, _mm512_permute_ps(t0, 0xB1));
    t2 = _mm512_mask_add_ps(t2, avx512Int2Mask(0xAAAA), t1, _mm512_permute_ps(t1, 0xB1));

    t2 = _mm512_add_ps(t2, _mm512_shuffle_f32x4(t2, t2, 0x4E));
    t2 = _mm512_add_ps(t2, _mm512_shuffle_f32x4(t2, t2, 0xB1));

    t3 = _mm512_castps512_ps128(t2);
    t4 = _mm_load_ps(m);
    t4 = _mm_add_ps(t4, t3);
    _mm_store_ps(m, t4);

    t3 = _mm_add_ps(t3, _mm_permute_ps(t3, 0x4E));
    t3 = _mm_add_ps(t3, _mm_permute_ps(t3, 0xB1));

    return _mm_cvtss_f32(t3);
}

static inline SimdFloat gmx_simdcall loadDualHsimd(const float* m0, const float* m1)
{
    assert(std::size_t(m0) % 32 == 0);
    assert(std::size_t(m1) % 32 == 0);

    return { _mm512_castpd_ps(_mm512_insertf64x4(
            _mm512_castpd256_pd512(_mm256_load_pd(reinterpret_cast<const double*>(m0))),
            _mm256_load_pd(reinterpret_cast<const double*>(m1)),
            1)) };
}

static inline SimdFloat gmx_simdcall loadDuplicateHsimd(const float* m)
{
    assert(std::size_t(m) % 32 == 0);
    return { _mm512_castpd_ps(_mm512_broadcast_f64x4(_mm256_load_pd(reinterpret_cast<const double*>(m)))) };
}

static inline SimdFloat gmx_simdcall loadU1DualHsimd(const float* m)
{
    return { _mm512_shuffle_f32x4(
            _mm512_broadcastss_ps(_mm_load_ss(m)), _mm512_broadcastss_ps(_mm_load_ss(m + 1)), 0x44) };
}


static inline void gmx_simdcall storeDualHsimd(float* m0, float* m1, SimdFloat a)
{
    assert(std::size_t(m0) % 32 == 0);
    assert(std::size_t(m1) % 32 == 0);

    _mm256_store_ps(m0, _mm512_castps512_ps256(a.simdInternal_));
    _mm256_store_pd(reinterpret_cast<double*>(m1),
                    _mm512_extractf64x4_pd(_mm512_castps_pd(a.simdInternal_), 1));
}

static inline void gmx_simdcall incrDualHsimd(float* m0, float* m1, SimdFloat a)
{
    assert(std::size_t(m0) % 32 == 0);
    assert(std::size_t(m1) % 32 == 0);

    __m256 x;

    // Lower half
    x = _mm256_load_ps(m0);
    x = _mm256_add_ps(x, _mm512_castps512_ps256(a.simdInternal_));
    _mm256_store_ps(m0, x);

    // Upper half
    x = _mm256_load_ps(m1);
    x = _mm256_add_ps(x, _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(a.simdInternal_), 1)));
    _mm256_store_ps(m1, x);
}

static inline void gmx_simdcall decr3Hsimd(float* m, SimdFloat a0, SimdFloat a1, SimdFloat a2)
{
    decrHsimd(m, a0);
    decrHsimd(m + GMX_SIMD_FLOAT_WIDTH / 2, a1);
    decrHsimd(m + GMX_SIMD_FLOAT_WIDTH, a2);
}


template<int align>
static inline void gmx_simdcall gatherLoadTransposeHsimd(const float*       base0,
                                                         const float*       base1,
                                                         const std::int32_t offset[],
                                                         SimdFloat*         v0,
                                                         SimdFloat*         v1)
{
    assert(std::size_t(offset) % 32 == 0);
    assert(std::size_t(base0) % 8 == 0);
    assert(std::size_t(base1) % 8 == 0);

    static_assert(align == 2 || align == 4, "If more are needed use fastMultiply");

#if GMX_SIMD_X86_AVX_512_USE_HW_GATHER
    __m256i idx;
    __m512  tmp1, tmp2;

    idx = _mm256_load_si256(reinterpret_cast<const __m256i*>(offset));

    if (align == 4)
    {
        idx = _mm256_slli_epi32(idx, 1);
    }

    tmp1 = _mm512_castpd_ps(
            _mm512_i32gather_pd(idx, reinterpret_cast<const double*>(base0), sizeof(double)));
    tmp2 = _mm512_castpd_ps(
            _mm512_i32gather_pd(idx, reinterpret_cast<const double*>(base1), sizeof(double)));

    v0->simdInternal_ = _mm512_mask_moveldup_ps(tmp1, 0xAAAA, tmp2);
    v1->simdInternal_ = _mm512_mask_movehdup_ps(tmp2, 0x5555, tmp1);

    v0->simdInternal_ = _mm512_permutexvar_ps(
            _mm512_set_epi32(15, 13, 11, 9, 7, 5, 3, 1, 14, 12, 10, 8, 6, 4, 2, 0), v0->simdInternal_);
    v1->simdInternal_ = _mm512_permutexvar_ps(
            _mm512_set_epi32(15, 13, 11, 9, 7, 5, 3, 1, 14, 12, 10, 8, 6, 4, 2, 0), v1->simdInternal_);
#else
    // 8 pairs from each base with 64-bit loads, then one two-source permute per output:
    // v0 = { base0[align*offset[0..7]], base1[align*offset[0..7]] }, v1 likewise for +1.
    avx512DeinterleavePairs(avx512LoadPairs8<align>(base0, offset),
                            avx512LoadPairs8<align>(base1, offset),
                            &v0->simdInternal_,
                            &v1->simdInternal_);
#endif
}

static inline float gmx_simdcall reduceIncr4ReturnSumHsimd(float* m, SimdFloat v0, SimdFloat v1)
{
    __m512 t0, t1;
    __m128 t2, t3;

    assert(std::size_t(m) % 16 == 0);

    t0 = _mm512_shuffle_f32x4(v0.simdInternal_, v1.simdInternal_, 0x88);
    t1 = _mm512_shuffle_f32x4(v0.simdInternal_, v1.simdInternal_, 0xDD);
    t0 = _mm512_add_ps(t0, t1);
    t0 = _mm512_add_ps(t0, _mm512_permute_ps(t0, 0x4E));
    t0 = _mm512_add_ps(t0, _mm512_permute_ps(t0, 0xB1));
    t0 = _mm512_maskz_compress_ps(avx512Int2Mask(0x1111), t0);

    t3 = _mm512_castps512_ps128(t0);
    t2 = _mm_load_ps(m);
    t2 = _mm_add_ps(t2, t3);
    _mm_store_ps(m, t2);

    t3 = _mm_add_ps(t3, _mm_permute_ps(t3, 0x4E));
    t3 = _mm_add_ps(t3, _mm_permute_ps(t3, 0xB1));

    return _mm_cvtss_f32(t3);
}

static inline SimdFloat gmx_simdcall loadUNDuplicate4(const float* f)
{
    return { _mm512_permute_ps(_mm512_maskz_expandloadu_ps(0x1111, f), 0) };
}

static inline SimdFloat gmx_simdcall load4DuplicateN(const float* f)
{
    return { _mm512_broadcast_f32x4(_mm_load_ps(f)) };
}

static inline SimdFloat gmx_simdcall loadU4NOffset(const float* f, int offset)
{
#if GMX_SIMD_X86_AVX_512_USE_HW_GATHER
    const __m256i idx = _mm256_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3);
    const __m256i gdx = _mm256_add_epi32(_mm256_setr_epi32(0, 2, 0, 2, 0, 2, 0, 2),
                                         _mm256_mullo_epi32(idx, _mm256_set1_epi32(offset)));
    return { _mm512_castpd_ps(_mm512_i32gather_pd(gdx, reinterpret_cast<const double*>(f), sizeof(float))) };
#else
    return { avx512LoadQuads(f, f + offset, f + 2 * offset, f + 3 * offset) };
#endif
}

} // namespace gmx

#endif // GMX_SIMD_IMPL_X86_AVX_512_UTIL_FLOAT_H
