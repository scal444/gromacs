/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2015- The GROMACS Authors
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
 * Tests for gmx::CpuInfo
 *
 * \author Erik Lindahl <erik.lindahl@gmail.com>
 * \ingroup module_hardware
 */
#include "gmxpre.h"

#include "gromacs/hardware/cpuinfo.h"

#include "config.h"

#include <ostream>
#include <set>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace gmx
{
namespace test
{
namespace
{

TEST(CpuInfoTest, SupportLevel)
{
    // There is no way we can compare to any reference data since that
    // depends on the architecture, but we can at least make sure that it
    // works to execute the tests

    gmx::CpuInfo c(gmx::CpuInfo::detect());

    std::string commonMsg =
            "\nGROMACS might still work, but it will likely hurt your performance."
            "\nPlease make a post at the GROMACS development forum at"
            "\nhttps://gromacs.bioexcel.eu/c/gromacs-developers/10 so we can try to fix it.";

    // It is not the end of the world if any of these tests fail (Gromacs will
    // work fine without cpuinfo), but we might as well flag it so we add it to
    // our detection code
    EXPECT_GT(c.supportLevel(), gmx::CpuInfo::SupportLevel::None)
            << "No CPU information at all could be detected. " << commonMsg << std::endl;

#if GMX_TARGET_X86
    EXPECT_GE(c.supportLevel(), gmx::CpuInfo::SupportLevel::Features)
            << "No CPU features could be detected. " << commonMsg << std::endl;
#endif

    if (c.supportLevel() >= gmx::CpuInfo::SupportLevel::LogicalProcessorInfo)
    {
        // Make sure assigned numbers are reasonable if we have them
        for (const auto& l : c.logicalProcessors())
        {
            EXPECT_GE(l.packageIdInMachine, 0)
                    << "Impossible package index for logical processor. " << commonMsg << std::endl;
            EXPECT_GE(l.coreIdInPackage, 0)
                    << "Impossible core index for logical processor. " << commonMsg << std::endl;
            EXPECT_GE(l.puIdInCore, 0)
                    << "Impossible pu index for logical processor. " << commonMsg << std::endl;
        }
    }
}


//! Feature set as cpuid reports it on an AVX-512 machine
const std::set<CpuInfo::Feature> c_avx512CpuidFeatures = {
    CpuInfo::Feature::X86_Sse2,       CpuInfo::Feature::X86_Sse4_1,
    CpuInfo::Feature::X86_Avx,        CpuInfo::Feature::X86_Avx2,
    CpuInfo::Feature::X86_Fma,        CpuInfo::Feature::X86_F16C,
    CpuInfo::Feature::X86_Avx512F,    CpuInfo::Feature::X86_Avx512CD,
    CpuInfo::Feature::X86_Avx512BW,   CpuInfo::Feature::X86_Avx512VL,
    CpuInfo::Feature::X86_Avx512BF16, CpuInfo::Feature::X86_Avx512secondFMA
};

//! The features that do not need any OS-enabled extended state
const std::set<CpuInfo::Feature> c_sseFeatures = { CpuInfo::Feature::X86_Sse2, CpuInfo::Feature::X86_Sse4_1 };

//! The features that only need the OS-enabled SSE and AVX state
const std::set<CpuInfo::Feature> c_avxFeatures = {
    CpuInfo::Feature::X86_Sse2, CpuInfo::Feature::X86_Sse4_1, CpuInfo::Feature::X86_Avx,
    CpuInfo::Feature::X86_Avx2, CpuInfo::Feature::X86_Fma,    CpuInfo::Feature::X86_F16C
};

TEST(CpuInfoTest, X86FeaturesKeptWhenOsEnablesAllState)
{
    // x87, SSE, AVX, opmask, ZMM_Hi256 and Hi16_ZMM, as on bare-metal AVX-512 machines
    EXPECT_EQ(c_avx512CpuidFeatures,
              detail::filterX86FeaturesByOsSupport(c_avx512CpuidFeatures, true, 0xe7));
    // Additional state bits (here PKRU and AMX) do not matter
    EXPECT_EQ(c_avx512CpuidFeatures,
              detail::filterX86FeaturesByOsSupport(c_avx512CpuidFeatures, true, 0x602e7));
}

TEST(CpuInfoTest, X86Avx512FeaturesRemovedWithoutOsAvx512State)
{
    // e.g. a hypervisor that passes through the cpuid bits but disables the AVX-512 state
    EXPECT_EQ(c_avxFeatures, detail::filterX86FeaturesByOsSupport(c_avx512CpuidFeatures, true, 0x07));
}

TEST(CpuInfoTest, X86Avx512FeaturesRemovedWithPartialOsAvx512State)
{
    // Opmask and ZMM_Hi256 state, but no Hi16_ZMM state
    EXPECT_EQ(c_avxFeatures, detail::filterX86FeaturesByOsSupport(c_avx512CpuidFeatures, true, 0x67));
}

TEST(CpuInfoTest, X86AvxFeaturesRemovedWithoutOsXsave)
{
    // e.g. a Linux kernel booted with noxsave; XCR0 must then be ignored
    EXPECT_EQ(c_sseFeatures, detail::filterX86FeaturesByOsSupport(c_avx512CpuidFeatures, false, 0));
    EXPECT_EQ(c_sseFeatures, detail::filterX86FeaturesByOsSupport(c_avx512CpuidFeatures, false, 0xe7));
}

TEST(CpuInfoTest, X86AvxFeaturesRemovedWithoutOsAvxState)
{
    // OSXSAVE is set, but only the x87 and SSE state are enabled
    EXPECT_EQ(c_sseFeatures, detail::filterX86FeaturesByOsSupport(c_avx512CpuidFeatures, true, 0x03));
}

TEST(CpuInfoTest, X86AvxOnlyFeaturesNeedOnlyAvxState)
{
    const std::set<CpuInfo::Feature> amdFeatures = { CpuInfo::Feature::X86_Sse2,
                                                     CpuInfo::Feature::X86_Avx,
                                                     CpuInfo::Feature::X86_Fma4,
                                                     CpuInfo::Feature::X86_Xop };
    EXPECT_EQ(amdFeatures, detail::filterX86FeaturesByOsSupport(amdFeatures, true, 0x07));
    EXPECT_EQ(std::set<CpuInfo::Feature>{ CpuInfo::Feature::X86_Sse2 },
              detail::filterX86FeaturesByOsSupport(amdFeatures, true, 0x03));
}

} // namespace
} // namespace test
} // namespace gmx
