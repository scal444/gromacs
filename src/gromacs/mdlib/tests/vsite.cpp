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
 * Tests for virtual sites.
 *
 * \author berk Hess <hess@kth.se>
 * \ingroup module_mdlib
 */
#include "gmxpre.h"

#include "gromacs/mdlib/vsite.h"

#include <cctype>

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "gromacs/gmxlib/nrnb.h"
#include "gromacs/mdlib/gmx_omp_nthreads.h"
#include "gromacs/mdtypes/md_enums.h"
#include "gromacs/pbcutil/ishift.h"
#include "gromacs/pbcutil/pbc.h"
#include "gromacs/topology/idef.h"
#include "gromacs/topology/ifunc.h"
#include "gromacs/topology/topology.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/enumerationhelpers.h"
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

//! Two vsites with the higher depending on the lower one
gmx_moltype_t moltypeVSite12()
{
    gmx_moltype_t moltype = {};

    moltype.atoms.nr                                        = 3;
    moltype.ilist[InteractionFunction::VirtualSite1].iatoms = { 0, 1, 0 };
    moltype.ilist[InteractionFunction::VirtualSite2].iatoms = { 1, 2, 0, 1 };

    return moltype;
}

//! Two vsites with the lower depending on the higher one
gmx_moltype_t moltypeVSite21()
{
    gmx_moltype_t moltype = {};

    moltype.atoms.nr                                        = 3;
    moltype.ilist[InteractionFunction::VirtualSite2].iatoms = { 0, 2, 0, 1 };
    moltype.ilist[InteractionFunction::VirtualSite1].iatoms = { 1, 1, 2 };

    return moltype;
}

//! Two vsites of the same type with one depending on the other
gmx_moltype_t moltypeVSite11()
{
    gmx_moltype_t moltype = {};

    moltype.atoms.nr                                        = 3;
    moltype.ilist[InteractionFunction::VirtualSite1].iatoms = { 0, 1, 0, 1, 2, 1 };

    return moltype;
}

gmx_molblock_t oneMolblock()
{
    return { 0, 1, {}, {} };
}

//! Test fixture class
class VSiteTest : public ::testing::Test
{
public:
    //! Global toplogy to use in tests
    gmx_mtop_t mtop_;
};

TEST_F(VSiteTest, VSiteLowerConstructingWorks)
{
    mtop_.moltype.emplace_back(moltypeVSite12());
    mtop_.molblock.push_back(oneMolblock());
    EXPECT_NO_THROW(makeVirtualSitesHandler(mtop_, nullptr, PbcType::Xyz, {}));
}

TEST_F(VSiteTest, VSiteHigherConstructingThrows)
{
    mtop_.moltype.emplace_back(moltypeVSite21());
    mtop_.molblock.push_back(oneMolblock());
    EXPECT_THROW_GMX(makeVirtualSitesHandler(mtop_, nullptr, PbcType::Xyz, {}), InvalidInputError);
}

TEST_F(VSiteTest, VSiteEqualConstructingThrows)
{
    mtop_.moltype.emplace_back(moltypeVSite11());
    mtop_.molblock.push_back(oneMolblock());
    EXPECT_THROW_GMX(makeVirtualSitesHandler(mtop_, nullptr, PbcType::Xyz, {}), InvalidInputError);
}

//! Test parameters: the virtual site type, the PBC type and whether atoms are in different periodic images
using VSiteDerivativesTestParameters = std::tuple<InteractionFunction, PbcType, bool>;

/*! \brief Test fixture for checking that force spreading matches construction
 *
 * Uses a molecule of four atoms with a single virtual site of the type
 * given by the test parameters, constructed from as many of the atoms
 * as the type needs.
 */
class VSiteDerivativesTest : public ::testing::TestWithParam<VSiteDerivativesTestParameters>
{
public:
    VSiteDerivativesTest()
    {
        bool atomsInDifferentImages;
        std::tie(functionType_, pbcType_, atomsInDifferentImages) = GetParam();

        clear_mat(box_);
        box_[XX][XX] = 2.0;
        box_[YY][YY] = 2.2;
        box_[ZZ][ZZ] = 2.4;

        // Set up the parameters and atom list for the virtual site
        gmx_moltype_t moltype    = {};
        moltype.atoms.nr         = c_numAtoms + 1;
        std::vector<int>& iatoms = moltype.ilist[functionType_].iatoms;
        if (functionType_ == InteractionFunction::VirtualSiteN)
        {
            // Each constructing atom has its own parameter type with its weight
            const std::array<real, 3> weights = { 0.5, 0.2, 0.3 };
            for (int i = 0; i < gmx::ssize(weights); i++)
            {
                t_iparams iparams;
                iparams.vsiten.n = gmx::ssize(weights);
                iparams.vsiten.a = weights[i];
                mtop_.ffparams.iparams.push_back(iparams);
                mtop_.ffparams.functype.push_back(functionType_);
                iatoms.insert(iatoms.end(), { i, c_numAtoms, i });
            }
        }
        else
        {
            t_iparams iparams;
            switch (functionType_)
            {
                // The parameters for the fixed distance types include lengths in nm
                case InteractionFunction::VirtualSite2FixedDistance: iparams.vsite.a = 0.07; break;
                case InteractionFunction::VirtualSite3FixedDistance:
                    iparams.vsite.a = 0.4;
                    iparams.vsite.b = 0.08;
                    break;
                case InteractionFunction::VirtualSite3FixedAngleDistance:
                    iparams.vsite.a = 0.05;
                    iparams.vsite.b = 0.06;
                    break;
                case InteractionFunction::VirtualSite3Outside:
                    iparams.vsite.a = 0.2;
                    iparams.vsite.b = 0.3;
                    iparams.vsite.c = 1.5;
                    break;
                case InteractionFunction::VirtualSite4FixedDistance:
                case InteractionFunction::VirtualSite4FixedDistanceNormalization:
                    iparams.vsite.a = 0.3;
                    iparams.vsite.b = 0.4;
                    iparams.vsite.c = 0.09;
                    break;
                default:
                    iparams.vsite.a = 0.3;
                    iparams.vsite.b = 0.4;
                    break;
            }
            mtop_.ffparams.iparams.push_back(iparams);
            mtop_.ffparams.functype.push_back(functionType_);
            iatoms.insert(iatoms.end(), { 0, c_numAtoms });
            for (int i = 0; i < NRAL(functionType_) - 1; i++)
            {
                iatoms.push_back(i);
            }
        }
        ilists_[functionType_].iatoms = iatoms;
        mtop_.moltype.push_back(moltype);
        mtop_.molblock.push_back(oneMolblock());
        mtop_.natoms = c_numAtoms + 1;

        // We don't call gmx_omp_nthreads_init(), so we set the number of threads here
        gmx_omp_nthreads_set(ModuleMultiThread::VirtualSite, 1);
        vsiteHandler_ = makeVirtualSitesHandler(mtop_, nullptr, pbcType_, {});
        std::vector<ParticleType> particleTypes(c_numAtoms + 1, ParticleType::Atom);
        particleTypes[c_numAtoms] = ParticleType::VSite;
        vsiteHandler_->setVirtualSites(ilists_, c_numAtoms + 1, c_numAtoms + 1, particleTypes);

        // Construct the virtual site for the whole molecule. Its initial position
        // only matters for the periodic image it is put in.
        x_.push_back(x_[0]);
        vsiteHandler_->construct(x_, {}, box_, VSiteOperation::Positions);
        xWhole_ = x_;

        if (atomsInDifferentImages)
        {
            x_[1] += RVec(box_[XX]);
            x_[2] -= RVec(box_[YY]);
            x_[3] += RVec(box_[ZZ]);
            // A virtual site of type 1 coincides with its constructing atom and
            // spreading relies on it being in the same periodic image.
            if (functionType_ != InteractionFunction::VirtualSite1)
            {
                x_[c_numAtoms] += RVec(box_[XX]) - RVec(box_[ZZ]);
            }
        }
    }

    //! Returns the position of the virtual site for the given atom coordinates and box
    RVec vsitePosition(ArrayRef<const RVec> xAtoms, const matrix box) const
    {
        // The initial position of the vsite determines the periodic image it is put in
        std::vector<RVec> x(xAtoms.begin(), xAtoms.end());
        x.push_back(x_[c_numAtoms]);
        vsiteHandler_->construct(x, {}, box, VSiteOperation::Positions);
        return x[c_numAtoms];
    }

    //! Output of spreading the force on the virtual site
    struct SpreadOutput
    {
        //! The forces on the atoms
        std::vector<RVec> forces;
        //! The shift forces
        std::vector<RVec> shiftForces;
        //! The virial contribution
        matrix virial = { { 0 } };
    };

    //! Spreads the force on the virtual site and returns the result
    SpreadOutput spreadForce(const VirtualSitesHandler::VirialHandling virialHandling) const
    {
        SpreadOutput output;
        output.forces.resize(c_numAtoms + 1, { 0.0_real, 0.0_real, 0.0_real });
        output.forces[c_numAtoms] = vsiteForce_;
        output.shiftForces.resize(c_numShiftVectors, { 0.0_real, 0.0_real, 0.0_real });
        t_nrnb nrnb;
        vsiteHandler_->spreadForces(
                x_, output.forces, virialHandling, output.shiftForces, output.virial, &nrnb, box_, nullptr);
        output.forces.resize(c_numAtoms);
        return output;
    }

protected:
    //! The number of atoms, excluding the virtual site
    static constexpr int c_numAtoms = 4;

    InteractionFunction                                         functionType_;
    PbcType                                                     pbcType_;
    matrix                                                      box_;
    gmx_mtop_t                                                  mtop_;
    gmx::EnumerationArray<InteractionFunction, InteractionList> ilists_;
    std::unique_ptr<VirtualSitesHandler>                        vsiteHandler_;

    //! Coordinates of the atoms, followed by that of the virtual site
    std::vector<RVec> x_ = { { 1.0, 1.1, 1.2 }, { 1.12, 1.07, 1.25 }, { 0.93, 1.19, 1.28 }, { 1.03, 1.01, 1.08 } };
    //! The coordinates with all atoms in the same periodic image
    std::vector<RVec> xWhole_;
    //! The force on the virtual site
    RVec vsiteForce_ = { 1.3, -0.7, 2.1 };
};

/*! \brief Checks that spreading is the transpose of the derivative of construction
 *
 * For a potential that is linear in the position of the virtual site,
 * i.e. a constant force on the virtual site, the spread forces should
 * be minus the derivative of that potential with respect to the positions
 * of the constructing atoms. With PBC, the forces and shift forces should
 * in addition match its derivative with respect to strain.
 */
TEST_P(VSiteDerivativesTest, SpreadForcesMatchConstruction)
{
    if (!c_canTestWithFiniteDifferences)
    {
        GTEST_SKIP() << "Numerical derivatives are only accurate enough in double precision";
    }

    auto energy = [&](ArrayRef<const RVec> xAtoms, const matrix box) -> double
    { return -iprod(vsiteForce_, vsitePosition(xAtoms, box)); };

    const SpreadOutput output = spreadForce(VirtualSitesHandler::VirialHandling::Pbc);

    const ArrayRef<const RVec> xAtoms = constArrayRefFromArray(x_.data(), c_numAtoms);

    checkForcesMatchEnergy(
            xAtoms, output.forces, [&](ArrayRef<const RVec> xTrial) { return energy(xTrial, box_); });

    checkVirialMatchesEnergy(
            xAtoms, box_, numPbcDimensions(pbcType_), output.forces, output.shiftForces, energy);
}

/*! \brief Checks the virial correction for non-linear constructions
 *
 * When the virial has been computed from the force on the virtual site,
 * as if it were an atom, spreading has to replace that contribution by
 * those of the forces on the constructing atoms.
 */
TEST_P(VSiteDerivativesTest, NonLinearVirialCorrectionIsCorrect)
{
    const SpreadOutput output = spreadForce(VirtualSitesHandler::VirialHandling::NonLinear);

    const FloatingPointTolerance tolerance =
            relativeToleranceAsFloatingPoint(1.0, GMX_DOUBLE ? 1e-12 : 1e-5);
    for (int d1 = 0; d1 < DIM; d1++)
    {
        for (int d2 = 0; d2 < DIM; d2++)
        {
            real expectedVirial = 0.5 * xWhole_[c_numAtoms][d1] * vsiteForce_[d2];
            for (int atom = 0; atom < c_numAtoms; atom++)
            {
                expectedVirial -= 0.5 * xWhole_[atom][d1] * output.forces[atom][d2];
            }
            EXPECT_REAL_EQ_TOL(output.virial[d1][d2], expectedVirial, tolerance)
                    << "for virial element " << d1 << "," << d2;
        }
    }
}

//! Returns a name for a test with parameters \c info
std::string nameOfTest(const ::testing::TestParamInfo<VSiteDerivativesTestParameters>& info)
{
    std::string name = formatString("%s_pbc_%s_%s",
                                    interaction_function[std::get<0>(info.param)].name,
                                    c_pbcTypeNames[std::get<1>(info.param)].c_str(),
                                    std::get<2>(info.param) ? "DifferentImages" : "SameImage");
    std::replace_if(name.begin(), name.end(), [](const char c) { return !std::isalnum(c); }, '_');
    return name;
}

INSTANTIATE_TEST_SUITE_P(WithoutPbc,
                         VSiteDerivativesTest,
                         ::testing::Combine(::testing::ValuesIn(vSiteFunctionTypes),
                                            ::testing::Values(PbcType::No),
                                            ::testing::Values(false)),
                         nameOfTest);

INSTANTIATE_TEST_SUITE_P(WithPbc,
                         VSiteDerivativesTest,
                         ::testing::Combine(::testing::ValuesIn(vSiteFunctionTypes),
                                            ::testing::Values(PbcType::Xyz),
                                            ::testing::Bool()),
                         nameOfTest);

} // namespace
} // namespace test
} // namespace gmx
