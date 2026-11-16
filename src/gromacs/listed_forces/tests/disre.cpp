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
 * Tests that forces and virial of distance restraints are consistent
 * with the energy
 *
 * \author Kevin Boyd <kevin44boyd@gmail.com>
 * \ingroup module_listed_forces
 */
#include "gmxpre.h"

#include "gromacs/listed_forces/disre.h"

#include <cctype>

#include <algorithm>
#include <string>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "gromacs/mdtypes/fcdata.h"
#include "gromacs/mdtypes/md_enums.h"
#include "gromacs/pbcutil/ishift.h"
#include "gromacs/pbcutil/pbc.h"
#include "gromacs/topology/idef.h"
#include "gromacs/utility/arrayref.h"
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

//! Test parameters: the PBC type and the upper bound up1 of the restraints
using DistanceRestraintsTestParameters = std::tuple<PbcType, real>;

/*! \brief Test fixture for distance restraints without time or ensemble averaging
 *
 * Uses two restraints, one with a single atom pair and one with three
 * atom pairs. Pairs within a restraint use conservative weighting,
 * as with equal weighting the force is not the derivative of the potential.
 */
class DistanceRestraintsTest : public ::testing::TestWithParam<DistanceRestraintsTestParameters>
{
public:
    DistanceRestraintsTest()
    {
        real up1;
        std::tie(pbcType_, up1) = GetParam();

        clear_mat(box_);
        box_[XX][XX] = 1.5;
        box_[YY][YY] = 1.6;
        box_[ZZ][ZZ] = 1.7;

        // One restraint with a single pair and one with three pairs
        const std::vector<int> numPairsPerRestraint = { 1, 3 };
        for (int restraint = 0; restraint < gmx::ssize(numPairsPerRestraint); restraint++)
        {
            t_iparams iparams;
            iparams.disres.label = restraint;
            iparams.disres.type  = 1;
            iparams.disres.npair = numPairsPerRestraint[restraint];
            iparams.disres.low   = up1 - 0.05;
            iparams.disres.up1   = up1;
            iparams.disres.up2   = up1 + 0.03;
            iparams.disres.kfac  = 1 + 0.5 * restraint;
            iparams_.push_back(iparams);
        }
        // The single pair, followed by the three pairs of the second restraint
        iatoms_ = { 0, 0, 1, 1, 2, 3, 1, 2, 4, 1, 3, 5 };

        const int numPairs = gmx::ssize(iatoms_) / 3;
        rt_.resize(numPairs);
        rm3tav_.resize(numPairs);
        // Three consecutive arrays, as the kernels expect the last two to be consecutive
        restraintData_.resize(3 * iparams_.size());

        disresData_.dr_weighting    = DistanceRestraintWeighting::Conservative;
        disresData_.dr_bMixed       = false;
        disresData_.dr_fc           = 800;
        disresData_.dr_tau          = 0;
        disresData_.ETerm           = 0;
        disresData_.ETerm1          = 1;
        disresData_.exp_min_t_tau   = 0;
        disresData_.nres            = iparams_.size();
        disresData_.npair           = numPairs;
        disresData_.type_min        = 0;
        disresData_.sumviol         = 0;
        disresData_.rt              = rt_.data();
        disresData_.rm3tav          = rm3tav_.data();
        disresData_.Rtl_6           = restraintData_.data();
        disresData_.Rt_6            = restraintData_.data() + iparams_.size();
        disresData_.Rtav_6          = restraintData_.data() + 2 * iparams_.size();
        disresData_.nsystems        = 1;
        disresData_.forceatomsStart = nullptr;
    }

    //! Output of the distance restraint kernel
    struct Output
    {
        //! The potential energy
        real energy = 0;
        //! The forces
        std::vector<RVec> forces;
        //! The shift forces
        std::vector<RVec> shiftForces;
    };

    //! Computes the restraint energy and forces for coordinates \c x and \c box
    Output computeRestraints(ArrayRef<const RVec> x, const matrix box)
    {
        t_pbc pbc;
        set_pbc(&pbc, pbcType_, box);
        const t_pbc* pbcPointer = (pbcType_ == PbcType::No ? nullptr : &pbc);

        calc_disres_R_6(
                nullptr, nullptr, iatoms_.size(), iatoms_.data(), as_rvec_array(x.data()), pbcPointer, &disresData_, nullptr);

        // We cannot store C-style array rvec4 in an std::vector, so we use real and reinterpret
        std::vector<real> forcesStorage(x.size() * 4, 0.0_real);
        rvec4*            forces = reinterpret_cast<rvec4*>(forcesStorage.data());

        Output output;
        output.shiftForces.resize(c_numShiftVectors, { 0.0_real, 0.0_real, 0.0_real });
        real dvdlambda = 0;
        output.energy  = ta_disres(iatoms_.size(),
                                  iatoms_.data(),
                                  iparams_.data(),
                                  as_rvec_array(x.data()),
                                  forces,
                                  as_rvec_array(output.shiftForces.data()),
                                  pbcPointer,
                                  0,
                                  &dvdlambda,
                                   {},
                                  nullptr,
                                  &disresData_,
                                  nullptr,
                                  nullptr);
        for (Index i = 0; i < x.ssize(); i++)
        {
            output.forces.emplace_back(forces[i][XX], forces[i][YY], forces[i][ZZ]);
        }

        return output;
    }

protected:
    PbcType                pbcType_;
    matrix                 box_;
    std::vector<t_iparams> iparams_;
    std::vector<int>       iatoms_;
    t_disresdata           disresData_;
    std::vector<real>      rt_;
    std::vector<real>      rm3tav_;
    std::vector<real>      restraintData_;

    /*! \brief Coordinates
     *
     * The distance for the first restraint is 0.3 nm and the effective
     * distance for the second restraint is 0.366 nm. The last two atoms
     * are in a different periodic image than their partners, which
     * only matters with PBC. */
    std::vector<RVec> x_ = { { 0.3, 0.4, 0.5 }, { 0.5, 0.6, 0.6 },       { 0.9, 0.3, 0.2 },
                             { 1.0, 0.8, 0.7 }, { 1.2, 0.5 - 1.6, 0.3 }, { 0.8 + 1.5, 0.8, 1.2 } };
};

TEST_P(DistanceRestraintsTest, DerivativesMatchEnergy)
{
    if (!c_canTestWithFiniteDifferences)
    {
        GTEST_SKIP() << "Numerical derivatives of the energy are only accurate enough in double "
                        "precision";
    }

    const Output output = computeRestraints(x_, box_);

    checkForcesMatchEnergy(x_,
                           output.forces,
                           [&](ArrayRef<const RVec> xTrial)
                           { return computeRestraints(xTrial, box_).energy; });

    checkVirialMatchesEnergy(x_,
                             box_,
                             numPbcDimensions(pbcType_),
                             output.forces,
                             output.shiftForces,
                             [&](ArrayRef<const RVec> xTrial, const matrix boxTrial)
                             { return computeRestraints(xTrial, boxTrial).energy; });
}

//! Returns a name for a test with parameters \c info
std::string nameOfTest(const ::testing::TestParamInfo<DistanceRestraintsTestParameters>& info)
{
    std::string name = formatString(
            "pbc_%s_up1_%g", c_pbcTypeNames[std::get<0>(info.param)].c_str(), std::get<1>(info.param));
    std::replace_if(name.begin(), name.end(), [](const char c) { return !std::isalnum(c); }, '_');
    return name;
}

/* The upper bounds are chosen such that the restraints are, in different combinations,
 * below the lower bound, not violated and in the quadratic and linear parts of the potential.
 */
INSTANTIATE_TEST_SUITE_P(Restraints,
                         DistanceRestraintsTest,
                         ::testing::Combine(::testing::Values(PbcType::No, PbcType::Xyz),
                                            ::testing::Values(0.45, 0.345, 0.29, 0.2)),
                         nameOfTest);

} // namespace
} // namespace test
} // namespace gmx
