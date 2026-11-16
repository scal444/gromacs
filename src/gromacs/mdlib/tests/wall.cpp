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
 * Tests that forces, virial and dV/dlambda of wall interactions are
 * consistent with the energy
 *
 * \author Kevin Boyd <kevin44boyd@gmail.com>
 * \ingroup module_mdlib
 */
#include "gmxpre.h"

#include "gromacs/mdlib/wall.h"

#include <cctype>
#include <cmath>

#include <algorithm>
#include <numeric>
#include <string>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "gromacs/gmxlib/nrnb.h"
#include "gromacs/gpu_utils/hostallocator.h"
#include "gromacs/mdtypes/forceoutput.h"
#include "gromacs/mdtypes/forcerec.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/mdtypes/md_enums.h"
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

//! Test parameters: wall type, whether atoms are perturbed, wall-r-linpot and lambda
using WallTestParameters = std::tuple<WallType, bool, real, real>;

//! Test fixture for wall interactions
class WallTest : public ::testing::TestWithParam<WallTestParameters>
{
public:
    WallTest()
    {
        std::tie(ir_.wall_type, havePerturbedAtoms_, ir_.wall_r_linpot, lambda_) = GetParam();

        // Two walls, each with their own atom type and energy group
        ir_.nwall            = 2;
        ir_.wall_atomtype[0] = 2;
        ir_.wall_atomtype[1] = 3;
        ir_.wall_density[0]  = 50;
        ir_.wall_density[1]  = 70;
        ir_.opts.ngener      = c_numEnergyGroups;

        // Note that the parameter matrix includes derivative prefactors
        fr_.ntype = c_numAtomTypes;
        fr_.nbfp.resize(2 * c_numAtomTypes * c_numAtomTypes, 0.0_real);
        for (int atomType = 0; atomType < 2; atomType++)
        {
            for (int wall = 0; wall < 2; wall++)
            {
                const int index     = 2 * (c_numAtomTypes * ir_.wall_atomtype[wall] + atomType);
                fr_.nbfp[index]     = 6 * 0.002 * (1 + atomType) * (1 + 0.3 * wall);
                fr_.nbfp[index + 1] = 12 * 2e-6 * (1 + 2 * atomType) * (1 + 0.2 * wall);
            }
        }
        energyGroupFlags_.resize(c_numEnergyGroups * c_numEnergyGroups, 0);
        fr_.egp_flags = energyGroupFlags_.data();

        clear_mat(box_);
        box_[XX][XX] = 3;
        box_[YY][YY] = 3;
        box_[ZZ][ZZ] = 2.5;
    }

    //! Output of the wall kernel
    struct Output
    {
        //! The total wall energy
        double energy = 0;
        //! The derivative of the energy with respect to lambda
        real dvdlambda = 0;
        //! The forces
        std::vector<RVec> forces;
        //! The z-component of the virial
        real virialZ = 0;
    };

    //! Calls the wall kernel for coordinates \c x with z box length \c boxZ and \c lambda
    Output runKernel(ArrayRef<const RVec> x, const real boxZ, const real lambda)
    {
        matrix box;
        copy_mat(box_, box);
        box[ZZ][ZZ] = boxZ;

        Output output;
        output.forces.resize(x.size(), { 0.0_real, 0.0_real, 0.0_real });
        ForceWithVirial   forceWithVirial(output.forces, true);
        std::vector<real> energies(c_numEnergyGroups * c_numEnergyGroups, 0.0_real);
        t_nrnb            nrnb;

        output.dvdlambda = do_walls(ir_,
                                    fr_,
                                    box,
                                    typeA_,
                                    typeB_,
                                    energyGroups_,
                                    x.size(),
                                    havePerturbedAtoms_ ? 2 : 0,
                                    x,
                                    &forceWithVirial,
                                    lambda,
                                    energies,
                                    &nrnb);
        output.energy    = std::accumulate(energies.begin(), energies.end(), 0.0);
        output.virialZ   = forceWithVirial.getVirial()[ZZ][ZZ];

        return output;
    }

protected:
    //! The number of atom types, two for atoms and two for walls
    static constexpr int c_numAtomTypes = 4;
    //! The number of energy groups, one for atoms and two for walls
    static constexpr int c_numEnergyGroups = 3;

    t_inputrec       ir_;
    t_forcerec       fr_{ HostAllocationPolicy{} };
    std::vector<int> energyGroupFlags_;
    matrix           box_;
    bool             havePerturbedAtoms_;
    real             lambda_;

    //! Atoms at different distances from both walls, some closer than wall-r-linpot
    std::vector<RVec> x_ = { { 0.5, 1.0, 0.21 },
                             { 2.0, 0.3, 0.34 },
                             { 1.0, 2.0, 0.5 },
                             { 1.5, 1.5, 2.05 },
                             { 0.2, 2.5, 2.27 } };
    //! Atom types in state A
    std::vector<int> typeA_ = { 0, 1, 0, 1, 0 };
    //! Atom types in state B
    std::vector<int> typeB_ = { 1, 1, 0, 0, 1 };
    //! Energy group of each atom
    std::vector<unsigned short> energyGroups_ = { 0, 0, 0, 0, 0 };
};

TEST_P(WallTest, DerivativesMatchEnergy)
{
    if (!c_canTestWithFiniteDifferences)
    {
        GTEST_SKIP() << "Numerical derivatives of the energy are only accurate enough in double "
                        "precision";
    }

    const Output output = runKernel(x_, box_[ZZ][ZZ], lambda_);

    checkForcesMatchEnergy(x_,
                           output.forces,
                           [&](ArrayRef<const RVec> xTrial)
                           { return runKernel(xTrial, box_[ZZ][ZZ], lambda_).energy; });

    checkDvdlambdaMatchesEnergy(output.dvdlambda,
                                [&](const double lambdaChange) {
                                    return runKernel(x_, box_[ZZ][ZZ], lambda_ + lambdaChange).energy;
                                });

    // Scaling the coordinates and box along z changes the energy
    // by twice the virial.
    const FiniteDifferenceSettings settings;
    const real                     numericalVirial =
            0.5
            * finiteDifferenceDerivative(
                    [&](const double strain)
                    {
                        std::vector<RVec> xTrial = x_;
                        for (RVec& xAtom : xTrial)
                        {
                            xAtom[ZZ] *= 1 + strain;
                        }
                        return runKernel(xTrial, box_[ZZ][ZZ] * (1 + strain), lambda_).energy;
                    },
                    settings.strainStep);
    EXPECT_REAL_EQ_TOL(
            output.virialZ,
            numericalVirial,
            finiteDifferenceTolerance(std::max(1.0, std::abs(static_cast<double>(output.virialZ))),
                                      settings.relativeTolerance,
                                      std::abs(output.energy),
                                      settings.strainStep))
            << "Virial does not match the derivative of the energy with respect to scaling z";
}

//! Returns a name for a test with parameters \c info
std::string nameOfTest(const ::testing::TestParamInfo<WallTestParameters>& info)
{
    std::string name = formatString("%s_%s_Linpot%g_Lambda%g",
                                    enumValueToString(std::get<0>(info.param)),
                                    std::get<1>(info.param) ? "Perturbed" : "Unperturbed",
                                    std::get<2>(info.param),
                                    std::get<3>(info.param));
    std::replace_if(name.begin(), name.end(), [](const char c) { return !std::isalnum(c); }, '_');
    return name;
}

INSTANTIATE_TEST_SUITE_P(Walls,
                         WallTest,
                         ::testing::Combine(::testing::Values(WallType::NineThree,
                                                              WallType::TenFour,
                                                              WallType::TwelveSix),
                                            ::testing::Bool(),
                                            ::testing::Values(-1.0, 0.3),
                                            ::testing::Values(0.0, 0.4, 1.0)),
                         nameOfTest);

} // namespace
} // namespace test
} // namespace gmx
