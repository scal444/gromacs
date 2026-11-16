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
 * Tests that the forces of the enforced rotation potentials are
 * consistent with the energy
 *
 * \author Kevin Boyd <kevin44boyd@gmail.com>
 * \ingroup module_pulling
 */
#include "gmxpre.h"

#include "gromacs/pulling/pull_rotation.h"

#include <cctype>
#include <cmath>
#include <cstdint>

#include <algorithm>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "gromacs/commandline/filenm.h"
#include "gromacs/domdec/localatomsetmanager.h"
#include "gromacs/fileio/filetypes.h"
#include "gromacs/fileio/oenv.h"
#include "gromacs/math/units.h"
#include "gromacs/mdrunutility/handlerestart.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/mdtypes/md_enums.h"
#include "gromacs/mdtypes/mdrunoptions.h"
#include "gromacs/mdtypes/state.h"
#include "gromacs/pbcutil/pbc.h"
#include "gromacs/topology/atoms.h"
#include "gromacs/topology/topology.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/mpicomm.h"
#include "gromacs/utility/real.h"
#include "gromacs/utility/smalloc.h"
#include "gromacs/utility/stringutil.h"
#include "gromacs/utility/vec.h"
#include "gromacs/utility/vectypes.h"

#include "testutils/finitedifference.h"
#include "testutils/testasserts.h"
#include "testutils/testfilemanager.h"

namespace gmx
{
namespace test
{
namespace
{

//! Returns \c x rotated by \c angle (in degrees) around the axis along unit vector \c axis through \c center
RVec rotateAroundAxis(const RVec& x, const RVec& axis, const RVec& center, const real angle)
{
    const real cosAngle = std::cos(angle * gmx::c_deg2Rad);
    const real sinAngle = std::sin(angle * gmx::c_deg2Rad);
    const RVec r        = x - center;
    // Rodrigues' rotation formula
    return center + r * cosAngle + axis.cross(r) * sinAngle + axis * (axis.dot(r) * (1 - cosAngle));
}

//! Test parameters: rotation potential type and whether mass weighting is used
using EnforcedRotationTestParameters = std::tuple<EnforcedRotationGroupType, bool>;

/*! \brief Test fixture for enforced rotation
 *
 * Sets up a rotation group that is a subset of the atoms of the system,
 * with the reference rotated to an angle that differs from the angle
 * of the, also distorted, group.
 */
class EnforcedRotationTest : public ::testing::TestWithParam<EnforcedRotationTestParameters>
{
public:
    EnforcedRotationTest()
    {
        const auto [rotationType, useMassWeighting] = GetParam();
        useMassWeighting_                           = useMassWeighting;

        const int numAtoms = gmx::ssize(c_referencePositions);

        clear_mat(box_);
        box_[XX][XX] = 10;
        box_[YY][YY] = 11;
        box_[ZZ][ZZ] = 12;

        // Each atom is a separate molecule, so no molecules need to be made whole
        mtop_.moltype.resize(numAtoms);
        mtop_.molblock.resize(numAtoms);
        for (int atom = 0; atom < numAtoms; atom++)
        {
            init_t_atoms(&mtop_.moltype[atom].atoms, 1, false);
            mtop_.moltype[atom].atoms.atom[0].m = c_masses[atom];
            mtop_.molblock[atom].type           = atom;
            mtop_.molblock[atom].nmol           = 1;
        }
        mtop_.natoms = numAtoms;
        mtop_.finalize();

        // The rotation axis, which should be normalized
        const RVec axis = RVec(0.3, -0.5, 1.0).unitVector();

        // The current positions are the reference positions rotated around
        // an axis through a point that is neither the pivot nor the center
        // of the group, translated and distorted.
        for (int atom = 0; atom < numAtoms; atom++)
        {
            const RVec xRotated = rotateAroundAxis(
                    c_referencePositions[atom], axis, { 5.1, 4.95, 5.05 }, c_actualAngle);
            x_.push_back(xRotated + RVec(0.03, -0.02, 0.04) + c_distortions[atom]);
        }

        ir_.pbcType   = PbcType::Xyz;
        ir_.init_t    = 0;
        ir_.init_step = 0;
        ir_.delta_t   = 0.002;
        ir_.bRot      = true;
        ir_.rot       = std::make_unique<t_rot>();
        // Output is avoided by not calling the module at output steps
        ir_.rot->nstrout = 100;
        ir_.rot->nstsout = 100;
        ir_.rot->grp.resize(1);
        t_rotgrp& group = ir_.rot->grp[0];
        group.eType     = rotationType;
        group.bMassW    = useMassWeighting;
        group.nat       = gmx::ssize(c_groupAtoms);
        // The index is freed by the destructor of t_inputrec
        snew(group.ind, group.nat);
        for (int i = 0; i < group.nat; i++)
        {
            group.ind[i] = c_groupAtoms[i];
            group.x_ref_original.push_back(c_referencePositions[c_groupAtoms[i]]);
        }
        copy_rvec(axis, group.inputVec);
        group.rate      = 1;
        group.k         = 500;
        group.pivot[XX] = 4.9;
        group.pivot[YY] = 5.1;
        group.pivot[ZZ] = 5.0;
        group.eFittype  = RotationGroupFitting::Rmsd;
        group.slab_dist = 0.5;
        // The potential and forces of the flexible types are discontinuous
        // where the Gaussian weight of an atom in a slab passes this cut-off,
        // so we use a value that makes those discontinuities negligible.
        group.min_gaussian = 1e-12;
        group.eps          = 1e-3;

        t_state globalState;
        globalState.changeNumAtoms(numAtoms);
        globalState.addEntry(StateEntry::X);
        std::copy(x_.begin(), x_.end(), globalState.x.begin());
        copy_mat(box_, globalState.box);

        std::vector<t_filenm> fileNames = {
            { efXVG, "-ro", "rotation", ffOPTWR, { fileManager_.getTemporaryFilePath("rotation.xvg").string() } },
            { efLOG,
              "-ra",
              "rotangles",
              ffOPTWR,
              { fileManager_.getTemporaryFilePath("rotangles.log").string() } },
            { efLOG, "-rs", "rotslabs", ffOPTWR, { fileManager_.getTemporaryFilePath("rotslabs.log").string() } },
            { efLOG,
              "-rt",
              "rottorque",
              ffOPTWR,
              { fileManager_.getTemporaryFilePath("rottorque.log").string() } }
        };

        output_env_init_default(&oenv_);

        enforcedRotation_ = init_rot(nullptr,
                                     &ir_,
                                     gmx::ssize(fileNames),
                                     fileNames.data(),
                                     mpiComm_,
                                     &atomSets_,
                                     &globalState,
                                     mtop_,
                                     oenv_,
                                     MdrunOptions(),
                                     StartingBehavior::NewSimulation);
    }

    ~EnforcedRotationTest() override
    {
        // The module closes its output files when it is destroyed
        enforcedRotation_.reset();
        output_env_done(oenv_);
    }

    //! Output of the enforced rotation module
    struct Output
    {
        //! The energy
        double energy = 0;
        //! The forces
        std::vector<RVec> forces;
    };

    //! Computes the energy and forces for coordinates \c x
    Output compute(ArrayRef<const RVec> x)
    {
        Output output;
        output.forces.resize(x.size(), { 0.0_real, 0.0_real, 0.0_real });
        // The reference is rotated by the rate, which is 1 degree/ps, times the time
        const real    time = c_referenceAngle;
        const int64_t step = 1;
        do_rotation(mpiComm_, nullptr, enforcedRotation_->getLegacyEnfrot(), box_, x, time, step, true);
        output.energy = add_rot_forces(
                enforcedRotation_->getLegacyEnfrot(), output.forces, mpiComm_, step, time);
        return output;
    }

protected:
    //! The angle over which the reference is rotated, in degrees
    static constexpr real c_referenceAngle = 40;
    //! The angle over which the group is rotated with respect to the initial reference, in degrees
    static constexpr real c_actualAngle = 31;

    //! Initial reference positions of all atoms in the system, extending about 2 nm along the rotation axis
    const std::vector<RVec> c_referencePositions = {
        { 4.62, 5.31, 4.11 }, { 5.35, 4.71, 4.23 }, { 4.83, 4.58, 4.47 }, { 5.44, 5.29, 4.62 },
        { 4.55, 4.87, 4.79 }, { 5.12, 5.48, 4.95 }, { 5.27, 4.52, 5.08 }, { 4.69, 5.22, 5.21 },
        { 5.38, 5.07, 5.42 }, { 4.91, 4.61, 5.56 }, { 4.74, 5.39, 5.73 }, { 5.21, 4.83, 5.94 }
    };
    //! Distortions of the group with respect to a rigid rotation of the reference
    const std::vector<RVec> c_distortions = { { 0.04, -0.02, 0.01 },  { -0.03, 0.05, 0.02 },
                                              { 0.02, 0.03, -0.04 },  { -0.05, -0.01, 0.03 },
                                              { 0.01, 0.04, 0.02 },   { 0.03, -0.04, -0.02 },
                                              { -0.02, 0.02, 0.05 },  { 0.05, 0.01, -0.03 },
                                              { -0.04, -0.03, 0.01 }, { 0.02, -0.05, 0.04 },
                                              { -0.01, 0.03, -0.05 }, { 0.03, 0.02, 0.02 } };
    //! Masses of all atoms in the system
    const std::vector<real> c_masses = { 12, 1, 16, 14, 12, 32, 1, 12, 16, 14, 12, 35 };
    //! The atoms that are part of the rotation group
    const std::vector<int> c_groupAtoms = { 0, 1, 2, 4, 5, 6, 7, 8, 10, 11 };

    TestFileManager                   fileManager_;
    MpiComm                           mpiComm_{ MpiComm::SingleRank{} };
    LocalAtomSetManager               atomSets_;
    gmx_mtop_t                        mtop_;
    t_inputrec                        ir_;
    gmx_output_env_t*                 oenv_ = nullptr;
    matrix                            box_;
    bool                              useMassWeighting_;
    std::vector<RVec>                 x_;
    std::unique_ptr<EnforcedRotation> enforcedRotation_;
};

TEST_P(EnforcedRotationTest, ForcesMatchEnergy)
{
    if (!c_canTestWithFiniteDifferences)
    {
        GTEST_SKIP() << "Numerical derivatives of the energy are only accurate enough in double "
                        "precision";
    }

    const EnforcedRotationGroupType rotationType = std::get<0>(GetParam());

    Output output = compute(x_);

    // Check that the test actually exercises the potential
    EXPECT_GT(output.energy, 1.0);

    if (rotationType == EnforcedRotationGroupType::Flext || rotationType == EnforcedRotationGroupType::Flex2t)
    {
        /* The translation-tolerant flexible potentials are functions of
         * the positions relative to the center of the group, x_i - x_c.
         * The forces are, as documented, minus the derivative with respect
         * to x_i - x_c, neglecting that x_c depends on the positions.
         * The exact force is obtained by subtracting the weighted net force:
         * -dV/dx_j = F_j - m_j/M sum_i F_i
         * Applying this to the forces enables an exact check of
         * the terms that are not neglected.
         */
        RVec netForce  = { 0.0_real, 0.0_real, 0.0_real };
        real totalMass = 0;
        for (const int atom : c_groupAtoms)
        {
            netForce += output.forces[atom];
            totalMass += (useMassWeighting_ ? c_masses[atom] : 1.0_real);
        }
        // Without the approximation there would be no net force
        EXPECT_GT(netForce.norm(), 1.0);
        for (const int atom : c_groupAtoms)
        {
            output.forces[atom] -=
                    netForce * ((useMassWeighting_ ? c_masses[atom] : 1.0_real) / totalMass);
        }
    }

    checkForcesMatchEnergy(
            x_, output.forces, [&](ArrayRef<const RVec> xTrial) { return compute(xTrial).energy; });
}

//! Returns a name for a test with parameters \c info
std::string nameOfTest(const ::testing::TestParamInfo<EnforcedRotationTestParameters>& info)
{
    std::string name = formatString("%s_%s",
                                    enumValueToString(std::get<0>(info.param)),
                                    std::get<1>(info.param) ? "MassWeighted" : "NotMassWeighted");
    std::replace_if(name.begin(), name.end(), [](const char c) { return !std::isalnum(c); }, '_');
    return name;
}

INSTANTIATE_TEST_SUITE_P(AllTypes,
                         EnforcedRotationTest,
                         ::testing::Combine(::testing::Values(EnforcedRotationGroupType::Iso,
                                                              EnforcedRotationGroupType::Isopf,
                                                              EnforcedRotationGroupType::Pm,
                                                              EnforcedRotationGroupType::Pmpf,
                                                              EnforcedRotationGroupType::Rm,
                                                              EnforcedRotationGroupType::Rmpf,
                                                              EnforcedRotationGroupType::Rm2,
                                                              EnforcedRotationGroupType::Rm2pf,
                                                              EnforcedRotationGroupType::Flex,
                                                              EnforcedRotationGroupType::Flext,
                                                              EnforcedRotationGroupType::Flex2,
                                                              EnforcedRotationGroupType::Flex2t),
                                            ::testing::Bool()),
                         nameOfTest);

} // namespace
} // namespace test
} // namespace gmx
