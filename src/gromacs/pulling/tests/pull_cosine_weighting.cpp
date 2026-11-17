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
 * Tests COM pulling with a cosine-weighted pull group
 *
 * \author Kevin Boyd <kevin44boyd@gmail.com>
 * \ingroup module_pulling
 */
#include "gmxpre.h"

#include <cmath>

#include <numeric>
#include <vector>

#include <gtest/gtest.h>

#include "gromacs/domdec/localatomsetmanager.h"
#include "gromacs/mdtypes/forceoutput.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/mdtypes/md_enums.h"
#include "gromacs/mdtypes/pull_params.h"
#include "gromacs/pbcutil/pbc.h"
#include "gromacs/pulling/pull.h"
#include "gromacs/pulling/pull_internal.h"
#include "gromacs/topology/atoms.h"
#include "gromacs/topology/topology.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/gmxassert.h"
#include "gromacs/utility/mpicomm.h"
#include "gromacs/utility/real.h"
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

//! The number of atoms in the pulled group
constexpr int c_numAtomsInPulledGroup = 3;
//! The number of atoms in the slab that is the cosine-weighted reference group
constexpr int c_numAtomsInSlab = 8;
//! The index of the cosine-weighted pull group
constexpr int c_slabGroup = 2;

/*! \brief Test fixture for pulling a group along z relative to a cosine-weighted slab
 *
 * The slab is a group of atoms with different masses that is spread along z,
 * so its atoms get significantly different weights.
 */
class PullCosineWeightingTest : public ::testing::Test
{
public:
    PullCosineWeightingTest()
    {
        GMX_RELEASE_ASSERT(gmx::ssize(x_) == c_numAtomsInPulledGroup + c_numAtomsInSlab
                                   && x_.size() == masses_.size(),
                           "The coordinates and masses should match the group sizes");

        // Set up a topology that only provides the masses
        gmx_moltype_t& moltype = mtop_.moltype.emplace_back();
        init_t_atoms(&moltype.atoms, gmx::ssize(masses_), false);
        for (int i = 0; i < moltype.atoms.nr; i++)
        {
            moltype.atoms.atom[i].m  = masses_[i];
            moltype.atoms.atom[i].mB = masses_[i];
        }
        gmx_molblock_t& molblock = mtop_.molblock.emplace_back();
        molblock.nmol            = 1;
        molblock.type            = 0;
        mtop_.natoms             = moltype.atoms.nr;
        mtop_.finalize();

        // Group 0 is the absolute reference, a PBC atom of -1 selects cosine weighting
        params_.group.resize(3);
        params_.group[1].ind           = { 0, 1, 2 };
        params_.group[1].pbcatom       = 0;
        params_.group[1].pbcatom_input = 0;
        for (int i = 0; i < c_numAtomsInSlab; i++)
        {
            params_.group[c_slabGroup].ind.push_back(c_numAtomsInPulledGroup + i);
        }
        params_.group[c_slabGroup].pbcatom       = -1;
        params_.group[c_slabGroup].pbcatom_input = -1;
        params_.ngroup                           = gmx::ssize(params_.group);

        // An umbrella potential along z with the minimum 0.25 nm from the current value
        t_pull_coord coord;
        coord.eType      = PullingAlgorithm::Umbrella;
        coord.eGeom      = PullGroupGeometry::Direction;
        coord.ngroup     = 2;
        coord.group      = { c_slabGroup, 1 };
        coord.dim        = { 0, 0, 1 };
        coord.vec        = { 0, 0, 1 };
        coord.k          = 900;
        coord.kB         = 900;
        coord.coordIndex = 0;
        params_.coord.push_back(coord);
        params_.ncoord        = 1;
        params_.coord[0].init = evaluate(x_, false).coordValue - 0.25;
    }

    //! The output of a pull evaluation
    struct PullOutput
    {
        //! The pull energy
        real energy = 0;
        //! The value of the pull coordinate
        double coordValue = 0;
        //! The forces on the atoms, when requested
        std::vector<RVec> forces;
        //! The weights of the atoms in the slab
        std::vector<real> slabWeights;
        //! The properties of the slab group that are derived from the weights
        real mwscale = 0, wscale = 0, invtm = 0;
        //! The reference position of the slab
        RVec slabPosition = { 0, 0, 0 };
    };

    //! Computes the pull energy and, when requested, the forces for coordinates \c x
    PullOutput evaluate(ArrayRef<const RVec> x, const bool computeForces) const
    {
        t_inputrec ir;
        ir.pbcType = PbcType::Xyz;

        const MpiComm       mpiComm(MpiComm::SingleRank{});
        LocalAtomSetManager atomSets;
        pull_t* pull = init_pull(nullptr, &params_, &ir, mtop_, mpiComm, nullptr, &atomSets, 0.0_real);

        t_pbc pbc;
        set_pbc(&pbc, ir.pbcType, box_);

        PullOutput output;
        real       dvdlambda = 0;
        output.energy = pull_potential(pull, masses_, pbc, mpiComm, 0.0, 0.0_real, x, &dvdlambda);
        output.coordValue = pull->coord[0].spatialData.value;
        if (computeForces)
        {
            output.forces.resize(x.size(), { 0, 0, 0 });
            ForceWithVirial forceWithVirial(output.forces, false);
            pull_apply_forces(pull, masses_, mpiComm, &forceWithVirial);
        }

        const pull_group_work_t& slab = pull->group[c_slabGroup];
        GMX_RELEASE_ASSERT(slab.epgrppbc == epgrppbcCOS, "The slab should use cosine weighting");
        output.slabWeights  = slab.localWeights;
        output.mwscale      = slab.mwscale;
        output.wscale       = slab.wscale;
        output.invtm        = slab.invtm;
        output.slabPosition = { real(slab.x[XX]), real(slab.x[YY]), real(slab.x[ZZ]) };

        finish_pull(pull);

        return output;
    }

protected:
    /*! \brief Coordinates in nm
     *
     * The first three atoms are the pulled group. The last eight are the
     * slab: a 4 x 2 grid of atoms in the xy-plane that is tilted along z.
     * The atoms of the slab are at different heights, 0.8 to 1.5 nm in a
     * box of 6 nm along z, so their cosine weights differ significantly.
     * The pulled group is above the slab along z, the pull direction.
     */
    const std::vector<RVec> x_ = { { 1.9, 1.7, 2.0 }, { 2.0, 1.55, 2.1 }, { 2.1, 1.4, 2.4 },
                                   { 1.0, 1.2, 0.8 }, { 1.5, 1.2, 0.9 },  { 2.0, 1.2, 1.0 },
                                   { 2.5, 1.2, 1.1 }, { 1.0, 1.9, 1.2 },  { 1.5, 1.9, 1.3 },
                                   { 2.0, 1.9, 1.4 }, { 2.5, 1.9, 1.5 } };
    /*! \brief Masses in amu, in the order of the coordinates
     *
     * All masses differ, so that an error in the mass weighting of
     * the atoms within a group cannot go unnoticed.
     */
    const std::vector<real> masses_ = { 1.0, 7.5, 14.0, 5.5, 12.0, 3.5, 10.0, 16.5, 8.0, 14.5, 6.0 };
    //! Topology, only used for the masses
    gmx_mtop_t mtop_;
    //! The box
    const matrix box_ = { { 5, 0, 0 }, { 0, 5.5, 0 }, { 0, 0, 6 } };
    //! The pull parameters
    pull_params_t params_;
};

TEST_F(PullCosineWeightingTest, ForcesMatchEnergy)
{
    if (!c_canTestWithFiniteDifferences)
    {
        GTEST_SKIP() << "Numerical derivatives of the energy are only accurate enough in double "
                        "precision";
    }

    const PullOutput output = evaluate(x_, true);

    // Check that we are actually testing something
    ASSERT_GT(std::abs(output.energy), 1);

    checkForcesMatchEnergy(x_,
                           output.forces,
                           [&](ArrayRef<const RVec> xTrial)
                           { return evaluate(xTrial, false).energy; });
}

TEST_F(PullCosineWeightingTest, ForcesSumToZero)
{
    const PullOutput output = evaluate(x_, true);

    // Check that we are actually testing something
    const double largestForce = largestNorm(output.forces);
    ASSERT_GT(largestForce, 1);

    const RVec sumOfForces = std::accumulate(output.forces.begin(), output.forces.end(), RVec(0, 0, 0));

    // The sum can only differ from zero by rounding errors of the individual forces
    EXPECT_LE(sumOfForces.norm(), (GMX_DOUBLE ? 1e-12 : 1e-5) * largestForce);
}

/*! \brief Checks the weights and the effective mass of the cosine-weighted group
 *
 * The weights determine the forces, which is covered by the tests above,
 * but also the displacements and the effective mass with constraint pulling.
 */
TEST_F(PullCosineWeightingTest, WeightsAndEffectiveMassAreCorrect)
{
    const PullOutput output = evaluate(x_, false);

    const FloatingPointTolerance tolerance =
            relativeToleranceAsFloatingPoint(1.0, GMX_DOUBLE ? 1e-12 : 1e-5);

    // The weight is the cosine of the phase difference with the reference position
    const double twoPiOverBox         = 2 * M_PI / box_[ZZ][ZZ];
    double       sumMassWeight        = 0;
    double       sumMassWeightSquared = 0;
    ASSERT_EQ(gmx::ssize(output.slabWeights), c_numAtomsInSlab);
    for (int i = 0; i < c_numAtomsInSlab; i++)
    {
        const int    atom   = c_numAtomsInPulledGroup + i;
        const double weight = std::cos(twoPiOverBox * (x_[atom][ZZ] - output.slabPosition[ZZ]));
        EXPECT_REAL_EQ_TOL(output.slabWeights[i], weight, tolerance);
        sumMassWeight += masses_[atom] * weight;
        sumMassWeightSquared += masses_[atom] * weight * weight;
    }
    EXPECT_REAL_EQ_TOL(output.mwscale, 1 / sumMassWeight, tolerance);
    EXPECT_REAL_EQ_TOL(output.wscale, sumMassWeight / sumMassWeightSquared, tolerance);
    EXPECT_REAL_EQ_TOL(output.invtm * sumMassWeight, sumMassWeightSquared / sumMassWeight, tolerance);
}

} // namespace
} // namespace test
} // namespace gmx
