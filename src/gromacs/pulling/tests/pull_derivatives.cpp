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
 * Tests that forces, virial and dV/dlambda of COM pulling are
 * consistent with the pull energy
 *
 * \author Kevin Boyd <kevin44boyd@gmail.com>
 * \ingroup module_pulling
 */
#include "gmxpre.h"

#include "config.h"

#include <cctype>
#include <cmath>

#include <algorithm>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "gromacs/domdec/localatomsetmanager.h"
#include "gromacs/math/utilities.h"
#include "gromacs/mdtypes/forceoutput.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/mdtypes/md_enums.h"
#include "gromacs/mdtypes/pull_params.h"
#include "gromacs/pbcutil/pbc.h"
#include "gromacs/pulling/pull.h"
#include "gromacs/pulling/pull_internal.h"
#include "gromacs/topology/topology.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/enumerationhelpers.h"
#include "gromacs/utility/gmxassert.h"
#include "gromacs/utility/mpicomm.h"
#include "gromacs/utility/real.h"
#include "gromacs/utility/smalloc.h"
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

//! The pull setups to test, the potential always acts on the last pull coordinate
enum class PullSetup : int
{
    Distance,
    DistanceXY,
    DistanceAbsoluteReference,
    Direction,
    DirectionCosineWeightedReference,
    DirectionPeriodic,
    DirectionRelative,
    DirectionRelativeWithDihedral,
    CylinderZ,
    CylinderMinusZ,
    CylinderTilted,
    Angle,
    AngleAxis,
    Dihedral,
    TransformationOfDistanceAndAngle,
    TransformationNested,
    Count
};

//! Returns the name of \c pullSetup
const char* enumValueToString(const PullSetup pullSetup)
{
    static constexpr EnumerationArray<PullSetup, const char*> sc_names = {
        "Distance",
        "DistanceXY",
        "DistanceAbsoluteReference",
        "Direction",
        "DirectionCosineWeightedReference",
        "DirectionPeriodic",
        "DirectionRelative",
        "DirectionRelativeWithDihedral",
        "CylinderZ",
        "CylinderMinusZ",
        "CylinderTilted",
        "Angle",
        "AngleAxis",
        "Dihedral",
        "TransformationOfDistanceAndAngle",
        "TransformationNested"
    };
    return sc_names[pullSetup];
}

//! The name of the module that provides the external potential
const char* const c_externalPotentialProvider = "test";

/*! \brief The potentials to apply to the pull coordinate
 *
 * With External the scalar force is passed to the pull code as done by e.g. AWH,
 * and the potential, which is linear in the coordinate, is computed by the test.
 */
enum class PullPotential : int
{
    Umbrella,
    UmbrellaWithRate,
    FlatBottom,
    FlatBottomHigh,
    ConstantForce,
    External,
    Count
};

//! Returns the name of \c pullPotential
const char* enumValueToString(const PullPotential pullPotential)
{
    static constexpr EnumerationArray<PullPotential, const char*> sc_names = {
        "Umbrella", "UmbrellaWithRate", "FlatBottom", "FlatBottomHigh", "ConstantForce", "External"
    };
    return sc_names[pullPotential];
}

//! The periodic setups, with periodicity atoms of pull groups are put in different periodic images
enum class PeriodicSetup : int
{
    NoPbc,
    Rectangular,
    Triclinic,
    Count
};

//! Returns the name of \c periodicSetup
const char* enumValueToString(const PeriodicSetup periodicSetup)
{
    static constexpr EnumerationArray<PeriodicSetup, const char*> sc_names = { "NoPbc",
                                                                               "Rectangular",
                                                                               "Triclinic" };
    return sc_names[periodicSetup];
}

//! Returns whether \c pullSetup uses a transformation coordinate
bool isTransformation(const PullSetup pullSetup)
{
    return pullSetup == PullSetup::TransformationOfDistanceAndAngle
           || pullSetup == PullSetup::TransformationNested;
}

//! The number of atoms in the six small pull groups together
constexpr int c_numAtomsInSmallGroups = 13;
//! The number of atoms in the slab that is used as cylinder and cosine-weighted group
constexpr int c_numAtomsInSlab = 16;
//! The index of the pull group that consists of the slab
constexpr int c_slabGroup = 7;

//! Returns an RVec, avoids narrowing in mixed precision
RVec makeRVec(const double x, const double y, const double z)
{
    return { static_cast<real>(x), static_cast<real>(y), static_cast<real>(z) };
}

//! Returns a unit vector along (x, y, z)
RVec makeUnitVector(const double x, const double y, const double z)
{
    const double invNorm = 1 / std::sqrt(x * x + y * y + z * z);
    return makeRVec(x * invNorm, y * invNorm, z * invNorm);
}

//! Returns coordinates with all atoms of each pull group close together
std::vector<RVec> makeCoordinates()
{
    std::vector<RVec> x = {
        makeRVec(0.95, 1.25, 1.05), makeRVec(1.10, 1.15, 1.20),
        makeRVec(0.90, 1.30, 1.00), // group 1
        makeRVec(1.85, 1.75, 1.95), makeRVec(2.00, 1.60, 2.10),
        makeRVec(1.90, 1.80, 1.90),                             // group 2
        makeRVec(1.35, 2.55, 1.25), makeRVec(1.50, 2.40, 1.40), // group 3
        makeRVec(2.25, 2.15, 2.70), makeRVec(2.40, 2.30, 2.50), // group 4
        makeRVec(2.60, 1.20, 1.60),                             // group 5
        makeRVec(1.15, 2.05, 2.95), makeRVec(1.30, 1.90, 2.80)  // group 6
    };
    GMX_RELEASE_ASSERT(gmx::ssize(x) == c_numAtomsInSmallGroups, "Atom count should match");
    // An irregular slab normal to z below group 2. The spacing is such
    // that with the cylinder radius used here four atoms get a large
    // weight, eight a small weight and the four corners zero weight.
    for (int i = 0; i < c_numAtomsInSlab; i++)
    {
        x.push_back(makeRVec(1.9 + 0.55 * (i % 4 - 1.5) + 0.06 * std::sin(1.7 * i + 0.3),
                             1.7 + 0.55 * (i / 4 - 1.5) + 0.06 * std::cos(2.3 * i + 1.1),
                             0.8 + 0.2 * std::sin(0.9 * i + 0.5)));
    }
    return x;
}

//! Returns non-uniform masses
std::vector<real> makeMasses()
{
    std::vector<real> masses;
    for (int i = 0; i < c_numAtomsInSmallGroups + c_numAtomsInSlab; i++)
    {
        masses.push_back(1 + 3 * ((7 * i) % 5) + 0.5 * i);
    }
    return masses;
}

//! Returns a pull group
t_pull_group makeGroup(const std::vector<int>& atoms, const int pbcAtom, const std::vector<real>& weights = {})
{
    t_pull_group group;
    group.ind           = atoms;
    group.weight        = weights;
    group.pbcatom       = pbcAtom;
    group.pbcatom_input = pbcAtom;
    return group;
}

//! Adds a pull coordinate to \c params
void addCoord(pull_params_t*          params,
              const PullGroupGeometry geometry,
              const std::vector<int>& groups,
              const IVec&             dim,
              const RVec&             vec        = { 0, 0, 0 },
              const std::string&      expression = "")
{
    t_pull_coord coord;
    coord.eType      = PullingAlgorithm::Umbrella;
    coord.eGeom      = geometry;
    coord.expression = expression;
    coord.ngroup     = gmx::ssize(groups);
    std::copy(groups.begin(), groups.end(), coord.group.begin());
    coord.dim        = dim;
    coord.vec        = vec;
    coord.coordIndex = gmx::ssize(params->coord);
    params->coord.push_back(coord);
    params->ncoord = gmx::ssize(params->coord);
}

/*! \brief Returns pull parameters for \c pullSetup
 *
 * The caller should set up the potential for the last coordinate.
 * The other coordinates have a force constant of zero, unless noted.
 */
pull_params_t makePullParameters(const PullSetup pullSetup)
{
    pull_params_t params;
    params.cylinder_r = 1.0;

    std::vector<int> slabAtoms(c_numAtomsInSlab);
    for (int i = 0; i < c_numAtomsInSlab; i++)
    {
        slabAtoms[i] = c_numAtomsInSmallGroups + i;
    }
    // A PBC atom of -1 selects cosine weighting
    const int slabPbcAtom =
            (pullSetup == PullSetup::DirectionCosineWeightedReference ? -1 : slabAtoms[5]);

    // The absolute reference
    params.group.push_back(makeGroup({}, -1));
    params.group.push_back(makeGroup({ 0, 1, 2 }, 0));
    // A group with relative weights
    params.group.push_back(makeGroup({ 3, 4, 5 }, 4, { 0.5, 1.0, 2.0 }));
    params.group.push_back(makeGroup({ 6, 7 }, 6));
    params.group.push_back(makeGroup({ 8, 9 }, 9));
    params.group.push_back(makeGroup({ 10 }, 10));
    params.group.push_back(makeGroup({ 11, 12 }, 11));
    params.group.push_back(makeGroup(slabAtoms, slabPbcAtom));
    GMX_RELEASE_ASSERT(gmx::ssize(params.group) == c_slabGroup + 1, "Group count should match");
    params.ngroup = gmx::ssize(params.group);

    const IVec allDims  = { 1, 1, 1 };
    const IVec noDims   = { 0, 0, 0 };
    const RVec noVector = { 0, 0, 0 };

    switch (pullSetup)
    {
        case PullSetup::Distance:
            addCoord(&params, PullGroupGeometry::Distance, { 1, 2 }, allDims);
            break;
        case PullSetup::DistanceXY:
            addCoord(&params, PullGroupGeometry::Distance, { 1, 2 }, { 1, 1, 0 });
            break;
        case PullSetup::DistanceAbsoluteReference:
            addCoord(&params, PullGroupGeometry::Distance, { 0, 2 }, allDims);
            params.coord.back().origin = makeRVec(1.2, 2.3, 1.4);
            break;
        case PullSetup::Direction:
            addCoord(&params, PullGroupGeometry::Direction, { 1, 2 }, allDims, makeUnitVector(0.5, -0.3, 0.8));
            break;
        case PullSetup::DirectionCosineWeightedReference:
            addCoord(&params, PullGroupGeometry::Direction, { c_slabGroup, 2 }, { 0, 0, 1 }, { 0, 0, 1 });
            break;
        case PullSetup::DirectionPeriodic:
            addCoord(&params, PullGroupGeometry::DirectionPBC, { 1, 2 }, allDims, makeUnitVector(0.5, -0.3, 0.8));
            break;
        case PullSetup::DirectionRelative:
            addCoord(&params, PullGroupGeometry::DirectionRelative, { 1, 2, 3, 4 }, allDims);
            break;
        case PullSetup::DirectionRelativeWithDihedral:
            // Two active coordinates, the pull code processes the dihedral first,
            // which has forces on more groups than the direction-relative coordinate.
            addCoord(&params, PullGroupGeometry::DirectionRelative, { 1, 2, 3, 4 }, allDims);
            params.coord.back().k    = 500;
            params.coord.back().kB   = 800;
            params.coord.back().init = 0.3;
            addCoord(&params, PullGroupGeometry::Dihedral, { 1, 2, 3, 4, 5, 6 }, allDims);
            break;
        case PullSetup::CylinderZ:
            addCoord(&params, PullGroupGeometry::Cylinder, { c_slabGroup, 2 }, allDims, { 0, 0, 1 });
            break;
        case PullSetup::CylinderMinusZ:
            addCoord(&params, PullGroupGeometry::Cylinder, { c_slabGroup, 2 }, allDims, { 0, 0, -1 });
            break;
        case PullSetup::CylinderTilted:
            addCoord(&params,
                     PullGroupGeometry::Cylinder,
                     { c_slabGroup, 2 },
                     allDims,
                     makeUnitVector(0.25, -0.15, 0.9));
            break;
        case PullSetup::Angle:
            addCoord(&params, PullGroupGeometry::Angle, { 1, 2, 3, 6 }, allDims);
            break;
        case PullSetup::AngleAxis:
            addCoord(&params, PullGroupGeometry::AngleAxis, { 1, 2 }, allDims, makeUnitVector(0.6, 0.0, -0.8));
            break;
        case PullSetup::Dihedral:
            addCoord(&params, PullGroupGeometry::Dihedral, { 1, 2, 3, 4, 5, 6 }, allDims);
            break;
        case PullSetup::TransformationOfDistanceAndAngle:
            addCoord(&params, PullGroupGeometry::Distance, { 1, 2 }, allDims);
            addCoord(&params, PullGroupGeometry::Angle, { 1, 2, 3, 6 }, allDims);
            addCoord(&params,
                     PullGroupGeometry::Transformation,
                     {},
                     noDims,
                     noVector,
                     "x1^2 + 0.5*sin(x2) + 0.1*t");
            break;
        case PullSetup::TransformationNested:
            addCoord(&params, PullGroupGeometry::Distance, { 5, 2 }, allDims);
            addCoord(&params, PullGroupGeometry::Dihedral, { 1, 2, 3, 4, 5, 6 }, allDims);
            addCoord(&params, PullGroupGeometry::Transformation, {}, noDims, noVector, "x1*cos(x2)");
            addCoord(&params,
                     PullGroupGeometry::Transformation,
                     {},
                     noDims,
                     noVector,
                     "x3 + 1/x1 + 0.3*x2");
            break;
        default: GMX_RELEASE_ASSERT(false, "Unhandled pull setup");
    }

    return params;
}

//! Sets up \c mtop as a single molecule with atoms with masses \c masses
void fillTopology(gmx_mtop_t* mtop, ArrayRef<const real> masses)
{
    gmx_moltype_t& moltype = mtop->moltype.emplace_back();
    moltype.atoms.nr       = gmx::ssize(masses);
    snew(moltype.atoms.atom, moltype.atoms.nr);
    for (int i = 0; i < moltype.atoms.nr; i++)
    {
        moltype.atoms.atom[i].m  = masses[i];
        moltype.atoms.atom[i].mB = masses[i];
    }
    gmx_molblock_t& molblock = mtop->molblock.emplace_back();
    molblock.nmol            = 1;
    molblock.type            = 0;
    mtop->natoms             = moltype.atoms.nr;
    mtop->finalize();
}

//! The output of a pull evaluation
struct PullOutput
{
    //! The pull energy
    real energy = 0;
    //! The forces on the atoms
    std::vector<RVec> forces;
    //! The virial
    matrix virial = { { 0 } };
    //! The derivative of the energy with respect to lambda
    real dvdlambda = 0;
    //! The value of the last pull coordinate, in internal units
    double lastCoordValue = 0;
};

//! Test parameters
using PullDerivativesParameters = std::tuple<PullSetup, PullPotential, PeriodicSetup>;

//! Test fixture for derivatives of the pull energy
class PullDerivativesTest : public ::testing::TestWithParam<PullDerivativesParameters>
{
public:
    PullDerivativesTest() : x_(makeCoordinates()), masses_(makeMasses())
    {
        const PeriodicSetup periodicSetup = std::get<2>(GetParam());

        const matrix rectangularBox = { { 5, 0, 0 }, { 0, 5.5, 0 }, { 0, 0, 6 } };
        const matrix triclinicBox   = { { 5, 0, 0 }, { 0.7, 5.5, 0 }, { -0.9, 1.1, 6 } };
        copy_mat(periodicSetup == PeriodicSetup::Triclinic ? triclinicBox : rectangularBox, box_);
        pbcType_ = (periodicSetup == PeriodicSetup::NoPbc ? PbcType::No : PbcType::Xyz);

        xUnshifted_ = x_;
        if (pbcType_ != PbcType::No)
        {
            // Put atoms of each group in different periodic images.
            // The PBC atom of group 2 stays in the unit cell, as the cylinder
            // code assumes that atoms are at most one box vector away
            // from the periodic image closest to the pull group.
            const std::vector<std::pair<int, IVec>> shifts = {
                { 1, { 1, 0, 0 } },  { 5, { 0, -1, 0 } },  { 7, { 0, 0, 1 } },
                { 8, { 1, 1, 0 } },  { 10, { 0, 1, -1 } }, { 12, { -1, 1, 0 } },
                { 15, { 0, 1, 0 } }, { 19, { 1, 0, -1 } }, { 22, { 0, 0, 1 } },
                { 26, { -1, 0, 0 } }
            };
            for (const auto& [atom, shift] : shifts)
            {
                for (int d = 0; d < DIM; d++)
                {
                    for (int e = 0; e < DIM; e++)
                    {
                        x_[atom][e] += shift[d] * box_[d][e];
                    }
                }
            }
        }

        fillTopology(&mtop_, masses_);
    }

    /*! \brief Computes the pull energy and, when requested, the derivatives
     *
     * A new pull object is constructed at every call, so no state
     * can be carried over between evaluations.
     */
    PullOutput runPull(const pull_params_t& params,
                       ArrayRef<const RVec> x,
                       const matrix         box,
                       const real           lambda,
                       const double         t,
                       const bool           computeDerivatives) const
    {
        t_inputrec ir;
        ir.pbcType = pbcType_;

        const MpiComm       mpiComm(MpiComm::SingleRank{});
        LocalAtomSetManager atomSets;
        pull_t* pull = init_pull(nullptr, &params, &ir, mtop_, mpiComm, nullptr, &atomSets, 0.0_real);

        t_pbc pbc;
        set_pbc(&pbc, pbcType_, box);

        const t_pull_coord& lastCoord  = params.coord.back();
        const bool          isExternal = (lastCoord.eType == PullingAlgorithm::External);
        if (isExternal)
        {
            register_external_pull_potential(pull, lastCoord.coordIndex, c_externalPotentialProvider);
        }

        PullOutput output;
        output.energy = pull_potential(pull, masses_, pbc, mpiComm, t, lambda, x, &output.dvdlambda);
        if (isExternal)
        {
            // The pull code does not compute the value of coordinates with an external potential
            output.lastCoordValue = get_pull_coord_value(pull, lastCoord.coordIndex, pbc, t);
            output.energy += lastCoord.k * output.lastCoordValue;
            apply_external_pull_coord_force(pull, lastCoord.coordIndex, -lastCoord.k);
        }
        else
        {
            output.lastCoordValue = pull->coord.back().spatialData.value;
        }
        if (computeDerivatives)
        {
            output.forces.resize(x.size(), { 0, 0, 0 });
            ForceWithVirial forceWithVirial(output.forces, true);
            pull_apply_forces(pull, masses_, mpiComm, &forceWithVirial);
            copy_mat(forceWithVirial.getVirial(), output.virial);
        }

        finish_pull(pull);

        return output;
    }

protected:
    //! Topology, only used for the masses
    gmx_mtop_t mtop_;
    //! Coordinates
    std::vector<RVec> x_;
    //! The coordinates before putting atoms in different periodic images
    std::vector<RVec> xUnshifted_;
    //! Masses
    std::vector<real> masses_;
    //! The box
    matrix box_;
    //! The type of periodicity
    PbcType pbcType_;
};

TEST_P(PullDerivativesTest, DerivativesMatchEnergy)
{
    if (!c_canTestWithFiniteDifferences)
    {
        GTEST_SKIP() << "Numerical derivatives of the energy are only accurate enough in double "
                        "precision";
    }

    const auto [pullSetup, pullPotential, periodicSetup] = GetParam();

    if (isTransformation(pullSetup) && !HAVE_MUPARSER)
    {
        GTEST_SKIP() << "Transformation pull coordinates require muParser";
    }
    if (pullSetup == PullSetup::DirectionCosineWeightedReference && periodicSetup != PeriodicSetup::Rectangular)
    {
        GTEST_SKIP() << "Cosine weighting requires periodicity and is not supported along "
                        "triclinic dimensions";
    }

    const real   lambda = 0.3;
    const double t      = 0.7;

    pull_params_t params = makePullParameters(pullSetup);

    // Get the current value of the coordinate, so we can set a reference
    // value that is far enough from it to avoid the kink of flat-bottomed
    // potentials and to get significant forces.
    t_pull_coord& coord           = params.coord.back();
    const double  internalToInput = pull_conversion_factor_internal2userinput(coord);
    const bool    isAngle         = (internalToInput != 1);
    const double  valueInInputUnits =
            runPull(params, x_, box_, lambda, t, false).lastCoordValue * internalToInput;
    if (coord.eGeom == PullGroupGeometry::Angle || coord.eGeom == PullGroupGeometry::AngleAxis)
    {
        // The derivative of the angle is singular at 0 and 180 degrees
        ASSERT_GT(valueInInputUnits, 30);
        ASSERT_LT(valueInInputUnits, 150);
    }
    if (coord.eGeom == PullGroupGeometry::Dihedral)
    {
        // Stay away from the periodic jump
        ASSERT_LT(std::abs(valueInInputUnits), 150);
    }
    const double deviation = (isAngle ? 15 : 0.25);
    const double reference =
            valueInInputUnits + (pullPotential == PullPotential::FlatBottomHigh ? deviation : -deviation);

    switch (pullPotential)
    {
        case PullPotential::Umbrella:
        case PullPotential::UmbrellaWithRate: coord.eType = PullingAlgorithm::Umbrella; break;
        case PullPotential::FlatBottom: coord.eType = PullingAlgorithm::FlatBottom; break;
        case PullPotential::FlatBottomHigh: coord.eType = PullingAlgorithm::FlatBottomHigh; break;
        case PullPotential::ConstantForce: coord.eType = PullingAlgorithm::ConstantForce; break;
        case PullPotential::External:
            coord.eType                     = PullingAlgorithm::External;
            coord.externalPotentialProvider = c_externalPotentialProvider;
            break;
        default: GMX_RELEASE_ASSERT(false, "Unhandled potential");
    }
    if (pullPotential == PullPotential::ConstantForce || pullPotential == PullPotential::External)
    {
        coord.k  = 120;
        coord.kB = 200;
    }
    else
    {
        coord.k  = 900;
        coord.kB = 1400;
    }
    // Energy and forces are always evaluated at the same time t
    coord.rate = (pullPotential == PullPotential::UmbrellaWithRate ? (isAngle ? 8 : 0.2) : 0);
    coord.init = reference - coord.rate * t;

    const PullOutput output = runPull(params, x_, box_, lambda, t, true);

    // Check that we are actually testing something
    ASSERT_GT(std::abs(output.energy), 1);

    if (pbcType_ != PbcType::No)
    {
        EXPECT_REAL_EQ_TOL(output.energy,
                           runPull(params, xUnshifted_, box_, lambda, t, false).energy,
                           relativeToleranceAsFloatingPoint(output.energy, 1e-10))
                << "The energy should not depend on the periodic images of the atoms";
    }

    FiniteDifferenceSettings settings;
    if (isTransformation(pullSetup))
    {
        // The pull code uses one-sided finite differences for the
        // derivative of the transformation expression
        settings.relativeTolerance = 1e-5;
    }

    checkForcesMatchEnergy(
            x_,
            output.forces,
            [&](ArrayRef<const RVec> xTrial)
            { return runPull(params, xTrial, box_, lambda, t, false).energy; },
            settings);

    checkDvdlambdaMatchesEnergy(
            output.dvdlambda,
            [&](const double lambdaChange)
            { return runPull(params, x_, box_, lambda + lambdaChange, t, false).energy; },
            settings);

    // The virial is -0.5 sum_i x_i (x) f_i, so element [d][e] should be half
    // the derivative of the energy with respect to a homogeneous strain of
    // the coordinates and the box that displaces along dimension e
    // proportional to the coordinate along dimension d. An absolute
    // reference position is part of the system that is strained.
    // With geometry direction-periodic the pull code does not compute
    // the virial, as documented in the mdp options.
    if (coord.eGeom != PullGroupGeometry::DirectionPBC)
    {
        double maxElement = 1;
        for (int d = 0; d < DIM; d++)
        {
            for (int e = 0; e < DIM; e++)
            {
                maxElement = std::max(maxElement, std::abs(static_cast<double>(output.virial[d][e])));
            }
        }
        const FloatingPointTolerance tolerance = finiteDifferenceTolerance(
                maxElement, settings.relativeTolerance, std::abs(output.energy), settings.strainStep);

        for (int forceDim = 0; forceDim < DIM; forceDim++)
        {
            // With periodicity only strains that keep the box lower-triangular are valid
            const int startDim = (pbcType_ != PbcType::No ? forceDim : 0);
            for (int distanceDim = startDim; distanceDim < DIM; distanceDim++)
            {
                const real numericalVirial =
                        0.5
                        * finiteDifferenceDerivative(
                                [&, forceDim = forceDim, distanceDim = distanceDim](const double strain)
                                {
                                    std::vector<RVec> xTrial = x_;
                                    for (Index atom = 0; atom < gmx::ssize(x_); atom++)
                                    {
                                        xTrial[atom][forceDim] += strain * x_[atom][distanceDim];
                                    }
                                    matrix boxTrial;
                                    copy_mat(box_, boxTrial);
                                    for (int d = 0; d < DIM; d++)
                                    {
                                        boxTrial[d][forceDim] += strain * box_[d][distanceDim];
                                    }
                                    pull_params_t paramsTrial = params;
                                    for (t_pull_coord& coordTrial : paramsTrial.coord)
                                    {
                                        coordTrial.origin[forceDim] +=
                                                strain * coordTrial.origin[distanceDim];
                                    }
                                    return runPull(paramsTrial, xTrial, boxTrial, lambda, t, false).energy;
                                },
                                settings.strainStep);
                EXPECT_REAL_EQ_TOL(output.virial[distanceDim][forceDim], numericalVirial, tolerance)
                        << "Virial element " << distanceDim << "," << forceDim
                        << " does not match the derivative of the energy with respect to strain";
            }
        }
    }
}

/*! \brief Checks the weights and the effective mass of a cosine-weighted group
 *
 * The weights determine the forces, which is covered by the test above,
 * but also the displacements and the effective mass with constraint pulling.
 */
TEST(PullCosineWeightingTest, WeightsAndMassAreCorrect)
{
    const std::vector<RVec> x      = makeCoordinates();
    const std::vector<real> masses = makeMasses();
    gmx_mtop_t              mtop;
    fillTopology(&mtop, masses);
    const matrix box = { { 5, 0, 0 }, { 0, 5.5, 0 }, { 0, 0, 6 } };

    const pull_params_t params = makePullParameters(PullSetup::DirectionCosineWeightedReference);

    t_inputrec ir;
    ir.pbcType = PbcType::Xyz;
    const MpiComm       mpiComm(MpiComm::SingleRank{});
    LocalAtomSetManager atomSets;
    pull_t* pull = init_pull(nullptr, &params, &ir, mtop, mpiComm, nullptr, &atomSets, 0.0_real);
    t_pbc   pbc;
    set_pbc(&pbc, ir.pbcType, box);
    real dvdlambda = 0;
    pull_potential(pull, masses, pbc, mpiComm, 0.0, 0.0_real, x, &dvdlambda);

    const pull_group_work_t& group = pull->group[c_slabGroup];
    ASSERT_EQ(group.epgrppbc, epgrppbcCOS);

    const FloatingPointTolerance tolerance =
            relativeToleranceAsFloatingPoint(1.0, GMX_DOUBLE ? 1e-12 : 1e-5);

    // The weight is the cosine of the phase difference with the reference position
    const double twoPiOverBox         = 2 * M_PI / box[ZZ][ZZ];
    double       sumMassWeight        = 0;
    double       sumMassWeightSquared = 0;
    for (int i = 0; i < c_numAtomsInSlab; i++)
    {
        const int    atom   = params.group[c_slabGroup].ind[i];
        const double weight = std::cos(twoPiOverBox * (x[atom][ZZ] - group.x[ZZ]));
        EXPECT_REAL_EQ_TOL(group.localWeights[i], weight, tolerance);
        sumMassWeight += masses[atom] * weight;
        sumMassWeightSquared += masses[atom] * weight * weight;
    }
    EXPECT_REAL_EQ_TOL(group.mwscale, 1 / sumMassWeight, tolerance);
    EXPECT_REAL_EQ_TOL(group.wscale, sumMassWeight / sumMassWeightSquared, tolerance);
    EXPECT_REAL_EQ_TOL(group.invtm * sumMassWeight, sumMassWeightSquared / sumMassWeight, tolerance);

    finish_pull(pull);
}

//! Returns a name for a test with parameters \c info
std::string nameOfTest(const ::testing::TestParamInfo<PullDerivativesParameters>& info)
{
    std::string name = formatString("%s_%s_%s",
                                    enumValueToString(std::get<0>(info.param)),
                                    enumValueToString(std::get<1>(info.param)),
                                    enumValueToString(std::get<2>(info.param)));
    std::replace_if(name.begin(), name.end(), [](const char c) { return !std::isalnum(c); }, '_');
    return name;
}

INSTANTIATE_TEST_SUITE_P(Pull,
                         PullDerivativesTest,
                         ::testing::Combine(::testing::ValuesIn(EnumerationWrapper<PullSetup>{}),
                                            ::testing::ValuesIn(EnumerationWrapper<PullPotential>{}),
                                            ::testing::ValuesIn(EnumerationWrapper<PeriodicSetup>{})),
                         nameOfTest);

} // namespace
} // namespace test
} // namespace gmx
