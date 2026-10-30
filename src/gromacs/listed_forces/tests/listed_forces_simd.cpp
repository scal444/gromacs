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
 * consider code modification as a contribution to GROMACS.
 *
 * To help us fund GROMACS development, we humbly ask that you cite
 * the research papers on the package. Check out https://www.gromacs.org.
 */
/*! \internal \file
 * \brief
 * Tests that the SIMD listed-forces code paths agree with the plain-C ones.
 *
 * A chain molecule, with bonds, angles, Urey-Bradley, proper, RB and improper
 * dihedrals and LJ-14 pairs, is put in a small periodic box, so many
 * interactions cross the periodic boundaries. Parameter types are mixed
 * perturbed and non-perturbed and some atoms have perturbed charges or are
 * perturbed otherwise. The full ListedForces::calculate() is run with and
 * without SIMD kernels and forces, energies, dV/dlambda and the virial
 * (including the shift-force contribution) are compared.
 *
 * \ingroup module_listed_forces
 */
#include "gmxpre.h"

#include "config.h"

#include <cmath>
#include <cstring>

#include <algorithm>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "gromacs/gmxlib/nrnb.h"
#include "gromacs/gpu_utils/hostallocator.h"
#include "gromacs/listed_forces/bonded.h"
#include "gromacs/listed_forces/listed_forces.h"
#include "gromacs/math/arrayrefwithpadding.h"
#include "gromacs/math/paddedvector.h"
#include "gromacs/math/units.h"
#include "gromacs/mdtypes/atominfo.h"
#include "gromacs/mdtypes/enerdata.h"
#include "gromacs/mdtypes/fcdata.h"
#include "gromacs/mdtypes/forceoutput.h"
#include "gromacs/mdtypes/forcerec.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/mdtypes/interaction_const.h"
#include "gromacs/mdtypes/md_enums.h"
#include "gromacs/mdtypes/simulation_workload.h"
#include "gromacs/pbcutil/ishift.h"
#include "gromacs/pbcutil/pbc.h"
#include "gromacs/random/threefry.h"
#include "gromacs/random/uniformrealdistribution.h"
#include "gromacs/tables/forcetable.h"
#include "gromacs/topology/forcefieldparameters.h"
#include "gromacs/topology/idef.h"
#include "gromacs/topology/ifunc.h"
#include "gromacs/topology/topsort.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/booltype.h"
#include "gromacs/utility/enumerationhelpers.h"
#include "gromacs/utility/real.h"
#include "gromacs/utility/stringutil.h"
#include "gromacs/utility/vec.h"

#include "testutils/testasserts.h"

namespace gmx
{
namespace test
{
namespace
{

//! Which quantities are computed
enum class Computes
{
    Forces,
    ForcesAndEnergy,
    ForcesAndVirialAndEnergy
};

//! Box shapes
enum class BoxShape
{
    Rectangular,
    Triclinic
};

//! The output of one ListedForces::calculate() call
struct Output
{
    std::vector<RVec>                                                 f;
    tensor                                                            virial;
    tensor                                                            shiftVirial;
    gmx::EnumerationArray<InteractionFunction, real>                  energies;
    real                                                              coulomb14;
    real                                                              lj14;
    gmx::EnumerationArray<FreeEnergyPerturbationCouplingType, double> dvdl;
};

//! The bonded types tested, these all have 4 or fewer atoms along a chain
const std::vector<InteractionFunction> c_bondedTypes = { InteractionFunction::Bonds,
                                                         InteractionFunction::Angles,
                                                         InteractionFunction::UreyBradleyPotential,
                                                         InteractionFunction::ProperDihedrals,
                                                         InteractionFunction::RyckaertBellemansDihedrals,
                                                         InteractionFunction::ImproperDihedrals,
                                                         InteractionFunction::LennardJones14 };

//! Test parameters: number of atoms, number of threads, computed quantities, FEP, lambda, box shape
using SimdTestParameters = std::tuple<int, int, Computes, bool, real, BoxShape>;

class ListedForcesSimdTest : public ::testing::TestWithParam<SimdTestParameters>
{
protected:
    int      numAtoms_;
    int      numThreads_;
    Computes computes_;
    bool     haveFep_;
    real     lambda_;
    BoxShape boxShape_;

    gmx_ffparams_t                          ffparams_;
    std::unique_ptr<InteractionDefinitions> idef_;
    PaddedVector<RVec>                      x_;
    matrix                                  box_;
    std::vector<real>                       chargeA_;
    std::vector<real>                       chargeB_;
    std::vector<BoolType>                   atomIsPerturbed_;
    std::vector<unsigned short>             cENER_;
    int                                     nPerturbed_ = 0;

    t_lambda   fepVals_;
    t_forcerec fr_{ HostAllocationPolicy{} };

    ListedForcesSimdTest() :
        numAtoms_(std::get<0>(GetParam())),
        numThreads_(std::get<1>(GetParam())),
        computes_(std::get<2>(GetParam())),
        haveFep_(std::get<3>(GetParam())),
        lambda_(std::get<4>(GetParam())),
        boxShape_(std::get<5>(GetParam()))
    {
        DefaultRandomEngine           rng(1234 + numAtoms_);
        UniformRealDistribution<real> uniform;

        setupBox();
        setupCoordinates(&rng, &uniform);
        setupAtoms(&rng, &uniform);
        setupForcerec();
        setupTopology(&rng, &uniform);
    }

    void setupBox()
    {
        clear_mat(box_);
        const real boxSize = 1.3_real;
        box_[XX][XX]       = boxSize;
        box_[YY][YY]       = boxSize * 1.1_real;
        box_[ZZ][ZZ]       = boxSize * 0.9_real;
        if (boxShape_ == BoxShape::Triclinic)
        {
            box_[YY][XX] = 0.3_real * boxSize;
            box_[ZZ][XX] = -0.2_real * boxSize;
            box_[ZZ][YY] = 0.4_real * boxSize;
        }
    }

    //! Generates a random chain with reasonable bond lengths and angles, then puts it in the box
    void setupCoordinates(DefaultRandomEngine* rng, UniformRealDistribution<real>* uniform)
    {
        const real bondLength = 0.15;
        x_.resizeWithPadding(numAtoms_);
        x_[0] = { 0.5_real * box_[XX][XX], 0.5_real * box_[YY][YY], 0.5_real * box_[ZZ][ZZ] };
        RVec prevBond = { 1, 0, 0 };
        for (int a = 1; a < numAtoms_; a++)
        {
            RVec bond;
            while (true)
            {
                for (int d = 0; d < DIM; d++)
                {
                    bond[d] = 2 * (*uniform)(*rng) - 1;
                }
                const real norm = bond.norm();
                if (norm < 0.1 || norm > 1)
                {
                    continue;
                }
                bond *= 1.0_real / norm;
                // The angle between the two bonds should be between 75 and 150 degrees
                const real cosAngle = -iprod(bond, prevBond);
                if (cosAngle < std::cos(150 * c_deg2Rad) || cosAngle > std::cos(75 * c_deg2Rad))
                {
                    continue;
                }
                break;
            }
            x_[a]    = x_[a - 1] + bond * (bondLength * (0.9_real + 0.2_real * (*uniform)(*rng)));
            prevBond = bond;
        }
        put_atoms_in_box(PbcType::Xyz, box_, x_.arrayRefWithPadding().unpaddedArrayRef());
    }

    void setupAtoms(DefaultRandomEngine* rng, UniformRealDistribution<real>* uniform)
    {
        chargeA_.resize(numAtoms_);
        chargeB_.resize(numAtoms_);
        atomIsPerturbed_.resize(numAtoms_, false);
        cENER_.resize(numAtoms_, 0);
        for (int a = 0; a < numAtoms_; a++)
        {
            chargeA_[a]       = 1.6 * (*uniform)(*rng) - 0.8;
            chargeB_[a]       = chargeA_[a];
            const real random = (*uniform)(*rng);
            if (haveFep_ && random < 0.2)
            {
                // Perturbed charge
                chargeB_[a]         = 1.6 * (*uniform)(*rng) - 0.8;
                atomIsPerturbed_[a] = true;
            }
            else if (haveFep_ && random < 0.3)
            {
                // Perturbed, but not the charge, e.g. the mass
                atomIsPerturbed_[a] = true;
            }
        }
        for (int a = 0; a < numAtoms_; a++)
        {
            nPerturbed_ += atomIsPerturbed_[a] ? 1 : 0;
        }
    }

    void setupForcerec()
    {
        fepVals_.sc_alpha                = 0.3;
        fepVals_.sc_power                = 1;
        fepVals_.sc_r_power              = 6.0;
        fepVals_.sc_sigma                = 0.3;
        fepVals_.sc_sigma_min            = 0.3;
        fepVals_.bScCoul                 = true;
        fepVals_.scGapsysScaleLinpointLJ = 0.85;
        fepVals_.scGapsysScaleLinpointQ  = 0.3;
        fepVals_.scGapsysSigmaLJ         = 0.3;
        fepVals_.softcoreFunction        = SoftcoreType::Beutler;

        fr_.ic = std::make_unique<interaction_const_t>();
        fr_.ic->softCoreParameters = std::make_unique<interaction_const_t::SoftCoreParameters>(fepVals_);
        fr_.ic->coulomb.epsfac = c_one4PiEps0;
        fr_.pairsTable         = make_tables(nullptr, *fr_.ic, nullptr, 2.0, GMX_MAKETABLES_14ONLY);
        fr_.efep    = haveFep_ ? FreeEnergyPerturbationType::Yes : FreeEnergyPerturbationType::No;
        fr_.fudgeQQ = 0.5;
        fr_.bMolPBC = true;
        fr_.pbcType = PbcType::Xyz;
    }

    //! Adds a parameter type and returns its index
    int addType(InteractionFunction ftype, const t_iparams& iparams)
    {
        ffparams_.functype.push_back(ftype);
        ffparams_.iparams.push_back(iparams);
        return ffparams_.numTypes() - 1;
    }

    //! Returns parameter types for \p ftype, the first one(s) are not perturbed, the last one is
    std::vector<int> addTypes(InteractionFunction ftype)
    {
        std::vector<int> types;
        t_iparams        ip;
        std::memset(&ip, 0, sizeof(ip));
        switch (ftype)
        {
            case InteractionFunction::Bonds:
                ip.harmonic = { 0.15, 3e5, 0.15, 3e5 };
                types.push_back(addType(ftype, ip));
                ip.harmonic = { 0.14, 2e5, 0.16, 3.5e5 };
                types.push_back(addType(ftype, ip));
                break;
            case InteractionFunction::Angles:
                ip.harmonic = { 109.5, 400, 109.5, 400 };
                types.push_back(addType(ftype, ip));
                ip.harmonic = { 120, 300, 120, 300 };
                types.push_back(addType(ftype, ip));
                ip.harmonic = { 112, 350, 125, 250 };
                types.push_back(addType(ftype, ip));
                break;
            case InteractionFunction::UreyBradleyPotential:
                ip.u_b = { 110, 300, 0.25, 2e4, 110, 300, 0.25, 2e4 };
                types.push_back(addType(ftype, ip));
                ip.u_b = { 105, 350, 0.24, 1e4, 115, 250, 0.26, 3e4 };
                types.push_back(addType(ftype, ip));
                break;
            case InteractionFunction::ProperDihedrals:
                ip.pdihs = { 0, 5, 3, 0, 5 };
                types.push_back(addType(ftype, ip));
                ip.pdihs = { 180, 2, 2, 180, 2 };
                types.push_back(addType(ftype, ip));
                ip.pdihs = { 60, 3, 1, 40, 4 };
                types.push_back(addType(ftype, ip));
                break;
            case InteractionFunction::RyckaertBellemansDihedrals:
            {
                const real rbcA[NR_RBDIHS] = { -5.35, 13.6, 8.4, -16.7, 0.3, 12.4 };
                const real rbcB[NR_RBDIHS] = { -6.35, 12.6, 8.1, -10.7, 0.9, 15.4 };
                std::copy(rbcA, rbcA + NR_RBDIHS, ip.rbdihs.rbcA);
                std::copy(rbcA, rbcA + NR_RBDIHS, ip.rbdihs.rbcB);
                types.push_back(addType(ftype, ip));
                std::copy(rbcB, rbcB + NR_RBDIHS, ip.rbdihs.rbcB);
                types.push_back(addType(ftype, ip));
                break;
            }
            case InteractionFunction::ImproperDihedrals:
                ip.harmonic = { 10, 50, 10, 50 };
                types.push_back(addType(ftype, ip));
                ip.harmonic = { 10, 50, 20, 40 };
                types.push_back(addType(ftype, ip));
                break;
            case InteractionFunction::LennardJones14:
                ip.lj14 = { 2e-3, 2e-6, 2e-3, 2e-6 };
                types.push_back(addType(ftype, ip));
                // A type with zero repulsion, which would use soft-core with perturbed atoms
                ip.lj14 = { 1e-4, 0, 1e-4, 0 };
                types.push_back(addType(ftype, ip));
                ip.lj14 = { 2e-3, 2e-6, 1e-3, 1e-6 };
                types.push_back(addType(ftype, ip));
                break;
            default: GMX_RELEASE_ASSERT(false, "Unhandled type");
        }
        return types;
    }

    void setupTopology(DefaultRandomEngine* rng, UniformRealDistribution<real>* uniform)
    {
        std::vector<std::vector<int>> types;
        for (const auto ftype : c_bondedTypes)
        {
            types.push_back(addTypes(ftype));
        }
        idef_ = std::make_unique<InteractionDefinitions>(ffparams_);

        for (size_t t = 0; t < c_bondedTypes.size(); t++)
        {
            const InteractionFunction ftype = c_bondedTypes[t];
            const int                 nral  = NRAL(ftype);
            // LJ-14 pairs are atoms 0 and 3 of a dihedral
            const int span = (ftype == InteractionFunction::LennardJones14 ? 4 : nral);
            for (int a = 0; a + span <= numAtoms_; a++)
            {
                // Without FEP we only use the non-perturbed types (the last type is perturbed)
                const int numTypes = types[t].size() - (haveFep_ ? 0 : 1);
                const int type =
                        types[t][std::min(static_cast<int>((*uniform)(*rng) * numTypes), numTypes - 1)];
                auto& iatoms = idef_->il[ftype].iatoms;
                iatoms.push_back(type);
                if (ftype == InteractionFunction::LennardJones14)
                {
                    iatoms.push_back(a);
                    iatoms.push_back(a + 3);
                }
                else
                {
                    for (int i = 0; i < nral; i++)
                    {
                        iatoms.push_back(a + i);
                    }
                }
            }
        }

        std::vector<int32_t> atomInfo(numAtoms_, 0);
        for (int a = 0; a < numAtoms_; a++)
        {
            if (atomIsPerturbed_[a])
            {
                atomInfo[a] |= sc_atomInfo_FreeEnergyPerturbation;
            }
            if (chargeB_[a] != chargeA_[a])
            {
                atomInfo[a] |= sc_atomInfo_HasPerturbedCharge;
            }
        }
        gmx_sort_ilist_fe(idef_.get(), atomInfo);
        if (haveFep_)
        {
            EXPECT_EQ(idef_->ilsort, ilsortFE_SORTED);
        }
        else
        {
            EXPECT_EQ(idef_->ilsort, ilsortNO_FE);
        }
    }

    //! Runs the listed forces calculation with or without SIMD kernels
    Output run(const bool useSimd)
    {
        fr_.use_simd_kernels = useSimd;

        t_nrnb       nrnb;
        ListedForces listedForces(ffparams_,
                                  1,
                                  1,
                                  numThreads_,
                                  ListedForces::interactionSelectionAll(),
                                  nullptr,
                                  nullptr,
                                  nullptr,
                                  nullptr,
                                  fr_,
                                  &nrnb);
        listedForces.setup(*idef_, numAtoms_, false, {});

        StepWorkload stepWork;
        stepWork.computeForces       = true;
        stepWork.computeListedForces = true;
        stepWork.computeEnergy       = (computes_ != Computes::Forces);
        stepWork.computeVirial       = (computes_ == Computes::ForcesAndVirialAndEnergy);
        stepWork.computeDhdl         = haveFep_;

        PaddedVector<RVec> f(numAtoms_, { 0, 0, 0 });
        std::vector<RVec>  fshift(c_numShiftVectors, { 0, 0, 0 });
        std::vector<RVec>  fDirectVirial(numAtoms_, { 0, 0, 0 });
        ForceWithShiftForces forceWithShiftForces(f.arrayRefWithPadding(), stepWork.computeVirial, fshift);
        ForceWithVirial forceWithVirial(fDirectVirial, stepWork.computeVirial);
        ForceOutputs    forceOutputs(forceWithShiftForces, false, forceWithVirial);

        t_pbc pbc;
        set_pbc(&pbc, PbcType::Xyz, box_);

        t_disresdata disres;
        std::memset(&disres, 0, sizeof(disres));
        t_fcdata fcdata;
        fcdata.disres = &disres;

        gmx_enerdata_t enerd(1, nullptr);

        std::vector<real> lambda(static_cast<int>(FreeEnergyPerturbationCouplingType::Count), lambda_);
        std::vector<int> globalAtomIndex(numAtoms_);
        for (int a = 0; a < numAtoms_; a++)
        {
            globalAtomIndex[a] = a;
        }

        listedForces.calculate(box_,
                               x_.constArrayRefWithPadding(),
                               {},
                               &fcdata,
                               nullptr,
                               &forceOutputs,
                               &pbc,
                               &enerd,
                               lambda,
                               chargeA_,
                               chargeB_,
                               haveFep_ ? makeConstArrayRef(atomIsPerturbed_) : ArrayRef<const bool>{},
                               cENER_,
                               nPerturbed_,
                               globalAtomIndex.data(),
                               stepWork);

        Output out;
        out.f.assign(f.begin(), f.begin() + numAtoms_);
        out.energies  = enerd.term;
        out.coulomb14 = enerd.grpp.energyGroupPairTerms[NonBondedEnergyTerms::Coulomb14][0];
        out.lj14      = enerd.grpp.energyGroupPairTerms[NonBondedEnergyTerms::LJ14][0];
        out.dvdl      = enerd.dvdl_nonlin;

        clear_mat(out.virial);
        clear_mat(out.shiftVirial);
        if (stepWork.computeVirial)
        {
            std::vector<RVec> shiftVectors(c_numShiftVectors);
            calc_shifts(box_, shiftVectors);
            for (int s = 0; s < c_numShiftVectors; s++)
            {
                for (int d1 = 0; d1 < DIM; d1++)
                {
                    for (int d2 = 0; d2 < DIM; d2++)
                    {
                        out.shiftVirial[d1][d2] -= 0.5 * shiftVectors[s][d1] * fshift[s][d2];
                    }
                }
            }
            for (int a = 0; a < numAtoms_; a++)
            {
                for (int d1 = 0; d1 < DIM; d1++)
                {
                    for (int d2 = 0; d2 < DIM; d2++)
                    {
                        out.virial[d1][d2] -= 0.5 * x_[a][d1] * out.f[a][d2];
                    }
                }
            }
            m_add(out.virial, out.shiftVirial, out.virial);
        }

        return out;
    }
};

/*! \brief Relative tolerance for comparing SIMD and plain-C results
 *
 * On energy steps the plain-C LJ-14 kernel uses cubic-spline tables, whereas the SIMD
 * kernel is analytical. In double precision the table error, a few 1e-7, dominates.
 */
real relativeTolerance(const Computes computes)
{
    if (GMX_DOUBLE)
    {
        return computes == Computes::Forces ? 1e-8 : 2e-6;
    }
    else
    {
        return 2e-4;
    }
}

TEST_P(ListedForcesSimdTest, SimdMatchesPlainC)
{
    const Output ref  = run(false);
    const Output simd = run(true);

    const real tolerance = relativeTolerance(computes_);
    // Returns whether a and b agree within the tolerance times scale
    auto closeEnough = [tolerance](real a, real b, real scale)
    { return std::abs(a - b) <= tolerance * std::max(scale, real(1)); };

    // Forces, compared relative to the largest force component
    real maxForce = 0;
    for (const auto& force : ref.f)
    {
        for (int d = 0; d < DIM; d++)
        {
            maxForce = std::max(maxForce, std::abs(force[d]));
        }
    }
    EXPECT_GT(maxForce, 0);
    for (int a = 0; a < numAtoms_; a++)
    {
        for (int d = 0; d < DIM; d++)
        {
            EXPECT_TRUE(closeEnough(ref.f[a][d], simd.f[a][d], maxForce)) << formatString(
                    "atom %d dim %d: plain-C %.12g SIMD %.12g max %g", a, d, ref.f[a][d], simd.f[a][d], maxForce);
        }
    }

    if (computes_ != Computes::Forces)
    {
        for (const auto ftype : c_bondedTypes)
        {
            if (ftype != InteractionFunction::LennardJones14)
            {
                EXPECT_TRUE(closeEnough(ref.energies[ftype], simd.energies[ftype], std::abs(ref.energies[ftype])))
                        << formatString("%s energy: plain-C %.12g SIMD %.12g",
                                        interaction_function[ftype].longname,
                                        ref.energies[ftype],
                                        simd.energies[ftype]);
            }
        }
        // The plain-C pair kernel uses tables, so we need a looser tolerance there
        EXPECT_TRUE(closeEnough(ref.coulomb14, simd.coulomb14, 10 * std::abs(ref.coulomb14)))
                << formatString("Coulomb-14: plain-C %g SIMD %g", ref.coulomb14, simd.coulomb14);
        EXPECT_TRUE(closeEnough(ref.lj14, simd.lj14, 10 * std::abs(ref.lj14)))
                << formatString("LJ-14: plain-C %g SIMD %g", ref.lj14, simd.lj14);
    }

    if (haveFep_)
    {
        for (const auto i : keysOf(ref.dvdl))
        {
            EXPECT_TRUE(closeEnough(ref.dvdl[i], simd.dvdl[i], std::abs(ref.dvdl[i]))) << formatString(
                    "dV/dl %d: plain-C %g SIMD %g", static_cast<int>(i), ref.dvdl[i], simd.dvdl[i]);
        }
    }

    if (computes_ == Computes::ForcesAndVirialAndEnergy)
    {
        real virialScale = 0;
        for (int a = 0; a < numAtoms_; a++)
        {
            virialScale += x_[a].norm() * ref.f[a].norm();
        }
        bool haveShifts = false;
        for (int d1 = 0; d1 < DIM; d1++)
        {
            for (int d2 = 0; d2 < DIM; d2++)
            {
                haveShifts = haveShifts || ref.shiftVirial[d1][d2] != 0;
                EXPECT_TRUE(closeEnough(ref.virial[d1][d2], simd.virial[d1][d2], virialScale)) << formatString(
                        "virial[%d][%d]: plain-C %g SIMD %g", d1, d2, ref.virial[d1][d2], simd.virial[d1][d2]);
                EXPECT_TRUE(closeEnough(ref.shiftVirial[d1][d2], simd.shiftVirial[d1][d2], virialScale))
                        << formatString("shift virial[%d][%d]: plain-C %g SIMD %g",
                                        d1,
                                        d2,
                                        ref.shiftVirial[d1][d2],
                                        simd.shiftVirial[d1][d2]);
            }
        }
        // Check that the test actually tests periodic shifts
        EXPECT_TRUE(numAtoms_ < 20 || haveShifts);
    }
}

//! Formats the test name
std::string testName(const ::testing::TestParamInfo<SimdTestParameters>& info)
{
    const char* computesNames[] = { "Forces", "ForcesAndEnergy", "ForcesAndVirialAndEnergy" };
    std::string lambda          = formatString("%g", std::get<4>(info.param));
    std::replace(lambda.begin(), lambda.end(), '.', '_');
    return formatString("%datoms_%dthreads_%s_%s_lambda%s_%s",
                        std::get<0>(info.param),
                        std::get<1>(info.param),
                        computesNames[static_cast<int>(std::get<2>(info.param))],
                        std::get<3>(info.param) ? "FEP" : "noFEP",
                        lambda.c_str(),
                        std::get<5>(info.param) == BoxShape::Rectangular ? "rect" : "tric");
}

// Atom counts give interaction counts that are not a multiple of the SIMD width,
// the counts with 3 threads give thread ranges straddling the perturbed boundary.
INSTANTIATE_TEST_SUITE_P(NoFep,
                         ListedForcesSimdTest,
                         ::testing::Combine(::testing::Values(5, 61),
                                            ::testing::Values(1, 3),
                                            ::testing::Values(Computes::Forces,
                                                              Computes::ForcesAndEnergy,
                                                              Computes::ForcesAndVirialAndEnergy),
                                            ::testing::Values(false),
                                            ::testing::Values(0.0),
                                            ::testing::Values(BoxShape::Rectangular, BoxShape::Triclinic)),
                         testName);

INSTANTIATE_TEST_SUITE_P(Fep,
                         ListedForcesSimdTest,
                         ::testing::Combine(::testing::Values(7, 61),
                                            ::testing::Values(1, 3),
                                            ::testing::Values(Computes::Forces,
                                                              Computes::ForcesAndEnergy,
                                                              Computes::ForcesAndVirialAndEnergy),
                                            ::testing::Values(true),
                                            ::testing::Values(0.0, 0.4),
                                            ::testing::Values(BoxShape::Rectangular, BoxShape::Triclinic)),
                         testName);

} // namespace
} // namespace test
} // namespace gmx
