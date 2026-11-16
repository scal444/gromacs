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
 * Tests that essential dynamics flooding forces are consistent with
 * the flooding energy
 *
 * \author Kevin Boyd <kevin44boyd@gmail.com>
 * \ingroup module_essentialdynamics
 */
#include "gmxpre.h"

#include <cmath>
#include <cstdint>
#include <cstdio>

#include <algorithm>
#include <fstream>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "gromacs/essentialdynamics/edsam.h"
#include "gromacs/fileio/oenv.h"
#include "gromacs/math/units.h"
#include "gromacs/mdlib/constr.h"
#include "gromacs/mdlib/makeconstraints.h"
#include "gromacs/mdrunutility/handlerestart.h"
#include "gromacs/mdtypes/edsamhistory.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/mdtypes/observableshistory.h"
#include "gromacs/mdtypes/state.h"
#include "gromacs/pbcutil/pbc.h"
#include "gromacs/topology/atoms.h"
#include "gromacs/topology/topology.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/logger.h"
#include "gromacs/utility/mpicomm.h"
#include "gromacs/utility/real.h"
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

//! A vector with an element for each atom in the essential dynamics group
using CollectiveVector = std::vector<RVec>;

//! Returns the inner product of two collective vectors
real innerProduct(const CollectiveVector& a, const CollectiveVector& b)
{
    real product = 0;
    for (Index i = 0; i < gmx::ssize(a); i++)
    {
        product += a[i].dot(b[i]);
    }
    return product;
}

//! Removes the components of \c vector along the vectors in \c orthonormalBasis and normalizes it
void orthonormalize(CollectiveVector* vector, ArrayRef<const CollectiveVector> orthonormalBasis)
{
    for (const CollectiveVector& basisVector : orthonormalBasis)
    {
        const real component = innerProduct(*vector, basisVector);
        for (Index i = 0; i < gmx::ssize(*vector); i++)
        {
            (*vector)[i] -= basisVector[i] * component;
        }
    }
    const real invNorm = 1 / std::sqrt(innerProduct(*vector, *vector));
    for (RVec& element : *vector)
    {
        element *= invNorm;
    }
}

//! Returns \c x rotated by \c angle (in degrees) around unit vector \c axis through the origin
RVec rotateAroundAxis(const RVec& x, const RVec& axis, const real angle)
{
    const real cosAngle = std::cos(angle * gmx::c_deg2Rad);
    const real sinAngle = std::sin(angle * gmx::c_deg2Rad);
    // Rodrigues' rotation formula
    return x * cosAngle + axis.cross(x) * sinAngle + axis * (axis.dot(x) * (1 - cosAngle));
}

/*! \brief Test parameters: whether the potential is harmonic, whether the projections on
 * the eigenvectors are mass weighted and whether the fit is mass weighted */
using FloodingTestParameters = std::tuple<bool, bool, bool>;

/*! \brief Test fixture for essential dynamics flooding
 *
 * The flooding potential is a function of the projections of the
 * positions, after fitting them to a reference structure, on a set of
 * eigenvectors. The forces neglect that the translation and rotation
 * of the fit depend on the positions. This has no effect when the
 * eigenvectors are orthogonal to translations and rotations of the fitted
 * structure, as eigenvectors from a covariance analysis of fitted
 * structures are. To be able to do an exact check, we construct such
 * eigenvectors and use positions that can be fitted exactly on the
 * reference structure.
 *
 * The flooding energy is only available in the output file,
 * so we read it from there, which limits its precision.
 */
class FloodingTest : public ::testing::TestWithParam<FloodingTestParameters>
{
public:
    FloodingTest()
    {
        const auto [isHarmonic, useMassWeightedProjection, useMassWeightedFit] = GetParam();

        const int numAtoms      = gmx::ssize(c_masses);
        const int numGroupAtoms = gmx::ssize(c_groupAtoms);

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

        // Put the center (of mass) of the reference structure at the origin,
        // which is the frame in which the fitted positions are projected.
        CollectiveVector reference       = c_reference;
        RVec             referenceCenter = { 0.0_real, 0.0_real, 0.0_real };
        real             totalMass       = 0;
        for (int i = 0; i < numGroupAtoms; i++)
        {
            const real mass = (useMassWeightedFit ? c_masses[c_groupAtoms[i]] : 1.0_real);
            referenceCenter += reference[i] * mass;
            totalMass += mass;
        }
        for (RVec& x : reference)
        {
            x -= referenceCenter / totalMass;
        }

        // The projection of a displacement dx on eigenvector v is
        // sum_j w_j v_j . dx_j, with w_j the square root of the mass with
        // mass weighting and otherwise 1. Translations and rotations of
        // the reference structure give no projection when v is orthogonal
        // to the vectors with elements w_j e_k and w_j e_k x y_j,
        // where e_k is the unit vector along dimension k
        // and y_j are the reference positions.
        std::vector<real> weights;
        for (const int atom : c_groupAtoms)
        {
            weights.push_back(useMassWeightedProjection ? std::sqrt(c_masses[atom]) : 1.0_real);
        }
        std::vector<CollectiveVector> basis;
        for (const bool isRotation : { false, true })
        {
            for (int d = 0; d < DIM; d++)
            {
                RVec unitVector = { 0.0_real, 0.0_real, 0.0_real };
                unitVector[d]   = 1;
                CollectiveVector rigidBodyMotion;
                for (int i = 0; i < numGroupAtoms; i++)
                {
                    rigidBodyMotion.push_back(
                            (isRotation ? unitVector.cross(reference[i]) : unitVector) * weights[i]);
                }
                orthonormalize(&rigidBodyMotion, basis);
                basis.push_back(rigidBodyMotion);
            }
        }
        std::vector<CollectiveVector> eigenvectors = c_rawEigenvectors;
        for (CollectiveVector& eigenvector : eigenvectors)
        {
            orthonormalize(&eigenvector, basis);
            basis.push_back(eigenvector);
        }

        // Write the essential dynamics input file as gmx make_edi does.
        // Note that numbers are read with a field width of at most 12 characters.
        const std::string ediFileName = fileManager_.getTemporaryFilePath("sam.edi").string();
        edoFileName_                  = fileManager_.getTemporaryFilePath("edsam.xvg").string();
        {
            std::string edi =
                    formatString("#MAGIC\n 670\n#NINI\n %d\n#FITMAS\n %d\n#ANALYSIS_MAS\n %d\n",
                                 numAtoms,
                                 useMassWeightedFit ? 1 : 0,
                                 useMassWeightedProjection ? 1 : 0);
            edi += "#OUTFRQ\n 1\n#MAXLEN\n 1000\n#SLOPECRIT\n 0.0\n";
            // No adaptive flooding
            edi += "#PRESTEPS\n 0\n#DELTA_F0\n 0.0\n#INIT_DELTA_F\n 0.0\n#TAU\n 0.0\n";
            // The flooding strength, negative for harmonic restraints
            edi += formatString("#EFL_NULL\n %f\n#ALPHA2\n 1.0\n#KT\n 2.5\n#HARMONIC\n %d\n",
                                isHarmonic ? -150.0 : 100.0,
                                isHarmonic ? 1 : 0);
            edi += "#CONST_FORCE_FLOODING\n 0\n";
            edi += formatString("#NREF, XREF\n %d\n", numGroupAtoms);
            for (int i = 0; i < numGroupAtoms; i++)
            {
                edi += formatString("%d  %.8f  %.8f  %.8f\n",
                                    c_groupAtoms[i] + 1,
                                    reference[i][XX],
                                    reference[i][YY],
                                    reference[i][ZZ]);
            }
            // The average structure, which differs from the reference
            edi += formatString("#NAV, XAV\n %d\n", numGroupAtoms);
            for (int i = 0; i < numGroupAtoms; i++)
            {
                const RVec average = reference[i] + c_averageMinusReference[i];
                edi += formatString(
                        "%d  %.8f  %.8f  %.8f\n", c_groupAtoms[i] + 1, average[XX], average[YY], average[ZZ]);
            }
            // No essential dynamics sampling
            for (int group = 1; group <= 6; group++)
            {
                edi += formatString("# NUMBER OF EIGENVECTORS + COMPONENTS GROUP %d\n 0\n", group);
            }
            edi += formatString("# NUMBER OF EIGENVECTORS + COMPONENTS GROUP 7\n %zu\n",
                                eigenvectors.size());
            for (Index i = 0; i < gmx::ssize(eigenvectors); i++)
            {
                // Index of the eigenvector and the inverse of the eigenvalue
                edi += formatString("%d  %f", int(i + 1), c_inverseEigenvalues[i]);
                if (isHarmonic)
                {
                    // Reference projection and its rate of change
                    edi += formatString("  %f  0.0", c_referenceProjections[i]);
                }
                edi += "\n";
            }
            for (const CollectiveVector& eigenvector : eigenvectors)
            {
                for (const RVec& element : eigenvector)
                {
                    edi += formatString("%.5e  %.5e  %.5e\n", element[XX], element[YY], element[ZZ]);
                }
            }
            edi += "#NTARGET, XTARGET\n 0\n#NORIGIN, XORIGIN\n 0\n";

            std::ofstream ediFile(ediFileName);
            ediFile << edi;
        }

        // The current positions are the reference structure, rotated and translated
        const RVec axis = RVec(1.0, 0.4, -0.7).unitVector();
        x_ = { { 3.1, 5.2, 4.3 }, { 2.9, 4.1, 3.4 }, { 4.4, 3.6, 5.2 }, { 5.3, 4.7, 3.8 },
               { 3.5, 4.4, 4.9 }, { 4.8, 5.3, 4.6 }, { 4.1, 3.9, 3.3 }, { 3.7, 5.5, 5.4 } };
        for (int i = 0; i < numGroupAtoms; i++)
        {
            x_[c_groupAtoms[i]] = rotateAroundAxis(reference[i], axis, 37) + RVec(4.1, 4.4, 4.2);
        }

        clear_mat(box_);
        box_[XX][XX] = 8;
        box_[YY][YY] = 9;
        box_[ZZ][ZZ] = 10;

        ir_.pbcType = PbcType::Xyz;
        ir_.init_t  = 0;
        ir_.delta_t = 0.002;

        t_state globalState;
        globalState.changeNumAtoms(numAtoms);
        globalState.addEntry(StateEntry::X);
        std::copy(x_.begin(), x_.end(), globalState.x.begin());
        copy_mat(box_, globalState.box);

        // Essential dynamics requires a constraints object
        constraints_ = makeConstraints(
                mtop_, ir_, nullptr, false, true, nullptr, mpiComm_, nullptr, false, false, nullptr, nullptr, nullptr, false, nullptr);

        output_env_init_default(&oenv_);

        essentialDynamics_ = init_edsam(MDLogger(),
                                        ediFileName.c_str(),
                                        edoFileName_.c_str(),
                                        mtop_,
                                        ir_,
                                        mpiComm_,
                                        nullptr,
                                        constraints_.get(),
                                        &globalState,
                                        &observablesHistory_,
                                        oenv_,
                                        StartingBehavior::NewSimulation);
    }

    ~FloodingTest() override
    {
        // The module closes its output file when it is destroyed
        essentialDynamics_.reset();
        output_env_done(oenv_);
    }

    //! Output of the flooding module
    struct Output
    {
        //! The flooding energy
        double energy = 0;
        //! The forces
        std::vector<RVec> forces;
    };

    //! Computes the flooding energy and forces for coordinates \c x
    Output compute(ArrayRef<const RVec> x)
    {
        Output output;
        output.forces.resize(x.size(), { 0.0_real, 0.0_real, 0.0_real });
        do_flood(mpiComm_, ir_, x, output.forces, essentialDynamics_->getLegacyED(), box_, step_, true);
        step_++;

        // The energy is in the last line of the output file, which has columns:
        // time, RMSD and for each eigenvector the projection, the energy and the force.
        // Flush all files, as we do not have access to the file pointer.
        std::fflush(nullptr);
        std::ifstream edoFile(edoFileName_);
        std::string   line;
        std::string   lastLine;
        while (std::getline(edoFile, line))
        {
            lastLine = line;
        }
        const std::vector<std::string> columns = splitString(lastLine);
        EXPECT_EQ(columns.size(), 2 + 3 * c_rawEigenvectors.size());
        if (columns.size() > 3)
        {
            output.energy = std::stod(columns[3]);
        }
        return output;
    }

protected:
    //! Masses of all atoms in the system
    const std::vector<real> c_masses = { 12, 1, 16, 14, 32, 1, 12, 35 };
    //! The atoms that are part of the essential dynamics group
    const std::vector<int> c_groupAtoms = { 0, 1, 3, 4, 5, 7 };
    //! The reference structure for fitting
    const CollectiveVector c_reference = { { 0.12, 0.31, -0.24 }, { -0.35, 0.11, 0.23 },
                                           { 0.33, -0.28, 0.17 }, { -0.14, -0.37, -0.29 },
                                           { 0.41, 0.26, 0.32 },  { -0.27, 0.19, -0.36 } };
    //! The difference between the average structure, from which projections are computed, and the reference
    const CollectiveVector c_averageMinusReference = {
        { 0.21, -0.12, 0.08 },  { -0.15, 0.24, 0.11 },  { 0.09, 0.18, -0.22 },
        { -0.19, -0.07, 0.16 }, { 0.13, -0.23, -0.14 }, { -0.11, 0.06, 0.25 }
    };
    //! Vectors from which eigenvectors are generated
    const std::vector<CollectiveVector> c_rawEigenvectors = { { { 0.5, -0.2, 0.3 },
                                                                { -0.4, 0.6, 0.1 },
                                                                { 0.2, 0.3, -0.7 },
                                                                { -0.6, -0.1, 0.4 },
                                                                { 0.1, -0.5, -0.3 },
                                                                { 0.3, 0.2, 0.6 } },
                                                              { { -0.3, 0.4, 0.2 },
                                                                { 0.6, 0.1, -0.5 },
                                                                { -0.1, -0.6, 0.3 },
                                                                { 0.2, 0.5, 0.4 },
                                                                { -0.5, 0.3, -0.2 },
                                                                { 0.4, -0.2, -0.1 } } };
    //! The inverse of the eigenvalues
    const std::vector<real> c_inverseEigenvalues = { 3.0, 5.0 };
    //! The center of the potential for harmonic restraints
    const std::vector<real> c_referenceProjections = { 0.1, -0.2 };

    TestFileManager                    fileManager_;
    std::string                        edoFileName_;
    MpiComm                            mpiComm_{ MpiComm::SingleRank{} };
    gmx_mtop_t                         mtop_;
    t_inputrec                         ir_;
    gmx_output_env_t*                  oenv_ = nullptr;
    matrix                             box_;
    std::vector<RVec>                  x_;
    ObservablesHistory                 observablesHistory_;
    std::unique_ptr<Constraints>       constraints_;
    std::unique_ptr<EssentialDynamics> essentialDynamics_;
    int64_t                            step_ = 0;
};

TEST_P(FloodingTest, ForcesMatchEnergy)
{
    if (!c_canTestWithFiniteDifferences)
    {
        GTEST_SKIP() << "Numerical derivatives of the energy are only accurate enough in double "
                        "precision";
    }

    const Output output = compute(x_);

    // Check that the test actually exercises the potential
    EXPECT_GT(std::abs(output.energy), 1.0);

    /* The energy is read from the output file with six digits.
     * To limit the effect of that on the numerical derivative,
     * we use a large step size and a large tolerance.
     */
    FiniteDifferenceSettings settings;
    settings.coordinateStep    = 0.05;
    settings.relativeTolerance = 1e-2;
    checkForcesMatchEnergy(
            x_, output.forces, [&](ArrayRef<const RVec> xTrial) { return compute(xTrial).energy; }, settings);
}

//! Returns a name for a test with parameters \c info
std::string nameOfTest(const ::testing::TestParamInfo<FloodingTestParameters>& info)
{
    return formatString("%s_%s_%s",
                        std::get<0>(info.param) ? "Harmonic" : "Gaussian",
                        std::get<1>(info.param) ? "MassWeightedProjection" : "PlainProjection",
                        std::get<2>(info.param) ? "MassWeightedFit" : "PlainFit");
}

INSTANTIATE_TEST_SUITE_P(Flooding,
                         FloodingTest,
                         ::testing::Combine(::testing::Bool(), ::testing::Bool(), ::testing::Bool()),
                         nameOfTest);

} // namespace
} // namespace test
} // namespace gmx
