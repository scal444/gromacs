/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2020, by the GROMACS development team, led by
 * Mark Abraham, David van der Spoel, Berk Hess, and Erik Lindahl,
 * and including many others, as listed in the AUTHORS file in the
 * top-level source directory and at http://www.gromacs.org.
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
 * http://www.gnu.org/licenses, or write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
 *
 * If you want to redistribute modifications to GROMACS, please
 * consider that scientific software is very special. Version
 * control is crucial - bugs must be traceable. We will be happy to
 * consider code for inclusion in the official distribution, but
 * derived work must not be called official GROMACS. Details are found
 * in the README & COPYING files - if they are missing, get the
 * official version at http://www.gromacs.org.
 *
 * To help us fund GROMACS development, we humbly ask that you cite
 * the research papers on the package. Check out http://www.gromacs.org.
 */
/*! \internal \file
 * \brief
 * Tests for functionality of the "msd" trajectory analysis module.
 *
 * \author Kevin Boyd <kevin44boyd@gmail.com>
 * \ingroup module_trajectoryanalysis
 */
#include "gmxpre.h"

#include "gromacs/trajectoryanalysis/modules/msd.h"

#include <gtest/gtest.h>

#include "gromacs/gmxpreprocess/grompp.h"
#include "gromacs/utility/path.h"
#include "gromacs/utility/stringutil.h"
#include "gromacs/utility/textstream.h"

#include "testutils/cmdlinetest.h"
#include "testutils/refdata.h"
#include "testutils/testasserts.h"
#include "testutils/testfilemanager.h"
#include "testutils/textblockmatchers.h"
#include "testutils/xvgtest.h"

#include "moduletest.h"

namespace gmx::test
{

namespace
{

using gmx::test::CommandLine;
using gmx::test::XvgMatch;

/********************************************************************
 * Tests for gmx::analysismodules::PairDistance.
 */

//! Test fixture for the select analysis module.

class MsdModuleTest : public gmx::test::TrajectoryAnalysisModuleTestFixture<gmx::analysismodules::MsdInfo>
{
public:
    MsdModuleTest()
    {
        setInputFile("-f", "msd_traj.xtc");
        setInputFile("-s", "msd_coords.gro");
        setInputFile("-n", "msd.ndx");
    }
};

// ----------------------------------------------
// These tests match against the MSD vs tau plot.
// ----------------------------------------------
TEST_F(MsdModuleTest, threeDimensionalDiffusion)
{
    setOutputFile("-o", "msd.xvg", XvgMatch());

    const char* const cmdline[] = {
        "msd", "-trestart", "200", "-sel", "0",
    };
    runTest(CommandLine(cmdline));
}

TEST_F(MsdModuleTest, twoDimensionalDiffusion)
{
    setOutputFile("-o", "msd.xvg", XvgMatch());
    const char* const cmdline[] = { "msd", "-trestart", "200", "-lateral", "z", "-sel", "0" };
    runTest(CommandLine(cmdline));
}

TEST_F(MsdModuleTest, oneDimensionalDiffusion)
{
    setOutputFile("-o", "msd.xvg", XvgMatch());
    const char* const cmdline[] = { "msd", "-trestart", "200", "-type", "x", "-sel", "0" };
    runTest(CommandLine(cmdline));
}

// -------------------------------------------------------------------------
// These tests use a custom matcher to check reported diffusion coefficients
// ------------------------------------------------------------------------

class MsdMatcher  : public ITextBlockMatcher
{
public:
    MsdMatcher() = default;

    void checkStream(TextInputStream* stream, TestReferenceChecker* checker) override {
        TestReferenceChecker dCoefficientChecker(checker->checkCompound("XvgLegend", "DiffusionCoefficient"));
        std::string line;
        while (stream->readLine(&line)) {
            if (!startsWith(line, "# D[")) {
                continue;
            }
            dCoefficientChecker.checkString(stripString(line.substr(1)), nullptr);
        }
    }
};

class MsdMatch : public ITextBlockMatcherSettings
{
public:
    TextBlockMatcherPointer createMatcher() const override{
        return std::make_unique<MsdMatcher>();
    }
};

// for 3D, (8 + 4 + 0) / 3 should yield 4 cm^2 / s
TEST_F(MsdModuleTest, threeDimensionalDiffusionCoefficient)
{
    setOutputFile("-o", "msd.xvg", MsdMatch());

    const char* const cmdline[] = {
        "msd", "-trestart", "200", "-sel", "0",
    };
    runTest(CommandLine(cmdline));
}

// for lateral z, (8 + 4) / 2 should yield 6 cm^2 /s
TEST_F(MsdModuleTest, twoDimensionalDiffusionCoefficient)
{
    setOutputFile("-o", "msd.xvg", MsdMatch());
    const char* const cmdline[] = { "msd", "-trestart", "200", "-lateral", "z", "-sel", "0" };
    runTest(CommandLine(cmdline));
}

// for type x, should yield 8 cm^2 / s
TEST_F(MsdModuleTest, oneDimensionalDiffusionCoefficient)
{
    setOutputFile("-o", "msd.xvg", MsdMatch());
    const char* const cmdline[] = { "msd", "-trestart", "200", "-type", "x", "-sel", "0" };
    runTest(CommandLine(cmdline));
}


class MsdMolTest : public gmx::test::TrajectoryAnalysisModuleTestFixture<gmx::analysismodules::MsdInfo>
{
public:
    MsdMolTest()
    {
        double    tolerance = 1e-5;
        XvgMatch  xvg;
        XvgMatch& toler = xvg.tolerance(gmx::test::relativeToleranceAsFloatingPoint(1, tolerance));
        setOutputFile("-mol", "msdmol.xvg", toler);
    }

    void executeTest(const CommandLine& args, const char* ndxfile, const std::string& simulationName)
    {
        setInputFile("-f", simulationName + ".pdb");
        std::string tpr = fileManager().getTemporaryFilePath(".tpr");
        std::string mdp = fileManager().getTemporaryFilePath(".mdp");
        FILE*       fp  = fopen(mdp.c_str(), "w");
        fprintf(fp, "cutoff-scheme = verlet\n");
        fprintf(fp, "rcoulomb      = 0.85\n");
        fprintf(fp, "rvdw          = 0.85\n");
        fprintf(fp, "rlist         = 0.85\n");
        fclose(fp);

        // Prepare a .tpr file
        {
            CommandLine caller;
            auto        simDB = gmx::test::TestFileManager::getTestSimulationDatabaseDirectory();
            auto        base  = gmx::Path::join(simDB, simulationName);
            caller.append("grompp");
            caller.addOption("-maxwarn", 0);
            caller.addOption("-f", mdp.c_str());
            std::string gro = (base + ".pdb");
            caller.addOption("-c", gro.c_str());
            std::string top = (base + ".top");
            caller.addOption("-p", top.c_str());
            std::string ndx = (base + ".ndx");
            caller.addOption("-n", ndx.c_str());
            caller.addOption("-o", tpr.c_str());
            ASSERT_EQ(0, gmx_grompp(caller.argc(), caller.argv()));
        }
        // Run the MSD analysis
        {
            setInputFile("-n", ndxfile);
            CommandLine& cmdline = commandLine();
            cmdline.merge(args);
            cmdline.addOption("-s", tpr.c_str());
            runTest(cmdline);
        }
    }
};


// Test the diffusion per molecule output, mass weighted
TEST_F(MsdMolTest, diffMolMassWeighted)
{
    const char* const cmdline[] = { "msd", "-trestart", "200" };
    executeTest(CommandLine(cmdline), "spc5.ndx", "spc5");
}

// Test the diffusion per molecule output, non-mass weighted
TEST_F(MsdMolTest, diffMolNonMassWeighted)
{
    const char* const cmdline[] = { "msd", "-trestart", "200", "-mw", "no" };
    executeTest(CommandLine(cmdline), "spc5.ndx", "spc5");
}


} // namespace

}  // namespace gmx::test