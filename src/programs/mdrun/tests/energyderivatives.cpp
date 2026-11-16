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
 * Tests that the dV/dlambda and the virial reported by mdrun are the
 * derivatives of the potential energy reported by mdrun.
 *
 * The derivatives of individual interactions are tested at the unit
 * level, but most of the bookkeeping that assembles them (soft-core
 * kernels, PME mesh, long-range and dispersion corrections, virtual
 * sites, scaling of reference coordinates) is only exercised by a
 * complete simulation. Here single-point energies of complete
 * simulation systems are differentiated numerically with respect to
 * lambda and to a homogeneous strain of coordinates and box.
 *
 * \author Kevin Boyd <kevin44boyd@gmail.com>
 * \ingroup module_mdrun_integration_tests
 */
#include "gmxpre.h"

#include "config.h"

#include <cctype>
#include <cmath>

#include <algorithm>
#include <array>
#include <filesystem>
#include <limits>
#include <ostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "gromacs/fileio/confio.h"
#include "gromacs/fileio/enxio.h"
#include "gromacs/fileio/trrio.h"
#include "gromacs/math/units.h"
#include "gromacs/pbcutil/pbc.h"
#include "gromacs/topology/topology.h"
#include "gromacs/trajectory/energyframe.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/real.h"
#include "gromacs/utility/smalloc.h"
#include "gromacs/utility/stringutil.h"
#include "gromacs/utility/textwriter.h"
#include "gromacs/utility/vec.h"
#include "gromacs/utility/vectypes.h"

#include "testutils/cmdlinetest.h"
#include "testutils/finitedifference.h"
#include "testutils/testasserts.h"
#include "testutils/testfilemanager.h"

#include "moduletest.h"

namespace gmx
{
namespace test
{
namespace
{

//! Energy terms of one energy-file frame by name
using EnergyTerms = std::unordered_map<std::string, double>;

//! Returns all energy terms in the first frame of energy file \c filename
EnergyTerms readFirstEnergyFrame(const std::string& filename)
{
    ener_file_t energyFile = open_enx(filename, "r");
    if (!energyFile)
    {
        GMX_THROW(FileIOError("Could not open energy file " + filename + " for reading"));
    }
    const std::vector<gmx_enxnm_t> names = readEnxNames(energyFile);
    t_enxframe                     frame;
    init_enxframe(&frame);
    EnergyTerms terms;
    if (do_enx(energyFile, &frame))
    {
        for (int i = 0; i < std::min<int>(frame.nre, gmx::ssize(names)); i++)
        {
            terms[names[i].name] = frame.ener[i].e;
        }
    }
    free_enxframe(&frame);
    done_ener_file(energyFile);
    if (terms.empty())
    {
        GMX_THROW(FileIOError("Could not read an energy frame from " + filename));
    }
    return terms;
}

//! Mdp options shared by all single-point calculations
const char* const c_singlePointMdpOptions =
        "nsteps                  = 0\n"
        "dt                      = 0.001\n"
        "nstcalcenergy           = 1\n"
        "nstenergy               = 1\n"
        "comm-mode               = none\n"
        "cutoff-scheme           = Verlet\n"
        "verlet-buffer-tolerance = -1\n"
        "tcoupl                  = no\n"
        "pcoupl                  = no\n"
        "gen-vel                 = no\n";

//! Test fixture that can run several single-point calculations
class SinglePointEnergyTest : public MdrunTestFixture
{
public:
    /*! \brief Runs grompp and a zero-step mdrun and returns the energy terms
     *
     * \param[in] tag            Unique name for the output files of this calculation
     * \param[in] mdpContents    The mdp options
     * \param[in] gromppOptions  Extra command-line options for grompp
     */
    EnergyTerms singlePointEnergies(const std::string& tag,
                                    const std::string& mdpContents,
                                    const CommandLine& gromppOptions = CommandLine())
    {
        runner_.tprFileName_ = fileManager_.getTemporaryFilePath(tag + ".tpr").string();
        runner_.edrFileName_ = fileManager_.getTemporaryFilePath(tag + ".edr").string();
        runner_.useStringAsMdpFile(mdpContents);
        if (runner_.callGrompp(gromppOptions) != 0)
        {
            GMX_THROW(InternalError("grompp failed for " + tag));
        }
        if (runner_.callMdrun() != 0)
        {
            GMX_THROW(InternalError("mdrun failed for " + tag));
        }
        return readFirstEnergyFrame(runner_.edrFileName_);
    }
};

/********************************************************************
 * dV/dlambda
 */

//! Description of a free-energy single-point calculation
struct DhdlTestParams
{
    //! Name of the system in the free-energy simulation database
    std::string system;
    //! Mdp options the system needs, e.g. for decoupling
    std::string systemMdpOptions;
    //! The lambda component to differentiate with respect to: fep, coul, vdw, bonded or restraint
    std::string component;
    //! The value of lambda for \c component
    double lambda;
    //! Values of lambda components that are kept constant
    std::vector<std::pair<std::string, double>> fixedLambdas;
    //! Name of the set of mdp options for the interactions
    std::string interactions;
    //! Name of the set of mdp options for the soft-core treatment
    std::string softCore;
    //! Value for the constraints mdp option
    std::string constraints;
    //! Number of grompp warnings to accept
    int maxWarnings;
    //! Optional label to distinguish cases that differ in \c systemMdpOptions
    std::string label;
};

//! Mdp options for PME electrostatics
const std::string c_pmeMdpOptions =
        "coulombtype = pme\nrcoulomb = 1.0\nfourierspacing = 0.12\npme-order = 4\newald-rtol = "
        "1e-6\n";
//! Mdp options for Lennard-Jones with force switch
const std::string c_forceSwitchMdpOptions =
        "vdwtype = cut-off\nvdw-modifier = force-switch\nrvdw-switch = 0.8\nrvdw = 1.0\nrlist = "
        "1.0\n";

//! Mdp options for the non-bonded interactions by name
const std::unordered_map<std::string, std::string> c_interactionMdpOptions = {
    { "Pme", c_pmeMdpOptions + c_forceSwitchMdpOptions + "dispcorr = enerpres\n" },
    { "PmeNoDispCorr", c_pmeMdpOptions + c_forceSwitchMdpOptions + "dispcorr = no\n" },
    { "PmeLJPme", c_pmeMdpOptions + "vdwtype = pme\nvdw-modifier = potential-shift\nrvdw = 1.0\nrlist = 1.0\ndispcorr = no\n" },
    { "ReactionField",
      "coulombtype = reaction-field\nepsilon-rf = 0\nrcoulomb = 1.0\n" + c_forceSwitchMdpOptions
              + "dispcorr = enerpres\n" },
    { "PlainCutoff",
      "coulombtype = reaction-field\nepsilon-rf = 1\nrcoulomb = 1.0\n"
      "vdwtype = cut-off\nvdw-modifier = potential-shift\nrvdw = 1.0\nrlist = 1.0\n"
      "dispcorr = enerpres\n" },
};

//! Mdp options for the soft-core treatment by name
const std::unordered_map<std::string, std::string> c_softCoreMdpOptions = {
    { "Beutler", "sc-function = beutler\nsc-alpha = 0.5\nsc-power = 1\nsc-sigma = 0.3\n" },
    { "BeutlerPower2", "sc-function = beutler\nsc-alpha = 0.5\nsc-power = 2\nsc-sigma = 0.3\n" },
    { "BeutlerCoulomb",
      "sc-function = beutler\nsc-alpha = 0.5\nsc-power = 1\nsc-sigma = 0.3\nsc-coul = yes\n" },
    { "Gapsys", "sc-function = gapsys\nsc-alpha = 0.5\n" },
    { "NoSoftCore", "sc-alpha = 0\n" },
};

//! Helper to give test cases readable names
std::string dhdlTestName(const testing::TestParamInfo<DhdlTestParams>& info)
{
    const DhdlTestParams& params = info.param;
    std::string name = params.system + params.label + "_" + params.component + "_" + params.interactions
                       + "_" + params.softCore + "_constraints_" + params.constraints;
    std::replace_if(name.begin(), name.end(), [](const char c) { return std::isalnum(c) == 0; }, '_');
    return name;
}

//! Lets gtest print the parameters
void PrintTo(const DhdlTestParams& params, std::ostream* os)
{
    *os << params.system << params.label << " " << params.component << " " << params.interactions
        << " " << params.softCore << " constraints=" << params.constraints;
}

//! Returns the five-point stencil derivative of energy term \c name
double derivativeOfTerm(const std::array<EnergyTerms, 4>& stencil, const std::string& name, const double stepSize)
{
    return (8.0 * (stencil[2].at(name) - stencil[1].at(name)) - (stencil[3].at(name) - stencil[0].at(name)))
           / (12.0 * stepSize);
}

/*! \brief Returns a description of how all energy terms change over a five-point stencil
 *
 * This helps to find out which interaction is responsible for a mismatch.
 */
std::string describeDerivativesOfTerms(const EnergyTerms&                center,
                                       const std::array<EnergyTerms, 4>& stencil,
                                       const double                      stepSize)
{
    std::vector<std::string> names;
    for (const auto& [name, value] : center)
    {
        names.push_back(name);
    }
    std::sort(names.begin(), names.end());
    std::string description = "\nEnergy terms and their numerical derivatives:\n";
    for (const auto& name : names)
    {
        description += formatString("%-20s %18.9f %18.9f\n",
                                    name.c_str(),
                                    center.at(name),
                                    derivativeOfTerm(stencil, name, stepSize));
    }
    return description;
}

/*! \brief Test fixture for checking dV/dlambda against the potential energy
 *
 * The potential energy of one configuration is computed at five
 * equally spaced values of lambda. The (sum of the) dV/dlambda
 * term(s) reported at the central value should match the numerical
 * derivative of the potential energy.
 *
 * The lambda values are passed as a lambda array, so one component
 * can be changed while others are kept constant. Components without
 * array use the values for fep-lambdas, which are zero when not set.
 *
 * Notes on what is compared:
 * - The contribution of constraints to dH/dlambda is not a derivative
 *   of the potential energy. The systems are therefore run without
 *   constraints (but with rigid water, which is not perturbed) or with
 *   only bonds constrained that have the same length in both states.
 *   With continuation = yes the configuration is used as is.
 * - The kinetic contribution dEkin/dlambda is not checked. Topologies
 *   that perturb masses get a constant mass-lambdas array, so it is
 *   reported separately.
 */
class DhdlMatchesEnergyTest : public SinglePointEnergyTest, public ::testing::WithParamInterface<DhdlTestParams>
{
};

TEST_P(DhdlMatchesEnergyTest, Works)
{
    if (!c_canTestWithFiniteDifferences)
    {
        GTEST_SKIP() << "Numerical derivatives of the energy are only accurate in double precision";
    }
    const DhdlTestParams& params = GetParam();

    /* The energies are of the order of 1e3 kJ/mol and the terms that depend
     * on lambda are smooth, so with this spacing the truncation error of the
     * stencil is negligible and the rounding error is of the order of 1e-9.
     * The largest relative deviation observed over all combinations of systems,
     * interactions and soft-core settings is 5e-9.
     */
    const double lambdaStep        = 1e-3;
    const double relativeTolerance = 1e-7;
    const int    numStates         = 5;
    const int    centerState       = numStates / 2;

    std::string mdp = c_singlePointMdpOptions;
    // The leap-frog integrator is run by the legacy simulator
    mdp += "integrator = md\ncontinuation = yes\n";
    mdp += "free-energy = yes\nnstdhdl = 1\ndhdl-derivatives = yes\nseparate-dhdl-file = no\n";
    mdp += "constraints = " + params.constraints + "\n";
    mdp += params.systemMdpOptions;
    mdp += c_interactionMdpOptions.at(params.interactions);
    mdp += c_softCoreMdpOptions.at(params.softCore);
    mdp += params.component + "-lambdas =";
    for (int state = 0; state < numStates; state++)
    {
        mdp += formatString(" %.17g", params.lambda + (state - centerState) * lambdaStep);
    }
    mdp += "\n";
    for (const auto& [component, value] : params.fixedLambdas)
    {
        mdp += component + "-lambdas =";
        for (int state = 0; state < numStates; state++)
        {
            mdp += formatString(" %.17g", value);
        }
        mdp += "\n";
    }

    const std::string systemPath = "freeenergy/" + params.system;
    runner_.topFileName_ = TestFileManager::getInputFilePath(systemPath + "/topol.top").string();
    runner_.groFileName_ = TestFileManager::getInputFilePath(systemPath + "/conf.gro").string();
    runner_.setMaxWarn(params.maxWarnings);

    auto energiesForState = [&](const int state)
    {
        return singlePointEnergies(formatString("state%d", state),
                                   mdp + formatString("init-lambda-state = %d\n", state));
    };
    const EnergyTerms                center  = energiesForState(centerState);
    const std::array<EnergyTerms, 4> stencil = { energiesForState(centerState - 2),
                                                 energiesForState(centerState - 1),
                                                 energiesForState(centerState + 1),
                                                 energiesForState(centerState + 2) };

    double dvdl = 0;
    if (params.component == "fep")
    {
        // All components without their own lambda array change together
        for (const auto& [name, value] : center)
        {
            if (name.rfind("dV", 0) == 0)
            {
                dvdl += value;
            }
        }
    }
    else
    {
        const std::string termName = "dV" + params.component + "/dl";
        ASSERT_EQ(center.count(termName), 1) << "Missing energy term " << termName;
        dvdl = center.at(termName);
    }

    const double numericalDvdl = derivativeOfTerm(stencil, "Potential", lambdaStep);

    const FloatingPointTolerance tolerance = finiteDifferenceTolerance(
            std::max(1.0, std::abs(dvdl)), relativeTolerance, std::abs(center.at("Potential")), lambdaStep);
    EXPECT_REAL_EQ_TOL(dvdl, numericalDvdl, tolerance)
            << "dV/dlambda does not match the derivative of the potential energy"
            << describeDerivativesOfTerms(center, stencil, lambdaStep);
}

//! Returns the test cases for dV/dlambda
std::vector<DhdlTestParams> dhdlTestCases()
{
    using Lambdas = std::vector<std::pair<std::string, double>>;
    //! Combination of interaction and soft-core settings
    using Settings = std::pair<std::string, std::string>;

    /* All 25 combinations of the interaction settings and soft-core settings were found to work for
     * all systems below. To limit the run time, we test selections in which each setting occurs.
     */
    const std::vector<Settings> manySettings = {
        { "Pme", "Beutler" },
        { "Pme", "Gapsys" },
        { "PmeNoDispCorr", "NoSoftCore" },
        { "PmeLJPme", "BeutlerCoulomb" },
        { "PmeLJPme", "Gapsys" },
        { "ReactionField", "BeutlerPower2" },
        { "ReactionField", "Gapsys" },
        { "PlainCutoff", "Beutler" },
    };
    /* grompp does not accept changing Coulomb and Van der Waals together
     * with soft-core for Van der Waals only when lambda arrays are used for
     * more than fep-lambdas, which we need for systems with perturbed masses.
     */
    const std::vector<Settings> manySettingsWithSoftCoreCoulomb = {
        { "Pme", "BeutlerCoulomb" },       { "Pme", "Gapsys" },
        { "PmeNoDispCorr", "NoSoftCore" }, { "PmeLJPme", "BeutlerCoulomb" },
        { "ReactionField", "Gapsys" },     { "PlainCutoff", "BeutlerCoulomb" },
    };
    const std::vector<Settings> fewSettings                    = { { "Pme", "Beutler" },
                                                                   { "ReactionField", "Gapsys" },
                                                                   { "PmeLJPme", "BeutlerCoulomb" } };
    const std::vector<Settings> fewSettingsWithSoftCoreCoulomb = { { "Pme", "BeutlerCoulomb" },
                                                                   { "ReactionField", "Gapsys" } };
    const std::vector<Settings> oneSetting                     = { { "Pme", "Beutler" } };

    const std::string decoupleAsn =
            "couple-moltype = ASN\ncouple-lambda0 = none\ncouple-lambda1 = vdw-q\n";
    const std::string decoupleChloride =
            "couple-moltype = CL\ncouple-lambda0 = none\ncouple-lambda1 = vdw-q\n";
    const std::string positionRestraints = "define = -DPOSRES\n";
    // Keeps the kinetic energy independent of lambda
    const Lambdas fixedMasses = { { "mass", 0.0 } };

    //! A system with the lambda component to change, constant lambdas and the settings to use
    struct SystemCase
    {
        std::string           system;
        std::string           mdpOptions;
        std::string           component;
        double                lambda;
        Lambdas               fixedLambdas;
        std::vector<Settings> settings;
        int                   maxWarnings;
        std::string           label = "";
    };
    const std::vector<SystemCase> systemCases = {
        // Decoupling of a molecule through mdp options
        { "coulandvdwsequential_coul", decoupleAsn + "couple-intramol = no\n", "coul", 0.4, { { "vdw", 1.0 } }, manySettings, 1 },
        { "coulandvdwsequential_vdw", decoupleAsn + "couple-intramol = no\n", "vdw", 0.6, { { "coul", 0.0 } }, manySettings, 1 },
        { "coulandvdwtogether", decoupleAsn + "couple-intramol = no\n", "fep", 0.5, {}, manySettings, 1 },
        { "coulandvdwintramol", decoupleAsn + "couple-intramol = yes\n", "fep", 0.5, {}, fewSettings, 1 },
        { "coulandvdwtogether-net-charge", decoupleAsn + "couple-intramol = no\n", "fep", 0.5, {}, fewSettings, 2 },
        { "coulandvdwtogether-decouple-counter-charge",
          decoupleChloride + "couple-intramol = no\n",
          "fep",
          0.5,
          {},
          fewSettings,
          2 },
        // Perturbations through the topology
        { "vdwalone", "", "fep", 0.25, {}, fewSettings, 1 },
        { "transformAtoB", "", "fep", 0.6, fixedMasses, manySettingsWithSoftCoreCoulomb, 1 },
        { "relative", "", "fep", 0.3, fixedMasses, manySettingsWithSoftCoreCoulomb, 11 },
        { "relative", "", "coul", 0.5, { { "vdw", 0.0 }, { "bonded", 0.0 } }, fewSettings, 11 },
        { "relative", "", "vdw", 0.6, { { "coul", 1.0 }, { "bonded", 0.6 } }, fewSettings, 11 },
        { "relative", "", "bonded", 0.6, { { "coul", 1.0 }, { "vdw", 0.6 } }, oneSetting, 11 },
        { "relative-position-restraints",
          positionRestraints,
          "restraint",
          0.5,
          { { "coul", 1.0 }, { "vdw", 0.6 }, { "bonded", 0.6 } },
          oneSetting,
          11 },
        { "relative-position-restraints",
          positionRestraints + "refcoord-scaling = com\n",
          "fep",
          0.4,
          fixedMasses,
          fewSettingsWithSoftCoreCoulomb,
          11,
          "_RefcoordScalingCom" },
        { "relative-position-restraints",
          positionRestraints + "refcoord-scaling = all\n",
          "fep",
          0.4,
          fixedMasses,
          fewSettingsWithSoftCoreCoulomb,
          11,
          "_RefcoordScalingAll" },
        { "restraints", "", "restraint", 0.4, {}, oneSetting, 1 },
        { "restraints", "", "fep", 0.4, {}, oneSetting, 1 },
        { "cmap-perturbation", "", "fep", 0.5, {}, oneSetting, 1 },
    };

    std::vector<DhdlTestParams> cases;
    for (const SystemCase& systemCase : systemCases)
    {
        for (const auto& [interactions, softCore] : systemCase.settings)
        {
            cases.push_back({ systemCase.system,
                              systemCase.mdpOptions,
                              systemCase.component,
                              systemCase.lambda,
                              systemCase.fixedLambdas,
                              interactions,
                              softCore,
                              "none",
                              systemCase.maxWarnings,
                              systemCase.label });
        }
    }
    /* With decoupling through mdp options no bonded parameters are perturbed,
     * so constraining bonds does not contribute to dH/dlambda.
     */
    for (const std::string intramolecular : { "no", "yes" })
    {
        cases.push_back({ intramolecular == "yes" ? "coulandvdwintramol" : "coulandvdwtogether",
                          decoupleAsn + "couple-intramol = " + intramolecular + "\n",
                          "fep",
                          0.5,
                          {},
                          "Pme",
                          "Beutler",
                          "h-bonds",
                          1,
                          "" });
    }
    return cases;
}

INSTANTIATE_TEST_SUITE_P(FreeEnergySystems, DhdlMatchesEnergyTest, ::testing::ValuesIn(dhdlTestCases()), dhdlTestName);

/********************************************************************
 * Virial and pressure
 */

//! Description of a single-point calculation for checking the virial
struct VirialTestParams
{
    //! Name of the test case
    std::string name;
    //! Name of the system in the simulation database
    std::string system;
    //! Contents for a topology file to use instead of the one in the database, when not empty
    std::string topology;
    //! The mdp options
    std::string mdpOptions;
    //! Magnitude in nm of displacements to apply to the coordinates
    double displacement;
    //! The strain elements to test, as pairs of the dimension displaced and the dimension it is proportional to
    std::vector<std::pair<int, int>> strains;
    //! Number of grompp warnings to accept
    int maxWarnings;
};

//! Helper to give test cases readable names
std::string virialTestName(const testing::TestParamInfo<VirialTestParams>& info)
{
    return info.param.name;
}

//! Lets gtest print the parameters
void PrintTo(const VirialTestParams& params, std::ostream* os)
{
    *os << params.name;
}

//! A configuration
struct Configuration
{
    //! Coordinates
    std::vector<RVec> x;
    //! Box
    matrix box;
};

//! Reads a configuration from \c filename
Configuration readConfiguration(const std::filesystem::path& filename)
{
    t_topology    topology;
    rvec*         x = nullptr;
    PbcType       pbcType;
    Configuration configuration;
    read_tps_conf(filename, &topology, &pbcType, &x, nullptr, configuration.box, FALSE);
    configuration.x.assign(x, x + topology.atoms.nr);
    sfree(x);
    done_top(&topology);
    return configuration;
}

/*! \brief Test fixture for checking virial and pressure against the potential energy
 *
 * A homogeneous strain that displaces the coordinates and box vectors
 * along dimension i proportional to their component along dimension j
 * changes the potential energy by -sum_atoms f_i x_j times the strain,
 * which is twice element j,i of the virial. Without kinetic energy
 * the pressure is minus two times the virial divided by the volume.
 * Only strains that keep the box lower triangular can be applied.
 *
 * The strained configurations are written to a full-precision
 * trajectory, from which grompp reads coordinates and box for a
 * single-point calculation. Thus reference positions for position
 * restraints, which come from the unstrained structure file, are
 * treated by mdrun as when the box changes due to pressure coupling.
 * Virtual sites are constructed by mdrun from the strained positions
 * of the constructing atoms.
 *
 * Notes on what is compared:
 * - Constraints are not used, as the constraint virial is not a derivative
 *   of the potential energy.
 * - The velocities are zero and the md-vv integrator reports the kinetic
 *   energy for the velocities at the time of the configuration, so
 *   the reported pressure has no kinetic part.
 * - The mesh part of (LJ-)PME is not an exact function of the coordinates,
 *   but the reported virial is the exact derivative of the mesh energy
 *   under strain at a fixed number of grid points, which is what we
 *   compare with.
 * - With cut-offs, pairs can cross the cut-off within the stencil.
 *   The energy is continuous, but its derivative is only continuous
 *   when the force is zero at the cut-off. With the small strain
 *   step used here few pairs cross, and we only use interactions for
 *   which the force is small at the cut-off. With plain cut-off
 *   electrostatics a single crossing pair gives an error of several
 *   kJ/mol, so that is not tested.
 * - With dispcorr = Ener the correction is, as requested, not applied to
 *   the pressure, so the identity does not hold and that is not tested.
 */
class VirialMatchesEnergyTest :
    public SinglePointEnergyTest,
    public ::testing::WithParamInterface<VirialTestParams>
{
};

TEST_P(VirialMatchesEnergyTest, Works)
{
    if (!c_canTestWithFiniteDifferences)
    {
        GTEST_SKIP() << "Numerical derivatives of the energy are only accurate in double precision";
    }
    const VirialTestParams& params = GetParam();

    /* With energies up to 1e5 kJ/mol the rounding error of the numerical
     * derivative is of the order of 1e-4 kJ/mol with this step, on virials of
     * up to 1e5 kJ/mol. With a 100 times larger step the deviations are
     * 100 times larger, as more pairs cross the cut-off within the stencil.
     * The largest deviation observed, relative to the largest virial element
     * of the system, is 2e-8 for unperturbed systems. It is 1e-7 for
     * perturbed interactions with PME, where the perturbed kernel interpolates
     * energy and force of the Ewald correction from tables.
     */
    const double strainStep        = 1e-6;
    const double relativeTolerance = 5e-7;

    runner_.useTopGroAndNdxFromDatabase(params.system);
    runner_.setMaxWarn(params.maxWarnings);
    if (!params.topology.empty())
    {
        runner_.topFileName_ = fileManager_.getTemporaryFilePath("custom.top").string();
        TextWriter::writeFileFromString(runner_.topFileName_, params.topology);
    }

    Configuration reference = readConfiguration(runner_.groFileName_);
    const int     numAtoms  = gmx::ssize(reference.x);
    // Displace the atoms in a reproducible, irregular fashion
    for (int atom = 0; atom < numAtoms; atom++)
    {
        for (int d = 0; d < DIM; d++)
        {
            reference.x[atom][d] += params.displacement * std::sin(1.7 * atom + 2.3 * d + 0.4);
        }
    }

    /* Write all configurations as frames of a full-precision trajectory,
     * of which grompp can select a frame by its time. Frame 0 is
     * the reference configuration, the others are strained.
     */
    const std::string trajectoryFileName = fileManager_.getTemporaryFilePath("strained.trr").string();
    const std::array<int, 4> stencilPoints = { -2, -1, 1, 2 };
    {
        t_fileio* trajectoryFile = gmx_trr_open(trajectoryFileName, "w");
        int       frameIndex     = 0;
        gmx_trr_write_frame(trajectoryFile,
                            frameIndex,
                            frameIndex,
                            0,
                            reference.box,
                            numAtoms,
                            as_rvec_array(reference.x.data()),
                            nullptr,
                            nullptr);
        frameIndex++;
        std::vector<RVec> x(numAtoms);
        for (const auto& [strainDim, distanceDim] : params.strains)
        {
            for (const int stencilPoint : stencilPoints)
            {
                const double strain = stencilPoint * strainStep;
                for (int atom = 0; atom < numAtoms; atom++)
                {
                    x[atom] = reference.x[atom];
                    x[atom][strainDim] += strain * reference.x[atom][distanceDim];
                }
                matrix box;
                copy_mat(reference.box, box);
                for (int boxVector = 0; boxVector < DIM; boxVector++)
                {
                    box[boxVector][strainDim] += strain * reference.box[boxVector][distanceDim];
                }
                gmx_trr_write_frame(
                        trajectoryFile, frameIndex, frameIndex, 0, box, numAtoms, as_rvec_array(x.data()), nullptr, nullptr);
                frameIndex++;
            }
        }
        gmx_trr_close(trajectoryFile);
    }

    // Note that virtual sites are only constructed at the start with continuation = no
    const std::string mdp = std::string(c_singlePointMdpOptions)
                            + "integrator = md-vv\nconstraints = none\ncontinuation = no\n"
                            + params.mdpOptions;

    auto energiesForFrame = [&](const int frameIndex)
    {
        CommandLine gromppOptions;
        gromppOptions.addOption("-t", trajectoryFileName);
        gromppOptions.addOption("-time", static_cast<double>(frameIndex));
        return singlePointEnergies(formatString("frame%d", frameIndex), mdp, gromppOptions);
    };

    const EnergyTerms center = energiesForFrame(0);
    ASSERT_EQ(center.at("Kinetic En."), 0.0)
            << "The pressure can only be compared without kinetic energy";
    const double volume = det(reference.box);

    // Collect the results for all strains first, as we need the magnitude of the virial
    struct StrainResult
    {
        std::string                element;
        double                     energyDerivative;
        std::array<EnergyTerms, 4> stencil;
    };
    std::vector<StrainResult> results;
    double                    virialMagnitude = 1.0;
    int                       frameIndex      = 1;
    const char* const         dimNames        = "XYZ";
    for (const auto& [strainDim, distanceDim] : params.strains)
    {
        StrainResult result;
        for (EnergyTerms& terms : result.stencil)
        {
            terms = energiesForFrame(frameIndex++);
        }
        result.energyDerivative = derivativeOfTerm(result.stencil, "Potential", strainStep);
        // The virial is -0.5 sum_atoms x (x) f
        result.element = std::string(1, dimNames[distanceDim]) + dimNames[strainDim];
        virialMagnitude = std::max(virialMagnitude, std::abs(2 * center.at("Vir-" + result.element)));
        results.push_back(result);
    }

    const double virialTolerance = relativeTolerance * virialMagnitude
                                   + 100 * std::numeric_limits<double>::epsilon()
                                             * std::abs(center.at("Potential")) / strainStep;
    for (const StrainResult& result : results)
    {
        const double twiceVirial       = 2 * center.at("Vir-" + result.element);
        const double pressure          = center.at("Pres-" + result.element);
        const double numericalPressure = -c_presfac * result.energyDerivative / volume;

        EXPECT_REAL_EQ_TOL(twiceVirial, result.energyDerivative, absoluteTolerance(virialTolerance))
                << "Virial element " << result.element
                << " does not match the strain derivative of the potential energy"
                << describeDerivativesOfTerms(center, result.stencil, strainStep);
        EXPECT_REAL_EQ_TOL(pressure, numericalPressure, absoluteTolerance(virialTolerance * c_presfac / volume))
                << "Pressure element " << result.element
                << " does not match the strain derivative of the potential energy";
    }
}

//! Strain elements that keep a rectangular box rectangular
const std::vector<std::pair<int, int>> c_diagonalStrains = { { XX, XX }, { YY, YY }, { ZZ, ZZ } };
//! All strain elements that keep the box lower triangular
const std::vector<std::pair<int, int>> c_allStrains = { { XX, XX }, { YY, YY }, { ZZ, ZZ },
                                                        { XX, YY }, { XX, ZZ }, { YY, ZZ } };
//! The strain elements that can be applied with walls
const std::vector<std::pair<int, int>> c_wallStrains = { { XX, XX }, { YY, YY }, { ZZ, ZZ }, { XX, YY } };

//! Returns the test cases for the virial
std::vector<VirialTestParams> virialTestCases()
{
    // 216 flexible water molecules in a cubic box of 1.86 nm
    const std::string flexibleWater = "define = -DFLEXIBLE\n";
    const std::string cutoffs       = "rlist = 0.9\nrcoulomb = 0.9\nrvdw = 0.9\n";
    const std::string ljForceSwitch =
            "vdwtype = cut-off\nvdw-modifier = force-switch\nrvdw-switch = 0.7\n";
    // The force is zero at the cut-off
    const std::string reactionField = "coulombtype = reaction-field\nepsilon-rf = 0\n";
    const std::string pme =
            "coulombtype = pme\nfourier-nx = 16\nfourier-ny = 16\nfourier-nz = 16\n";
    const std::string water = flexibleWater + cutoffs;

    // Water with anisotropic or flat-bottomed position restraints on the oxygens
    const std::string restrainedWaterTopology =
            "#include \"oplsaa.ff/forcefield.itp\"\n"
            "#include \"oplsaa.ff/tip3p.itp\"\n"
            "[ position_restraints ]\n"
            "#ifdef FLAT_BOTTOM\n"
            "1 2 1 0.01 3000\n"
            "#else\n"
            "1 1 1000 2000 3000\n"
            "#endif\n"
            "[ system ]\nRestrained water\n"
            "[ molecules ]\nSOL 216\n";

    // All water molecules half decoupled, so all interactions go through the perturbed kernel
    const std::string decoupleWater =
            "free-energy = yes\ninit-lambda = 0.5\ncouple-moltype = SOL\ncouple-lambda0 = "
            "vdw-q\n"
            "couple-lambda1 = none\ncouple-intramol = no\nsc-alpha = 0.5\nsc-power = "
            "1\nsc-sigma = "
            "0.3\n";

    // One hexadecane molecule with virtual sites of all types in a box of 5 nm
    const std::string virtualSites = "define = -DEXCLUDE_COINCIDING_SITE\n";

    // Two dipoles of two charged atoms each in a box of 4 nm
    const std::string dipoles =
            "rlist = 1.5\nrcoulomb = 1.5\nrvdw = 1.5\ncoulombtype = pme\npme-order = 4\n"
            "fourier-nx = 20\nfourier-ny = 20\n"
            + ljForceSwitch;
    const std::string walls =
            "pbc = xy\nnwall = 2\nwall-atomtype = C C\nwall-type = 9-3\n"
            "wall-density = 20 20\nwall-ewald-zfac = 2\nfourier-nz = 40\n";

    std::vector<VirialTestParams> cases;
    cases.push_back(
            { "Water_ReactionField", "spc216", "", water + ljForceSwitch + reactionField, 0.0, c_allStrains, 0 });
    cases.push_back({ "Water_ReactionField_LJPotentialShift_DispCorrEnerPres",
                      "spc216",
                      "",
                      water + reactionField + "dispcorr = enerpres\n",
                      0.0,
                      c_diagonalStrains,
                      0 });
    cases.push_back({ "Water_ReactionField_LJForceSwitch_DispCorrEnerPres",
                      "spc216",
                      "",
                      water + ljForceSwitch + reactionField + "dispcorr = enerpres\n",
                      0.0,
                      c_diagonalStrains,
                      0 });
    cases.push_back({ "Water_Pme", "spc216", "", water + ljForceSwitch + pme, 0.0, c_allStrains, 0 });
    cases.push_back({ "Water_PmeOrder5",
                      "spc216",
                      "",
                      water + ljForceSwitch + pme + "pme-order = 5\newald-rtol = 1e-7\n",
                      0.0,
                      c_allStrains,
                      0 });
    cases.push_back(
            { "Water_Pme_LJPme", "spc216", "", water + pme + "vdwtype = pme\n", 0.0, c_allStrains, 0 });
    cases.push_back({ "Water_Pme_LJPmeLorentzBerthelot",
                      "spc216",
                      "",
                      water + pme + "vdwtype = pme\nlj-pme-comb-rule = lorentz-berthelot\n",
                      0.0,
                      c_allStrains,
                      1 });
    cases.push_back({ "Water_Ewald",
                      "spc216",
                      "",
                      water + ljForceSwitch + "coulombtype = ewald\nfourier-nx = 12\nfourier-ny = 12\nfourier-nz = 12\n",
                      0.0,
                      c_allStrains,
                      0 });
    cases.push_back({ "Water_Perturbed_Pme_Beutler",
                      "spc216",
                      "",
                      water + ljForceSwitch + pme + decoupleWater + "dispcorr = enerpres\n",
                      0.0,
                      c_allStrains,
                      0 });
    cases.push_back({ "Water_Perturbed_ReactionField_Gapsys",
                      "spc216",
                      "",
                      water + ljForceSwitch + reactionField + decoupleWater + "sc-function = gapsys\n",
                      0.0,
                      c_allStrains,
                      0 });
    cases.push_back({ "Water_Perturbed_Pme_LJPme",
                      "spc216",
                      "",
                      water + pme + "vdwtype = pme\n" + decoupleWater,
                      0.0,
                      c_allStrains,
                      0 });
    for (const std::string scaling : { "no", "all", "com" })
    {
        cases.push_back({ "Water_PositionRestraints_RefcoordScaling_" + scaling,
                          "spc216",
                          restrainedWaterTopology,
                          water + ljForceSwitch + reactionField + "refcoord-scaling = " + scaling + "\n",
                          0.02,
                          c_allStrains,
                          0 });
        cases.push_back({ "Water_FlatBottomedRestraints_RefcoordScaling_" + scaling,
                          "spc216",
                          restrainedWaterTopology,
                          "define = -DFLEXIBLE -DFLAT_BOTTOM\n" + cutoffs + ljForceSwitch
                                  + reactionField + "refcoord-scaling = " + scaling + "\n",
                          0.02,
                          c_allStrains,
                          0 });
    }
    cases.push_back({ "VirtualSites_ReactionField",
                      "vsite_test",
                      "",
                      virtualSites + cutoffs + ljForceSwitch + reactionField,
                      0.0,
                      c_allStrains,
                      0 });
    cases.push_back(
            { "VirtualSites_Pme",
              "vsite_test",
              "",
              virtualSites + "rlist = 1.2\nrcoulomb = 1.2\nrvdw = 1.2\n" + ljForceSwitch
                      + "coulombtype = pme\nfourier-nx = 32\nfourier-ny = 32\nfourier-nz = 32\n",
              0.0,
              c_allStrains,
              0 });
    cases.push_back({ "Dipoles_Pme", "dipoles", "", dipoles + "fourier-nz = 20\n", 0.0, c_allStrains, 0 });
    cases.push_back({ "Dipoles_Pme_EpsilonSurface",
                      "dipoles",
                      "",
                      dipoles + "fourier-nz = 20\nepsilon-surface = 1\n",
                      0.0,
                      c_allStrains,
                      0 });
    cases.push_back({ "Dipoles_Pme_Walls", "dipoles", "", dipoles + walls, 0.0, c_wallStrains, 1 });
    cases.push_back(
            { "Dipoles_Pme_Walls_3DC", "dipoles", "", dipoles + walls + "ewald-geometry = 3dc\n", 0.0, c_wallStrains, 0 });
    cases.push_back({ "Dipoles_Pme_Walls_3DC_LinearPotential",
                      "dipoles",
                      "",
                      dipoles + walls + "ewald-geometry = 3dc\nwall-r-linpot = 1.1\n",
                      0.0,
                      c_wallStrains,
                      0 });
    return cases;
}

INSTANTIATE_TEST_SUITE_P(Systems, VirialMatchesEnergyTest, ::testing::ValuesIn(virialTestCases()), virialTestName);

} // namespace
} // namespace test
} // namespace gmx
