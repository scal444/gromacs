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
 * Tests that the long-range dispersion correction for the energy matches
 * an independent integration of the modified potential and that the
 * corrections for the pressure and dV/dlambda are consistent with the energy
 *
 * \author Kevin Boyd <kevin44boyd@gmail.com>
 * \ingroup module_mdlib
 */
#include "gmxpre.h"

#include "gromacs/mdlib/dispersioncorrection.h"

#include <cctype>
#include <cmath>

#include <algorithm>
#include <array>
#include <string>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "gromacs/math/units.h"
#include "gromacs/math/utilities.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/mdtypes/interaction_const.h"
#include "gromacs/mdtypes/md_enums.h"
#include "gromacs/topology/atoms.h"
#include "gromacs/topology/forcefieldparameters.h"
#include "gromacs/topology/topology.h"
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

//! The Van der Waals treatment: interaction type, modifier and LJ-PME combination rule
struct VdwTreatment
{
    VanDerWaalsType      type;
    InteractionModifiers modifier;
    LongRangeVdW         ljPmeCombinationRule;
};

//! Test parameters: Van der Waals treatment, correction type, whether atom types are perturbed and lambda
using DispersionCorrectionTestParameters = std::tuple<VdwTreatment, DispersionCorrectionType, bool, real>;

//! The functional form of the modification of the potential within the cut-off
enum class ModifiedForm
{
    PlainCutoff,
    PotentialShift,
    ForceSwitch,
    PotentialSwitch
};

//! Returns the functional form used for \c vdwTreatment
ModifiedForm modifiedForm(const VdwTreatment& vdwTreatment)
{
    if (vdwTreatment.type == VanDerWaalsType::Shift)
    {
        return ModifiedForm::ForceSwitch;
    }
    if (vdwTreatment.type == VanDerWaalsType::Switch)
    {
        return ModifiedForm::PotentialSwitch;
    }
    switch (vdwTreatment.modifier)
    {
        case InteractionModifiers::PotShift: return ModifiedForm::PotentialShift;
        case InteractionModifiers::ForceSwitch: return ModifiedForm::ForceSwitch;
        case InteractionModifiers::PotSwitch: return ModifiedForm::PotentialSwitch;
        default: return ModifiedForm::PlainCutoff;
    }
}

/*! \brief The part of r^-power that is missing from the modified interaction within the cut-off
 *
 * These are the differences between the plain and the modified
 * interactions as given in the reference manual, written such that
 * there is no cancellation at short distances.
 */
double missingPotentialWithinCutoff(const ModifiedForm form,
                                    const int          power,
                                    const double       r,
                                    const double       rSwitch,
                                    const double       rCutoff)
{
    switch (form)
    {
        case ModifiedForm::PlainCutoff: return 0;
        case ModifiedForm::PotentialShift: return std::pow(rCutoff, -power);
        case ModifiedForm::ForceSwitch:
        {
            const double width = rCutoff - rSwitch;
            const double a     = -power * ((power + 4) * rCutoff - (power + 1) * rSwitch)
                             / (std::pow(rCutoff, power + 2) * width * width);
            const double b = power * ((power + 3) * rCutoff - (power + 1) * rSwitch)
                             / (std::pow(rCutoff, power + 2) * width * width * width);
            const double c = std::pow(rCutoff, -power) - a / 3 * width * width * width
                             - b / 4 * width * width * width * width;
            const double d = std::max(r - rSwitch, 0.0);
            return a / 3 * d * d * d + b / 4 * d * d * d * d + c;
        }
        case ModifiedForm::PotentialSwitch:
        {
            const double t = std::max(r - rSwitch, 0.0) / (rCutoff - rSwitch);
            // One minus the switch function
            return std::pow(r, -power) * t * t * t * (10 - 15 * t + 6 * t * t);
        }
    }
    return 0;
}

/*! \brief Returns the integral over all space of the part of r^-power that is missing from the modified interaction
 *
 * Uses Simpson's rule for the part within the cut-off.
 */
double integralOfMissingPotential(const ModifiedForm form, const int power, const double rSwitch, const double rCutoff)
{
    const int    numIntervals = 20000;
    const double spacing      = rCutoff / numIntervals;
    double       sum          = 0;
    for (int i = 1; i <= numIntervals; i++)
    {
        // The integrand is zero at zero distance
        const double r      = i * spacing;
        const double weight = (i == numIntervals ? 1 : (i % 2 == 1 ? 4 : 2));
        sum += weight * r * r * missingPotentialWithinCutoff(form, power, r, rSwitch, rCutoff);
    }
    const double withinCutoff = sum * spacing / 3;
    const double beyondCutoff = std::pow(rCutoff, 3 - power) / (power - 3);

    return 4 * M_PI * (withinCutoff + beyondCutoff);
}

//! Test fixture for the dispersion correction
class DispersionCorrectionTest : public ::testing::TestWithParam<DispersionCorrectionTestParameters>
{
public:
    DispersionCorrectionTest()
    {
        std::tie(vdwTreatment_, ir_.eDispCorr, haveFep_, lambda_) = GetParam();

        ir_.vdwtype                = vdwTreatment_.type;
        ir_.ljpme_combination_rule = vdwTreatment_.ljPmeCombinationRule;
        ir_.efep = (haveFep_ ? FreeEnergyPerturbationType::Yes : FreeEnergyPerturbationType::No);

        ic_.vdw.type               = vdwTreatment_.type;
        ic_.vdw.modifier           = vdwTreatment_.modifier;
        ic_.vdw.cutoff             = c_cutoff;
        ic_.vdw.switchDistance     = c_switchDistance;
        ic_.vdw.ewaldCoeff         = 3;
        ic_.vdw.pmeCombinationRule = vdwTreatment_.ljPmeCombinationRule;

        // Parameters that do not follow a combination rule
        mtop_.ffparams.atnr = c_numAtomTypes;
        mtop_.ffparams.iparams.resize(c_numAtomTypes * c_numAtomTypes);
        for (int i = 0; i < c_numAtomTypes; i++)
        {
            for (int j = 0; j < c_numAtomTypes; j++)
            {
                t_iparams& iparams = mtop_.ffparams.iparams[i * c_numAtomTypes + j];
                iparams.lj.c6      = 0.002 * (1 + i) * (1 + j) * (i == j ? 1 : 1.1);
                iparams.lj.c12     = 2e-6 * (1 + 2 * i) * (1 + 2 * j) * (i == j ? 1 : 0.8);
            }
        }

        // Molecules of three atoms, with exclusions and perturbed types, and single atoms
        mtop_.moltype.resize(2);
        {
            const std::array<int, 3> typeA = { 0, 1, 0 };
            const std::array<int, 3> typeB = { 0, 2, 1 };
            t_atoms&                 atoms = mtop_.moltype[0].atoms;
            init_t_atoms(&atoms, 3, false);
            for (int i = 0; i < 3; i++)
            {
                atoms.atom[i].type  = typeA[i];
                atoms.atom[i].typeB = (haveFep_ ? typeB[i] : typeA[i]);
            }
            mtop_.moltype[0].excls.pushBack(std::vector<int>{ 0, 1 });
            mtop_.moltype[0].excls.pushBack(std::vector<int>{ 0, 1, 2 });
            mtop_.moltype[0].excls.pushBack(std::vector<int>{ 1, 2 });
        }
        {
            t_atoms& atoms = mtop_.moltype[1].atoms;
            init_t_atoms(&atoms, 1, false);
            atoms.atom[0].type  = 2;
            atoms.atom[0].typeB = 2;
            mtop_.moltype[1].excls.pushBack(std::vector<int>{ 0 });
        }
        mtop_.molblock.resize(2);
        mtop_.molblock[0].type = 0;
        mtop_.molblock[0].nmol = 20;
        mtop_.molblock[1].type = 1;
        mtop_.molblock[1].nmol = 15;
        mtop_.natoms           = 20 * 3 + 15;

        clear_mat(box_);
        box_[XX][XX] = 3.0;
        box_[YY][XX] = 0.4;
        box_[YY][YY] = 3.2;
        box_[ZZ][XX] = -0.3;
        box_[ZZ][YY] = 0.5;
        box_[ZZ][ZZ] = 3.5;
    }

    //! Returns the C6 used on the LJ-PME grid for a pair of types
    double gridC6(const int typeI, const int typeJ) const
    {
        const auto&  iparams = mtop_.ffparams.iparams;
        const double c6I     = iparams[typeI * (c_numAtomTypes + 1)].lj.c6;
        const double c6J     = iparams[typeJ * (c_numAtomTypes + 1)].lj.c6;
        if (vdwTreatment_.ljPmeCombinationRule == LongRangeVdW::Geom)
        {
            return std::sqrt(c6I * c6J);
        }
        else
        {
            const double c12I   = iparams[typeI * (c_numAtomTypes + 1)].lj.c12;
            const double c12J   = iparams[typeJ * (c_numAtomTypes + 1)].lj.c12;
            const double sigmaI = std::pow(c12I / c6I, 1.0 / 6.0);
            const double sigmaJ = std::pow(c12J / c6J, 1.0 / 6.0);
            // Four times epsilon
            const double epsilonI = c6I * c6I / c12I;
            const double epsilonJ = c6J * c6J / c12J;
            return std::sqrt(epsilonI * epsilonJ) * std::pow(0.5 * (sigmaI + sigmaJ), 6);
        }
    }

    /*! \brief Returns the average C6 and C12 that are not accounted for beyond the cut-off
     *
     * Loops over all non-excluded atom pairs in the system.
     */
    std::array<double, 2> averageParameters(const bool useStateB) const
    {
        // Atom type and molecule index for all atoms
        std::vector<int> type;
        std::vector<int> molecule;
        // Index within the molecule of all atoms
        std::vector<int> indexInMolecule;
        int              numMolecules = 0;
        for (const gmx_molblock_t& molblock : mtop_.molblock)
        {
            const t_atoms& atoms = mtop_.moltype[molblock.type].atoms;
            for (int mol = 0; mol < molblock.nmol; mol++)
            {
                for (int i = 0; i < atoms.nr; i++)
                {
                    type.push_back(useStateB ? atoms.atom[i].typeB : atoms.atom[i].type);
                    molecule.push_back(numMolecules);
                    indexInMolecule.push_back(i);
                }
                numMolecules++;
            }
        }

        double sumC6    = 0;
        double sumC12   = 0;
        int    numPairs = 0;
        for (size_t i = 0; i < type.size(); i++)
        {
            for (size_t j = i + 1; j < type.size(); j++)
            {
                // All atoms within the molecule of three atoms exclude each other,
                // except for the first and the last
                if (molecule[i] == molecule[j] && indexInMolecule[j] - indexInMolecule[i] == 1)
                {
                    continue;
                }
                const t_iparams& iparams = mtop_.ffparams.iparams[type[i] * c_numAtomTypes + type[j]];
                sumC6 += iparams.lj.c6;
                if (usingLJPme(vdwTreatment_.type))
                {
                    sumC6 -= gridC6(type[i], type[j]);
                }
                sumC12 += iparams.lj.c12;
                numPairs++;
            }
        }

        return { sumC6 / numPairs, sumC12 / numPairs };
    }

    //! Returns the average C6 and C12 at \c lambda
    std::array<double, 2> averageParameters(const double lambda) const
    {
        const std::array<double, 2> stateA = averageParameters(false);
        const std::array<double, 2> stateB = averageParameters(true);

        return { (1 - lambda) * stateA[0] + lambda * stateB[0],
                 (1 - lambda) * stateA[1] + lambda * stateB[1] };
    }

    //! Returns whether repulsion is corrected for
    bool correctRepulsion() const
    {
        return ir_.eDispCorr == DispersionCorrectionType::AllEner
               || ir_.eDispCorr == DispersionCorrectionType::AllEnerPres;
    }

    //! Returns whether the pressure is corrected
    bool correctPressure() const
    {
        return ir_.eDispCorr == DispersionCorrectionType::EnerPres
               || ir_.eDispCorr == DispersionCorrectionType::AllEnerPres;
    }

    //! Returns the correction for the box scaled by \c boxScaling at \c lambda
    DispersionCorrection::Correction calculate(const double boxScaling, const real lambda) const
    {
        const DispersionCorrection dispersionCorrection(mtop_, ir_, false, ic_, "unused.xvg");

        matrix box;
        msmul(box_, boxScaling, box);

        return dispersionCorrection.calculate(box, lambda);
    }

protected:
    static constexpr int  c_numAtomTypes   = 3;
    static constexpr real c_cutoff         = 1.0;
    static constexpr real c_switchDistance = 0.75;

    VdwTreatment        vdwTreatment_;
    bool                haveFep_;
    real                lambda_;
    t_inputrec          ir_;
    interaction_const_t ic_;
    gmx_mtop_t          mtop_;
    matrix              box_;
};

TEST_P(DispersionCorrectionTest, EnergyMatchesIntegralOfMissingPotential)
{
    const ModifiedForm          form       = modifiedForm(vdwTreatment_);
    const std::array<double, 2> parameters = averageParameters(lambda_);
    const double                density    = mtop_.natoms / det(box_);

    // Each atom misses half of the interaction with the average density
    // around it. Within the cut-off the pair density is lower than
    // the average, as there is one atom fewer to interact with.
    auto missingEnergyPerAtom = [&](const int power)
    {
        return 0.5
               * (density * integralOfMissingPotential(form, power, c_switchDistance, c_cutoff)
                  - missingPotentialWithinCutoff(
                          form, power, 0.5 * c_switchDistance, c_switchDistance, c_cutoff));
    };
    double referenceEnergy = -parameters[0] * mtop_.natoms * missingEnergyPerAtom(6);
    if (correctRepulsion())
    {
        referenceEnergy += parameters[1] * mtop_.natoms * missingEnergyPerAtom(12);
    }

    // The correction uses a table with limited resolution
    const FloatingPointTolerance tolerance =
            relativeToleranceAsFloatingPoint(referenceEnergy, GMX_DOUBLE ? 1e-7 : 1e-4);
    EXPECT_REAL_EQ_TOL(calculate(1.0, lambda_).energy, real(referenceEnergy), tolerance);
}

TEST_P(DispersionCorrectionTest, DerivativesMatchEnergy)
{
    if (!c_canTestWithFiniteDifferences)
    {
        GTEST_SKIP() << "Numerical derivatives of the energy are only accurate enough in double "
                        "precision";
    }

    const DispersionCorrection::Correction correction = calculate(1.0, lambda_);

    if (haveFep_)
    {
        checkDvdlambdaMatchesEnergy(correction.dvdl,
                                    [&](const double lambdaChange)
                                    { return calculate(1.0, lambda_ + lambdaChange).energy; });
    }
    else
    {
        EXPECT_EQ(correction.dvdl, 0);
    }

    if (!correctPressure())
    {
        EXPECT_EQ(correction.pressure, 0);
        EXPECT_EQ(correction.virial, 0);

        return;
    }

    const double volume = det(box_);

    // The pressure is twice the virial, which is added to each diagonal element, per volume
    EXPECT_REAL_EQ_TOL(correction.pressure,
                       -2 * correction.virial / volume * c_presfac,
                       relativeToleranceAsFloatingPoint(correction.pressure, 1e-12));

    // Scaling the box by a factor 1 + strain changes the volume by 3 * strain * volume
    // The energy and virial corrections for switched interactions are integrals
    // over a table with limited resolution, which are consistent up to 1e-7
    FiniteDifferenceSettings settings;
    settings.relativeTolerance = 1e-6;
    const double energyDerivative =
            finiteDifferenceDerivative([&](const double strain)
                                       { return calculate(1 + strain, lambda_).energy; },
                                       settings.strainStep)
            / (3 * volume);

    // An interaction that is continuous at the cut-off only contributes to the
    // pressure through the volume dependence of the energy. With a plain cut-off
    // the impulsive force at the cut-off, which would act on the average number
    // of atoms at the cut-off distance, is not part of the simulated system,
    // so its contribution to the pressure should not be corrected for.
    double pressureFromDiscontinuity = 0;
    if (modifiedForm(vdwTreatment_) == ModifiedForm::PlainCutoff)
    {
        const std::array<double, 2> parameters = averageParameters(lambda_);
        const double                density    = mtop_.natoms / volume;
        const double                numPairsWithinCutoff =
                0.5 * mtop_.natoms * density * 4 * M_PI / 3 * std::pow(c_cutoff, 3);
        double potentialAtCutoff = -parameters[0] * std::pow(c_cutoff, -6);
        if (correctRepulsion())
        {
            potentialAtCutoff += parameters[1] * std::pow(c_cutoff, -12);
        }
        pressureFromDiscontinuity = numPairsWithinCutoff * potentialAtCutoff / volume;
    }
    const double referencePressure = (-energyDerivative + pressureFromDiscontinuity) * c_presfac;

    EXPECT_REAL_EQ_TOL(correction.pressure,
                       referencePressure,
                       finiteDifferenceTolerance(std::max(1.0, std::abs(referencePressure)),
                                                 settings.relativeTolerance,
                                                 std::abs(correction.energy) * c_presfac / (3 * volume),
                                                 settings.strainStep))
            << "Pressure correction does not match the volume derivative of the energy correction";
}

//! Returns a name for a test with parameters \c info
std::string nameOfTest(const ::testing::TestParamInfo<DispersionCorrectionTestParameters>& info)
{
    const VdwTreatment& vdwTreatment = std::get<0>(info.param);
    std::string         name         = formatString(
            "%s_%s_%s_%s_%s_Lambda%g",
            enumValueToString(vdwTreatment.type),
            enumValueToString(vdwTreatment.modifier),
            usingLJPme(vdwTreatment.type) ? enumValueToString(vdwTreatment.ljPmeCombinationRule) : "",
            enumValueToString(std::get<1>(info.param)),
            std::get<2>(info.param) ? "Perturbed" : "Unperturbed",
            std::get<3>(info.param));
    std::replace_if(name.begin(), name.end(), [](const char c) { return !std::isalnum(c); }, '_');
    return name;
}

//! The Van der Waals treatments for which dispersion correction is supported without table files
const VdwTreatment c_vdwTreatments[] = {
    { VanDerWaalsType::Cut, InteractionModifiers::None, LongRangeVdW::Geom },
    { VanDerWaalsType::Cut, InteractionModifiers::PotShift, LongRangeVdW::Geom },
    { VanDerWaalsType::Cut, InteractionModifiers::ForceSwitch, LongRangeVdW::Geom },
    { VanDerWaalsType::Cut, InteractionModifiers::PotSwitch, LongRangeVdW::Geom },
    { VanDerWaalsType::Shift, InteractionModifiers::None, LongRangeVdW::Geom },
    { VanDerWaalsType::Switch, InteractionModifiers::None, LongRangeVdW::Geom },
    { VanDerWaalsType::Pme, InteractionModifiers::None, LongRangeVdW::Geom },
    { VanDerWaalsType::Pme, InteractionModifiers::PotShift, LongRangeVdW::Geom },
    { VanDerWaalsType::Pme, InteractionModifiers::None, LongRangeVdW::LB },
    { VanDerWaalsType::Pme, InteractionModifiers::PotShift, LongRangeVdW::LB }
};

INSTANTIATE_TEST_SUITE_P(Unperturbed,
                         DispersionCorrectionTest,
                         ::testing::Combine(::testing::ValuesIn(c_vdwTreatments),
                                            ::testing::Values(DispersionCorrectionType::Ener,
                                                              DispersionCorrectionType::EnerPres,
                                                              DispersionCorrectionType::AllEner,
                                                              DispersionCorrectionType::AllEnerPres),
                                            ::testing::Values(false),
                                            ::testing::Values(0.0)),
                         nameOfTest);

INSTANTIATE_TEST_SUITE_P(Perturbed,
                         DispersionCorrectionTest,
                         ::testing::Combine(::testing::ValuesIn(c_vdwTreatments),
                                            ::testing::Values(DispersionCorrectionType::Ener,
                                                              DispersionCorrectionType::EnerPres,
                                                              DispersionCorrectionType::AllEner,
                                                              DispersionCorrectionType::AllEnerPres),
                                            ::testing::Values(true),
                                            ::testing::Values(0.0, 0.4, 1.0)),
                         nameOfTest);

} // namespace
} // namespace test
} // namespace gmx
