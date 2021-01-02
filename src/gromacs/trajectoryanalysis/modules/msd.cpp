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
 * Defines the trajectory analysis module for mean squared displacement calculations.
 *
 * \author Kevin Boyd <kevin44boyd@gmail.com>
 * \ingroup module_trajectoryanalysis
 */

#include "gmxpre.h"

#include "msd.h"

#include <numeric>

#include "gromacs/analysisdata/analysisdata.h"
#include "gromacs/analysisdata/modules/average.h"
#include "gromacs/analysisdata/modules/plot.h"
#include "gromacs/fileio/oenv.h"
#include "gromacs/fileio/trxio.h"
#include "gromacs/fileio/xvgr.h"
#include "gromacs/math/functions.h"
#include "gromacs/math/vectypes.h"
#include "gromacs/options/basicoptions.h"
#include "gromacs/options/filenameoption.h"
#include "gromacs/options/ioptionscontainer.h"
#include "gromacs/pbcutil/pbc.h"
#include "gromacs/selection/selectionoption.h"
#include "gromacs/statistics/statistics.h"
#include "gromacs/trajectoryanalysis/analysissettings.h"
#include "gromacs/trajectoryanalysis/topologyinformation.h"
#include "gromacs/trajectory/trajectoryframe.h"
#include "gromacs/utility/programcontext.h"
#include "gromacs/utility.h"

namespace gmx::analysismodules
{

namespace {

// Convert nm^2/ps to 10e-5 cm^2/s
constexpr double c_diffusionConversionFactor = 1000.0;
// Used in diffusion coefficient calculations
constexpr double c_3DdiffusionDimensionFactor = 6.0;
constexpr double c_2DdiffusionDimensionFactor = 4.0;
constexpr double c_1DdiffusionDimensionFactor = 2.0;


class MsdData {
public:
    void AddPoint(size_t tau_index, real value);
    [[nodiscard]] std::vector<real> AverageMsds() const;

private:
    // Results - first indexed by tau, then data points
    std::vector<std::vector<real>> msds_;
};

void MsdData::AddPoint(size_t tau_index, real value) {
    if (msds_.size() <= tau_index) {
        msds_.resize(tau_index + 1);
    }
    msds_[tau_index].push_back(value);
}

std::vector<real> MsdData::AverageMsds() const {
    std::vector<real> msdSums;
    msdSums.reserve(msds_.size());
    for (gmx::ArrayRef<const real> msd_vals : msds_) {
        if (msd_vals.empty()) {
            msdSums.push_back(0.0);
            continue;
        }
        msdSums.push_back(
                std::accumulate(msd_vals.begin(),
                                msd_vals.end(),
                                0.0,
                                std::plus<>()
                ) / msd_vals.size());
    }
    return msdSums;
}

template<bool x, bool y, bool z>
inline real CalcSingleMsd(RVec c1, RVec c2) {
    real result = 0;
    if constexpr (x) {
        result += (c1[XX] - c2[XX]) * (c1[XX] - c2[XX]);
    }
    if constexpr (y) {
        result += (c1[YY] - c2[YY]) * (c1[YY] - c2[YY]);
    }
    if constexpr (z) {
        result += (c1[ZZ] - c2[ZZ]) * (c1[ZZ] - c2[ZZ]);
    }
    return result;
}

template<bool x, bool y, bool z>
real CalcMsds(const RVec* c1, const RVec* c2, const int num_vals) {
    real result = 0;
    for (int i = 0; i < num_vals; i++) {
        result += CalcSingleMsd<x,y,z>(c1[i], c2[i]);
    }
    return result / num_vals;
}

}  // namespace

// Describes 1D MSDs, in the given dimension.
enum class SingleDimDiffType : int {
    unused = 0,
    x,
    y,
    z,
    Count,
};

// Describes 2D MSDs, in the plane normal to the given dimension.
enum class TwoDimDiffType : int {
    unused = 0,
    xNormal,
    yNormal,
    zNormal,
    Count,
};

class Msd : public TrajectoryAnalysisModule
{
public:
    Msd();
    ~Msd() override;

    void initOptions(IOptionsContainer* options, TrajectoryAnalysisSettings* settings) override;
    void initAfterFirstFrame(const TrajectoryAnalysisSettings& settings, const t_trxframe& fr) override;
    void initAnalysis(const TrajectoryAnalysisSettings& settings, const TopologyInformation& top) override;
    void analyzeFrame(int frnr, const t_trxframe& fr, t_pbc* pbc, TrajectoryAnalysisModuleData* pdata) override;
    void finishAnalysis(int nframes) override;
    void writeOutput() override;
private:

    //! Selections for MSD output
    SelectionList sel_;

    // MSD type information
    SingleDimDiffType singleDimType_ = SingleDimDiffType::unused;
    TwoDimDiffType twoDimType_ = TwoDimDiffType::unused;
    double            diffusionCoefficientDimensionFactor_     = c_3DdiffusionDimensionFactor;
    std::function<real(const RVec*, const RVec*, int)> calcFn_ = CalcMsds<true, true, true>;

    // Defaults - to hook up to option machinery when ready
    //! Picoseconds between restarts
    real trestart_ = 10.0;
    real t0_ = 0;
    real dt_ = -1;
    real beginFit_ = -1.0 ;
    real endFit_ = -1.0 ;
    // Coordinates - first indexed by group, then by frame, then by atom
    std::vector<std::vector<std::vector<RVec>>> frames_;
    std::vector<std::vector<real>> frame_times_;

    // Previous coordinates (indexed by group) - used for PBC correction.
    std::vector<std::vector<RVec>> previousFrames_;
    // Timestamp associated with coordinates
    std::vector<real> times_;
    //! Result accumulator indexed by group
    std::vector<MsdData> msds_;
    //! Summed and averaged MSDs - indexed by group, then by tau.
    std::vector<std::vector<real>> msd_sums_;
    //! Calculated Diffusion coefficients, per group, as well as error estimates.
    std::vector<real> diffusionCoefficients_;
    std::vector<real> sigmas_;

    //! Taus for output - won't know the size until the end.
    std::vector<real> taus_;

    // MSD per-molecule stuff
    bool mol_selected_ = false;
    // per group per mol atom count for COM calculation.
    std::vector<std::vector<int>> mol_atom_counts_;
    // Per group mol index vector
    std::vector<std::vector<int>> mol_index_mappings_;
    // Stores the msd data of each molecule for group 1, indexed by molecule
    std::vector<MsdData> per_mol_msds_;
    std::vector<real> perMolDiffusionCoefficients_;

    //! Output stuff
    std::string out_file_;
    std::string mol_file_;
    gmx_output_env_t* oenv_ = nullptr;
};


Msd::Msd() = default;
Msd::~Msd() {
    output_env_done(oenv_);
}



void Msd::initOptions(IOptionsContainer* options, TrajectoryAnalysisSettings* settings)
{
    static const char* const desc[] = {
        "[THISMODULE] computes MSDs via GPU. PBC not yet supported, nor any other features."
    };

    settings->setHelpText(desc);

    options->addOption(FileNameOption("o")
                               .filetype(eftPlot)
                               .outputFile()
                               .store(&out_file_)
                               .defaultBasename("msdout")
                               .description("MSD output"));
    options->addOption(RealOption("trestart").
                       description("Time between restarting points in trajectory (ps)").
                       defaultValue(10.0).
                       store(&trestart_));
    options->addOption(FileNameOption("mol").filetype(eftPlot).outputFile()
                               .store(&mol_file_).storeIsSet(&mol_selected_).defaultBasename("diff_mol")
                               .description("Report diffusion coefficients for each molecule in selection"));

    EnumerationArray<SingleDimDiffType, const char*> enumTypeNames = {"unselected", "x", "y", "z"};
    EnumerationArray<TwoDimDiffType, const char*> enumLateralNames = {"unselected", "x", "y", "z"};
    options->addOption(EnumOption<SingleDimDiffType>("type").enumValue(enumTypeNames).store(&singleDimType_).defaultValue(SingleDimDiffType::unused));
    options->addOption(EnumOption<TwoDimDiffType>("lateral").enumValue(enumLateralNames).store(&twoDimType_).defaultValue(TwoDimDiffType::unused));
    options->addOption(SelectionOption("sel").storeVector(&sel_).required().onlyStatic().multiValue().description(
            "Selections to compute MSDs for from the reference"));
}

void Msd::initAnalysis(const TrajectoryAnalysisSettings& gmx_unused settings, const TopologyInformation& top)
{
    const int numSelections = sel_.size();
    // Accumulated frames and results
    msds_.resize(numSelections);
    frames_.resize(numSelections);
    frame_times_.resize(numSelections);
    previousFrames_.resize(numSelections);

    // Processed result structures
    msd_sums_.resize(numSelections);
    diffusionCoefficients_.resize(numSelections);
    sigmas_.resize(numSelections);

    output_env_init(&oenv_, getProgramContext(), settings.timeUnit(), FALSE, settings.plotSettings().plotFormat(), 0);

    // Parse dimensionality and assign the MSD calculating function.
    if (singleDimType_ != SingleDimDiffType::unused && twoDimType_ != TwoDimDiffType::unused) {
        std::string errorMessage =
                "Options -type and -lateral are mutually exclusive. Choose one or neither.";
        GMX_THROW(InconsistentInputError(errorMessage.c_str()));
    }

    switch (singleDimType_) {
        case SingleDimDiffType::x:
            calcFn_ = CalcMsds<true, false, false>;
            diffusionCoefficientDimensionFactor_ = c_1DdiffusionDimensionFactor;
            break;
        case SingleDimDiffType::y:
            calcFn_ = CalcMsds<false, true, false>;
            diffusionCoefficientDimensionFactor_ = c_1DdiffusionDimensionFactor;
            break;
        case SingleDimDiffType::z:
            calcFn_ = CalcMsds<false, false, true>;
            diffusionCoefficientDimensionFactor_ = c_1DdiffusionDimensionFactor;
            break;
        default:
            break;
    }
    switch (twoDimType_) {
        case TwoDimDiffType::xNormal:
            calcFn_ = CalcMsds<false, true, true>;
            diffusionCoefficientDimensionFactor_ = c_2DdiffusionDimensionFactor;
            break;
        case TwoDimDiffType::yNormal:
            calcFn_ = CalcMsds<true, false, true>;
            diffusionCoefficientDimensionFactor_ = c_2DdiffusionDimensionFactor;
            break;
        case TwoDimDiffType::zNormal:
            calcFn_ = CalcMsds<true, true, false>;
            diffusionCoefficientDimensionFactor_ = c_2DdiffusionDimensionFactor;
            break;
        default:
            break;
    }

    // TODO validate that we have mol info and not atom only.
    if (mol_selected_) {
        mol_index_mappings_.resize(numSelections);
        mol_atom_counts_.resize(numSelections);
        for (int g = 0; g < numSelections; g++) {
            int nMol = sel_[g].initOriginalIdsToGroup(top.mtop(), INDEX_MOL);
            if (g == 0)
            {
                per_mol_msds_.resize(nMol);
                perMolDiffusionCoefficients_.resize(nMol);
            }
            gmx::ArrayRef<const int> mapped_ids = sel_[0].mappedIds();
            mol_index_mappings_[g].resize(sel_[g].posCount());
            std::copy(mapped_ids.begin(), mapped_ids.end(), mol_index_mappings_[g].begin());

            mol_atom_counts_[g].resize(nMol, 0);
            for (int i = 0; i < sel_[g].posCount(); i++) {
                mol_atom_counts_[g][mapped_ids[i]]++;
            }
        }
    }
}

void Msd::initAfterFirstFrame(const TrajectoryAnalysisSettings& gmx_unused settings, const t_trxframe& fr)
{
    t0_ = std::round(fr.time);
    for (size_t g = 0; g < sel_.size(); g++) {
        if (mol_selected_) {
            previousFrames_[g].resize(mol_atom_counts_[g].size());
        } else {
            previousFrames_[g].resize(sel_[g].posCount());
        }
    }
}


void Msd::analyzeFrame(int gmx_unused frnr, const t_trxframe& fr, t_pbc* gmx_unused pbc, TrajectoryAnalysisModuleData* gmx_unused pdata)
{
    const real time = std::round(fr.time);
    // Need to populate dt on frame 2;
    if (dt_ < 0 && !times_.empty()) {
        dt_ = time - times_[0];
    }
    times_.push_back(time);
    // Each frame will get a tau between it and frame 0, and all other frame combos should be
    // covered by this.
    // TODO this will no longer hold exactly when maxtau is added
    taus_.push_back(time - times_[0]);

    for (size_t g = 0; g < sel_.size(); g++)
    {
        std::vector<RVec> coords(sel_[g].coordinates().begin(), sel_[g].coordinates().end());
        // Do PBC removal


        if (mol_selected_) {
            // Do COM gathering for group 0 to get mol stuff. Note that pbc/com removal is already done
            // operate on vector[RVec] where each RVec is a mol COM
            // go to vector[molInd (0 based)]vector[tau]vector[entry] for summation.

            // First create a clear buffer
            std::vector<RVec> mol_positions(mol_atom_counts_[g].size(), {0.0, 0.0, 0.0});

            // Sum up all positions
            // TODO mass weight.
            for(int i = 0; i < sel_[g].posCount(); i++) {
                const int mol_index = mol_index_mappings_[g][i];
                mol_positions[mol_index] += coords[i] / mol_atom_counts_[g][mol_index];
            }

            // Override the current coordinates.
            coords = std::move(mol_positions);
        }

        // There are two types of "pbc removal" in gmx msd. The first happens in the trajectoryanalysis
        // framework, which makes molecules whole across periodic boundaries and is done
        // automatically where the inputs support it. This lambda performs the second PBC correction, where
        // any "jump" across periodic boundaries BETWEEN FRAMES is put back. The order of these
        // operations is important - since the first transformation may only apply to part of a
        // molecule (e.g., one half in/out of the box is put on one side of the box), the
        // subsequent step needs to be applied to the molecule COM rather than individual atoms, or
        // we'd have a clash where the per-mol PBC removal moves an atom that gets put back into
        // it's original position by the second transformation. Therefore, this second transformation
        // is applied *after* per molecule coordinates have been consolidated into COMs.
        auto pbcRemover = [pbc] (RVec in, RVec prev)
        {
            rvec dx;
            pbc_dx(pbc, in, prev, dx);
            return prev + dx;
        };
        // If on frame 0, set up as "previous" frame. We can't set up on initAfterFirstFrame since
        // selections haven't been evaluated. No cross-box pbc removal for the first frame - note
        // that while initAfterFirstFrame does not do per-mol PBC removal, it is done prior to
        // AnalyzeFrame for frame 0, so when we store frame 0 there's no wonkiness.
        if (frnr != 0) {

            std::transform(coords.begin(), coords.end(), previousFrames_[g].begin(), coords.begin(), pbcRemover);
        }
        // Update "previous frame" for next rounds pbc removal. Note that for msd mol these coords
        // are actually the molecule COM.
        std::copy(coords.begin(), coords.end(), previousFrames_[g].begin());

        // For each preceding frame, calculate tau and do comparison.
        // NOTE - as currently construed, one element is added to msds_ for each frame
        for (size_t i = 0; i < frames_[g].size(); i++)
        {
            real tau       = time - frame_times_[g][i];
            long tau_index = gmx::roundToInt64(tau / dt_);
            msds_[g].AddPoint(tau_index,  calcFn_(
                                                 coords.data(),
                                                 frames_[g][i].data(),
                                                 mol_selected_ ? per_mol_msds_.size() : sel_[g].posCount()));
            if (g == 0 && mol_selected_) {
                for (size_t molInd = 0; molInd < per_mol_msds_.size(); molInd++) {
                    per_mol_msds_[molInd].AddPoint(tau_index, calcFn_(&coords[molInd], &frames_[g][i][molInd], 1));
                }
            }
        }


        // We only store the frame for the future if it's a restart per -trestart.
        if (bRmod(time, t0_, trestart_))
        {
            frames_[g].push_back(std::move(coords));
            frame_times_[g].push_back(time);
        }
    }
}

void Msd::finishAnalysis(int gmx_unused nframes) {

    // If unspecified, calculate beginfit and endfit as 10% and 90% indices.
    int beginFitIndex = 0;
    int endFitIndex = 0;
    // TODO - the else clause when beginfit and endfit are supported.
    if (beginFit_ < 0) {
        beginFitIndex = gmx::roundToInt(taus_.size() * 0.1);
        beginFit_ = taus_[beginFitIndex];
    }
    if (endFit_ < 0) {
        const size_t maybeMaxIndex = gmx::roundToInt(taus_.size() * 0.9);
        endFitIndex = maybeMaxIndex >= taus_.size() ? taus_.size() - 1 : maybeMaxIndex;
        endFit_ = taus_[endFitIndex];
    }
    const int numTaus = 1 + endFitIndex - beginFitIndex;

    // These aren't used, except for corrCoef, which is used to estimate error if enough points are
    // available.
    real b = 0.0, corrCoef =0.0, chiSquared = 0.0;

    for (size_t g = 0; g < sel_.size(); g++) {
        msd_sums_[g] = msds_[g].AverageMsds();


        if (numTaus >= 4)
        {
            const int halfNumTaus = numTaus / 2;
            const int secondaryStartIndex = beginFitIndex + halfNumTaus;
            // Split the fit in 2, and compare the results of each fit;
            real a = 0.0, a2 = 0.0;
            lsq_y_ax_b(halfNumTaus, &taus_[beginFitIndex], &msd_sums_[g][beginFitIndex], &a, &b, &corrCoef, &chiSquared);
            lsq_y_ax_b(halfNumTaus,
                       &taus_[secondaryStartIndex], &msd_sums_[g][secondaryStartIndex], &a2, &b, &corrCoef, &chiSquared);
            sigmas_[g] = std::abs(a - a2);
        }
        lsq_y_ax_b(numTaus, &taus_[beginFitIndex], &msd_sums_[g][beginFitIndex], &diffusionCoefficients_[g], &b, &corrCoef, &chiSquared);
        diffusionCoefficients_[g] *= c_diffusionConversionFactor / diffusionCoefficientDimensionFactor_;
        sigmas_[g] *= c_diffusionConversionFactor / diffusionCoefficientDimensionFactor_;
    }

    if (!mol_selected_) {
        return;
    }

    for (size_t i = 0; i < per_mol_msds_.size(); i++) {
        std::vector<real> msds = per_mol_msds_[i].AverageMsds();
        lsq_y_ax_b(numTaus, &taus_[beginFitIndex], &msds[beginFitIndex], &perMolDiffusionCoefficients_[i], &b, &corrCoef, & chiSquared);
        perMolDiffusionCoefficients_[i] *= c_diffusionConversionFactor / diffusionCoefficientDimensionFactor_;
    }





}

void Msd::writeOutput() {

    // Ideally we'd use the trajectory analysis framework with a plot module for output.
    // Unfortunately MSD is one of the few analyses where the number of datasets and data columns
    // can't be determined until simulation end, so AnalysisData objects can't be easily used here.
    // Since the plotting modules are completely wired into the analysis data, we can't use the nice
    // plotting functionality.
    FILE* out = xvgropen(out_file_.c_str(), "Mean Square Displacement",  output_env_get_xvgr_tlabel(oenv_),
                         "MSD (nm\\S2\\N)", oenv_);
    fprintf(out, "# MSD gathered over %g %s with %zu restarts\n", times_.back() - times_[0],
            output_env_get_time_unit(oenv_).c_str(), frames_[0].size());
    fprintf(out, "# Diffusion constants fitted from time %g to %g %s\n", beginFit_, endFit_,
            output_env_get_time_unit(oenv_).c_str());
    for (size_t g = 0; g < sel_.size(); g++) {
        if (diffusionCoefficients_[g]  > 0.01 && diffusionCoefficients_[g] < 1e4)
        {
            fprintf(out, "# D[%10s] = %.4f (+/- %.4f) (1e-5 cm^2/s)\n",
                    sel_[g].name(), diffusionCoefficients_[g] , sigmas_[g]);
        }
        else
        {
            fprintf(out, "# D[%10s] = %.4g (+/- %.4f) (1e-5 cm^2/s)\n",
                    sel_[g].name(), diffusionCoefficients_[g] , sigmas_[g]);
        }
    }

    for (size_t i = 0; i < taus_.size(); i++) {
        fprintf(out, "%10g", taus_[i]);
        for (size_t g = 0; g < sel_.size(); g++) {
            fprintf(out, "  %10g", msd_sums_[g][i]);
        }
        fprintf(out, "\n");
    }
    xvgrclose(out);

    // Handle per mol stuff if needed.
    if (mol_selected_) {
        out = xvgropen(mol_file_.c_str(), "Diffusion Coefficients / Molecule", "Molecule", "D (1e-5 cm^2/s)", oenv_);
        for (size_t i = 0; i < perMolDiffusionCoefficients_.size(); i++) {
            fprintf(out, "%10zu  %10g\n", i, perMolDiffusionCoefficients_[i]);
        }
        xvgrclose(out);
    }
}


const char MsdInfo::name[]             = "msd";
const char MsdInfo::shortDescription[] = "Compute mean squared displacements";
TrajectoryAnalysisModulePointer MsdInfo::create()
{
    return TrajectoryAnalysisModulePointer(new Msd);
}



}  // namespace gmx::analysismodules