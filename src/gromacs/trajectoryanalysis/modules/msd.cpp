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
#include "gromacs/fileio/trxio.h"
#include "gromacs/math/vectypes.h"
#include "gromacs/options/basicoptions.h"
#include "gromacs/options/filenameoption.h"
#include "gromacs/options/ioptionscontainer.h"
#include "gromacs/selection/selectionoption.h"
#include "gromacs/statistics/statistics.h"
#include "gromacs/trajectoryanalysis/analysissettings.h"
#include "gromacs/trajectory/trajectoryframe.h"
#include "gromacs/utility/futil.h"

namespace gmx::analysismodules
{

namespace {

// Convert nm^2/ps to 10e-5 cm^2/s
constexpr double c_diffusionConversionFactor = 1000.0;
// 6 For 3D, 4 for 2D, 2 for 1D diffusion
// TODO genericize
constexpr double c_diffusionDimensionFactor = 6.0;

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
        msdSums.push_back(
                std::accumulate(msd_vals.begin(),
                                msd_vals.end(),
                                0.0,
                                std::plus<>()
                ) / msd_vals.size());
    }
    return msdSums;
}



real MeanSquaredDisplacement(const RVec* c1, const RVec* c2, int num_vals) {
    real result = 0;
    for (int i = 0; i < num_vals; i++) {
        RVec displacement = c1[i] - c2[i];
        result += displacement.dot(displacement);
    }
    return result / num_vals;
}

}  // namespace

class Msd : public TrajectoryAnalysisModule
{
public:
    Msd();

    void initOptions(IOptionsContainer* options, TrajectoryAnalysisSettings* settings) override;
    void initAfterFirstFrame(const TrajectoryAnalysisSettings& settings, const t_trxframe& fr) override;
    void initAnalysis(const TrajectoryAnalysisSettings& settings, const TopologyInformation& top) override;
    void analyzeFrame(int frnr, const t_trxframe& fr, t_pbc* pbc, TrajectoryAnalysisModuleData* pdata) override;
    void finishAnalysis(int nframes) override;
    void writeOutput() override;
private:

    //! Selections for MSD output
    SelectionList sel_;

    // Defaults - to hook up to option machinery when ready
    //! Picoseconds between restarts
    int trestart_ = 10;
    long t0_ = 0;
    long dt_ = -1;
    // real beginFit_ = -1 ;
    // real endFit_ = -1 ;
    // Coordinates - first indexed by group, then by frame, then by atom
    std::vector<std::vector<std::vector<RVec>>> frames_;
    // Timestamp associated with coordinates
    std::vector<long> times_;
    //! Result accumulator indexed by group
    std::vector<MsdData> msds_;
    //! Summed and averaged MSDs - indexed by group, then by tau.
    std::vector<std::vector<real>> msd_sums_;
    //! Calculated Diffusion coefficients, per group, as well as error estimates.
    std::vector<real> diffusionCoefficients_;
    std::vector<real> sigmas_;

    //! Taus for output - won't know the size until the end.
    std::vector<real> taus_;

    //! Output stuff
    std::string out_file_;
};


Msd::Msd() = default;


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
    options->addOption(SelectionOption("sel").storeVector(&sel_).required().onlyStatic().multiValue().description(
            "Selections to compute MSDs for from the reference"));


}

void Msd::initAnalysis(const TrajectoryAnalysisSettings& gmx_unused settings, const TopologyInformation& /* top */)
{
    // Accumulated frames and results
    msds_.resize(sel_.size());
    frames_.resize(sel_.size());

    // Processed result structures
    msd_sums_.resize(sel_.size());
    diffusionCoefficients_.resize(sel_.size());
    sigmas_.resize(sel_.size());
}

void Msd::initAfterFirstFrame(const TrajectoryAnalysisSettings& gmx_unused settings, const t_trxframe& fr)
{
    t0_ = std::round(fr.time);
}


void Msd::analyzeFrame(int gmx_unused frnr, const t_trxframe& fr, t_pbc* gmx_unused pbc, TrajectoryAnalysisModuleData* gmx_unused pdata)
{

    long time = std::round(fr.time);
    if (!bRmod(time, t0_, trestart_)) {

        return;
    }

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
        frames_[g].push_back(std::move(coords));

        // For each preceding frame, calculate tau and do comparison.
        // NOTE - as currently construed, one element is added to msds_ for each frame
        for (size_t i = 0; i < frames_[g].size(); i++)
        {
            long tau       = time - times_[i];
            long tau_index = tau / dt_;
            msds_[g].AddPoint(tau_index,  MeanSquaredDisplacement(
                                                 frames_[g].back().data(),
                                                 frames_[g][i].data(),
                                                 sel_[g].posCount()));
        }
    }
}

void Msd::finishAnalysis(int gmx_unused nframes) {
    const int numTaus = taus_.size();

    for (size_t g = 0; g < sel_.size(); g++) {
        msd_sums_[g] = msds_[g].AverageMsds();



        // These aren't used, except for corrCoef, which is used to estimate error if enough points are
        // available.
        real b = 0.0, corrCoef =0.0, chiSquared = 0.0;
        if (numTaus >= 4)
        {
            // Split the fit in 2, and compare the results of each fit;
            real a = 0.0, a2 = 0.0;
            lsq_y_ax_b(numTaus / 2, taus_.data(), msd_sums_[g].data(), &a, &b, &corrCoef, &chiSquared);
            lsq_y_ax_b(numTaus / 2,
                       taus_.data() + numTaus / 2, msd_sums_[g].data() + numTaus/ 2, &a2, &b, &corrCoef, &chiSquared);
            sigmas_[g] = std::abs(a - a2);
        }
        lsq_y_ax_b(numTaus, taus_.data(), msd_sums_[g].data(), &diffusionCoefficients_[g], &b, &corrCoef, &chiSquared);
        diffusionCoefficients_[g] *= c_diffusionConversionFactor / c_diffusionDimensionFactor;
        sigmas_[g] *= c_diffusionConversionFactor / c_diffusionDimensionFactor;
    }





}

void Msd::writeOutput() {

    // Ideally we'd use the trajectory analysis framework with a plot module for output.
    // Unfortunately MSD is one of the few analyses where the number of datasets and data columns
    // can't be determined until simulation end, so AnalysisData objects can't be easily used here.
    FILE* out = gmx_ffopen(out_file_, "w");
    fprintf(out, "# MSD gathered over PLACEHOLDER ps with %zul restarts\n", frames_.size());
    for (size_t g = 0; g < sel_.size(); g++) {
        // fprintf(out, "# Diffusion constants fitted from time %g to %g %s\n", beginfit, endfit,
        if (diffusionCoefficients_[g]  > 0.01 && diffusionCoefficients_[g] < 1e4)
        {
            fprintf(out, "# D[%10s] %.4f (+/- %.4f) 1e-5 cm^2/s\n",
                    sel_[g].name(), diffusionCoefficients_[g] , sigmas_[g]);
        }
        else
        {
            fprintf(out, "# D[%10s] %.4g (+/- %.4f) 1e-5 cm^2/s\n",
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
    gmx_ffclose(out);
}


const char MsdInfo::name[]             = "msd";
const char MsdInfo::shortDescription[] = "Compute mean squared displacements";
TrajectoryAnalysisModulePointer MsdInfo::create()
{
    return TrajectoryAnalysisModulePointer(new Msd);
}



}  // namespace gmx::analysismodules