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
#include "msder.h"

#include <deque>

#include "gromacs/analysisdata/analysisdata.h"
#include "gromacs/analysisdata/modules/average.h"
#include "gromacs/analysisdata/modules/plot.h"
#include "gromacs/fileio/trxio.h"
#include "gromacs/gpu_utils/device_context.h"
#include "gromacs/gpu_utils/device_stream.h"
#include "gromacs/hardware/device_information.h"
#include "gromacs/hardware/device_management.h"
#include "gromacs/math/vectypes.h"
#include "gromacs/options/basicoptions.h"
#include "gromacs/options/filenameoption.h"
#include "gromacs/options/ioptionscontainer.h"
#include "gromacs/selection/selectionoption.h"
#include "gromacs/trajectoryanalysis/analysissettings.h"
#include "gromacs/trajectoryanalysis/modules/msder.h"
#include "gromacs/trajectory/trajectoryframe.h"

namespace gmx::analysismodules
{

namespace {

/*
const float3* asConstFloat3(const rvec* const in)
{
    static_assert(sizeof(in[0]) == sizeof(float3),
                  "Size of the host-side data-type is different from the size of the device-side "
                  "counterpart.");
    return reinterpret_cast<const float3*>(in);
}
*/

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
    //! GPU to run on.
    int gpu_id_ = 0;
    //! GPU stream
    std::unique_ptr<DeviceStream> stream_ = nullptr;

    //! Selections for rdf output
    Selection sel_;

    //! Coordinates to do le msd.
    // std::deque<DeviceBuffer<float3>> frame_holder_;
    //! Output file
    std::string out_file_;

    // Defaults - to hook up to option machinery when ready
    //! Picoseconds between restarts
    int trestart_ = 10;
    long t0_ = 0;
    long dt_ = 0;
    // int natoms_ = 0;
    std::unique_ptr<BufferManager> manager_;
    // Timestamp associated with coordinates
    std::vector<long> times_;
    // Results - first indexed by tau, then just data points
    std::vector<std::vector<real>> msds_;
};


Msd::Msd() = default;


void Msd::initOptions(IOptionsContainer* options, TrajectoryAnalysisSettings* settings)
{
    static const char* const desc[] = {
        "[THISMODULE] computes MSDs via GPU. PBC not yet supported, nor any other features."
    };

    settings->setHelpText(desc);

    options->addOption(IntegerOption("gpu_id")
                               .store(&gpu_id_)
                               .description("Which GPU to run on"));
    options->addOption(FileNameOption("o")
                               .filetype(eftPlot)
                               .outputFile()
                               .store(&out_file_)
                               .defaultBasename("msdout")
                               .description("MSD output"));
    // TODO  - allow multiple selections
    options->addOption(SelectionOption("sel").store(&sel_).required().onlyStatic().description(
            "Selections to compute MSDs for from the reference"));
}

void Msd::initAnalysis(const TrajectoryAnalysisSettings& gmx_unused settings, const TopologyInformation& /* top */)
{
    DeviceContext dummy_context(DeviceInformation{});
    for (std::unique_ptr<DeviceInformation>& device : findDevices())
    {
        if (device->id == gpu_id_) {
            setActiveDevice(*device);
            stream_ = std::make_unique<DeviceStream>(dummy_context, DeviceStreamPriority::High, false);
            break;
        }
    }
    GMX_ASSERT(stream_ != nullptr, "Didn't initiate cuda stream.");
}

void Msd::initAfterFirstFrame(const TrajectoryAnalysisSettings& gmx_unused settings, const t_trxframe& fr)
{
    DeviceContext dummy_context(DeviceInformation{});
    t0_ = std::round(fr.time);
    manager_ = std::make_unique<BufferManager>(sel_.posCount());
}


void Msd::analyzeFrame(int gmx_unused frnr, const t_trxframe& fr, t_pbc* gmx_unused pbc, TrajectoryAnalysisModuleData* gmx_unused pdata)
{
    DeviceContext dummy_context(DeviceInformation{});
    long time = std::round(fr.time);
    if (!bRmod(fr.time, t0_, trestart_)) {
        return;
    }

    // Need to populate dt on frame 2;
    if (dt_ == 0 && !times_.empty()) {
        dt_ = time - times_[0];
    }
    std::vector<float3> local_coords(reinterpret_cast<const float3*>(&sel_.coordinates()[0]),
                                     reinterpret_cast<const float3*>(&sel_.coordinates()[0]) + sel_.posCount());
    manager_->AddFrame(&local_coords[0]);
    // For each preceding frame, calculate tau and do comparison.
    // NOTE - as currently construed, one element is added to msds_ for each frame
    msds_.emplace_back();
    for (int i = 0; i < frnr; i++) {
        long tau_index = (time - times_[i]) / dt_;
        msds_[tau_index].push_back(manager_->GetMsd(tau_index, frnr));
    }

    //
    times_.push_back(time);
    // Always copy over
    // DeviceBuffer<float3> coords = nullptr;
    // allocateDeviceBuffer(&coords,  natoms_, dummy_context);
    // copyToDeviceBuffer(&coords, asConstFloat3(sel_.coordinates().data()), 0, natoms_, *stream_,  GpuApiCallBehavior::Sync, nullptr);
    // Kick off analysis
    // Free if not a restart frame
    // frame_holder_.push_back(coords);

}

void Msd::finishAnalysis(int gmx_unused nframes) {
}

void Msd::writeOutput() {
    long tau = times_[0];
    for (gmx::ArrayRef<const real> msd_vals : msds_) {
        real sum = 0.0;
        for (real val : msd_vals) {
            sum += val;
        }
        sum /= msd_vals.size();
        fprintf(stdout, "MSD at tau %li = %f", tau, sum);
        tau += dt_;
    }
}


const char MsdInfo::name[]             = "msd";
const char MsdInfo::shortDescription[] = "Compute mean squared displacements";
TrajectoryAnalysisModulePointer MsdInfo::create()
{
    return TrajectoryAnalysisModulePointer(new Msd);
}



}  // namespace gmx::analysismodules