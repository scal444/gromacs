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
 * Defines the trajectory analysis module for RMSD calculations.
 *
 * \author Kevin Boyd <kevin44boyd@gmail.com>
 * \ingroup module_trajectoryanalysis
 */

#include "gmxpre.h"

#include "rms.h"

#include "gromacs/analysisdata/analysisdata.h"
#include "gromacs/analysisdata/modules/average.h"
#include "gromacs/analysisdata/modules/plot.h"
#include "gromacs/gpu_utils/devicebuffer.h"
#include "gromacs/gpu_utils/device_context.h"
#include "gromacs/gpu_utils/device_stream.h"
#include "gromacs/gpu_utils/transformations.cuh"
#include "gromacs/gpu_utils/typecasts.cuh"
#include "gromacs/hardware/device_information.h"
#include "gromacs/hardware/device_management.h"
#include "gromacs/options/basicoptions.h"
#include "gromacs/options/filenameoption.h"
#include "gromacs/options/ioptionscontainer.h"
#include "gromacs/selection/selectionoption.h"
#include "gromacs/trajectoryanalysis/analysissettings.h"
#include "gromacs/trajectory/trajectoryframe.h"

namespace gmx::analysismodules
{


class Rms : public TrajectoryAnalysisModule
{
public:
    Rms();

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

    //! Selection for alignment.
    Selection refSel_;
    //! Selections for rdf output
    SelectionList sel_;

    //! Buffer for reference positions
    DeviceBuffer<float3> reference_ = nullptr;
    //! Buffer for all positions
    DeviceBuffer<float3> all_positions_ = nullptr;
    //! Indices for coordinates to compare against reference.
    std::vector<DeviceBuffer<int>> coord_buffers_;
    //! Per-frame result
    AnalysisData rmsds_;

    //! Output file
    std::string out_file_;
};


Rms::Rms()
{
    rmsds_.setDataSetCount(1);
    rmsds_.setColumnCount(0, 1);
    registerAnalysisDataset(&rmsds_, "rmsd");
}


void Rms::initOptions(IOptionsContainer* options, TrajectoryAnalysisSettings* settings)
{
    static const char* const desc[] = {
        "[THISMODULE] computes RMSDs via GPU. PBC not yet supported, nor any other features."
    };

    settings->setHelpText(desc);

    options->addOption(IntegerOption("gpu_id")
                               .store(&gpu_id_)
                               .description("Which GPU to run on"));
    options->addOption(FileNameOption("o")
                               .filetype(eftPlot)
                               .outputFile()
                               .store(&out_file_)
                               .defaultBasename("rmsdout")
                               .description("Average distances as function of time"));
    options->addOption(SelectionOption("ref").store(&refSel_).required().description(
            "Reference selection for pre-RMSD alignment"));
    options->addOption(SelectionOption("sel").storeVector(&sel_).required().multiValue().description(
            "Selections to compute RMSDs for from the reference"));
}

void Rms::initAnalysis(const TrajectoryAnalysisSettings& gmx_unused settings, const TopologyInformation& /* top */)
{
    // Set up output
    auto plotm = std::make_shared<AnalysisDataPlotModule>(settings.plotSettings());
    plotm->setFileName(out_file_);
    plotm->setTitle("RMSD");
    plotm->setXAxisIsTime();
    plotm->setYLabel("RMSD (units)");
    for (const auto& sel: sel_)
    {
        plotm->appendLegend(sel.name());
    }
    rmsds_.addModule(plotm);

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
    allocateDeviceBuffer(&reference_, refSel_.posCount(), dummy_context);
    const int nSels = sel_.size();
    coord_buffers_.reserve(nSels);
    for (int i = 0; i < nSels; i++)
    {
        allocateDeviceBuffer(&coord_buffers_[i], sel_[i].posCount(), dummy_context);
        copyToDeviceBuffer(&coord_buffers_[i], sel_[i].atomIndices().data(),0, sel_[i].posCount(),
                           *stream_,  GpuApiCallBehavior::Sync, nullptr);
    }
}

void Rms::initAfterFirstFrame(const TrajectoryAnalysisSettings& gmx_unused settings, const t_trxframe& fr)
{
    DeviceContext dummy_context(DeviceInformation{});
    // Set up all frames now that we know what they are. Could also get it from topo prolly.
    allocateDeviceBuffer(&all_positions_, fr.natoms, dummy_context);

    // Here, put the reference data on the GPU.
    copyToDeviceBuffer(&reference_, asConstFloat3(refSel_.coordinates().data()), 0,
                       refSel_.posCount(), *stream_,
                       GpuApiCallBehavior::Sync, nullptr);
    DeviceBuffer<float3> com_device = nullptr;

    allocateDeviceBuffer(&com_device, 1, dummy_context);
    center_of_mass(reference_, nullptr, nullptr, refSel_.posCount(), com_device);
    float3 com;
    copyFromDeviceBuffer(&com, &com_device, 0, 1, *stream_,
                         GpuApiCallBehavior::Sync, nullptr);
    translate(reference_, refSel_.posCount(),  {com.x * -1.0F, com.y * -1.0F, com.z * -1.0F});
}


void Rms::analyzeFrame(int frnr, const t_trxframe& fr, t_pbc* gmx_unused pbc, TrajectoryAnalysisModuleData* pdata)
{
    DeviceContext dummy_context(DeviceInformation{});

    AnalysisDataHandle   dh   = pdata->dataHandle(rmsds_);
    dh.startFrame(frnr, fr.time);

    // Copy ALL positions over. Future optimizations could be just the ones needed and a bunch of
    // reindexing
    copyToDeviceBuffer(&all_positions_, asConstFloat3(fr.x), 0, fr.natoms, *stream_, GpuApiCallBehavior::Sync, nullptr);

    // Center new positions based on reference indices.
    DeviceBuffer<float3> com_device = nullptr;

    allocateDeviceBuffer(&com_device, 1, dummy_context);
    center_of_mass(all_positions_, nullptr, refSel_.atomIndices().data(), refSel_.posCount(), com_device);
    float3 com;
    copyFromDeviceBuffer(&com, &com_device, 0, 1, *stream_,
                         GpuApiCallBehavior::Sync, nullptr);
    translate(all_positions_, fr.natoms, {com.x * -1.0F, com.y * -1.0F, com.z * -1.0F});
    // Do fit/alignment to existing reference

    // Do calculation

    // DUMMY - frnr squared
    dh.setPoint(0, frnr * frnr);
    dh.finishFrame();
}

void Rms::finishAnalysis(int gmx_unused nframes) {
}

void Rms::writeOutput() {

}


const char RmsInfo::name[]             = "rms";
const char RmsInfo::shortDescription[] = "Compute root mean squared deviations";
TrajectoryAnalysisModulePointer RmsInfo::create()
{
    return TrajectoryAnalysisModulePointer(new Rms);
}



}  // namespace gmx::analysismodules