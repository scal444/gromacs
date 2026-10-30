/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2010- The GROMACS Authors
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
 * Implements gmx::TrajectoryAnalysisCommandLineRunner.
 *
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 * \ingroup module_trajectoryanalysis
 */
#include "gmxpre.h"

#include "gromacs/trajectoryanalysis/cmdlinerunner.h"

#include <cstdio>

#include <condition_variable>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include "gromacs/analysisdata/paralleloptions.h"
#include "gromacs/commandline/cmdlinemodulemanager.h"
#include "gromacs/commandline/cmdlineoptionsmodule.h"
#include "gromacs/options/ioptionscontainer.h"
#include "gromacs/options/timeunitmanager.h"
#include "gromacs/pbcutil/pbc.h"
#include "gromacs/selection/selectioncollection.h"
#include "gromacs/selection/selectionoptionbehavior.h"
#include "gromacs/trajectory/trajectoryframe.h"
#include "gromacs/trajectoryanalysis/analysismodule.h"
#include "gromacs/trajectoryanalysis/analysissettings.h"
#include "gromacs/trajectoryanalysis/topologyinformation.h"
#include "gromacs/utility/filestream.h"
#include "gromacs/utility/vectypes.h"

#include "runnercommon.h"

namespace gmx
{
class CommandLineModuleSettings;

namespace
{

/********************************************************************
 * RunnerModule
 */

class RunnerModule : public ICommandLineOptionsModule
{
public:
    explicit RunnerModule(TrajectoryAnalysisModulePointer module) :
        module_(std::move(module)), common_(&settings_)
    {
    }

    void init(CommandLineModuleSettings* /*settings*/) override {}
    void initOptions(IOptionsContainer* options, ICommandLineOptionsModuleSettings* settings) override;
    void optionsFinished() override;
    int  run() override;

    //! Analyzes all frames in the calling thread, returning the frame count.
    int analyzeFramesSerial();
    //! Analyzes frames concurrently in \p threadCount threads, returning the frame count.
    int analyzeFramesParallel(int threadCount);

    TrajectoryAnalysisModulePointer module_;
    TrajectoryAnalysisSettings      settings_;
    TrajectoryAnalysisRunnerCommon  common_;
    SelectionCollection             selections_;
};

void RunnerModule::initOptions(IOptionsContainer* options, ICommandLineOptionsModuleSettings* settings)
{
    std::shared_ptr<TimeUnitBehavior>        timeUnitBehavior(new TimeUnitBehavior());
    std::shared_ptr<SelectionOptionBehavior> selectionOptionBehavior(
            new SelectionOptionBehavior(&selections_, common_.topologyProvider()));
    settings->addOptionsBehavior(timeUnitBehavior);
    settings->addOptionsBehavior(selectionOptionBehavior);
    IOptionsContainer& commonOptions = options->addGroup();
    IOptionsContainer& moduleOptions = options->addGroup();

    settings_.setOptionsModuleSettings(settings);
    module_->initOptions(&moduleOptions, &settings_);
    settings_.setOptionsModuleSettings(nullptr);
    common_.initOptions(&commonOptions, timeUnitBehavior.get());
    selectionOptionBehavior->initOptions(&commonOptions);
}

void RunnerModule::optionsFinished()
{
    common_.optionsFinished();
    module_->optionsFinished(&settings_);
}

/*! \brief
 * Copy of a trajectory frame that owns its coordinate buffers.
 *
 * Fields other than coordinates, velocities and forces (e.g., atoms and
 * index) point to the data of the frame that was copied, which must stay
 * valid and unchanged while the copy is used.
 */
struct FrameCopy
{
    //! Copies \p source into this object, reusing earlier allocated buffers.
    void assign(const t_trxframe& source)
    {
        frame   = source;
        frame.x = copyVectors(source.bX ? source.x : nullptr, source.natoms, &x_);
        frame.v = copyVectors(source.bV ? source.v : nullptr, source.natoms, &v_);
        frame.f = copyVectors(source.bF ? source.f : nullptr, source.natoms, &f_);
    }

    //! Frame data passed to analysis.
    t_trxframe frame;

private:
    static rvec* copyVectors(const rvec* source, int count, std::vector<RVec>* buffer)
    {
        if (source == nullptr)
        {
            return nullptr;
        }
        buffer->assign(source, source + count);
        return as_rvec_array(buffer->data());
    }

    std::vector<RVec> x_;
    std::vector<RVec> v_;
    std::vector<RVec> f_;
};

int RunnerModule::run()
{
    common_.initTopology();
    const TopologyInformation& topology = common_.topologyInformation();
    module_->initAnalysis(settings_, topology);

    // Load first frame.
    common_.initFirstFrame();
    common_.initFrameIndexGroup();
    module_->initAfterFirstFrame(settings_, common_.frame());

    const bool canRunParallel = settings_.hasFlag(TrajectoryAnalysisSettings::efFrameParallel)
                                && common_.hasTrajectory();
    const int threadCount = canRunParallel ? common_.threadCount() : 1;
    const int nframes = threadCount > 1 ? analyzeFramesParallel(threadCount) : analyzeFramesSerial();

    if (common_.hasTrajectory())
    {
        fprintf(stderr, "Analyzed %d frames, last time %.3f\n", nframes, common_.frame().time);
    }
    else
    {
        fprintf(stderr, "Analyzed topology coordinates\n");
    }

    // Restore the maximal groups for dynamic selections.
    selections_.evaluateFinal(nframes);

    module_->finishAnalysis(nframes);
    module_->writeOutput();

    return 0;
}

int RunnerModule::analyzeFramesSerial()
{
    const TopologyInformation& topology = common_.topologyInformation();

    t_pbc  pbc;
    t_pbc* ppbc = settings_.hasPBC() ? &pbc : nullptr;

    int                                 nframes = 0;
    AnalysisDataParallelOptions         dataOptions;
    TrajectoryAnalysisModuleDataPointer pdata(module_->startFrames(dataOptions, selections_));
    do
    {
        common_.initFrame();
        t_trxframe& frame = common_.frame();
        if (ppbc != nullptr)
        {
            set_pbc(ppbc, topology.pbcType(), frame.box);
        }

        selections_.evaluate(&frame, ppbc);
        module_->analyzeFrame(nframes, frame, ppbc, pdata.get());
        module_->finishFrameSerial(nframes);

        ++nframes;
    } while (common_.readNextFrame());
    module_->finishFrames(pdata.get());
    if (pdata.get() != nullptr)
    {
        pdata->finish();
    }
    pdata.reset();

    return nframes;
}

/*! \brief
 * Runs analyzeFrame() for multiple frames concurrently.
 *
 * The calling thread reads frames, copies them into a ring of
 * \c windowSize frame buffers and queues them for the worker threads.
 * Each worker owns a copy of the selection collection and a
 * TrajectoryAnalysisModuleData object.  Frames are finished with
 * TrajectoryAnalysisModule::finishFrameSerial() in the calling thread, in
 * frame order.  At most \c windowSize frames are in progress (queued,
 * being analyzed, or analyzed but not yet finished serially) at any time,
 * which is the guarantee that AnalysisDataParallelOptions requires.
 */
class ParallelFrameAnalyzer
{
public:
    ParallelFrameAnalyzer(TrajectoryAnalysisModule*         module,
                          const TrajectoryAnalysisSettings& settings,
                          const TopologyInformation&        topology,
                          const SelectionCollection&        selections,
                          int                               threadCount) :
        module_(*module),
        pbcType_(topology.pbcType()),
        hasPbc_(settings.hasPBC()),
        windowSize_(2 * threadCount),
        frames_(windowSize_),
        isAnalyzed_(windowSize_, false)
    {
        const AnalysisDataParallelOptions dataOptions(windowSize_);
        workers_.reserve(threadCount);
        for (int i = 0; i < threadCount; ++i)
        {
            auto worker        = std::make_unique<Worker>();
            worker->selections = std::make_unique<SelectionCollection>(selections);
            worker->data       = module_.startFrames(dataOptions, *worker->selections);
            workers_.push_back(std::move(worker));
        }
    }

    ~ParallelFrameAnalyzer() { stopThreads(); }

    //! Starts the worker threads.
    void start()
    {
        for (auto& worker : workers_)
        {
            Worker* w      = worker.get();
            worker->thread = std::thread([this, w] { workerLoop(w); });
        }
    }

    //! Queues a frame for analysis, blocking until a frame buffer is free.
    void analyzeFrame(const t_trxframe& frame)
    {
        waitUntilFinishedBefore(nextFrame_ - windowSize_ + 1);
        frames_[nextFrame_ % windowSize_].assign(frame);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.push_back(nextFrame_);
        }
        ++nextFrame_;
        workAvailable_.notify_one();
    }

    /*! \brief
     * Waits for all queued frames, and finishes the threads and
     * thread-local data.
     *
     * \returns Number of analyzed frames.
     */
    int finish(SelectionCollection* selections)
    {
        waitUntilFinishedBefore(nextFrame_);
        stopThreads();
        for (auto& worker : workers_)
        {
            selections->addCoveredFractionSums(*worker->selections);
            module_.finishFrames(worker->data.get());
            if (worker->data != nullptr)
            {
                worker->data->finish();
            }
            worker->data.reset();
        }
        return nextFrame_;
    }

private:
    //! Per-thread state.
    struct Worker
    {
        std::unique_ptr<SelectionCollection> selections;
        TrajectoryAnalysisModuleDataPointer  data;
        t_pbc                                pbc;
        std::thread                          thread;
    };

    void workerLoop(Worker* worker)
    {
        while (true)
        {
            int frameIndex = 0;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                workAvailable_.wait(lock, [this] { return !queue_.empty() || bStop_ || error_; });
                if (queue_.empty() || error_)
                {
                    return;
                }
                frameIndex = queue_.front();
                queue_.pop_front();
            }
            try
            {
                t_trxframe& frame = frames_[frameIndex % windowSize_].frame;
                t_pbc*      ppbc  = hasPbc_ ? &worker->pbc : nullptr;
                if (ppbc != nullptr)
                {
                    set_pbc(ppbc, pbcType_, frame.box);
                }
                worker->selections->evaluate(&frame, ppbc);
                module_.analyzeFrame(frameIndex, frame, ppbc, worker->data.get());
            }
            catch (...)
            {
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    if (!error_)
                    {
                        error_ = std::current_exception();
                    }
                }
                workAvailable_.notify_all();
                frameAnalyzed_.notify_all();
                return;
            }
            {
                std::lock_guard<std::mutex> lock(mutex_);
                isAnalyzed_[frameIndex % windowSize_] = true;
            }
            frameAnalyzed_.notify_one();
        }
    }

    /*! \brief
     * Finishes analyzed frames in order until all frames before \p frameIndex
     * have been finished.
     *
     * Rethrows any exception thrown in a worker thread.
     */
    void waitUntilFinishedBefore(int frameIndex)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        while (nextToFinish_ < frameIndex)
        {
            frameAnalyzed_.wait(
                    lock, [this] { return error_ || isAnalyzed_[nextToFinish_ % windowSize_]; });
            if (error_)
            {
                std::rethrow_exception(error_);
            }
            isAnalyzed_[nextToFinish_ % windowSize_] = false;
            lock.unlock();
            module_.finishFrameSerial(nextToFinish_);
            lock.lock();
            ++nextToFinish_;
        }
    }

    void stopThreads()
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            // Frames still queued here are only abandoned after an error.
            queue_.clear();
            bStop_ = true;
        }
        workAvailable_.notify_all();
        for (auto& worker : workers_)
        {
            if (worker->thread.joinable())
            {
                worker->thread.join();
            }
        }
    }

    TrajectoryAnalysisModule& module_;
    const PbcType             pbcType_;
    const bool                hasPbc_;
    //! Maximum number of frames in progress at any time.
    const int windowSize_;
    //! Frame buffers, frame \c i is stored at \c i % windowSize_.
    std::vector<FrameCopy> frames_;
    //! Per-thread state; the pointers keep the addresses stable.
    std::vector<std::unique_ptr<Worker>> workers_;

    //! Guards \a queue_, \a isAnalyzed_, \a bStop_ and \a error_.
    std::mutex              mutex_;
    std::condition_variable workAvailable_;
    std::condition_variable frameAnalyzed_;
    //! Frames waiting to be analyzed.
    std::deque<int> queue_;
    //! Whether the frame in each buffer has been analyzed.
    std::vector<bool> isAnalyzed_;
    //! Whether worker threads should exit.
    bool bStop_ = false;
    //! First exception thrown in a worker thread.
    std::exception_ptr error_;

    //! Index of the next frame to queue (only accessed by the calling thread).
    int nextFrame_ = 0;
    //! Index of the next frame to finish serially (only modified by the calling thread).
    int nextToFinish_ = 0;
};

int RunnerModule::analyzeFramesParallel(int threadCount)
{
    ParallelFrameAnalyzer analyzer(
            module_.get(), settings_, common_.topologyInformation(), selections_, threadCount);
    analyzer.start();
    do
    {
        common_.initFrame();
        analyzer.analyzeFrame(common_.frame());
    } while (common_.readNextFrame());
    return analyzer.finish(&selections_);
}

} // namespace

/********************************************************************
 * TrajectoryAnalysisCommandLineRunner
 */

// static
int TrajectoryAnalysisCommandLineRunner::runAsMain(int argc, char* argv[], const ModuleFactoryMethod& factory)
{
    auto runnerFactory = [factory] { return createModule(factory()); };
    return ICommandLineOptionsModule::runAsMain(argc, argv, nullptr, nullptr, runnerFactory);
}

// static
void TrajectoryAnalysisCommandLineRunner::registerModule(CommandLineModuleManager*  manager,
                                                         const char*                name,
                                                         const char*                description,
                                                         const ModuleFactoryMethod& factory)
{
    auto runnerFactory = [factory] { return createModule(factory()); };
    ICommandLineOptionsModule::registerModuleFactory(manager, name, description, runnerFactory);
}

// static
std::unique_ptr<ICommandLineOptionsModule>
TrajectoryAnalysisCommandLineRunner::createModule(TrajectoryAnalysisModulePointer module)
{
    return ICommandLineOptionsModulePointer(new RunnerModule(std::move(module)));
}

} // namespace gmx
