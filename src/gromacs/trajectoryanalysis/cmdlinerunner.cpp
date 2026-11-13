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
    common_.moduleOptionsFinished();
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

/*! \brief
 * State for analyzing frames in one thread.
 *
 * Holds the selections, the thread-local module data and the PBC
 * structure used for calls to TrajectoryAnalysisModule::analyzeFrame()
 * from one thread.
 */
class FrameAnalysisContext
{
public:
    /*! \brief
     * Starts thread-local data for analyzing frames.
     *
     * \p selections must stay valid for the lifetime of this object, and
     * are evaluated for each analyzed frame.
     */
    FrameAnalysisContext(TrajectoryAnalysisModule*          module,
                         const AnalysisDataParallelOptions& dataOptions,
                         SelectionCollection*               selections,
                         PbcType                            pbcType,
                         bool                               hasPbc) :
        module_(*module),
        selections_(*selections),
        pbcType_(pbcType),
        hasPbc_(hasPbc),
        data_(module->startFrames(dataOptions, *selections))
    {
    }

    //! Evaluates the selections for \p frame and analyzes it.
    void analyze(int frameIndex, t_trxframe* frame)
    {
        t_pbc* ppbc = hasPbc_ ? &pbc_ : nullptr;
        if (ppbc != nullptr)
        {
            set_pbc(ppbc, pbcType_, frame->box);
        }
        selections_.evaluate(frame, ppbc);
        module_.analyzeFrame(frameIndex, *frame, ppbc, data_.get());
    }

    //! Finishes the thread-local data after all frames have been analyzed.
    void finish()
    {
        module_.finishFrames(data_.get());
        if (data_ != nullptr)
        {
            data_->finish();
        }
        data_.reset();
    }

private:
    TrajectoryAnalysisModule&           module_;
    SelectionCollection&                selections_;
    const PbcType                       pbcType_;
    const bool                          hasPbc_;
    TrajectoryAnalysisModuleDataPointer data_;
    t_pbc                               pbc_;
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

    const bool isFrameParallel = settings_.hasFlag(TrajectoryAnalysisSettings::efFrameParallel)
                                 && common_.hasTrajectory();
    const int threadCount = settings_.threadCount();
    if (threadCount > 1 && !isFrameParallel
        && !settings_.hasFlag(TrajectoryAnalysisSettings::efInternalParallel))
    {
        fprintf(stderr, "Note: Frames are analyzed serially with these options; -nt is ignored.\n");
    }
    const int nframes = (isFrameParallel && threadCount > 1) ? analyzeFramesParallel(threadCount)
                                                             : analyzeFramesSerial();

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
    FrameAnalysisContext context(module_.get(),
                                 AnalysisDataParallelOptions(),
                                 &selections_,
                                 common_.topologyInformation().pbcType(),
                                 settings_.hasPBC());
    int                  nframes = 0;
    do
    {
        common_.initFrame();
        context.analyze(nframes, &common_.frame());
        module_->finishFrameSerial(nframes);
        ++nframes;
    } while (common_.readNextFrame());
    context.finish();
    return nframes;
}

/*! \brief
 * Runs analyzeFrame() for multiple frames concurrently.
 *
 * The calling thread reads frames, copies them into a ring of
 * \c windowSize frame buffers and makes them available to the worker
 * threads, which take frames in order.  Each worker owns a copy of the
 * selection collection and a FrameAnalysisContext.  Frames are finished with
 * TrajectoryAnalysisModule::finishFrameSerial() in the calling thread, in
 * frame order.  At most \c windowSize frames are in progress (waiting, being
 * analyzed, or analyzed but not yet finished serially) at any time, which is
 * the guarantee that AnalysisDataParallelOptions requires.
 *
 * The destructor stops and joins the worker threads, so that exceptions
 * thrown in the calling thread propagate normally.
 */
class ParallelFrameAnalyzer
{
public:
    ParallelFrameAnalyzer(TrajectoryAnalysisModule*         module,
                          const TrajectoryAnalysisSettings& settings,
                          const TopologyInformation&        topology,
                          const SelectionCollection&        selections,
                          int                               threadCount) :
        module_(*module), windowSize_(2 * threadCount), frames_(windowSize_), isAnalyzed_(windowSize_, false)
    {
        const AnalysisDataParallelOptions dataOptions(windowSize_);
        workers_.reserve(threadCount);
        for (int i = 0; i < threadCount; ++i)
        {
            auto worker        = std::make_unique<Worker>();
            worker->selections = std::make_unique<SelectionCollection>(selections);
            worker->context    = std::make_unique<FrameAnalysisContext>(
                    module, dataOptions, worker->selections.get(), topology.pbcType(), settings.hasPBC());
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

    //! Makes a frame available for analysis, blocking until a frame buffer is free.
    void analyzeFrame(const t_trxframe& frame)
    {
        waitUntilFinishedBefore(frameCount_ - windowSize_ + 1);
        frames_[frameCount_ % windowSize_].assign(frame);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            ++frameCount_;
        }
        workAvailable_.notify_one();
    }

    /*! \brief
     * Waits for all frames, and finishes the threads and thread-local data.
     *
     * Covered fractions computed by the workers are added to \p selections.
     *
     * \returns Number of analyzed frames.
     */
    int finish(SelectionCollection* selections)
    {
        waitUntilFinishedBefore(frameCount_);
        stopThreads();
        for (auto& worker : workers_)
        {
            selections->addCoveredFractionSums(*worker->selections);
            worker->context->finish();
        }
        return frameCount_;
    }

private:
    //! Per-thread state.
    struct Worker
    {
        std::unique_ptr<SelectionCollection>  selections;
        std::unique_ptr<FrameAnalysisContext> context;
        std::thread                           thread;
    };

    void workerLoop(Worker* worker)
    {
        while (true)
        {
            int frameIndex = 0;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                workAvailable_.wait(
                        lock, [this] { return nextToAnalyze_ < frameCount_ || bStop_ || error_; });
                if (nextToAnalyze_ >= frameCount_ || error_)
                {
                    return;
                }
                frameIndex = nextToAnalyze_++;
            }
            try
            {
                worker->context->analyze(frameIndex, &frames_[frameIndex % windowSize_].frame);
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
            // Frames not yet taken by a worker here are only abandoned
            // after an error.
            nextToAnalyze_ = frameCount_;
            bStop_         = true;
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
    //! Maximum number of frames in progress at any time.
    const int windowSize_;
    //! Frame buffers, frame \c i is stored at \c i % windowSize_.
    std::vector<FrameCopy> frames_;
    //! Per-thread state; the pointers keep the addresses stable.
    std::vector<std::unique_ptr<Worker>> workers_;

    //! Guards the members below, except where noted.
    std::mutex              mutex_;
    std::condition_variable workAvailable_;
    std::condition_variable frameAnalyzed_;
    /*! \brief
     * Number of frames made available for analysis.
     *
     * Only modified by the calling thread, which can read it without
     * locking.
     */
    int frameCount_ = 0;
    //! Index of the next frame for a worker to analyze.
    int nextToAnalyze_ = 0;
    //! Whether the frame in each buffer has been analyzed.
    std::vector<bool> isAnalyzed_;
    //! Whether worker threads should exit.
    bool bStop_ = false;
    //! First exception thrown in a worker thread.
    std::exception_ptr error_;
    /*! \brief
     * Index of the next frame to finish serially.
     *
     * Only accessed by the calling thread.
     */
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
