/*
 *
 *                This source code is part of
 *
 *                 G   R   O   M   A   C   S
 *
 *          GROningen MAchine for Chemical Simulations
 *
 * Written by David van der Spoel, Erik Lindahl, Berk Hess, and others.
 * Copyright (c) 1991-2000, University of Groningen, The Netherlands.
 * Copyright (c) 2001-2009, The GROMACS development team,
 * check out http://www.gromacs.org for more information.

 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * If you want to redistribute modifications, please consider that
 * scientific software is very special. Version control is crucial -
 * bugs must be traceable. We will be happy to consider code for
 * inclusion in the official distribution, but derived work must not
 * be called official GROMACS. Details are found in the README & COPYING
 * files - if they are missing, get the official version at www.gromacs.org.
 *
 * To help us fund GROMACS development, we humbly ask that you cite
 * the papers on the package - you can find them in the top README file.
 *
 * For more info, check our website at http://www.gromacs.org
 */
/*! \internal \file
 * \brief
 * Declares mock implementation of gmx::AnalysisDataModuleInterface.
 *
 * Requires Google Mock.
 *
 * \author Teemu Murtola <teemu.murtola@cbr.su.se>
 * \ingroup module_analysisdata
 */
#ifndef GMX_ANALYSISDATA_TESTS_MOCK_MODULE_H
#define GMX_ANALYSISDATA_TESTS_MOCK_MODULE_H

#include <gmock/gmock.h>

#include "gromacs/analysisdata/datamodule.h"

class MockModule : public gmx::AnalysisDataModuleInterface
{
    public:
        explicit MockModule(int flags) : _flags(flags) {}

        int flags() const { return _flags; }

        MOCK_METHOD1(dataStarted, void(gmx::AbstractAnalysisData *data));
        MOCK_METHOD2(frameStarted, void(real x, real dx));
        MOCK_METHOD7(pointsAdded, void(real x, real dx, int firstcol, int n,
                                       const real *y, const real *dy,
                                       const bool *present));
        MOCK_METHOD0(frameFinished, void());
        MOCK_METHOD0(dataFinished, void());

    private:
        int         _flags;
};

#endif
