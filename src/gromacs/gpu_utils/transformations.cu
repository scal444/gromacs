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
/*! \libinternal \file
 *  \brief Defines  3D transformation functions for coordinate systems.
 *
 *  \author Kevin Boyd <kevin44boyd@gmail.com>
 *  \inlibraryapi
 */

#include "gromacs/gpu_utils/vectype_ops.cuh"


/*! \brief
 * Calculates the center of mass for a set of positions.
 *
 * \param[in] positions      3D coordinates
 * \param[in] masses         (optional) particle masses. If nullptr - will take unweighted average
 * \param[in] indices        (optional) index array. Non-indexed positions/masses will be ignored
 * \param[in] num_positions  number of indices, if indices != nullptr. Number of total atoms,
 *                           indexed from 0 otherwise
 */
__host__ __device__ float3 center_of_mass( const float3 * positions, const float * masses, const int* indices, const int num_positions)
{
    float3 com = {0.0,0.0,0.0};
    float total_mass = 0.0;
    // Mass-weighted COM
    if (masses != nullptr)
    {
        for (int i = 0; i < num_positions; i++)
        {
            const int idx = indices != nullptr ? indices[i] : i;
            com += positions[idx] * masses[idx];
            total_mass += masses[idx];
        }
    }
        // Non-mass-weighted COM.
    else {
        for (int i = 0; i < num_positions; i++)
        {
            const int idx = indices != nullptr ? indices[i] : i;
            com += positions[idx];
        }
    }
    if (masses != nullptr) {
        com /= total_mass;
    }
    return com;
}
/*! \brief
 * Applies a 3D translation to the given coordinates
 *
 * \param[in] positions      3D coordinates
 * \param[in] num_positions  The number of atoms to translate
 * \param[in] translation    The distance to move
 */
__host__ __device__ void translate(float3 * positions, const int num_positions, const float3 translation)
{
    for (int i = 0; i < num_positions; i++)
    {
        positions[i] += translation;
    }

}