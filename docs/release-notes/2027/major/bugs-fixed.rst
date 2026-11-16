Bugs fixed
^^^^^^^^^^

.. Note to developers!
   Please use """"""" to underline the individual entries for fixed issues in the subfolders,
   otherwise the formatting on the webpage is messed up.
   Also, please use the syntax :issue:`number` to reference issues on GitLab, without
   a space between the colon and number!

Require at least 100 integration steps for Nose-Hoover thermostat
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

The minimum required integration steps for the Nose-Hoover thermostat of 20
was too low. This could lead to an overestimate of the fluctuation of the
kinetic energy and temperature by up to 25%. Note that the averages were
not affected.

:issue:`5783`

PDB with a box of size 1 Angstrom are interpreted as having no PBC
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

PDB files with structures not determined from X-ray should have a unit-cell with P=1
and dimensions of 1 Angstrom to indicate no periodic boundary conditions.
|Gromacs| now interprets such structures as not having PBC.

:issue:`4645`, :issue:`5679`

PDB trajectory output now writes correct per-frame PBC type
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

When writing PDB trajectories, the PBC type of the system was ignored
and instead guessed from the box. As a result, the ``CRYST1`` record
was written incorrectly for screw PBC (wrong space group and cell length).

Analysis tools now always process time values in double precision
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

This avoids picking incorrect frames when times values are large.

Request the MPI thread support required by CP2K
"""""""""""""""""""""""""""""""""""""""""""""""

With external MPI, setting the new environment variable
``GMX_USE_MPI_THREAD_MULTIPLE`` to any value requests ``MPI_THREAD_MULTIPLE``.
The default MPI initialization is unchanged. QM/MM setup checks the linked
CP2K version and reports a helpful fatal error if it requires
``MPI_THREAD_MULTIPLE`` but MPI was initialized with insufficient thread support.
This applies to CP2K 2027 or later, CP2K 2026.2 development builds, and
unrecognized CP2K versions. Users are instructed to set the environment
variable before launching |Gromacs|.

Collective variables module (Colvars) fixes
"""""""""""""""""""""""""""""""""""""""""""

Below are issues fixed in the (`Colvars <https://colvars.github.io>`_) library (also included in the minor versions of the |Gromacs| 2026 release).

- Improved restart consistency and fixes for multi-replica well-tempered metadynamics (Colvars repo pull request `820 <https://github.com/Colvars/colvars/pull/820>`_)
- Fixed wrapping of extended-system variable coupled to orientationAngle (Colvars repo pull request `943 <https://github.com/Colvars/colvars/pull/943>`_)

Correct dV/dlambda for perturbed GROMOS-96 bonds
""""""""""""""""""""""""""""""""""""""""""""""""

The derivative of the potential energy with respect to lambda for
perturbed GROMOS-96 (quartic) bonds was a factor of two too large. Forces and
energies were correct. Free-energy differences computed from dH/dlambda
or its derived quantities with such bonds perturbed were affected.

Correct dV/dlambda for perturbed linear angles
""""""""""""""""""""""""""""""""""""""""""""""

The contribution to dV/dlambda from perturbing the parameter a of the
linear-angle potential had the wrong sign. Forces and energies were correct.

Correct dV/dlambda for perturbed angle restraints
"""""""""""""""""""""""""""""""""""""""""""""""""

The contribution to dV/dlambda from perturbing the reference angle of
angle restraints, with or without z-axis, had the wrong sign and lacked
a factor of the multiplicity. Forces and energies were correct.

Correct virial contribution of angle restraints across periodic boundaries
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

The shift forces of angle restraints had the wrong sign. This led to
an incorrect virial, and thus pressure, when the two atoms defining a
restraint vector were in different periodic images. Forces and energies
were correct.

Correct wall energies and dV/dlambda with wall-r-linpot and perturbed atoms
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

With perturbed atoms closer to a wall than :mdp:`wall-r-linpot`,
the linear part of the wall potential was scaled twice with the lambda
factor. This led to incorrect wall energies and dV/dlambda. Forces were
correct.

Correct wall virial with wall-r-linpot
""""""""""""""""""""""""""""""""""""""

For atoms closer to a wall than :mdp:`wall-r-linpot`, the virial was computed
using that distance instead of the actual distance to the wall, which led
to an incorrect pressure along z.

Correct forces with Ewald geometry 3DC, net charge and perturbed charges
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

With :mdp:`ewald-geometry` set to ``3dc``, a system with net charge and
perturbed charges, the force correction for the net charge in the B state
was not multiplied by the B-state charge of the atom.

Correct the virial of the Ewald surface and 3DC corrections
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

The virial contribution of the Ewald dipole corrections used with
:mdp:`epsilon-surface` or :mdp:`ewald-geometry` set to ``3dc``
was divided by the box volume instead of multiplied by a half. Without
perturbed charges it was also incorrectly scaled by one minus lambda.

Correct dV/dlambda with Gapsys soft-core and potential-switched Van der Waals
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

With the Gapsys soft-core function and a potential-switch modifier for
Van der Waals interactions, the soft-core contribution to dV/dlambda was not
multiplied by the switch function. This only affected pairs at distances
that are both in the switching range and below the soft-core linearization
point, which does not occur with commonly used settings.

Correct the Ewald 3DC energy for systems with net charge with domain decomposition
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

With :mdp:`ewald-geometry` set to ``3dc`` and a system with net charge,
the energy correction for the net charge, and the corresponding dV/dlambda,
only included the atoms that are local to the main rank when running with
domain decomposition over multiple ranks. Forces were correct.

Add missing off-diagonal virial for Ewald surface and 3DC corrections
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

The virial contribution of the Ewald dipole corrections used with
:mdp:`epsilon-surface` or :mdp:`ewald-geometry` set to ``3dc``
lacked the off-diagonal elements, which affected the off-diagonal
elements of the pressure tensor.
Correct the virial of the Ewald correction for systems with net charge
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

With Ewald or PME electrostatics and a system with net charge, the
virial contribution of the energy correction for the net charge was
divided by the box volume instead of multiplied by a half. The pressure
contribution of this term was therefore too small by a factor of half
the box volume in cubic nanometers. The error in the pressure is
proportional to the square of the net charge divided by the square of
the volume, so it mainly affected small systems with a large net charge.
The energy and forces were correct.

Correct the Ewald correction for net charge with walls
""""""""""""""""""""""""""""""""""""""""""""""""""""""

With two walls and :mdp:`pbc` set to ``xy``, the Ewald unit cell is
elongated along z by :mdp:`wall-ewald-zfac`, but the energy correction for
systems with net charge used the volume of the simulation box. The
correction, its contribution to dV/dlambda when the net charge is
perturbed and its virial were thus a factor :mdp:`wall-ewald-zfac` too
large. Forces were correct.
Correct pull forces on groups with cosine weighting
"""""""""""""""""""""""""""""""""""""""""""""""""""

With COM pulling using a pull group with cosine weighting
(``pull-group?-pbcatom = -1``), the forces on the atoms of that group
were scaled by a factor that, depending on the location of the group in
the box, was up to 25% smaller than one. The pull energy and the forces
on other pull groups were correct, so the pull forces were not conservative
and did not sum to zero.

Correct the effective mass of cosine-weighted pull groups with constraint pulling
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

The effective mass of a pull group with cosine weighting
(``pull-group?-pbcatom = -1``) was underestimated by up to 25%,
depending on the location of the group in the box.
This affected the constraint force with constraint pulling.

Correct radial pull forces with cylinder geometry and a pull vector not along a positive axis
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

With :mdp-value:`pull-coord1-geometry=cylinder` the radial forces,
which arise from the radial dependence of the weights of the atoms
in the cylinder group, were incorrect when the pull vector was not
along the positive x, y or z direction, e.g. with a vector of ``0 0 -1``.
The pull energy and the forces along the pull vector were correct.

Avoid incorrect forces with direction-relative pulling
""""""""""""""""""""""""""""""""""""""""""""""""""""""

With :mdp-value:`pull-coord1-geometry=direction-relative`, uninitialized
values were added to the forces on, and to the virial of, the two pull
groups that define the pull direction. In practice these values were often
zero, but when a pull coordinate with geometry angle or dihedral with
a higher index was present, the forces of that coordinate were applied
a second time to the groups defining the direction.

Correct the virial with direction-relative pulling
""""""""""""""""""""""""""""""""""""""""""""""""""

With :mdp-value:`pull-coord1-geometry=direction-relative`, the forces
on the two pull groups that define the pull direction were not included
in the virial. This led to an incorrect pressure, although the trace
of the pressure tensor was not affected.

Correct off-diagonal virial elements of pull forces
"""""""""""""""""""""""""""""""""""""""""""""""""""

The off-diagonal elements of the virial contribution of COM pulling
with a potential were transposed, which matters for pull geometries
direction and angle-axis, where the pull force is not parallel to
the distance vector between the groups. Furthermore, off-diagonal
elements were missing when pulling was not applied in all dimensions.
This affected the off-diagonal elements of the pressure tensor. The
diagonal elements, and thus simulations with isotropic or semi-isotropic
pressure coupling or a rectangular box, were not affected.

Correct the virial with cylinder pulling
""""""""""""""""""""""""""""""""""""""""

With :mdp-value:`pull-coord1-geometry=cylinder`, only the force along
the pull vector acting between the pull group and the axial location
of the center of mass of the cylinder group was included in the virial.
The contributions of the radial forces and of the radial offset
of the cylinder group were missing. With a cylinder along z, this
affected all elements of the virial, and thus the pressure,
except for the zz element.

Center the cylinder on the pull group with cylinder pulling and an external potential
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

With :mdp-value:`pull-coord1-geometry=cylinder` combined with
:mdp-value:`pull-coord1-type=external-potential`, e.g. when using AWH,
the cylinder that selects and weights the atoms of the reference
group was centered on the center of mass of the whole reference group
instead of on the pull group. The pull coordinate value was therefore
not the distance to the local part of the reference group and
the radial forces did not match the coordinate.
Correct density-guided simulation forces with non-unit amplitudes
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

The density-guided simulation force on an atom was multiplied twice by
the amplitude with which the atom is spread on the density grid. With
:mdp:`density-guided-simulation-normalize-densities` set to ``yes``,
which is the default, and unit amplitudes, the amplitude is one over
the number of fitted atoms, so the forces were too small by that factor
compared to the reported energy. With mass or charge as amplitude, the
force on each atom was additionally scaled by its (normalized) mass
or charge. Energies were correct. Note that, to obtain the same forces
as before with unit amplitudes and normalization,
:mdp:`density-guided-simulation-force-constant` needs to be divided by
the number of atoms in the fitting group.

Correct density-guided simulations with two domains
"""""""""""""""""""""""""""""""""""""""""""""""""""

With exactly two particle-particle MPI ranks, the simulated density was
not summed over the ranks, so each rank computed the similarity and the
forces from the density of its own atoms only. This led to incorrect
density-guided simulation forces and energies when atoms of the fitting
group were present on both ranks. Runs with one rank or with more than
two ranks were not affected.

Correct essential dynamics flooding forces with mass-weighted eigenvectors
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

With eigenvectors from a mass-weighted covariance analysis, the flooding
potential is a function of mass-weighted projections on the eigenvectors,
but the flooding force on each atom lacked the corresponding factor of
the square root of the mass of the atom. The forces were thus not
consistent with the flooding energy, also for harmonic restraints and
constant-force flooding. Flooding with eigenvectors from an analysis
without mass weighting was not affected.

Correct periodic image selection for density-guided simulation with non-cubic densities
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

With periodic boundary conditions, atoms are placed in the periodic image
closest to the center of the reference density. The center was computed
with the number of voxels along x and z interchanged. With reference densities
with different extents along x and z, in boxes not much larger than the
density, atoms could be placed in a periodic image outside the density, where
they do not contribute to the similarity and experience no force.
