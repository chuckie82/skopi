import numpy as np
import skopi.geometry as psg
from skopi.reciprocal_detector import ReciprocalDetector


class Experiment(object):
    mesh_size = 151
    
    def __init__(self, det, beam, particles):
        self.det = det
        self.beam = beam
        self.particles = particles
        self.n_particle_kinds = len(particles)

        # Create mesh
        highest_k_beam = self.beam.get_highest_wavenumber_beam()
        recidet = ReciprocalDetector(self.det, highest_k_beam)
        mesh, self.voxel_length = recidet.get_reciprocal_mesh(
            voxel_number_1d=self.mesh_size)

        # Create volumes
        import skopi.gpu as pg
        self.volumes = []
        for particle in particles:
            if particle is None:
                self.volumes.append(np.zeros(mesh.shape[:-1], np.complex128))
                continue
            self.volumes.append(
                pg.calculate_diffraction_pattern_gpu(mesh, particle, return_type='complex_field'))

        # soon obsolete flag to handle multi particle hit
        self.multi_particle_hit = False

        # set up list to track diplacements from beam center
        self.beam_displacements = list()

    def generate_image(self, return_orientation=False):
        if return_orientation:
            img_stack, orientation = self.generate_image_stack(return_orientation=return_orientation)
            return self.det.assemble_image_stack(img_stack), orientation
        else:
            img_stack = self.generate_image_stack()
            return self.det.assemble_image_stack(img_stack)

    def generate_image_stack(self, return_photons=None,
                             return_intensities=False,
                             return_orientation=False,
                             always_tuple=False, noise={}):
        """
        Generate and return a snapshot of the experiment.

        By default, return a photon snapshot.
        That behavior can be changed by setting:
          - return_photons
          - return_intensities
        to True or False.

        If more than one is requested, the function returns a tuple of
        arrays instead of the array itself.
        To return a tuple even if only one array is requested, set
        always_tuple to True.

        Noise is introduced based on contents of noise parameter. Dark
        noise, beam jitter, and static noise are currently implemented.
        """
        if return_photons is None and return_intensities is False:
            return_photons = True

        beam_spectrum = self.beam.generate_new_state()

        sample_state = self.generate_new_sample_state()

        intensities_stack = 0.

        orientations = sample_state[0][1]

        if ('jitter' in noise.keys()) and (noise['jitter']!=0):
            displacement = self.det.add_beam_jitter(noise['jitter'])
            self.beam_displacements.append(displacement)

        for spike in beam_spectrum:
            recidet = ReciprocalDetector(self.det, spike)

            group_complex_pattern = 0.
            for i, particle_group in enumerate(sample_state):
                next_pattern = self._generate_group_complex_pattern(
                    recidet, i, particle_group)
                if np.sum(next_pattern) == 0:
                    print("Using direct calculation instead")
                    next_pattern = self._direct_calculate(recidet, particle_group)
                group_complex_pattern += next_pattern

            group_pattern = np.abs(group_complex_pattern)**2

            # corrections are based on miscentered beam if there's jitter
            group_intensities = self.det.add_correction(group_pattern) 
            intensities_stack += group_intensities

        # add static noise to sum of all spikes and particles
        if 'static' in noise.keys() and noise['static'] is True:
            intensities_stack = self.det.add_static_noise(intensities_stack)

        # add sloped background incoherently
        if 'sloped' in noise.keys():
            if noise['sloped'].shape != self.det.shape:
                noise['sloped'] = self.det.disassemble_image_stack(noise['sloped'])
            intensities_stack += noise['sloped']

        # We are summing up intensities then converting to photons as opposed to converting to photons then summing up.
        # Note: We may want to revisit the correctness of this procedure.
        photons_stack = recidet.add_quantization(intensities_stack)

        ret = []
        if return_photons:
            ret.append(photons_stack)
        if return_intensities:
            ret.append(intensities_stack)

        if return_orientation:
            if len(ret) == 1:
                return ret[0], orientations
            return tuple(ret), orientations
        else:
            if len(ret) == 1:
                return ret[0]
            return tuple(ret)

    def _generate_group_complex_pattern(self, recidet, i, particle_group):
        positions, orientations = particle_group

        slices = psg.take_n_slices(
            volume=self.volumes[i],
            voxel_length=self.voxel_length,
            pixel_momentum=recidet.pixel_position_reciprocal,
            orientations=orientations,
            inverse=True)

        for j, position in enumerate(positions):
            slices[j] = recidet.add_phase_shift(slices[j], position)

        return slices.sum(axis=0)

    def _direct_calculate(self, recidet, particle_group):
        """
        Compute patterns using the Detector class, so directly at reciprocal
        space positions of interest.
    
        :param recidet: ReciprocalDetector object
        :param particle_group: list of positions and orientations per particle
        :return pattern: complex field for particle group
        """
        positions, orientations = particle_group
        pattern = 0
    
        for j in range(len(orientations)):
            self.particles[j].rotate(orientations[j])
            next_slice = recidet.get_pattern_without_corrections(self.particles[j], return_type='complex_field')
            pattern += recidet.add_phase_shift(next_slice, positions[j])

            # unrotate particle
            rot_mat = psg.convert.quaternion2rot3d(orientations[j])
            rot_mat_inv = np.linalg.inv(rot_mat)
            quat_inv = psg.convert.rotmat_to_quaternion(rot_mat_inv)
            self.particles[j].rotate(quat_inv)
        
        return pattern

    def generate_new_sample_state(self):
        """
        Return a list of "particle group"

        Each group is a tuple (positions, orientations)
        where the positions and orientations have one line
        per particle in the sample at this state.
        """
        raise NotImplementedError

    def set_multi_particle_hit(self, multi_particle_hit):
        self.multi_particle_hit = multi_particle_hit
