import numpy as np
import pysingfel.geometry as psg
from pysingfel.reciprocal_detector import ReciprocalDetector


class Experiment(object):
    mesh_size = 151

    def __init__(self, det, beam, particles):
        self.det = det
        self.beam = beam
        self.n_particle_kinds = len(particles)

        # Create mesh
        highest_k_beam = self.beam.get_highest_wavenumber_beam()
        recidet = ReciprocalDetector(self.det, highest_k_beam)
        mesh, self.voxel_length = recidet.get_reciprocal_mesh(
            voxel_number_1d=self.mesh_size)

        # Create volumes
        import pysingfel.gpu as pg
        self.volumes = []
        for particle in particles:
            if particle is None:
                self.volumes.append(np.zeros(mesh.shape[:-1], np.complex128))
                continue
            self.volumes.append(
                pg.calculate_diffraction_pattern_gpu(mesh, particle, return_type='complex_field'))

    def generate_image(self):
        img_stack = self.generate_image_stack()
        return self.det.assemble_image_stack(img_stack)

    def generate_image_stack(self, return_photons=True,
                             return_intensities=False):
        beam_spectrum = self.beam.generate_new_state()
        sample_state = self.generate_new_sample_state()
        intensities_stack = 0.
        photons_stack = 0.

        for spike in beam_spectrum:
            recidet = ReciprocalDetector(self.det, spike)

            group_complex_pattern = 0.
            for i, particle_group in enumerate(sample_state):
                group_complex_pattern += self._generate_group_complex_pattern(
                    recidet, i, particle_group)

            group_pattern = np.abs(group_complex_pattern)**2

            group_intensities = recidet.add_correction(group_pattern)
            intensities_stack += group_intensities

            group_photons = recidet.add_quantization(group_intensities)
            photons_stack += group_photons

        ret = []
        if return_photons:
            ret.append(photons_stack)
        if return_intensities:
            ret.append(intensities_stack)

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

    def generate_new_sample_state(self):
        """
        Return a list of "particle group"

        Each group is a tuple (positions, orientations)
        where the positions and orientations have one line
        per particle in the sample at this state.
        """
        raise NotImplementedError
