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
            self.volumes.append(
                pg.calculate_diffraction_pattern_gpu(mesh, particle, return_type='complex_field'))

    def generate_image(self):
        beam_spectrum = self.beam.generate_new_state()
        img_stack = 0.

        for spike in beam_spectrum:
            recidet = ReciprocalDetector(self.det, spike)

            slice_ = 0.
            for i, particle_group in enumerate(self.generate_new_sample_state()):
                positions, orientations = particle_group

                slices = psg.take_n_slices(
                    volume=self.volumes[i],
                    voxel_length=self.voxel_length,
                    pixel_momentum=recidet.pixel_position_reciprocal,
                    orientations=orientations,
                    inverse=True)

                for j, position in enumerate(positions):
                    recidet.add_phase_shift(slices[j], position)

                slice_ += slices.sum(axis=0)

            intens_slice = np.abs(slice_)**2
            raw_img = recidet.add_correction_and_quantization(intens_slice)
            img_stack += raw_img

        return self.det.assemble_image_stack(img_stack)

    def generate_new_sample_state(self):
        """
        Return a list of "particle group"

        Each group is a tuple (positions, orientations)
        where the positions and orientations have one line
        per particle in the sample at this state.
        """
        raise NotImplementedError
