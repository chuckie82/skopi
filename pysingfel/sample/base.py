class Sample(object):
    def __init__(self, particles):
        self.particle_kinds = particles

    @property
    def n_particle_kinds(self):
        """
        Return the number of kinds of particles in the sample.
        """
        return len(self.particle_kinds)

    def generate_new_state(self):
        """
        Return a list of "particle group"

        Each group is a tuple (positions, orientations)
        where the positions and orientations have one line
        per particle in the sample at this state.
        """
        raise NotImplementedError
