import numpy as np

import pysingfel as ps
import pysingfel.constants as cst


def test_particle_rotate_x():
    """Test particle rotation around z."""
    particle_1 = ps.Particle()
    particle_1.create_from_atoms([
        ("O", cst.vecy),
        ("O", 2*cst.vecz)
    ])

    particle_2 = ps.Particle()
    particle_2.create_from_atoms([
        ("O", cst.vecz),
        ("O", -2*cst.vecy)
    ])

    particle_1.rotate(cst.quatx90)
    assert np.allclose(particle_1.atom_pos, particle_2.atom_pos, atol=1e-20)


def test_particle_rotate_y():
    """Test particle rotation around z."""
    particle_1 = ps.Particle()
    particle_1.create_from_atoms([
        ("O", cst.vecz),
        ("O", 2*cst.vecx)
    ])

    particle_2 = ps.Particle()
    particle_2.create_from_atoms([
        ("O", cst.vecx),
        ("O", -2*cst.vecz)
    ])

    particle_1.rotate(cst.quaty90)
    assert np.allclose(particle_1.atom_pos, particle_2.atom_pos, atol=1e-20)


def test_particle_rotate_z():
    """Test particle rotation around z."""
    particle_1 = ps.Particle()
    particle_1.create_from_atoms([
        ("O", cst.vecx),
        ("O", 2*cst.vecy)
    ])

    particle_2 = ps.Particle()
    particle_2.create_from_atoms([
        ("O", cst.vecy),
        ("O", -2*cst.vecx)
    ])

    particle_1.rotate(cst.quatz90)
    assert np.allclose(particle_1.atom_pos, particle_2.atom_pos, atol=1e-20)
