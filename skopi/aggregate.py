import numpy as np
from skopi.particleCollection import ParticleCollection

def build_bpca(num_pcles=1024, radius=0.5, overlap=None, output=True):
    """
    Build a simple ballistic particle cluster aggregate by generating particle and
    allowing it to stick where it first intersects another particle.
    
    If overlap is set to a value between 0 and 1, monomers will be allowed to overlap
    by 0.5*overlap*(radius1+radius2).
    """
    debug=False
    if overlap is not None:
        if (overlap<0.) or (overlap>1.):
            print('ERROR: overlap must be either None, or 0<overlap<1')
            return None

    pc = ParticleCollection(max_pcles=num_pcles, debug=debug)
    pc.add( (0.,0.,0.), radius)

    # generate a "proposed" particle and trajectory, and see where it intersects the
    # aggregate. add the new particle at this point!
    for n in range(num_pcles-1):

        success = False
        while not success:
            
            if output: print('Generating particle %d of %d' % (n+2, num_pcles))

            first = random_sphere() * max(pc.farthest() * 2.0, radius *4.)
            second = random_sphere() * max(pc.farthest() * 2.0, radius *4.)
            direction = (second - first)
            direction = direction/np.linalg.norm(direction)
            ids, hit = pc.intersect(first, direction, closest=True)
            if hit is None: continue

            # shift the origin along the line from the particle centre to the intersect
            new = hit + (hit-pc.pos[np.where(pc.idx==ids)[0][0]])

            # add to the simulation, checking for overlap with existing partilces (returns False if overlap detected)
            success = pc.check(new, radius)
            if not success: continue

            # if requested, move the monomer back an amount
            if overlap is not None:
                new = hit + (hit-pc.pos[ids])*(1.-overlap)

            pc.add(new, radius)


            # if proposed particle is acceptable, add to the sim and carry on
            if success & debug: print('Adding particle at distance %f' % np.linalg.norm(hit))

    return pc

def random_sphere():
    """
    Returns a random point on a unit sphere.
    """

    u = np.random.uniform(-1,1)
    theta = np.random.uniform(0,2*np.pi)
    x = np.cos(theta)*np.sqrt(1-u*u)
    y = np.sin(theta)*np.sqrt(1-u*u)
    z = u

    return np.array([x,y,z])
