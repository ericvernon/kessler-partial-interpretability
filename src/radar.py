import numpy as np

DEFAULT_RADAR_ZONES = [100, 300, 500]


def get_radar(centered_asteroids, asteroid_radii, radar_zones=None):
    """
    Given a list of asteroid positions **relative to some point** (e.g. the ship) and asteroid sizes,
    return a "radar" like view of the region surrounding the reference point.
    :param centered_asteroids: An (n,2) numpy array of asteroid positions. These positions should be relative to
                               some reference point (e.g. the ship) and in polar (rho, phi) format.
    :param asteroid_radii: An (n,) numpy array of asteroid radii.
    :param radar_zones: Optional, a (3,) array of the distances that are considered "near", "middle", and "far".
    :return: A (3,4) numpy array representing the radar. The radar is divided into twelve zones, i.e.:
                (Near, Middle, Far) X (Right, Front, Left, Rear)
             An index of [1, 2] would refer to "Middle-Left", while [0, 3] refers to "Near-Rear".
             Each entry corresponds to the total area of asteroids centered in that zone,
             divided by the total area of the zone, as a float in the range [0, 1].
             !! Note the phrasing, "centered in that zone." An asteroids entire area is assigned to the zone where its
             center is positioned, regardless of if any of its area "spills over" into other zones. Likewise, if two or
             more asteroids overlap, we simply sum their areas, instead of the actual area covered. This does mean, in
             certain cases, the total sum of asteroid areas might exceed the area of the radar zone itself -- however,
             we artificially cap the output of each radar value at 1.
    """
    asteroid_areas = np.pi * asteroid_radii * asteroid_radii
    rho, phi = centered_asteroids[:, 0], centered_asteroids[:, 1]
    if radar_zones is None:
        radar_zones = DEFAULT_RADAR_ZONES

    is_near = rho < radar_zones[0]
    is_medium = np.logical_and(rho < radar_zones[1], rho >= radar_zones[0])
    is_far = np.logical_and(rho < radar_zones[2], rho >= radar_zones[1])

    is_front = np.logical_or(phi < 0.25 * np.pi, phi >= 1.75 * np.pi)
    is_left = np.logical_and(phi < 0.75 * np.pi, phi >= 0.25 * np.pi)
    is_behind = np.logical_and(phi < 1.25 * np.pi, phi >= 0.75 * np.pi)
    is_right = np.logical_and(phi < 1.75 * np.pi, phi >= 1.25 * np.pi)

    inner_area = np.pi * radar_zones[0] * radar_zones[0]
    middle_area = np.pi * radar_zones[1] * radar_zones[1]
    outer_area = np.pi * radar_zones[2] * radar_zones[2]
    # The area of one slice in the outer, middle, and inner donuts
    slice_areas = [(outer_area - (middle_area + inner_area)) / 4, (middle_area - inner_area) / 4, inner_area / 4]

    radar_info = np.zeros(shape=(3,4))
    for idx, distance_mask in enumerate([is_near, is_medium, is_far]):
        slice_area = slice_areas[idx]
        for jdx, angle_mask in enumerate([is_right, is_front, is_left, is_behind]):
            mask = np.logical_and(distance_mask, angle_mask)
            total_asteroid_area = np.sum(asteroid_areas[mask])
            radar_info[idx, jdx] = min(1, total_asteroid_area / slice_area)

    return radar_info
