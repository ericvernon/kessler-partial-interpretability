import numpy as np


def center_coords(ship_position, ship_heading, asteroid_positions, map_size):
    """
    Given a ship's position and heading, find the polar coordinates of all asteroids relative to the ship.
    For sample usage, check the unit tests!
    :param ship_position: The cartesian (x, y) of the ship
    :param ship_heading: The heading of the ship, in radians. A zero-heading indicates the ship is facing right.
                        Note that kessler-lib uses degrees, convert before calling.
    :param asteroid_positions: An (n,2) numpy array of the asteroid (x,y) positions
    :param map_size: A (2,) numpy array of the map size in (x, y) units
    :return: An (n,2) numpy array of the asteroid (rho, phi) positions relative to the ship.
             An angle of 0 indicates the asteroid is directly in front of the ship.
             The angle will always be within the range [0, 2pi)
             !! IMPORTANT!! The map wraps around at the edges. This means each asteroid an asteroid which is visually
                            "in front" of the ship, is **also** behind the ship at a different distance. This function
                            will only return the position with the smallest rho-value (i.e. closest to the ship), which
                            may not correspond to the visual position you might see at first-glance due to the wrapping.

    """
    # I'm not confident this is the most efficient solution! But seems to work.
    # Move the ship to the center of the map
    center = map_size / 2
    offset = center - ship_position
    ship_position = ship_position.copy() + offset

    # Offset everything by the same amount, and adjust anything that's now "out of bounds"
    centered_asteroids = np.mod(asteroid_positions + offset, map_size)

    # The ship is in the middle now, so the shortest path from ship <--> asteroid can never wrap around edges
    # Now get the coordinates of each asteroid relative to the ship (i.e. the center)
    centered_asteroids -= ship_position

    # Convert cartesian coordinates to polar
    rho, phi = c2p(centered_asteroids[:, 0], centered_asteroids[:, 1])

    # Rotate everything relative to the ship's heading, keep in range [0, 2pi)
    phi -= ship_heading
    phi = np.mod(phi, 2 * np.pi)

    return np.stack([rho, phi], axis=-1)


def parse_game_state(ship_state, game_state, forecast_seconds=1):
    ship_position = np.array(ship_state['position'], dtype=np.float64)
    ship_heading = np.radians(ship_state['heading'])
    ship_velocity = np.array(ship_state['velocity'], dtype=np.float64)
    ship_speed = np.array([ship_state['speed']])

    asteroids = game_state['asteroids']
    asteroid_positions = np.array([asteroid['position'] for asteroid in asteroids], dtype=np.float64)
    asteroid_velocity = np.array([asteroid['velocity'] for asteroid in asteroids], dtype=np.float64)
    asteroid_radii = np.array([asteroid['radius'] for asteroid in asteroids])

    map_size = np.array(game_state['map_size'])

    ship_future_position = np.mod(ship_position + (forecast_seconds * ship_velocity), map_size)
    asteroid_future_positions = np.mod(asteroid_positions + (forecast_seconds * asteroid_velocity), map_size)

    centered_asteroids = center_coords(ship_position, ship_heading, asteroid_positions, map_size)
    centered_future_asteroids = center_coords(ship_future_position, ship_heading, asteroid_future_positions, map_size)

    return {
        'ship': {
            'ship_position': ship_position,
            'future_position': ship_future_position,
            'ship_heading': ship_heading,
            'ship_velocity': ship_velocity,
            'ship_speed': ship_speed,
            'is_respawning': ship_state['is_respawning'],
        },
        'asteroids': {
            'xy_positions': asteroid_positions,
            'xy_future_positions': asteroid_future_positions,
            'xy_velocity': asteroid_velocity,
            'polar_positions': centered_asteroids,
            'polar_future_positions': centered_future_asteroids,
            'radii': asteroid_radii,
            'python_obj': game_state['asteroids'],
        },
        'game': {
            'map_size': map_size,
            'time': game_state['time'],
            'delta_time': game_state['delta_time']
        }
    }


def c2p(x, y):
    z = x + 1j * y
    return np.abs(z), np.angle(z)


def p2c(rho, phi):
    return rho * np.cos(phi), rho * np.sin(phi)
