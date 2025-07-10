import numpy as np

from kesslergame import KesslerController, KesslerGame, Scenario
from typing import Dict, Tuple
from src.lib import parse_game_state


# This is an example of a simple, but "somewhat" intelligent controller:
# Try to navigate away from the nearest asteroid!! The strategy is simple:
# - If the asteroid is "in front" of the ship, try to turn towards it and move backwards
# - Otherwise, the asteroid is "behind" the ship, so turn away from it and move forwards
class OnlyRunController(KesslerController):
    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool, bool]:
        # Parse the game state
        state = parse_game_state(ship_state, game_state)

        # Polar coordinates make it easy to identify the nearest asteroid, and the relative angle!
        nearest_asteroid_idx = np.argmin(state['asteroids']['polar_positions'][:, 0])
        asteroid_distance = state['asteroids']['polar_positions'][nearest_asteroid_idx, 0]
        asteroid_angle = state['asteroids']['polar_positions'][nearest_asteroid_idx, 1]

        # The thrust should be a number between -480 and 480, with negative meaning backwards.
        # The closer the asteroid is, the more quickly we should try to move!
        thrust = np.max(480 - asteroid_distance, 0)

        # In our API, "directly in front" means 0 radians, while just a smidge to the front-right is 2pi-ε radians
        # But the Kessler API wants the turn angle as degrees, in the range [-180, 180] (negative meaning turn right)
        asteroid_angle = np.degrees(asteroid_angle)

        # Change the angles to the range [-180, 180], with "directly to the right" being -90
        if asteroid_angle > 180:
            asteroid_angle -= 360

        if -90 <= asteroid_angle <= 90:
            # It is in front of us, face towards the asteroid (match its angle) and go backwards
            thrust *= -1
            turn_angle = asteroid_angle
        else:
            if asteroid_angle < 0:
                # The asteroid is behind and to the right
                turn_angle = asteroid_angle + 180
            else:
                # It is behind and to the left
                turn_angle = asteroid_angle - 180

        fire = False
        drop_mine = False

        return thrust, turn_angle, fire, drop_mine

    @property
    def name(self) -> str:
        return "I am running away!!"


def main():
    controller = OnlyRunController()
    game = KesslerGame()
    game = game.run(scenario=Scenario(num_asteroids=10), controllers=[controller])


if __name__ == '__main__':
    main()


def main():
    print("Hello, World")


if __name__ == '__main__':
    main()
