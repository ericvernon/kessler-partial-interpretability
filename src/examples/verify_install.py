from kesslergame import KesslerController, KesslerGame, Scenario
from typing import Dict, Tuple


# See: https://github.com/ThalesGroup/kessler-game/blob/main/examples/test_controller.py
class TestController(KesslerController):
    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool, bool]:
        thrust = 50
        turn_rate = -90
        fire = True
        drop_mine = False

        return thrust, turn_rate, fire, drop_mine

    @property
    def name(self) -> str:
        return "Test Controller"


def main():
    controller = TestController()
    game = KesslerGame()
    game = game.run(scenario=Scenario(num_asteroids=1), controllers=[controller])


if __name__ == '__main__':
    main()


def main():
    print("Hello, World")


if __name__ == '__main__':
    main()
