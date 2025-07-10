import gymnasium as gym
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from src.envs import RadarEnv
from kesslergame import KesslerGame, Scenario, TrainerEnvironment, KesslerController
from typing import Dict, Tuple
import numpy as np

from src.lib import center_coords
from src.radar import get_radar

THRUST_SCALE, TURN_SCALE = 480.0, 180.0


def train():
    scenario = Scenario(num_asteroids=10, map_size=(600, 600), ship_states=[
        {
            'position': (100, 100),
        }
    ])
    kessler_env = Monitor(RadarEnv(scenario))
    model = PPO("MultiInputPolicy", kessler_env)
    mean_reward, _ = evaluate_policy(model, kessler_env, n_eval_episodes=10)
    print(f'        Mean reward: {mean_reward:.2f}')

    model.learn(5000)
    mean_reward, _ = evaluate_policy(model, kessler_env, n_eval_episodes=10)
    print(f'+5000   Mean reward: {mean_reward:.2f}')
    model.save("out/5k")

    model.learn(50000)
    mean_reward, _ = evaluate_policy(model, kessler_env, n_eval_episodes=10)
    print(f'+50000  Mean reward: {mean_reward:.2f}')
    model.save("out/50k")

    model.learn(100_000)
    mean_reward, _ = evaluate_policy(model, kessler_env, n_eval_episodes=10)
    print(f'+100000  Mean reward: {mean_reward:.2f}')
    model.save("out/100k")


    model.learn(500000)
    mean_reward, _ = evaluate_policy(model, kessler_env, n_eval_episodes=10)
    print(f'+500000 Mean reward: {mean_reward:.2f}')
    model.save("out/500k")


def run():
    kessler_game = KesslerGame()
    scenario = Scenario(num_asteroids=4, time_limit=180, map_size=(600, 600))
    controller = SuperDummyController()
    score, perf_list, state = kessler_game.run(scenario=scenario, controllers=[controller])
    # print(score)


class SuperDummyController(KesslerController):
    def __init__(self):
        self.model = PPO.load("out/100k")

    @property
    def name(self) -> str:
        return "Super Dummy"

    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool, bool]:
        obs = self._get_obs(game_state)
        action = self.model.predict(obs)
        thrust, turn = list(action[0])
#        print(action[0])
        return thrust * THRUST_SCALE, turn * TURN_SCALE, False, False

    def _get_obs(self, game_state):
        # For now, we are assuming only one ship (ours)
        ship = game_state['ships'][0]
        ship_position = np.array(ship['position'], dtype=np.float64)
        ship_heading = np.radians(ship['heading'])
        ship_velocity = np.array(ship['velocity'], dtype=np.float64)
        ship_speed = np.array([ship['speed']])

        asteroids = game_state['asteroids']
        asteroid_positions = np.array([asteroid['position'] for asteroid in asteroids], dtype=np.float64)
        asteroid_velocity = np.array([asteroid['velocity'] for asteroid in asteroids], dtype=np.float64)
        asteroid_radii = np.array([asteroid['radius'] for asteroid in asteroids])
        map_size = np.array(game_state['map_size'])

        ship_future_position = ship_position + (30 * ship_velocity)
        asteroid_future_positions = asteroid_positions + (30 * asteroid_velocity)

        centered_asteroids = center_coords(ship_position, ship_heading, asteroid_positions, map_size)
        centered_future_asteroids = center_coords(ship_future_position, ship_heading, asteroid_future_positions,
                                                  map_size)

        radar = get_radar(centered_asteroids, asteroid_radii, [100, 250, 400])
        forecast = get_radar(centered_future_asteroids, asteroid_radii, [100, 250, 400])

        obs = {
            "radar": radar,
            "forecast": forecast,
        }
        return obs


if __name__ == '__main__':
    #train()
    run()
