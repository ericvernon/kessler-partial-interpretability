import numpy as np
import gymnasium as gym
from gymnasium import spaces
from kesslergame import TrainerEnvironment, KesslerController
from typing import Dict, Tuple
from collections import deque
from src.lib import center_coords
from src.radar import get_radar

THRUST_SCALE, TURN_SCALE = 480.0, 180.0
SHIP_MAX_SPEED = 240
DEFAULT_RADAR_ZONES = [100, 250, 400]
DEFAULT_FORECAST_FRAMES = 30


class RadarEnv(gym.Env):
    def __init__(self, scenario, radar_zones=None,
                 forecast_frames=DEFAULT_FORECAST_FRAMES):
        if radar_zones is None:
            self.radar_zones = DEFAULT_RADAR_ZONES
        else:
            self.radar_zones = radar_zones
        self.forecast_frames = forecast_frames

        self.controller = DummyController()
        self.kessler_game = TrainerEnvironment()
        self.scenario = scenario
        self.game_generator = self.kessler_game.run_step(scenario=self.scenario, controllers=[self.controller])

        self.observation_space = spaces.Dict(
            {
                # Radar: Density of asteroids in each zone
                "radar": spaces.Box(low=0, high=1, shape=(3, 4)),
                "forecast": spaces.Box(low=0, high=1, shape=(3, 4)),
            }
        )
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.game_generator = self.kessler_game.run_step(scenario=self.scenario, controllers=[self.controller])
        score, perf_list, game_state = next(self.game_generator)
        obs = get_obs(game_state, forecast_frames=self.forecast_frames, radar_zones=self.radar_zones)
        return obs, self._get_info()

    def step(self, action):
        thrust, turn_rate, fire, drop_mine = action[0] * THRUST_SCALE, action[1] * TURN_SCALE, False, False
        self.controller.action_queue.append(tuple([thrust, turn_rate, fire, drop_mine]))
        try:
            score, perf_list, game_state = next(self.game_generator)
            terminated = False
        except StopIteration as exp:
            score, perf_list, game_state = list(exp.args[0])
            terminated = True
        obs = get_obs(game_state, forecast_frames=self.forecast_frames, radar_zones=self.radar_zones)
        reward = get_reward(game_state)
        return obs, reward, terminated, False, self._get_info()

    def _get_info(self):
        return {}


def get_obs(game_state, forecast_frames, radar_zones):
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

    ship_future_position = ship_position + (forecast_frames * ship_velocity)
    asteroid_future_positions = asteroid_positions + (forecast_frames * asteroid_velocity)

    centered_asteroids = center_coords(ship_position, ship_heading, asteroid_positions, map_size)
    centered_future_asteroids = center_coords(ship_future_position, ship_heading, asteroid_future_positions, map_size)

    radar = get_radar(centered_asteroids, asteroid_radii, radar_zones)
    forecast = get_radar(centered_future_asteroids, asteroid_radii, radar_zones)

    obs = {
        "radar": radar,
        "forecast": forecast,
    }

    return obs


def get_reward(game_state):
    # It seems best if the majority of the reward comes from simply staying alive,
    # and let reinforcement learning figure out how best to actually do that.
    # However, we do want to "gently" guide the ship to sparse areas -- if any exist.
    ship = game_state['ships'][0]
    ship_position = np.array(ship['position'], dtype=np.float64)
    asteroids = game_state['asteroids']
    asteroid_positions = np.array([asteroid['position'] for asteroid in asteroids], dtype=np.float64)
    dist = np.min(np.linalg.norm(asteroid_positions - ship_position, axis=1))
    return np.sqrt(dist)


class DummyController(KesslerController):
    def __init__(self):
        super(DummyController, self).__init__()
        self.action_queue = deque()

    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool, bool]:
        return self.action_queue.popleft()

    def name(self) -> str:
        return "Hello Mr"