import random
from itertools import permutations
from os import listdir
from os.path import isfile, join

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
from gymnasium.spaces.discrete import Discrete

from .render_utils import room_to_one_hot, room_to_tiny_world_rgb


def generate_room_from_ascii(ascii_map):
    room_fixed = []
    room_state = []

    targets = []
    boxes = []
    for row in ascii_map:
        room_f = []
        room_s = []

        for e in row:
            if e == "#":
                room_f.append("#")
                room_s.append(" ")

            elif e == "@":
                room_f.append(" ")
                room_s.append("@")

            elif e == "A":
                boxes.append((len(room_fixed), len(room_f)))
                room_f.append(" ")
                room_s.append("A")
            elif e == "a":
                targets.append((len(room_fixed), len(room_f)))
                room_f.append("a")
                room_s.append(" ")

            elif e == "B":
                boxes.append((len(room_fixed), len(room_f)))
                room_f.append(" ")
                room_s.append("B")
            elif e == "b":
                targets.append((len(room_fixed), len(room_f)))
                room_f.append("b")
                room_s.append(" ")

            elif e == "C":
                boxes.append((len(room_fixed), len(room_f)))
                room_f.append(" ")
                room_s.append("C")
            elif e == "c":
                targets.append((len(room_fixed), len(room_f)))
                room_f.append("c")
                room_s.append(" ")

            elif e == " ":
                room_f.append(" ")
                room_s.append(" ")

            else:
                raise Exception("Unknown map element: {}".format(e))

        room_fixed.append(room_f)
        room_state.append(room_s)

    # used for replay in room generation, unused here because pre-generated levels
    room_fixed = np.array(room_fixed, dtype="U1")
    room_state = np.array(room_state, dtype="U1")
    return room_fixed, room_state


class MapSelector:
    def __init__(self, custom_maps, curriculum_cutoff=1, hardcode_level=None):
        self.train_data_dir = custom_maps
        self.curriculum_scores = [10] * curriculum_cutoff
        # TODO rather than hardcoding 10, better to use the max_episode_steps from gym
        self.hardcode_level = hardcode_level

        generated_files = [f for f in listdir(self.train_data_dir) if isfile(join(self.train_data_dir, f))]
        source_file = join(self.train_data_dir, random.choice(generated_files))

        ascii_maps = []
        current_map = []
        with open(source_file, "r") as sf:
            for line in sf.readlines():
                if "#" == line[0]:
                    current_map.append(line.strip())
                else:
                    if current_map:
                        ascii_maps.append(current_map)
                        current_map = []
        if current_map:
            ascii_maps.append(current_map)

        # self.maps = [generate_room_from_ascii(am) for am in ascii_maps]
        self.maps = []
        for am in ascii_maps:
            original_room_fixed, original_room_state = generate_room_from_ascii(am)

            for flip in [True, False]:
                for rot in [0, 1, 2, 3]:
                    for permutation in permutations(["a", "b", "c"]):
                        room_fixed = original_room_fixed.copy()
                        room_state = original_room_state.copy()

                        if flip:
                            room_fixed = np.fliplr(room_fixed)
                            room_state = np.fliplr(room_state)

                        room_fixed = np.rot90(room_fixed, k=rot)
                        room_state = np.rot90(room_state, k=rot)

                        # randomly permute colors
                        old_room_fixed = room_fixed.copy()
                        old_room_state = room_state.copy()
                        for from_color, to_color in zip(["a", "b", "c"], permutation):
                            room_fixed[old_room_fixed == from_color] = to_color
                            room_state[old_room_state == from_color.upper()] = to_color.upper()

                        self.maps.append((room_fixed, room_state))

    def select_room(self):
        if self.hardcode_level is not None:
            map_index = self.hardcode_level
        else:
            # clip curriculum scores to the number of maps persisently
            if len(self.curriculum_scores) > len(self.maps):
                self.curriculum_scores = self.curriculum_scores[: len(self.maps)]

            # with probability 0.1 choose random map
            if np.random.rand() < 0.1:
                map_index = np.random.choice(len(self.curriculum_scores))
            else:
                # with probability 0.9 choose some challenging map
                for _ in range(40):
                    # choose map randomly
                    map_index = np.random.choice(len(self.curriculum_scores))
                    score = self.curriculum_scores[map_index]
                    # choose that map only if it's challenging
                    if score > 4:
                        break

        # print(map_index)
        room_fixed, room_state = self.maps[map_index]

        return room_fixed.copy(), room_state.copy(), map_index


class SokobanEnvUncertain(gym.Env):
    def __init__(self, map_selector, dim_room=(7, 7)):
        # General Configuration
        self.dim_room = dim_room
        self.map_selector = map_selector
        self.metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

        self.map_index = None

        # Penalties and Rewards
        self.penalty_for_step = -0.3
        self.penalty_box_off_target = -1
        self.reward_box_on_target = 1
        self.reward_finished = 1
        self.reward_last = 0

        # Other Settings
        self.action_space = Discrete(len(ACTION_LOOKUP))
        self.observation_space = Box(low=0, high=255, shape=(dim_room[0] - 2, dim_room[1] - 2, 10), dtype=np.uint8)

    def select_room(self):
        room_fixed, room_state, self.map_index = self.map_selector.select_room()

        # check at which index in room_state 5 (player) is
        self.player_position = np.argwhere(room_state == "@")[0]

        self.room_fixed = room_fixed
        self.room_state = room_state
        self.box_mapping = {}
        assert self.room_fixed.shape == self.dim_room
        assert self.room_state.shape == self.dim_room

    def reset(self, seed=None, options={}, second_player=False):
        # first, try to save the score of the previous map
        if self.map_index is not None:
            self.map_selector.curriculum_scores[self.map_index] = self.num_env_steps

        if seed is not None:
            np.random.seed(seed)
        self.select_room()

        self.num_env_steps = 0
        self.reward_last = 0
        self.boxes_on_target = 0

        starting_observation = self.get_observation()

        return starting_observation, {}

    def step(self, action, observation_mode="rgb_array"):
        assert action in ACTION_LOOKUP
        assert observation_mode in ["rgb_array", "tiny_rgb_array", "raw"]

        self.num_env_steps += 1

        self.new_box_position = None
        self.old_box_position = None

        moved_box = False

        if action == 0:
            moved_player = False
        # All push actions are in the range of [0, 3]
        elif action < 5:
            moved_player, moved_box = self._push(action)
        else:
            moved_player = self._move(action)

        self._calc_reward()

        observation = self.get_observation()

        info = {
            "action.name": ACTION_LOOKUP[action],
            "action.moved_player": moved_player,
            "action.moved_box": moved_box,
        }

        done = self._check_if_all_boxes_on_target()
        if done:
            info["all_boxes_on_target"] = True

        return observation, self.reward_last, done, False, info

    def _push(self, action):
        """
        Perform a push, if a box is adjacent in the right direction.
        If no box, can be pushed, try to move.
        :param action:
        :return: Boolean, indicating a change of the room's state
        """
        change = CHANGE_COORDINATES[(action - 1) % 4]
        new_position = self.player_position + change
        current_position = self.player_position.copy()

        # No push, if the push would get the box out of the room's grid
        # this section if probably unneeded bc we have the boundary walls
        new_box_position = new_position + change
        if new_box_position[0] >= self.room_state.shape[0] or new_box_position[1] >= self.room_state.shape[1]:
            return False, False

        can_push_box = self.room_state[*new_position] in ["A", "B", "C"]
        can_push_box &= self.room_state[*new_box_position] == " "
        can_push_box &= self.room_fixed[*new_box_position] != "#"
        if can_push_box:
            box_type = self.room_state[*new_position]

            self.new_box_position = tuple(new_box_position)
            self.old_box_position = tuple(new_position)

            # Move Player
            self.player_position = new_position
            self.room_state[*new_position] = "@"
            self.room_state[*current_position] = " "

            # Move Box
            self.room_state[*new_box_position] = box_type
            return True, True

        # Try to move if no box to push, available
        else:
            return self._move(action), False

    def _move(self, action):
        """
        Moves the player to the next field, if it is not occupied.
        :param action:
        :return: Boolean, indicating a change of the room's state
        """
        change = CHANGE_COORDINATES[(action - 1) % 4]
        new_position = self.player_position + change
        current_position = self.player_position.copy()

        # Move player if the field in the moving direction is either
        # an empty field or an empty box target.
        if self.room_state[*new_position] == " " and self.room_fixed[*new_position] != "#":
            self.player_position = new_position
            self.room_state[*new_position] = "@"
            self.room_state[*current_position] = " "

            return True

        return False

    def _calc_reward(self):
        """
        Calculate Reward Based on
        :return:
        """
        # Every step a small penalty is given, This ensures
        # that short solutions have a higher reward.
        self.reward_last = self.penalty_for_step

        # count boxes off or on the target
        current_boxes_on_target = 0
        for box_type in ["A", "B", "C"]:
            boxes = box_type == self.room_state
            targets = box_type.lower() == self.room_fixed
            homed_boxes = boxes & targets
            current_boxes_on_target += np.sum(homed_boxes)

        # Add the reward if a box is pushed on the target and give a
        # penalty if a box is pushed off the target.
        if current_boxes_on_target > self.boxes_on_target:
            self.reward_last += self.reward_box_on_target
        elif current_boxes_on_target < self.boxes_on_target:
            self.reward_last += self.penalty_box_off_target

        game_won = self._check_if_all_boxes_on_target()
        if game_won:
            self.reward_last += self.reward_finished

        self.boxes_on_target = current_boxes_on_target

    def _check_if_all_boxes_on_target(self):
        for box_type in ["A", "B", "C"]:
            boxes = box_type == self.room_state
            targets = box_type.lower() == self.room_fixed
            unhomed_boxes = boxes & (~targets)
            if np.any(unhomed_boxes):
                return False
        return True

    def render(self):
        img = room_to_tiny_world_rgb(self.room_state, self.room_fixed)
        # cut off the borders
        img = img[1:-1, 1:-1, :]
        return img

    def get_observation(self):
        obs = room_to_one_hot(self.room_state, self.room_fixed)
        obs = obs[1:-1, 1:-1, :]
        return obs

    # def get_action_lookup(self):
    #     return ACTION_LOOKUP

    # def get_action_meanings(self):
    #     return ACTION_LOOKUP


ACTION_LOOKUP = {
    0: "no operation",
    1: "push up",
    2: "push down",
    3: "push left",
    4: "push right",
    # 5: 'move up',
    # 6: 'move down',
    # 7: 'move left',
    # 8: 'move right',
}

# Moves are mapped to coordinate changes as follows
# 0: Move up
# 1: Move down
# 2: Move left
# 3: Move right
CHANGE_COORDINATES = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

RENDERING_MODES = ["rgb_array", "human", "tiny_rgb_array", "tiny_human", "raw"]

