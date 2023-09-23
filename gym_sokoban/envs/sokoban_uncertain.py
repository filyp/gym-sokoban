import os
import random
import zipfile
from os import listdir
from os.path import isfile, join

import numpy as np
import requests
from tqdm import tqdm

from .render_utils import room_to_rgb, room_to_tiny_world_rgb
from .sokoban_env import SokobanEnv


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
        
        self.maps = [generate_room_from_ascii(am) for am in ascii_maps]


    def select_room(self):
        if self.hardcode_level is not None:
            map_index = self.hardcode_level
        else:
            # clip curriculum scores to the number of maps persisently
            if len(self.curriculum_scores) > len(self.maps):
                self.curriculum_scores = self.curriculum_scores[: len(self.maps)]

            # # choose map with probability proportional to the score**2
            # _sc = (np.array(self.curriculum_scores)) ** 1
            # probs = _sc / np.sum(_sc)
            # map_index = np.random.choice(
            #     len(self.curriculum_scores),
            #     p=probs,
            # )
            
            # choose map randomly
            map_index = np.random.choice(len(self.curriculum_scores))

        # # with probability epsilon choose a random map
        # # otherwise choose among the unsolved maps
        # epsilon = 0.1
        # if np.random.rand() < epsilon:
        #     map_index = np.random.choice(len(self.curriculum_scores))
        # else:
        #     is_not_solved = np.array(self.curriculum_scores) == self.max_episode_steps
        #     if np.sum(is_not_solved) == 0:
        #         map_index = np.random.choice(len(self.curriculum_scores))
        #     else:
        #         # choose among the unsolved maps
        #         maps_to_choose_from = np.arange(len(self.curriculum_scores))[is_not_solved]
        #         map_index = np.random.choice(maps_to_choose_from)

        # print(map_index)
        room_fixed, room_state = self.maps[map_index]

        return room_fixed.copy(), room_state.copy(), map_index


class SokobanUncertainEnv(SokobanEnv):
    def __init__(self, map_selector, **kwargs):
        # self.max_episode_steps = 10  # TODO should be fetched from gym

        self.map_selector = map_selector

        self.verbose = False
        super(SokobanUncertainEnv, self).__init__(
            **kwargs,
        )

    def reset(self, seed=None, options={}):
        if seed is not None:
            np.random.seed(seed)
        self.select_room()

        self.num_env_steps = 0
        self.reward_last = 0
        self.boxes_on_target = 0

        starting_observation = self.get_image()

        return starting_observation, {}

    def select_room(self):
        room_fixed, room_state, self.map_index = self.map_selector.select_room()

        # random flip
        if np.random.choice([True, False]):
            room_fixed = np.fliplr(room_fixed)
            room_state = np.fliplr(room_state)
        # random rotation
        _rot = np.random.choice([0, 1, 2, 3])
        room_fixed = np.rot90(room_fixed, k=_rot)
        room_state = np.rot90(room_state, k=_rot)
        # randomly permute colors
        old_room_fixed = room_fixed.copy()
        old_room_state = room_state.copy()
        permutation = np.random.permutation(["a", "b", "c"])
        for from_color, to_color in zip(["a", "b", "c"], permutation):
            room_fixed[old_room_fixed == from_color] = to_color
            room_state[old_room_state == from_color.upper()] = to_color.upper()

        # check at which index in room_state 5 (player) is
        self.player_position = np.argwhere(room_state == "@")[0]

        self.room_fixed = room_fixed
        self.room_state = room_state
        self.box_mapping = {}
        assert self.room_fixed.shape == self.dim_room
        assert self.room_state.shape == self.dim_room

    def when_done_callback(self):
        self.map_selector.curriculum_scores[self.map_index] = self.num_env_steps
