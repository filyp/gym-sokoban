from .sokoban_env import SokobanEnv
from .render_utils import room_to_rgb, room_to_tiny_world_rgb
import os
from os import listdir
from os.path import isfile, join
import requests
import zipfile
from tqdm import tqdm
import random
import numpy as np

class BoxobanEnv(SokobanEnv):
    def __init__(self, difficulty='unfiltered', split='train', custom_maps=None, curriculum_cutoff=1, **kwargs):
        self.difficulty = difficulty
        self.split = split
        self.custom_maps = custom_maps
        self.curriculum_cutoff = curriculum_cutoff
        self.steps_per_level = dict()

        self.verbose = False
        super(BoxobanEnv, self).__init__(
            dim_room=(10, 10),
            num_boxes=4,
            **kwargs,
        )
        
    def reset(self):
        if self.custom_maps is None:
            # use DeepMind's pre-generated levels
            self.cache_path = '.sokoban_cache'
            self.train_data_dir = os.path.join(self.cache_path, 'boxoban-levels-master', self.difficulty, self.split)

            if not os.path.exists(self.cache_path):
               
                url = "https://github.com/deepmind/boxoban-levels/archive/master.zip"
                
                if self.verbose:
                    print('Boxoban: Pregenerated levels not downloaded.')
                    print('Starting download from "{}"'.format(url))

                response = requests.get(url, stream=True)

                if response.status_code != 200:
                    raise "Could not download levels from {}. If this problem occurs consistantly please report the bug under https://github.com/mpSchrader/gym-sokoban/issues. ".format(url)

                os.makedirs(self.cache_path)
                path_to_zip_file = os.path.join(self.cache_path, 'boxoban_levels-master.zip')
                with open(path_to_zip_file, 'wb') as handle:
                    for data in tqdm(response.iter_content()):
                        handle.write(data)

                zip_ref = zipfile.ZipFile(path_to_zip_file, 'r')
                zip_ref.extractall(self.cache_path)
                zip_ref.close()
        else:
            # don't download DeepMind's levels, use custom levels
            self.train_data_dir = self.custom_maps
        
        self.select_room()

        self.num_env_steps = 0
        self.reward_last = 0
        self.boxes_on_target = 0

        if self.use_tiny_world:
            starting_observation = room_to_tiny_world_rgb(self.room_state, self.room_fixed)
        else:
            starting_observation = room_to_rgb(self.room_state, self.room_fixed)

        return starting_observation, {}

    def select_room(self):
        
        generated_files = [f for f in listdir(self.train_data_dir) if isfile(join(self.train_data_dir, f))]
        source_file = join(self.train_data_dir, random.choice(generated_files))

        maps = []
        current_map = []
        
        with open(source_file, 'r') as sf:
            for line in sf.readlines():
                if ';' in line and current_map:
                    maps.append(current_map)
                    current_map = []
                if '#' == line[0]:
                    current_map.append(line.strip())
        
        maps.append(current_map)

        # use triangular distribution, to spend more time on challenging levels
        map_index = int(np.random.triangular(0, self.curriculum_cutoff, self.curriculum_cutoff))
        if map_index >= len(maps):
            map_index = np.random.choice(len(maps))
            print('Warning: map_index out of bounds, using random map instead.')
        self.map_index = map_index
        selected_map = maps[map_index]

        if self.verbose:
            print('Selected Level from File "{}"'.format(source_file))

        self.room_fixed, self.room_state, self.box_mapping = self.generate_room(selected_map)
        assert self.room_fixed.shape == self.dim_room

    def generate_room(self, select_map):
        room_fixed = []
        room_state = []

        targets = []
        boxes = []
        for row in select_map:
            room_f = []
            room_s = []

            for e in row:
                if e == '#':
                    room_f.append(0)
                    room_s.append(0)

                elif e == '@':
                    room_f.append(1)
                    room_s.append(5)

                elif e == '$':
                    boxes.append((len(room_fixed), len(room_f)))
                    room_f.append(1)
                    room_s.append(4)

                elif e == '.':
                    targets.append((len(room_fixed), len(room_f)))
                    room_f.append(2)
                    room_s.append(2)

                else:
                    room_f.append(1)
                    room_s.append(1)

            room_fixed.append(room_f)
            room_state.append(room_s)

        # used for replay in room generation, unused here because pre-generated levels
        box_mapping = {}
        room_fixed = np.array(room_fixed)
        room_state = np.array(room_state)

        # random flip
        if np.random.choice([True, False]):
            room_fixed = np.fliplr(room_fixed)
            room_state = np.fliplr(room_state)
        # random rotation
        _rot = np.random.choice([0,1,2,3])
        room_fixed = np.rot90(room_fixed, k=_rot)
        room_state = np.rot90(room_state, k=_rot)

        # check at which index in room_state 5 (player) is
        self.player_position = np.argwhere(room_state == 5)[0]
        return room_fixed, room_state, box_mapping
    
    def when_done_callback(self):
        if self.map_index not in self.steps_per_level:
            self.steps_per_level[self.map_index] = []
        self.steps_per_level[self.map_index].append(self.num_env_steps)
        



