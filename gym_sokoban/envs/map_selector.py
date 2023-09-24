from itertools import permutations
from os.path import isfile

import numpy as np


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


def get_all_transformations(original_room_fixed, original_room_state):
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

                yield room_fixed, room_state


class MapSelector:
    def __init__(
        self,
        custom_maps,
        curriculum_cutoff=None,
        hardcode_level=None,
        challenging_threshold=9,
        p_random_map=0.2,
    ):
        self.train_data_dir = custom_maps
        self.hardcode_level = hardcode_level
        self.challenging_threshold = challenging_threshold
        self.p_random_map = p_random_map

        ascii_maps = []
        for source_file in custom_maps:
            ascii_maps_in_file = []
            if not isfile(source_file):
                raise Exception("File not found: {}".format(source_file))
            current_map = []
            with open(source_file, "r") as sf:
                for line in sf.readlines():
                    if "#" == line[0]:
                        current_map.append(line.strip())
                    else:
                        if current_map:
                            ascii_maps_in_file.append(current_map)
                            current_map = []
            if current_map:
                ascii_maps_in_file.append(current_map)

            print(f"Loaded {len(ascii_maps_in_file)} maps from {source_file}")
            ascii_maps.extend(ascii_maps_in_file)

        self.maps = []
        for ascii_map in ascii_maps:
            # for each ascii map, generate 2*48 variants
            # there are 48 possible transformations of the map
            # and in each of them we either make it fully certain or not
            for certainty in [True, False]:
                room_fixed, room_state = generate_room_from_ascii(ascii_map)
                for var_fixed, var_state in get_all_transformations(room_fixed, room_state):
                    self.maps.append((var_fixed, var_state, {"certainty": certainty}))

        self.curriculum_scores = [10] * len(self.maps)
        if curriculum_cutoff is None:
            self.curriculum_cutoff = len(self.maps)
        else:
            self.curriculum_cutoff = curriculum_cutoff

    def select_room(self):
        if self.hardcode_level is not None:
            map_index = self.hardcode_level
        else:
            # with some probability choose random map
            if np.random.rand() < self.p_random_map:
                map_index = np.random.choice(self.curriculum_cutoff)
            else:
                # otherwise choose challenging map
                for _ in range(20):
                    # choose map randomly
                    map_index = np.random.choice(self.curriculum_cutoff)
                    score = self.curriculum_scores[map_index]
                    # choose that map only if it's challenging
                    if score > self.challenging_threshold:
                        break

        # print(map_index)
        room_fixed, room_state, params = self.maps[map_index]

        return room_fixed.copy(), room_state.copy(), params.copy(), map_index

    def grow_curriculum(self, n):
        self.curriculum_cutoff = min(len(self.maps), self.curriculum_cutoff + n)
