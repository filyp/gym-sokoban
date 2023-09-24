import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
from gymnasium.spaces.discrete import Discrete

from .render_utils import room_to_one_hot, room_to_tiny_world_rgb


class SokobanUncertainEnv(gym.Env):
    def __init__(self, map_selector, dim_room=(7, 7), num_uncertain_steps=2):
        # General Configuration
        self.map_selector = map_selector
        self.dim_room = dim_room
        self.num_uncertain_steps = num_uncertain_steps

        self.metadata = {"render_modes": ["rgb_array"], "render_fps": 4}
        self.map_index = None

        # Penalties and Rewards
        self.penalty_for_step = -0.1
        self.penalty_box_off_target = -1
        self.reward_box_on_target = 1
        self.reward_finished = 1
        self.reward_last = 0

        # Other Settings
        self.action_space = Discrete(len(ACTION_LOOKUP))
        self.observation_space = Box(
            low=0, high=255, shape=(dim_room[0] - 2, dim_room[1] - 2, 11), dtype=np.uint8
        )

    def reset(self, seed=None, options={}, second_player=False):
        # first, try to save the score of the previous map
        if self.map_index is not None:
            self.map_selector.curriculum_scores[self.map_index] = self.num_env_steps

        if seed is not None:
            np.random.seed(seed)

        # load a new map
        self.room_fixed, self.room_state, params, self.map_index = self.map_selector.select_room()
        assert self.room_fixed.shape == self.dim_room
        assert self.room_state.shape == self.dim_room
        # check at which index in room_state
        self.player_position = np.argwhere(self.room_state == "@")[0]
        assert len(np.argwhere(self.room_state == "@")) == 1

        self.num_env_steps = 0
        self.reward_last = 0
        self.boxes_on_target = 0

        # in some episodes we want complete certainty,
        # so that the agent can learn to solve colored sokoban
        self.complete_certainty = params["certainty"]

        starting_observation = self.get_observation()

        return starting_observation, {}

    def fog_of_uncertainty(self):
        if self.complete_certainty or self.num_env_steps >= self.num_uncertain_steps:
            return self.room_state
        else:
            # mask the colors of the boxes by replacing A, B, C with X
            masked_state = self.room_state.copy()
            masked_state[masked_state == "A"] = "X"
            masked_state[masked_state == "B"] = "X"
            masked_state[masked_state == "C"] = "X"
            return masked_state

    def render(self):
        perceived_state = self.fog_of_uncertainty()
        img = room_to_tiny_world_rgb(perceived_state, self.room_fixed)
        # cut off the borders
        img = img[1:-1, 1:-1, :]
        return img

    def get_observation(self):
        perceived_state = self.fog_of_uncertainty()
        obs = room_to_one_hot(perceived_state, self.room_fixed)
        obs = obs[1:-1, 1:-1, :]
        return obs

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
        if (
            new_box_position[0] >= self.room_state.shape[0]
            or new_box_position[1] >= self.room_state.shape[1]
        ):
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
        # TODO ! note that agent will receive the correct reward now, even if in principle
        # it should still be uncertain about which box goes on which target
        # in principle it could make use of that fact, if it has some access to the rewards it gets

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
