"""RoboTwin clean50 data configuration and dataset mixture."""

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import (
    StateActionToTensor,
    StateActionTransform,
)


class RoboTwinClean50DataConfig:
    """Three-camera, bimanual RoboTwin data with a 50-step action horizon."""

    video_keys = ["video.cam_high", "video.cam_left_wrist", "video.cam_right_wrist"]
    state_keys = [
        "state.left_joints",
        "state.right_joints",
        "state.left_gripper",
        "state.right_gripper",
    ]
    action_keys = [
        "action.left_joints",
        "action.right_joints",
        "action.left_gripper",
        "action.right_gripper",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(50))

    def modality_config(self):
        return {
            "video": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.video_keys,
            ),
            "state": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.state_keys,
            ),
            "action": ModalityConfig(
                delta_indices=self.action_indices,
                modality_keys=self.action_keys,
            ),
            "language": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.language_keys,
            ),
        }

    def transform(self):
        state_modes = {
            "state.left_joints": "min_max",
            "state.right_joints": "min_max",
            "state.left_gripper": "binary",
            "state.right_gripper": "binary",
        }
        action_modes = {
            "action.left_joints": "min_max",
            "action.right_joints": "min_max",
            "action.left_gripper": "binary",
            "action.right_gripper": "binary",
        }
        return ComposedModalityTransform(
            transforms=[
                StateActionToTensor(apply_to=self.state_keys),
                StateActionTransform(
                    apply_to=self.state_keys,
                    binary_threshold=0.49,
                    normalization_modes=state_modes,
                ),
                StateActionToTensor(apply_to=self.action_keys),
                StateActionTransform(
                    apply_to=self.action_keys,
                    binary_threshold=0.49,
                    normalization_modes=action_modes,
                ),
            ]
        )


class RoboTwinClean50DepthDataConfig(RoboTwinClean50DataConfig):
    """在原 clean50 样本上增加与三路 RGB 相机一一对应的真实深度。"""

    depth_keys = ["depth.cam_high", "depth.cam_left_wrist", "depth.cam_right_wrist"]

    def modality_config(self):
        config = super().modality_config()
        # 深度只取当前帧，动作标签仍保持原来的 50 步 chunk。
        config["depth"] = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.depth_keys,
        )
        return config


ROBOT_TYPE_CONFIG_MAP = {
    "robotwin50": RoboTwinClean50DataConfig(),
    "robotwin50_depth": RoboTwinClean50DepthDataConfig(),
}
ROBOT_TYPE_TO_EMBODIMENT_TAG = {}

_CLEAN50_TASKS = (
    "adjust_bottle",
    "beat_block_hammer",
    "blocks_ranking_rgb",
    "blocks_ranking_size",
    "click_alarmclock",
    "click_bell",
    "dump_bin_bigbin",
    "grab_roller",
    "handover_block",
    "handover_mic",
    "hanging_mug",
    "lift_pot",
    "move_can_pot",
    "move_pillbottle_pad",
    "move_playingcard_away",
    "move_stapler_pad",
    "open_laptop",
    "open_microwave",
    "pick_diverse_bottles",
    "pick_dual_bottles",
    "place_a2b_left",
    "place_a2b_right",
    "place_bread_basket",
    "place_bread_skillet",
    "place_burger_fries",
    "place_can_basket",
    "place_cans_plasticbox",
    "place_container_plate",
    "place_dual_shoes",
    "place_empty_cup",
    "place_fan",
    "place_mouse_pad",
    "place_object_basket",
    "place_object_scale",
    "place_object_stand",
    "place_phone_stand",
    "place_shoe",
    "press_stapler",
    "put_bottles_dustbin",
    "put_object_cabinet",
    "rotate_qrcode",
    "scan_object",
    "shake_bottle_horizontally",
    "shake_bottle",
    "stack_blocks_three",
    "stack_blocks_two",
    "stack_bowls_three",
    "stack_bowls_two",
    "stamp_seal",
    "turn_switch",
)

_RGBD10_TASKS = (
    "click_alarmclock",
    "click_bell",
    "move_pillbottle_pad",
    "place_phone_stand",
    "stack_blocks_three",
    "stamp_seal",
    "turn_switch",
    "open_microwave",
    "adjust_bottle",
    "blocks_ranking_rgb",
)

_RGBD_ARCH18_200_TASKS = _RGBD10_TASKS + (
    "press_stapler",
    "beat_block_hammer",
    "hanging_mug",
    "stack_blocks_two",
    "place_object_scale",
    "place_can_basket",
    "open_laptop",
    "lift_pot",
)

DATASET_NAMED_MIXTURES = {
    "robotwin_clean_50": [
        (f"Clean/{task_name}", 1.0, "robotwin50")
        for task_name in _CLEAN50_TASKS
    ],
    # 使用同一任务目录，但要求 parquet 中包含 observation.depths.* 三个无损深度字段。
    "robotwin_clean_50_depth": [
        (f"Clean/{task_name}", 1.0, "robotwin50_depth")
        for task_name in _CLEAN50_TASKS
    ],
    "robotwin_rgbd_10": [
        (f"Clean/{task_name}", 1.0, "robotwin50_depth")
        for task_name in _RGBD10_TASKS
    ],
    "robotwin_rgbd_arch18_200": [
        (f"Clean/{task_name}", 1.0, "robotwin50_depth")
        for task_name in _RGBD_ARCH18_200_TASKS
    ],
    # The depth-aware policy predicts geometry from cam_head RGB. Ground-truth
    # depth remains exclusive to the standalone stage-one supervision dataset.
    "robotwin_depthaware_arch18_200": [
        (f"Clean/{task_name}", 1.0, "robotwin50")
        for task_name in _RGBD_ARCH18_200_TASKS
    ],
    "robotwin_depthaware_clean50_360": [
        (f"Clean/{task_name}", 1.0, "robotwin50")
        for task_name in _CLEAN50_TASKS
    ],
}
