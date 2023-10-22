# ----------------------------------------------------------------------
# HAP: Structure-Aware Masked Image Modeling for Human-Centric Perception
# Written by Junkun Yuan (yuanjk0921@outlook.com)
# ----------------------------------------------------------------------
# Human body part-guided masking for pre-training
# ----------------------------------------------------------------------

import random


def get_patch(patch_choose, x_min, x_max, y_min, y_max, grid_size):
    """Return index of patches within a bounding-box."""
    for x_step_walk in range(x_min, x_max + 1):
        for y_step_walk in range(y_min, y_max + 1):
            if 0 <= x_step_walk <= grid_size[1] - 1 and 0 <= y_step_walk <= grid_size[0] - 1:
                patch_choose.add(int(y_step_walk * grid_size[1] + x_step_walk))
    return patch_choose


def mask_body_parts(noise, batch, patch_size, grid_size, keypoints_all, num_parts=6):
    """Mask parts of body."""
    body_part = [
        [[0, 1], [0, 2], [1, 2], [1, 3], [2, 4]],  # face
        [[5, 7], [7, 9]],  # left arm
        [[6, 8], [8, 10]],  # right arm
        [[11, 13], [13, 15]],  # left leg
        [[12, 14], [14, 16]],  # right leg
        [[6, 11], [5, 12]]  # upper body
        ]
    
    body_list = list(range(len(body_part)))

    for bs_idx in range(batch):
        random.shuffle(body_list)  # shuffle for each image

        num_parts_used = 1
        current_prior = 100  # mask priority
        num_parts = random.randint(0, num_parts)  # randomly choose number of parts to mask 

        for part_idx in body_list:  # choose body parts
            if num_parts_used > num_parts:
                break

            patch_choose = set()

            for _, sk in enumerate(body_part[part_idx]):  # choose skeleton
                # Get (x, y) position
                pos1 = (int(keypoints_all[bs_idx, sk[0], 0]), int(keypoints_all[bs_idx, sk[0], 1]))
                pos2 = (int(keypoints_all[bs_idx, sk[1], 0]), int(keypoints_all[bs_idx, sk[1], 1]))
                if pos1[0] < 0 or pos1[1] < 0 or pos2[0] < 0 or pos2[1] < 0:
                        continue

                # Get (x_step, y_step) of patches
                step1 = (int(pos1[0] / patch_size), int(pos1[1] / patch_size))
                step2 = (int(pos2[0] / patch_size), int(pos2[1] / patch_size))

                # Get the step range
                x_min, x_max = min(step1[0], step2[0]), max(step1[0], step2[0])
                y_min, y_max = min(step1[1], step2[1]), max(step1[1], step2[1])

                # Choose the body part
                patch_choose = get_patch(patch_choose, x_min, x_max, y_min, y_max, grid_size)

            if patch_choose:
                patch_choose = list(patch_choose)
                random.shuffle(patch_choose)
                mask_num = int(len(patch_choose) * random.uniform(0., 0.3))
                for pc in patch_choose[:mask_num]:
                    noise[bs_idx, pc] = current_prior
                
                num_parts_used += 1
                current_prior -= 1
            
    return noise
