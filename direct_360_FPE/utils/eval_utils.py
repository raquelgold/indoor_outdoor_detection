import os
import glob
import json
import cv2
import numpy as np
from skimage import measure
from utils.io import read_trajectory
from utils.metric import calc_corners_pr, calc_polygon_iou
import matplotlib.pyplot as plt


def rotate_by_axis_corners(points, axis_corners):
    points = points.copy()
    x, y = axis_corners[1] - axis_corners[0]
    yaw_angle = np.arctan(y / x)
    yaw_2d = np.array([[np.cos(yaw_angle), -np.sin(yaw_angle)],
                       [np.sin(yaw_angle),
                        np.cos(yaw_angle)]])

    points[:, :2] = np.matmul(points[:, :2], yaw_2d)
    if points.shape[-1] > 6:
        # has normal
        yaw_3d = np.eye(3)
        yaw_3d[:2, :2] = yaw_2d
        points[:, 6:] = np.matmul(points[:, 6:], yaw_3d)
    return points


def _safe_div(numerator, denominator):
    """Recall/precision are undefined with no GT (or no predictions) to
    match against — report 0.0 instead of crashing (dummy-GT scenes can
    legitimately have zero rooms/corners after axis-alignment)."""
    return numerator / denominator if denominator else 0.0


def evaluate_scene(room_corners_pred, room_corners_gt, points, axis_corners=None):
    '''
    Evaluate the scene with room and corner metric
        corners_pred: list of predicted room corners
        corners_gt: list of GT room corners
        points: point cloud for background visualization
        axis_corners: the two corners define the rotation for axis-alignment
    '''
    return_dict = {}
    images_dict = {}
    for iou_threshold in [0.3, 0.5, 0.7]:
        num_match, size_pred, size_gt, image_pred, image_gt = evaluate_rooms_pr(
            room_corners_pred, room_corners_gt,
            points, axis_corners,
            grid_size=512,
            iou_threshold=iou_threshold,
            room_name_list=None,
        )
        print(f'Room@{iou_threshold} recall: {_safe_div(num_match, size_gt)}, precision: {_safe_div(num_match, size_pred)}')
        return_dict[f'room@{iou_threshold}'] = dict(
            num_match=num_match,
            size_pred=size_pred,
            size_gt=size_gt,
        )
        images_dict[f'room@{iou_threshold}'] = dict(
            image_pred=image_pred,
            image_gt=image_gt,
        )

    corners_pred = np.concatenate(room_corners_pred, axis=0)
    corners_gt = np.concatenate(room_corners_gt, axis=0)
    for do_merge in [True, False]:
        num_match, size_pred, size_gt, image_pred, image_gt = evaluate_corners_pr(
            corners_pred, corners_gt,
            points, axis_corners,
            grid_size=256,
            merge_corners=do_merge,
            merge_dist=0.5,
            dist_threshold=10,      # 10 pixels
        )
        print(f'Corner recall: {_safe_div(num_match, size_gt)}, precision: {_safe_div(num_match, size_pred)}')
        key = 'corner_merge' if do_merge else 'corner_raw'
        return_dict[key] = dict(
            num_match=num_match,
            size_pred=size_pred,
            size_gt=size_gt,
        )
        images_dict[key] = dict(
            image_pred=image_pred,
            image_gt=image_gt,
        )
    return return_dict, images_dict


def dump_images(images_dict, save_dir):
    # Dump images
    scene = images_dict['scene']
    keys = sorted(list(images_dict.keys()))
    plt.imsave(os.path.join(save_dir, f'{scene}_room_id.png'), images_dict['room_id'])
    plt.imsave(os.path.join(save_dir, f'{scene}_final_fp.png'), images_dict['final_fp'])
    keys.remove('scene')
    keys.remove('room_id')
    keys.remove('final_fp')
    for key in keys:
        plt.imsave(os.path.join(save_dir, f'{scene}_{key}_gt.png'), images_dict[key]['image_gt'])
        plt.imsave(os.path.join(save_dir, f'{scene}_{key}_pred.png'), images_dict[key]['image_pred'])


def dump_result(result_list, save_dir):
    # Dump result to a csv file
    with open(os.path.join(save_dir, '360_dfpe_result.csv'), 'w') as f:
        keys = sorted(list(result_list[0].keys()))
        keys.remove('scene')
        # Write header
        f.write('scene')
        for key in keys:
            f.write(f',{key}_recall,{key}_prec')
        f.write('\n')

        for result_dict in result_list:
            # Write each row (repective to the result of a scene)
            scene = result_dict['scene']
            f.write(f'{scene}')
            for key in keys:
                recall = result_dict[key]['num_match'] / result_dict[key]['size_gt']
                prec = result_dict[key]['num_match'] / result_dict[key]['size_pred']
                f.write(f',{recall:.4f},{prec:.4f}')
            f.write('\n')

        # Compute average result for last row
        f.write('mean')
        for key in keys:
            all_recall = []
            all_prec = []
            for result_dict in result_list:
                recall = result_dict[key]['num_match'] / result_dict[key]['size_gt']
                prec = result_dict[key]['num_match'] / result_dict[key]['size_pred']
                all_recall.append(recall)
                all_prec.append(prec)
            f.write(f',{np.mean(all_recall):.4f},{np.mean(all_prec):.4f}')
        f.write('\n')


def dump_coords_json(room_corner_list, scale_recover, scene_name, results_dir):
    """
    Saves world-space XZ coordinates for final_fp.png and scale_recovery.jpg as JSON.

    room_corner_list : list of (N, 2) arrays — estimated room corner polygons in
                       world XZ (metres), one array per room.
    scale_recover    : ScaleRecover instance (provides vo_scale, gt_scale and
                       access to dt.list_ly for wall boundary points).
    scene_name       : string used to name the output file.
    results_dir      : directory where the file is written.

    Output file: <results_dir>/<scene_name>_coords.json
    """
    # --- room corners (final_fp.png) ---
    rooms = []
    for room_id, corners in enumerate(room_corner_list):
        # corners is (N, 2): columns are [x, z] in metres
        rooms.append({
            "room_id": room_id,
            "corners_xz": [[round(float(c[0]), 6), round(float(c[1]), 6)]
                           for c in corners],
        })

    # --- wall boundaries per keyframe (raw OCG input) ---
    wall_boundaries = []
    for ly in scale_recover.dt.list_ly:
        if ly.cam2boundary.max() >= 20:
            continue
        xz = ly.boundary[[0, 2], ::8].T
        xz = np.vstack([xz, xz[0]])
        wall_boundaries.append(
            [[round(float(p[0]), 5), round(float(p[1]), 5)] for p in xz]
        )

    # --- RANSAC-fitted wall plane segments + camera trajectory, grouped per frame ---
    wall_planes_frames = []
    camera_trajectory  = []
    for ly in scale_recover.dt.list_ly:
        if not ly.is_initialized:
            continue
        # camera position for this frame
        t = ly.pose.t
        camera_trajectory.append({
            "frame_idx": int(ly.idx),
            "pos_xz": [round(float(t[0]), 5), round(float(t[2]), 5)],
        })
        # plane segments for this frame
        segs = []
        for pl in ly.list_pl:
            xz = pl.boundary[[0, 2], :]
            if xz.shape[1] < 2:
                continue
            line_dir = np.array([pl.line_dir[0], pl.line_dir[2]], dtype=float)
            norm = np.linalg.norm(line_dir)
            if norm < 1e-9:
                continue
            line_dir /= norm
            t_proj = line_dir @ xz
            p1 = xz[:, np.argmin(t_proj)]
            p2 = xz[:, np.argmax(t_proj)]
            segs.append([
                [round(float(p1[0]), 5), round(float(p1[1]), 5)],
                [round(float(p2[0]), 5), round(float(p2[1]), 5)],
            ])
        if segs:
            wall_planes_frames.append({
                "frame_idx": int(ly.idx),
                "segments": segs,
            })
    # flat list kept for static map (plot_coords_maps.py)
    wall_segments = [seg for f in wall_planes_frames for seg in f["segments"]]

    data = {
        "scene": scene_name,
        "vo_scale": round(float(scale_recover.vo_scale), 6),
        "gt_scale": round(float(scale_recover.gt_scale), 6),
        "final_fp": {
            "description": "Estimated room corner polygons — world XZ coordinates in metres",
            "num_rooms": len(rooms),
            "rooms": rooms,
        },
        "scale_recovery": {
            "description": (
                "Per-keyframe wall boundary polygons — world XZ coordinates in metres. "
                "Each entry is a closed polygon (128 pts) for one keyframe."
            ),
            "num_frames": len(wall_boundaries),
            "wall_boundaries": wall_boundaries,
        },
        "wall_planes": {
            "description": (
                "RANSAC-fitted wall plane segments — world XZ endpoints in metres. "
                "'frames' groups segments by keyframe for animation; "
                "'segments' is the flat list for static maps."
            ),
            "num_segments": len(wall_segments),
            "segments": wall_segments,
            "frames": wall_planes_frames,
        },
        "camera_trajectory": {
            "description": "Camera XZ position (metres) per initialized keyframe.",
            "positions": camera_trajectory,
        },
    }

    filename = os.path.join(results_dir, f"{scene_name}_coords.json")
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Coordinates saved → {filename}")
    return filename


class FloorNetCornerAlignment:
    '''
        Align the corners based on the FloorNet and Floor-SP style.
    '''
    def __init__(
        self,
        points,
        axis_corners=None,
        padding_ratio=0.05,
    ):
        self.axis_corners = axis_corners
        if self.axis_corners is not None:
            points = rotate_by_axis_corners(points, self.axis_corners)

        assert len(points.shape) == 2
        if points.shape[1] > 2:
            # Use only the x, y coordinate
            points = points[:, :2]

        # Map the point coordinate into 0-1
        # where the point cloud is in the middle with a padding
        self.max_edge = np.max(points.max(0) - points.min(0))
        self.padding = self.max_edge * padding_ratio
        self.mins = (points.max(0) + points.min(0)) / 2 - self.max_edge / 2
        self.mins -= self.padding
        self.max_edge += self.padding * 2

        self.aligned_points = (points - self.mins) / self.max_edge

    def align_corners(self, corners):
        if self.axis_corners is not None:
            corners = rotate_by_axis_corners(corners, self.axis_corners)
        return (corners - self.mins) / self.max_edge

    def inverse_align_corners(self, corners):
        corners = corners * self.max_edge + self.mins
        if self.axis_corners is not None:
            # Inverse rotate
            corners = corners.copy()
            x, y = self.axis_corners[1] - self.axis_corners[0]
            yaw_angle = np.arctan(y / x)
            yaw_2d = np.array([[np.cos(yaw_angle), -np.sin(yaw_angle)],
                               [np.sin(yaw_angle),
                                np.cos(yaw_angle)]])
            corners[:, :2] = np.matmul(corners[:, :2], yaw_2d.T)
        return corners


def merge_neighbor_corners(corners, dist_threshold):
    '''
        Naively merge all corners by DFS
        corners: shape (N, 2)
        dist_threshold: float

        return: merged corners with shape (M, 2)
    '''
    dist_matrix = np.expand_dims(corners, axis=1) - np.expand_dims(
        corners, axis=0)  # N, N, 2
    dist_matrix = np.sqrt(np.sum(np.power(dist_matrix, 2), axis=-1))  # N, N
    mask = dist_matrix < dist_threshold

    def dfs(i, graph, used_node):
        collected = set()
        used_node[i] = True
        collected.add(i)

        for j in range(graph.shape[1]):
            if graph[i, j] and not used_node[j]:
                collected = collected | dfs(j, graph, used_node)
        return collected

    used_node = np.zeros((corners.shape[0], ))
    merged_corners = []
    for i in range(corners.shape[0]):
        if not used_node[i]:
            corners_ids = list(dfs(i, mask, used_node))
            merged_corners.append(np.mean(corners[corners_ids, :], axis=0))

    merged_corners = np.stack(merged_corners, axis=0)
    return merged_corners


def compute_grid_map(points, height=256, width=256, scale_fractor=1):
    size = np.array([width, height])
    coords = np.clip(np.round(points[:, :2] * size), 0,
                     size - 1).astype(np.int32)
    coords = coords[:, 0] * width + coords[:, 1]

    density = np.bincount(coords, minlength=height * width)
    density = density.reshape(height, width) / scale_fractor
    return density


def draw_corners(background, corners, color, r=2):
    '''
        background: [height, width, 3]
        corners: [N, 2]
        color: (r, g, b)
        r: radius
    '''
    for i in range(corners.shape[0]):
        x = int(round(corners[i, 0]))
        y = int(round(corners[i, 1]))
        background[x-r:x+r+1, y-r:y+r+1, :] = color
    return background


def add_opacity(image):
    height, width = image.shape[:2]
    alpha = np.zeros((height, width, 1), dtype=np.uint8)
    alpha.fill(255)
    image = np.concatenate([image, alpha], axis=-1)
    return image


def composite_rgba(background, input):
    # Based on https://stackoverflow.com/questions/10781953/determine-rgba-colour-received-by-combining-two-colours
    assert input.shape[:2] == background.shape[:2]
    H, W = background.shape[:2]
    input = input.astype(np.float32) / 255
    background = background.astype(np.float32) / 255

    if input.shape[2] == 4:
        input_alpha = input[:, :, 3].reshape(H, W, 1)
    else:
        input_alpha = np.ones((H, W, 1), dtype=np.float32)

    input_rgb = input[:, :, :3]
    if background.shape[2] == 4:
        bg_alpha = background[:, :, 3].reshape(H, W, 1)
    else:
        bg_alpha = np.ones((H, W, 1), dtype=np.float32)
    bg_rgb = background[:, :, :3]

    new_alpha = input_alpha + bg_alpha * (1 - input_alpha)
    new_rgb = input_rgb * input_alpha + bg_rgb * bg_alpha * (1 - input_alpha)
    new_rgb /= new_alpha
    new_img = np.concatenate([new_rgb, new_alpha], axis=-1)

    return np.round(new_img * 255).astype(np.uint8)


def draw_room(background, corners, alignment=None, color=None, opacity=180):
    height, width = background.shape[:2]
    if background.shape[2] == 3:
        # Add alpha channel
        background = add_opacity(background)
    room_map = np.zeros((height, width, 4), dtype=np.uint8)
    if alignment is not None:
        size = np.array([height, width])
        corners = alignment.align_corners(corners) * size
    mask = measure.grid_points_in_poly((height, width), corners)
    if color is None:
        color = np.array([255, 0, 0])       # Red
    room_map[mask, :3] = color
    room_map[mask, 3] = opacity

    return composite_rgba(background, room_map)


def evaluate_rooms_pr(
    room_corners_pred,
    room_corners_gt,
    points_gt,
    axis_corners,
    grid_size=512,
    room_name_list=None,
    iou_threshold=0.5,
):
    '''
        Evaluate the precision and recall for room IoU.
        NOTE: Evaluating the room IoU does not need axis-alignment. The axis-alignment is only for visualization.
        Parameters:
            room_corners_pred: List of estimated room corners (N_i, 2)
            room_corners_gt: List of GT room corners (M_i, 2)
            points_gt: GT point cloud
            axis_corners: two corners defining an edge that need to be aligned to x-axis (2, 2)
            grid_size: The size of the 2D occ grid in pixels
            iou_threshold: the threshold for computing matching for rooms
            room_name_list: List of room name for visualization (the same size of room_corners_gt)
        Returns:
            num_match: Number of matches
            size_pred: Total number of estimated rooms
            size_gt: Total number of GT rooms
            image_pred: visualization image result
            image_gt: visualization image result
    '''
    gt_map_dict = {}
    for gt_idx in range(len(room_corners_gt)):
        # sample points in gt
        for pred_idx in range(len(room_corners_pred)):

            iou = calc_polygon_iou(
                room_corners_gt[gt_idx],
                room_corners_pred[pred_idx],
                sample_size=512
            )

            if gt_idx in gt_map_dict:
                if iou > gt_map_dict[gt_idx][1]:
                    # Replace previous match with larger iou match
                    gt_map_dict[gt_idx] = (pred_idx, iou)
            else:
                gt_map_dict[gt_idx] = (pred_idx, iou)

    # num_match = len([iou for _, iou in gt_map_dict.values() if iou >= iou_threshold])
    num_match = 0
    pred_set = set()
    for (pred_idx, iou) in gt_map_dict.values():
        if iou >= iou_threshold and pred_idx not in pred_set:
            pred_set.add(pred_idx)
            num_match += 1

    points = points_gt[:, [0, 2]]
    alignment = FloorNetCornerAlignment(points, axis_corners)
    points = alignment.align_corners(points)
    density_map = compute_grid_map(points, height=grid_size, width=grid_size, scale_fractor=0.1)
    density_map = np.stack([density_map, density_map, density_map], axis=-1)
    density_map *= 5
    density_map = np.clip(np.round(density_map), 0, 255).astype(np.uint8)
    density_map = 255 - density_map

    image_gt = density_map.copy()
    image_pred = density_map.copy()
    matched_pred_idxs = []

    # The seed here make sure the visualization color will be the deterministic
    np.random.seed(255)
    for gt_idx in range(len(room_corners_gt)):
        room_corners = room_corners_gt[gt_idx]
        sample_color = np.random.choice(range(256), size=3)
        image_gt = draw_room(
            image_gt, room_corners,
            alignment=alignment, color=sample_color
        )

        if gt_idx in gt_map_dict and gt_map_dict[gt_idx][1] >= iou_threshold:
            pred_idx = gt_map_dict[gt_idx][0]
            room_corners = room_corners_pred[pred_idx]
            image_pred = draw_room(
                image_pred, room_corners,
                alignment=alignment, color=sample_color
            )

            matched_pred_idxs.append(pred_idx)

    np.random.seed(798)
    for pred_idx in range(len(room_corners_pred)):
        if pred_idx not in matched_pred_idxs:
            room_corners = room_corners_pred[pred_idx]
            sample_color = np.random.choice(range(256), size=3)
            image_pred = draw_room(
                image_pred, room_corners,
                alignment=alignment, color=sample_color
            )

    # Draw room name on image_pred
    if room_name_list is not None:
        for pred_idx in range(len(room_corners_pred)):
            corners = room_corners_pred[pred_idx]
            size = np.array([grid_size, grid_size])
            corners = alignment.align_corners(corners) * size
            center = corners.mean(0).astype(int)

            cv2.putText(
                image_pred,
                str(room_name_list[pred_idx]),
                (center[1], center[0]),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 0, 255), 1,
                cv2.LINE_AA
            )

    # Draw room corners
    for gt_idx in range(len(room_corners_gt)):
        corners = room_corners_gt[gt_idx]
        size = np.array([grid_size, grid_size])
        corners = alignment.align_corners(corners) * size
        image_gt = draw_corners(image_gt, corners, color=np.array([255, 20, 147, 255]))

    for pred_idx in range(len(room_corners_pred)):
        corners = room_corners_pred[pred_idx]
        size = np.array([grid_size, grid_size])
        corners = alignment.align_corners(corners) * size
        image_pred = draw_corners(image_pred, corners, color=np.array([255, 20, 147, 255]))

    size_pred = len(room_corners_pred)
    size_gt = len(room_corners_gt)
    return num_match, size_pred, size_gt, image_pred, image_gt


def evaluate_corners_pr(
    corners_pred, corners_gt,
    points_gt, axis_corners,
    grid_size=256,
    merge_corners=False,
    merge_dist=0.5,
    dist_threshold=10,
):
    '''
        Evaluate the corner in the FloorNet and Floor-SP way.
        It will calculate precision and recall for corners on a grid space.
        The corners that are too close in real-world distance are merged so that it follows the shared-corners designed in FloorNet.
        NOTE: This will have problem when scene is really big with dense corner.
        Parameters:
            corners_pred: Estimated corners (N, 2)
            corners_gt: GT corners (M, 2)
            points_gt: GT point cloud
            axis_corners: Two corners defining an edge that need to be aligned to x-axis (2, 2)
            grid_size: The size of the 2D occ grid in pixels
            merge_corners: Boolean, defines whether merge corners that are too close
            merge_dist: The distance for merging (meter)
            dist_threshold: the threshold for computing matching in pixels
        Return:
            num_match: Number of matches
            size_pred: Total number of estimated corners
            size_gt: Total number of GT corners
            image_pred: visualization image result
            image_gt: visualization image result
    '''
    points = points_gt[:, [0, 2]]
    aligment = FloorNetCornerAlignment(points, axis_corners)
    points = aligment.align_corners(points)
    if merge_corners:
        corners_gt = merge_neighbor_corners(corners_gt, merge_dist)
    corners_gt = aligment.align_corners(corners_gt)     # This will map the corner coordinates into 0-1
    # Map gt_corners to grid
    size = np.array([grid_size, grid_size])
    corners_gt = np.round(corners_gt * size).astype(np.int32)
    density_map = compute_grid_map(points, height=grid_size, width=grid_size, scale_fractor=0.1)
    density_map = np.stack([density_map, density_map, density_map], axis=-1)
    density_map = np.clip(np.round(density_map), 0, 255).astype(np.uint8)
    density_map = 255 - density_map
    image_gt = draw_corners(density_map.copy(), corners_gt, np.array([255, 0, 0]))

    if merge_corners:
        corners_pred = merge_neighbor_corners(corners_pred, merge_dist)
    corners_pred = aligment.align_corners(corners_pred)
    corners_pred = np.round(corners_pred * size).astype(np.int32)
    image_pred = draw_corners(density_map.copy(), corners_pred, np.array([0, 255, 0]))
    num_match, size_pred, size_gt = calc_corners_pr(corners_pred, corners_gt, dist_threshold)
    return num_match, size_pred, size_gt, image_pred, image_gt


def make_unvisited_area(base_dir,
                        scene_name,
                        gt_corners,
                        min_key_frames_per_room=5):
    """Mask out unvisited room masks and corners."""
    scene, version = scene_name.split('_')
    pose_fn = os.path.join(base_dir, scene, version, 'frm_ref.txt')
    kf_fn = os.path.join(base_dir, scene, version + '/*/keyframe_list.txt')
    kf_fn = glob.glob(kf_fn)

    assert len(kf_fn) == 1, 'Keyframe list found error. Size: ' + str(len(kf_fn))
    with open(kf_fn[0], 'r') as f:
        kf_list = [int(kf_id) for kf_id in f.read().split()]

    poses_gt = read_trajectory(pose_fn)
    gt_poses = [list(poses_gt.values())[idx - 1] for idx in kf_list]

    all_cam_poses = np.vstack([pose[0:3, 3] for pose in gt_poses])
    masked_in_corners = []
    masked_out_corners = []
    for local_corners_gt in gt_corners:
        mask = measure.points_in_poly(all_cam_poses[:, (0, 2)], local_corners_gt)
        if np.sum(mask) <= min_key_frames_per_room:
            masked_out_corners.append(local_corners_gt)
            continue
        masked_in_corners.append(local_corners_gt)
    return masked_in_corners, masked_out_corners
