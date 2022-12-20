from mmdet3d.datasets import build_dataset
from mmcv import Config, DictAction
import open3d as o3d
import numpy as np
import mmcv
import matplotlib.pyplot as plt
import argparse
import cowa3d_common


def render_box(renderables: list, box3d, labels, name):
    clr_map = plt.get_cmap('tab10').colors
    corners = box3d.corners
    cores = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
        (8, 4), (8, 5), (8, 6), (8, 7)
    ]
    ret = None
    for corners_i, label_i in zip(corners, labels):
        corners_i = corners_i.numpy().astype(np.float64)
        frontcenter = corners_i[[4, 5, 6, 7]].mean(axis=0, keepdims=True)
        heading = corners_i[4] - corners_i[0]
        frontcenter += 0.3 * heading / np.linalg.norm(heading)
        corners_i = np.concatenate((corners_i, frontcenter), axis=0)
        corners_i = o3d.utility.Vector3dVector(corners_i)
        corners_i = o3d.geometry.PointCloud(points=corners_i)

        box = o3d.geometry.LineSet.create_from_point_cloud_correspondences(
            corners_i,
            corners_i,
            cores)
        box.paint_uniform_color(clr_map[label_i % len(clr_map)])
        if ret is None:
            ret = box
        else:
            ret += box
    renderables.append(dict(name=name, geometry=ret))
    return ret


def render_points(renderables: list, points, name, label=None,
                  num_classes=None):
    xyz = points[:, :3].numpy().astype(np.float64)
    if label is not None:
        clr = plt.get_cmap('tab20')(label % 20)[:, :3]
    else:
        clr = plt.get_cmap('gist_rainbow')(points[:, 3])[:, :3]
    points = o3d.geometry.PointCloud(
        points=o3d.utility.Vector3dVector(xyz))
    points.colors = o3d.utility.Vector3dVector(clr)
    renderables.append(dict(name=name, geometry=points))


def render_axis(renderables: list):
    renderables.append(dict(
        name='axis',
        geometry=o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)))


def parse_args():
    parser = argparse.ArgumentParser(description='Browse a dataset')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction)
    parser.add_argument(
        '--remote',
        action='store_true')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    dataset = build_dataset(cfg.data.train)

    if args.remote:
        o3d.visualization.webrtc_server.enable_webrtc()

    len_dataset = len(dataset)
    dataset = mmcv.track_iter_progress(((d for d in dataset), len_dataset))

    def next_data(vis: o3d.visualization.O3DVisualizer = None):
        data = next(dataset, None)
        if data is None:
            return
        renderables = []
        points = data['points'].data
        labels = data.get('pts_semantic_mask', None)
        if labels is not None:
            labels = labels.data
        render_points(renderables, points, name='points', label=labels)
        box3d = data.get('gt_bboxes_3d', None)
        names = data.get('gt_labels_3d', None)
        if box3d is not None and names is not None:
            render_box(renderables, box3d.data, names.data,
                       name='gt_bboxes_3d')
        if vis is not None:
            for renderable in renderables:
                vis.remove_geometry(renderable['name'])
                vis.add_geometry(renderable)
        else:
            return renderables

    o3d.visualization.gui.Application.instance.initialize()
    w = o3d.visualization.O3DVisualizer('Remote Visualizer')
    if args.remote:
        w.enable_raw_mode(True)
    w.set_background((0.0, 0.0, 0.0, 0.0), None)
    w.point_size = 2
    for data in next_data():
        w.add_geometry(data)
    w.show_skybox(False)
    w.show_ground = True
    w.ground_plane = w.ground_plane.XY
    w.add_action('next', next_data)
    w.reset_camera_to_default()
    w.setup_camera(90, (0, 0, 0), (-30, 0, 30), (1, 0, 1))
    o3d.visualization.gui.Application.instance.add_window(w)
    o3d.visualization.gui.Application.instance.run()


if __name__ == '__main__':
    main()
