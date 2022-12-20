import os

container = 'crpilot2-x86-2.4'

comm_port = 6969

crload_args = dict(
    param='''
LidarCompensator:
  Component:
    class: "lidar_compensator:LidarCompensator"
    enable: true
    reader:
      channel: "/left/ruby/lidar_points"
      queue_size: 1
  max_points: 600000
  merge: 
    - "/left/ouster/lidar_points"
    - "/right/ouster/lidar_points"
''',
    offline=True,
    custom_args=['--logcout', '--loglevel', '-2']
)

data_files = ['/data/veg/06.30-10.01.29.489583571.record_to']

request_channels = [
    '/fusion/current_pose',
    '/left/ruby/lidar_points',
    '/left/ouster/lidar_points',
    '/right/ouster/lidar_points']

proxy_topics = [('/fusion/lidar_points', 'PointCloud3')]
blacklist_topics = ['/fusion/lidar_points']
pose_provider = 'FUSIONED'

filter_sensors = [0]

model_pth = 'work_dirs/cylinder3d_semantic_cowa/epoch_33.pth'
model_cfg = 'work_dirs/cylinder3d_semantic_cowa/cylinder3d_semantic_cowa.py'


def init():
    if __name__ == '__main__':
        import subprocess
        ret = subprocess.run(
            ['docker', 'cp', f'{__file__}',
             f'{container}:/tmp/data_extract.py'])
        assert ret.returncode == 0, ret
        ret = subprocess.run(
            f'docker exec {container} '
            '/bin/bash -c "source setup.bash;python -c \'import crpilot;print(crpilot.__file__)\'"',
            shell=True, capture_output=True)
        assert ret.returncode == 0, ret
        crpilot_inside_container = ret.stdout.decode().strip()

        ret = subprocess.run(
            ['docker', 'cp', f'{container}:{crpilot_inside_container}',
             '/tmp/crpilot.py'])
        assert ret.returncode == 0, ret
    import importlib
    import sys
    spec = importlib.util.spec_from_file_location('crpilot', '/tmp/crpilot.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules['crpilot'] = module
    CUDA_VISIBLE_DEVICES = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import open3d
    if CUDA_VISIBLE_DEVICES is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES
    else:
        os.environ.pop('CUDA_VISIBLE_DEVICES')


def launch_reader():
    import subprocess
    reader_proc = subprocess.Popen(
        f'docker exec {container} '
        '/bin/bash -c "source setup.bash;python /tmp/data_extract.py"',
        shell=True)
    return reader_proc


class Inference:
    def __init__(self, model_cfg):
        self.cfg = model_cfg
        self._model = None

    def model(self):
        from mmdet3d.models import build_model
        from mmcv import Config
        from mmcv.runner import wrap_fp16_model, load_checkpoint
        if self._model is not None:
            return self._model
        else:
            model_cfg = Config.fromfile(self.cfg)
            model = build_model(model_cfg.model,
                                test_cfg=model_cfg.get('test_cfg'))
            fp16_cfg = model_cfg.get('fp16', None)
            if fp16_cfg is not None:
                wrap_fp16_model(model)
            checkpoint = load_checkpoint(model, model_pth, map_location='cpu')
            model.CLASSES = checkpoint['meta']['CLASSES']
            model.cuda()
            model.eval()
            self._model = model
            return self._model

    def infer_loop(self, output_q):
        import torch
        model = self.model()
        s = crpilot.CRMWRaw(
            remote=crpilot.RemoteManager(port=comm_port))
        with torch.no_grad():
            while not s.IsRemoteDone():
                topic, data = s.Pop()
                if topic is not None and topic == '/fusion/lidar_points':
                    print(
                        f'{topic} @ {(data.timestamp % int(60000000000)) / 1e9:.2f}')
                    pc = data.point
                    if filter_sensors is not None:
                        sensor = pc['sensor']
                        sensor_mask = [sensor == s for s in
                                       filter_sensors]
                        sensor_mask = np.stack(sensor_mask,
                                               axis=0).any(
                            axis=0)
                        pc = pc[sensor_mask]
                    xyzi = np.stack(
                        [pc['x'].astype(np.float32),
                         pc['y'].astype(np.float32),
                         pc['z'].astype(np.float32),
                         pc['intensity'].astype(np.float32) / 255.0],
                        axis=-1)
                    xyzi = torch.from_numpy(xyzi).float().cuda()
                    dist = xyzi[:, :2].norm(dim=-1)
                    xyzi = xyzi[(dist < 50) & (dist > 1.5)]
                    seg = model(return_loss=False, rescale=True,
                                points=[[xyzi]],
                                img_metas=[[None]])[0]
                    segmented = []
                    for ci, cname in enumerate(model.CLASSES):
                        mask_c = (seg == ci)
                        xyz_c = np.ascontiguousarray(
                            xyzi[mask_c, :3].cpu().numpy().astype(np.float64))
                        if len(xyz_c) == 0:
                            xyz_c = np.full([1, 3], 10000,
                                            dtype=np.float64)
                        clr_c = plt.get_cmap('tab20')(ci)[:3]
                        clr_c = np.ascontiguousarray(
                            np.array([clr_c] * len(xyz_c), dtype=np.float64))
                        segmented.append((xyz_c, clr_c, cname))
                    output_q.put(segmented)
            s.Finish()

    def infer(self, window):
        q = Queue(maxsize=10)
        process = Process(target=self.infer_loop, args=(q,))
        process.start()
        vis_con = o3d.visualization.gui.Application.instance
        while True:
            try:
                segmented = q.get()
            except ValueError:
                break
            except OSError:
                break

            geodicts = []
            for (xyz_c, clr_c, cname) in segmented:
                points = o3d.geometry.PointCloud(
                    points=o3d.utility.Vector3dVector(xyz_c))
                points.colors = o3d.utility.Vector3dVector(clr_c)
                geodicts.append(dict(name=cname, geometry=points))

            def draw_task():
                for geodict in geodicts:
                    geo = window.get_geometry(geodict['name'])
                    vis = geo.is_visible if geo is not None else True
                    window.remove_geometry(geodict['name'])
                    window.add_geometry(**geodict, is_visible=vis)
                window.post_redraw()

            vis_con.post_to_main_thread(w, draw_task)
        process.join()

        def quit():
            w.quit()

        vis_con.post_to_main_thread(w, quit)


if 'CRPILOT_ROOT' in os.environ:
    from msgs.shared_msg_pb2 import PointCloud3, Image2
    import pycrmw
    import crpilot

    assert len(data_files) > 0, "no valid record file found"
    s = crpilot.RecordProxy(
        data_files,
        remote=crpilot.RemoteManager(crmw_side=True, port=comm_port))
    rec_info = s.record.GetInfo()

    for channel in request_channels:
        if channel not in rec_info:
            print(
                f"requested channel: {channel} not found in the record file"
                f" {data_files}!\nexisting topics:\n" +
                '\n'.join([f'{k}: {v} messages' for k, v in rec_info.items()]))
            s.remote.finish()
            exit(0)

    s.LoadComponents(**crload_args)
    for topic, typ in proxy_topics:
        s.AddTopicProxy('/fusion/lidar_points', eval(typ))
    s.SetBackList(blacklist_topics)
    s.AddPoseProvider(pycrmw.PoseProvider(
        eval(f'pycrmw.PoseProvider.{pose_provider}')))
    s.Run()

else:
    init()
    import numpy as np
    from torch.multiprocessing import Queue, Process
    import matplotlib.pyplot as plt
    import open3d as o3d
    import crpilot
    from functools import partial

    if __name__ == '__main__':
        reader_proc = launch_reader()

        inferer = Inference(model_cfg)

        o3d.visualization.gui.Application.instance.initialize()
        w = o3d.visualization.O3DVisualizer('Remote Visualizer')
        w.enable_raw_mode(True)
        w.set_background((0.0, 0.0, 0.0, 0.0), None)
        w.point_size = 2
        w.setup_camera(60, (0, 0, 0), (-5, 0, 5), (1, 0, 1))
        w.show_skybox(False)
        w.show_ground = True
        w.ground_plane = w.ground_plane.XY
        o3d.visualization.gui.Application.instance.add_window(w)

        o3d.visualization.gui.Application.instance.run_in_thread(
            partial(inferer.infer, w))
        o3d.visualization.gui.Application.instance.run()

        ret = reader_proc.wait(timeout=5)
        if ret is None:
            reader_proc.kill()
        elif ret != 0:
            raise RuntimeError(
                f'crmw reader process exits with return code {ret}!')
