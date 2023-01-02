import torch
import os
import torch
import matplotlib.pyplot as plt

# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, load_obj, load_ply

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights, 
    AmbientLights,
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesUV,
    TexturesVertex
)
import argparse

# add path for demo utils functions 
import sys
import os
import math
sys.path.append(os.path.abspath(''))

from plot_image_grid import image_grid

# Setup
device = 'cuda'

from pytorch3d.io import IO

from pathlib import Path
from tqdm import tqdm
import os
import json

def render(eye, dir, mesh):
    R, T = look_at_view_transform(eye=eye, at=eye+dir, up=[[0,0,1]])
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    raster_settings = RasterizationSettings(
        image_size=768, 
        blur_radius=0.0, 
        faces_per_pixel=10, 
        # max_faces_per_bin=200_000,
    )
    lights = AmbientLights(device=device)
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device, 
            cameras=cameras,
            lights=lights
        )
    )
    images = renderer(mesh)
    return images


def render_scene(scene_id):
    obj_filename = "dataset/scannetv2/scans/%s/%s_vh_clean.ply"%(scene_id, scene_id)
    if not os.path.exists(obj_filename):
        return
    if os.path.exists(os.path.join('rendered_images_new', scene_id)):
        if len(os.listdir(os.path.join('rendered_images_new', scene_id))) == 200:
            return
    mesh = IO().load_mesh(obj_filename, device=device)

    range = mesh.verts_padded().max(dim=1)[0]
    range_min = mesh.verts_padded().min(dim=1)[0]
    # xrange = torch.arange(0, range[0, 0], 0.5)
    xrange = torch.linspace(range_min[0, 0], range[0, 0], 6).to(device)
    xrange = (xrange[:-1] + xrange[1:]) / 2
    # yrange = torch.arange(0, range[0, 1], 0.5)
    yrange = torch.linspace(range_min[0, 1], range[0, 1], 6).to(device)
    yrange = (yrange[:-1] + yrange[1:]) / 2
    z = range[0, 2] * 0.85
    # print(range)
    # print(xrange)
    # print(yrange)

    Path(os.path.join('rendered_images_new', scene_id)).mkdir(exist_ok=True, parents=True)
    for x in xrange:
        for y in yrange:
            for dx, dy in [[0, 1], [0, -1], [1, 0], [1, -1], [1, 1], [-1, 0], [-1, -1], [-1, 1]]:
                images = render(torch.tensor([[x, y, z]]).to(device), torch.tensor([[dx, dy, -0.5]]).to(device), mesh).clamp(0, 1)
                plt.imsave(os.path.join('rendered_images_new', scene_id, f'{x:.2f}_{y:.2f}_{z:.2f}_{dx}_{dy}_0.jpg'), images[0, ..., :3].cpu().numpy())


if __name__=='__main__':
    scene_list = os.listdir('dataset/scannetv2/scans')
    # with open('/home/zhengmh/Projects/ScanQA-Alter/data/qa/ScanQA_v1.0_val.json') as f:
    #     ann = json.load(f)
    # scene_list = []
    # for x in ann:
    #     scene_id = x['scene_id'].split('_')[0]
    #     if scene_id+'_00' not in scene_list:
    #         scene_list.append(scene_id+'_00')
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default=None)
    args = parser.parse_args()

    print(len(scene_list))
    if args.split:
        s, tot = args.split.split(':')
        s = int(s)
        tot = int(tot)
        space = math.ceil(len(scene_list) / tot)
        print(s * space, (s + 1) * space)
        for scene_id in tqdm(scene_list[s * space:(s + 1) * space]):
            render_scene(scene_id)
    else:
        for scene_id in tqdm(scene_list):
            render_scene(scene_id)