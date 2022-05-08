
import os
import json
import torch
import numpy as np
from thirdparty.bop_toolkit.bop_toolkit_lib.inout import load_ply

def load_meshes(model_dir, obj_ids):
    meshes = {}
    for obj_id in obj_ids:
        fln = os.path.join(model_dir, f"obj_{obj_id:06d}.ply")
        print("Loading mesh file", fln, end='\r')
        meshes[obj_id] = load_ply(fln)
    print()
    return meshes

def load_mesh_db(model_dir):
    with open(os.path.join(model_dir, 'models_info.json'), 'r') as f:
        model_info = json.load(f)
    obj_ids = [int(obj_id) for obj_id in model_info.keys()]
    meshes = load_meshes(model_dir, obj_ids)
    mesh_db = {}
    for obj_id in obj_ids:
        info = model_info[str(obj_id)]
        is_sym_disc = len(info.get("symmetries_discrete",[])) > 0
        is_sym_cont = len(info.get("symmetries_continuous",[])) > 0
        cont_sym = []
        if is_sym_cont:
            cont_sym = info["symmetries_continuous"]
        points_np = np.array(meshes[obj_id]["pts"], dtype=np.float32)
        points = torch.tensor(points_np).to(torch.float32)
        if torch.cuda.is_available():
            points = points.cuda()
        mesh_db[obj_id] = {
                "is_symmetric": is_sym_disc or is_sym_cont,
                "continuous_sym": cont_sym,
                "diameter": info["diameter"],
                "points": points,
        }
        #print(f"Object {obj_id} is_symmetric: {is_sym_disc or is_sym_cont}")
        #for k, v in meshes[obj_id].items():
        #    mesh_db[k] = v
        #for k, v in model_info[str(obj_id)].items():
        #    mesh_db[k] = v
    return mesh_db

def load_meshes_DEBUG(model_dir, obj_ids, obj_map):
    meshes = {}
    for obj_id in obj_ids:
        fln = os.path.join(model_dir, obj_map[obj_id], "points.xyz")
        print("Loading mesh file", fln, end='\r')
        meshes[obj_id] = 1000 * np.loadtxt(fln, dtype=np.float32)
    print()
    return meshes


# For loading the original YCBV models for DEBUGGING ONLY
def load_mesh_db_DEBUG(model_dir, obj_map):
    obj_ids = [int(obj_id) for obj_id in obj_map.keys()]
    meshes = load_meshes_DEBUG(model_dir, obj_ids, obj_map)
    mesh_db = {}
    for obj_id in obj_ids:
        points_np = meshes[obj_id]
        points = torch.tensor(points_np).to(torch.float32)
        if torch.cuda.is_available():
            points = points.cuda()
        mesh_db[obj_id] = {
                "is_symmetric": False,
                "continuous_sym": False,
                "points": points,
                "points_np": points_np,
        }
        #print(f"Object {obj_id} is_symmetric: {is_sym_disc or is_sym_cont}")
        #for k, v in meshes[obj_id].items():
        #    mesh_db[k] = v
        #for k, v in model_info[str(obj_id)].items():
        #    mesh_db[k] = v
    return mesh_db
