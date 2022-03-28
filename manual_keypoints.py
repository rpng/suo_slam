#!/usr/bin/env python3

import argparse
import cv2
import numpy as np
import math
import pandas as pd
import time
import os
import json

# sudo apt install libfreetype6-dev
# sudo apt install libglfw3
from thirdparty.bop_toolkit.bop_toolkit_lib.renderer_py import RendererPython

from lib.labeling import kp_config

np.random.seed(666)

# Rendering image size.
IMG_WH = 420

def euler2R(euler):
    # Assume euler = {gamma, beta, alpha} (degrees)
    g, b, a = np.deg2rad(euler).astype(np.float64)
    cosa = np.cos(a)
    cosb = np.cos(b)
    cosg = np.cos(g)
    sina = np.sin(a)
    sinb = np.sin(b)
    sing = np.sin(g)
    R = np.array([[cosa*cosb, cosa*sinb*sing - sina*cosg, cosa*sinb*cosg + sina*sing],
                  [sina*cosb, sina*sinb*sing + cosa*cosg, sina*sinb*cosg - cosa*sing],
                  [-sinb, cosb*sing, cosb*cosg]], dtype=np.float64)
    return R

class SelectionGui():
    def __init__(self, ply_file, kp_config_file, r=7):
        """
        ply_file (str): The path to the ply mesh file
        kp_config_file (str): The path to the keypoint config CSV file containing
                              info about each object instance.
        r (int): Drawing circle radius (pixels)
        """
    
        # Camera focal length and center (square image)
        self.f = IMG_WH
        self.c = self.f/2
        

        self.K = np.array([[self.f, 0, self.c],
                      [0, self.f, self.c],
                      [0, 0, 1]], dtype=np.float64)
        
        # Best pose to view object
        self.view_pose = np.eye(4)
        self.view_pose[2,3] = 333

        print(f"Loading and setting up renderer for object from \"{ply_file}\"...")
        t = time.time()
        self.ply_file = ply_file
        self.renderer = RendererPython(self.f, self.f)
        self.renderer.add_object(0, ply_file)
        print(f"Done (took {time.time()-t:.3f} seconds)")
        
        print(f"Checking keypoint config file \"{kp_config_file}\"...")
        self.kp_config_file = kp_config_file
        # Note that this is 1-indexed
        self.object_id = int(ply_file.split("obj_")[-1].strip(".ply"))
        print(f"Found object ID {self.object_id}")
        self.load_kp_config()
        print("Done.")

        # Store the initial rotations
        self.rows = 2
        self.cols = 4
        # View angles with slight perturbation
        angles = np.zeros((self.rows*self.cols,3), dtype=np.float64)
        angles[:,2] = 180 # Objects are upside down 
        angles += np.random.normal(scale=10, size=angles.shape)
    
        self.Rs = np.zeros((self.rows, self.cols, 3, 3))
        for i in range(self.rows):
            for j in range(self.cols):
                self.Rs[i,j] = euler2R(angles[self.cols*i + j])
        # Each object will have the same position
        self.t = np.array([0,0,333], dtype=np.float64)

        # Drawing circle radius
        self.r = r

        # Number of keypoints (for this object anyways)
        n = len(self.kp_list)
        self.n = n
        self.k = np.zeros((self.rows,self.cols), dtype=np.int) # Current keypoint ID (per image)
        self.dones = np.zeros((self.rows,self.cols,self.n)).astype(np.bool)
        # Use full number of kp for colors so that they correspond
        self.colors = kp_config.kp_colors()
        self.curr_mouse = (0,0) # Mouse location
        
        # Store the last UVs, KP ID, and rotation mats
        self.selections = []

    def load_kp_config(self):
        data = pd.read_csv(self.kp_config_file)
        kp_config_df = data.iloc(0)[self.object_id-1]
        print("Found keypoint config:")
        print(kp_config_df)
        self.kp_map = kp_config.load_kp_config(data, self.object_id)
        self.kp_list = []
        for k in kp_config.kp_list:
            if k in self.kp_map.keys():
                self.kp_list.append(k)
            
        print("with keypoints:")
        print(self.kp_list)
        print("\nExit now if incorrect")

    # Render 8 views and panel them for user to select keypoints while 
    def render_views(self, Rs, t):
        m, n = self.rows, self.cols
        self.img = np.zeros((m*self.f, n*self.f, 3), dtype=np.uint8)
        self.depth_img = np.zeros((m*self.f, n*self.f), dtype=np.float32)
        # Make sure to get all sides of object
        for i in range(m):
            for j in range(n):
                R = Rs[i,j]
                rgbd = self.renderer.render_object(
                        0, R, t, self.f, self.f, self.c, self.c)
                    
                # ::-1 for RGB to BGR
                self.img[i*self.f:(i+1)*self.f, j*self.f:(j+1)*self.f,:] = rgbd['rgb'][...,::-1]
                self.depth_img[i*self.f:(i+1)*self.f, j*self.f:(j+1)*self.f] = rgbd['depth']

    def undo(self):
        # Grab the last UV. Undo has to be in order, so
        # only the last drawn KP can be undone.
        if len(self.selections) > 0:
            last = self.selections[-1]
            u, v = last["uv"]
            self.selections.pop()

            # Fix the current kp 
            k_last = self.k[v//self.f][u//self.f]
            if k_last > 0:
                self.dones[v//self.f,u//self.f,k_last-1] = False
                self.k[v//self.f][u//self.f] = k_last-1
    
    def kp_color_uv(self, u, v):
        return self.kp_color(self.k[v//self.f][u//self.f])
    
    def kp_color(self, k):
        k = min(self.n-1, k)
        return self.colors[self.kp_map[self.kp_list[k]]].tolist()

    def mouse_callback(self, event, u, v, flags, param):
        self.curr_mouse = (u,v)
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.depth_img[v,u] > 0:
                # Disallow extra KP if already set
                k = self.k[v//self.f][u//self.f]
                if k < self.n and not self.dones[v//self.f,u//self.f,k]:
                    # Project into object frame with GT depth and pose
                    # Transfer uv (of whole panel of images) to uv of just one image
                    un, vn = u % self.f, v % self.f

                    # Project to 3D in camera frame
                    z = self.depth_img[v, u]
                    x = z * (un - self.c) / self.f
                    y = z * (vn - self.c) / self.f
                    p_FinC = np.array([x, y, z], dtype=np.float64)[:,None]

                    p_OinC = self.t[:,None]
                    R_OtoC = self.Rs[v//self.f,u//self.f]
                    
                    # Put in object frame
                    p_FinO = R_OtoC.T @ (p_FinC - p_OinC)
                        
                    self.selections.append({
                        "uv": (u,v),
                        "kp_sample": p_FinO,
                        "k": k,
                        "kp_name": self.kp_list[min(self.n-1,self.k[v//self.f][u//self.f])]
                    }) 

                    # Increment color for this image
                    self.dones[v//self.f,u//self.f,self.k[v//self.f][u//self.f]] = True
                    self.k[v//self.f][u//self.f] = min(k+1, self.n)
                else:
                    print(f"Keypoint {min(self.n-1,k)} already set in this section!")
            else:
                print("Please pick a point on the object!")

    def kp_stats(self):
        # Average the 3D keypoint locations in object frame
        kp_avg = np.zeros((self.n,3))
        kp_n = np.zeros((self.n), dtype=np.int)
        kp_sample = np.zeros((8,self.n,3))
        for selection in self.selections:
            p_FinO = selection["kp_sample"]
            k = selection["k"]

            # Store the sample
            kp_sample[kp_n[k],k,:] = p_FinO[:,0]

            # add to avg
            kp_avg[k,:] += p_FinO[:,0]
            kp_n[k] += 1

        # Avg
        kp_avg /= np.maximum(np.ones_like(kp_n)[:,None], kp_n[:,None])

        # Sample covariance
        kp_cov = np.zeros((self.n, 3, 3))
        for k in range(self.n):
            # kp_n[k] samples avail for keypoint k
            res = kp_sample[:kp_n[k],k,:] - kp_avg[k:k+1,:]
            kp_cov[k,:,:] = np.sum(res[:,:,None] @ res[:,None,:], axis=0)

        # Median-based cov estimate
        kp_cov /= np.maximum(np.ones_like(kp_n)[:,None,None], (kp_n[:,None,None] - 1))
    
        return kp_n, kp_avg, kp_cov

    def inspect_results(self, kp_n, kp_avg, kp_cov, once=False):
        if not once:
            print("\n\nInspect the results!")
            print("Use the \"wasd\" to turn the object.")
            print("Press \"i\" to zoom in and \"o\" to zoom out.")
            print("Press \"Esc\" to go back, \"Enter\" to accept "
                    + "(saving keypoints and viewpoint for vizualization).")
            cv2.namedWindow('Inspect Results')

        # Small delta for the inspection to get the view_pose more accurate
        delta = 2
        while True:
            R, t = self.view_pose[:3,:3], self.view_pose[:3,3]
            rgbd = self.renderer.render_object(
                    0, R, t, self.f, self.f, self.c, self.c)
            
            img = np.copy(rgbd['rgb'][...,::-1])
            depth = rgbd['depth']
            mask = depth == 0
            #for i in range(3):
            #    mask = cv2.medianBlur(mask.astype(np.uint8), 7)
            img[mask.astype(np.bool)] = 255

            # Connect the dots for text keypoints
            connected = {}
            for key in kp_config.instance_texture_kps["brand_name"]:
                connected[key] = None

            # Draw the keypoints with covariance ellipse and normal
            img_normal = np.copy(img)
            for k in range(self.n):
                if kp_n[k] > 0:
                    p_FinC = R @ kp_avg[k,:,None] + t[:,None]
                    uvz = self.K @ p_FinC
                    u = uvz[0,0] / uvz[2,0]
                    v = uvz[1,0] / uvz[2,0]
                    u, v = int(u+.5), int(v+.5)
                    cv2.circle(img_normal, (u,v), self.r, self.kp_color(k), -1)
                    if self.kp_list[k] in connected.keys():
                        connected[self.kp_list[k]] = (u,v), self.kp_color(k)

                    if not once:
                        # Draw the keypoint name next to keypoint
                        img_normal = cv2.putText(img_normal, 
                                self.kp_list[min(self.n-1,k)], 
                                (u+10, v), cv2.FONT_HERSHEY_PLAIN, 1, 
                                (255,255,255), 1, cv2.LINE_AA)
                
                if kp_n[k] > 2: # Need at least 3 points to sample cov    
                    # Covariance propagate with first-order linear approximate of uv
                    duv_duvz = np.array([[1/uvz[2,0],0,-uvz[0,0]/uvz[2,0]**2],
                                         [1/uvz[2,0],0,-uvz[1,0]/uvz[2,0]**2]])
                    
                    S = duv_duvz @ self.K @ R
                    cov = S @ kp_cov[k] @ S.T

                    lamb, V = np.linalg.eig(cov)
                    # Ellipse angle is based on direction of first eigenvector
                    angle = np.arctan2(V[1,0], V[0,0])
                    s = 1.0 # Make cov look bigger/smaller
                    axes_length = (int(round(s*2*np.sqrt(5.991*lamb[0]))), 
                                   int(round(s*2*np.sqrt(5.991*lamb[1]))))
                    img = cv2.ellipse(img, (u,v), axes_length, 
                            180*angle/np.pi, 0, 360, self.kp_color(k), -1) 
           
            # Draw the lines for text keypoints here
            text_uvs = [connected[key] for key in kp_config.instance_texture_kps["brand_name"]]
            for i in range(len(text_uvs)):
                if text_uvs[i] is not None and text_uvs[i-1] is not None:
                    cv2.line(img_normal, text_uvs[i-1][0], text_uvs[i][0], (0,0,255), 2)
                    # Redraw the points over the line
                    for di in [-1,0]:
                        uv, color = text_uvs[i+di]
                        cv2.circle(img_normal, uv, self.r, color, -1)
            
            # If "once" we are doing a big vizualiation, so just return the normal image
            if once:
                return img_normal

            img_normal = cv2.putText(img_normal, "Unscaled", (3,33), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (10,10,255), 2, cv2.LINE_AA)
            img = cv2.putText(img, "Scaled w/ Cov", (3,33), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (10,255,10), 2, cv2.LINE_AA)
                
            img = np.concatenate((img_normal, img), axis=1)
            cv2.imshow('Inspect Results', img)
            k = cv2.waitKey(20) & 0xFF
            if k == 27: # Esc to quit
                cv2.destroyWindow('Inspect Results')
                return False
            elif k == 13: # Return to accept
                good = True
                for k in range(self.n):
                    if kp_n[k] < 3:
                        print(f"ERROR: Not enough selections for keypoint {k} (at least 3 needed)!")
                        print("Please try again or reduce the number of keypoints to select.")
                        good = False
                if good:
                    self.view_pose[:3,:] = np.concatenate((R,t[:,None]), axis=-1)
                    cv2.destroyWindow('Inspect Results')
                    return True
            elif k == ord('w'):
                self.view_pose[:3,:3] = euler2R([-delta,0,0]) @ self.view_pose[:3,:3]
            elif k == ord('a'):
                self.view_pose[:3,:3] = euler2R([0,delta,0]) @ self.view_pose[:3,:3]
            elif k == ord('s'):
                self.view_pose[:3,:3] = euler2R([delta,0,0]) @ self.view_pose[:3,:3]
            elif k == ord('d'):
                self.view_pose[:3,:3] = euler2R([0,-delta,0]) @ self.view_pose[:3,:3]
            elif k == ord('i'):
                self.view_pose[2,3] -= delta
            elif k == ord('o'):
                self.view_pose[2,3] += delta
    
    def kp_save_path(self):
        # E.g., blah.ply becomes ../kp_info/blah_kp_info.json
        kp_dir = os.path.abspath(os.path.join(os.path.dirname(self.ply_file),
                '../kp_info'))
        if not os.path.exists(kp_dir):
            os.makedirs(kp_dir)
        return os.path.join(kp_dir,
               os.path.basename(self.ply_file).split('.')[0] + "_kp_info.json")

    def save_kp(self, kp_avg, kp_cov):
        # E.g., blah.ply becomes ../kp_info/blah_kp_info.json
        kp_dir = os.path.abspath(os.path.join(os.path.dirname(self.ply_file), '../kp_info'))
        if not os.path.exists(kp_dir):
            os.makedirs(kp_dir)
        fname = os.path.join(kp_dir,
                os.path.basename(self.ply_file).split('.')[0] + "_kp_info.json")
        assert kp_avg.shape[0] == len(self.kp_list)
        kp_data = {
            "keypoints": {},
            "view_pose": self.view_pose.reshape(-1).tolist()
        }
        for i, kp_name in enumerate(self.kp_list):
            kp_data["keypoints"][kp_name] = {
                    "pos_mean": kp_avg[i].reshape(-1).tolist(),
                    "pos_cov": kp_cov[i].reshape(-1).tolist(),
            }
        with open(fname, 'w') as f:
            json.dump(kp_data, f, indent=4)
        print(f"Keypoint stats (avg, cov) in json format saved to:\n\t{fname}")
    
    def inspect_from_file(self, once=False):
        kp_dir = os.path.abspath(os.path.join(os.path.dirname(self.ply_file), '../kp_info'))
        fname = os.path.join(kp_dir, 
                os.path.basename(self.ply_file).split('.')[0] + "_kp_info.json")
        with open(fname, 'r') as f:
            kp_data = json.load(f)
        kp_avg = np.empty((self.n, 3))
        kp_cov = np.empty((self.n, 3, 3))
        for i, kp_name in enumerate(self.kp_list):
            kp_avg[i] = kp_data["keypoints"][kp_name]["pos_mean"]
            kp_cov[i] = np.array(kp_data["keypoints"][kp_name]["pos_cov"]).reshape(3,3)
        self.view_pose = np.array(kp_data["view_pose"]).reshape(4,4)
        kp_n = 7 * np.ones((self.n), dtype=np.int) # Magic number 7 is >2
        ret = self.inspect_results(kp_n, kp_avg, kp_cov, once=once)
        if not once and ret:
            self.save_kp(kp_avg, kp_cov)
        return ret

    # OpenCV GUI for keypoint selection on rendered images
    def run(self):
        print("\n\n============= Welcome ===============")
        print("Select the keypoints with a left click!")
        print("Use the \"wasd\" to turn the objects.")
        print("Press \"i\" to zoom in and \"o\" to zoom out.")
        print("Make sure that the keypoint colors match between all views.")
        print("Messed up? Just press 'u' to undo.")
        print("Press \"Enter\" to finish and save the keypoints")
        print("Press \"Esc\" to just quit")
        cv2.namedWindow('Select Keypoints')
        cv2.setMouseCallback('Select Keypoints', self.mouse_callback)
        
        delta = 10

        while True:
            self.render_views(self.Rs, self.t)           
            
            # Draw lines for brand name for convenience. Since the selections
            # are in any order, figure out which dots should be connected with this.
            # TODO if any other text keypoints, add them to this logic
            connected = [[None] * self.cols] * self.rows
            for i in range(self.rows):
                for j in range(self.cols):
                    connected[i][j] = {} # Store the UV's or None
                    for key in kp_config.instance_texture_kps["brand_name"]:
                        connected[i][j][key] = None

            # Draw selected kps
            for selection in self.selections:
                u, v = selection["uv"]
                i, j = v//self.f, u//self.f
                p_FinC = self.Rs[i,j] @ selection["kp_sample"] + self.t[:,None]
                uvz = self.K @ p_FinC
                u = uvz[0,0] / uvz[2,0] + u//self.f * self.f
                v = uvz[1,0] / uvz[2,0] + v//self.f * self.f
                # Check if it's in the same section as originally selected.
                if i == v//self.f and j == u//self.f:
                    cv2.circle(self.img,(int(u+.5), int(v+.5)), self.r, 
                            self.kp_color(selection["k"]), -1)
                    if selection["kp_name"] in kp_config.instance_texture_kps["brand_name"]:
                        connected[i][j][selection["kp_name"]] = (int(u+.5), int(v+.5))
            
            # Draw the connecting lines for brand_name
            for i in range(self.rows):
                for j in range(self.cols):
                    text_uvs = [connected[i][j][key] for key in \
                            kp_config.instance_texture_kps["brand_name"]]
                    for k in range(len(text_uvs)):
                        if text_uvs[k] is not None and text_uvs[k-1] is not None:
                            cv2.line(self.img, text_uvs[k-1], text_uvs[k], (0,0,255), 2)
                            # Redraw the points over the line
                            for dk in [-1,0]:
                                color = kp_config.kp_color(
                                        kp_config.instance_texture_kps["brand_name"][k+dk]).tolist()
                                cv2.circle(self.img, text_uvs[k+dk], self.r, color, -1)

            
            # Draw the keypoint under the mouse
            cv2.circle(self.img, self.curr_mouse, self.r, self.kp_color_uv(*self.curr_mouse), -1)
            # Draw the keypoint name next to mouse
            u, v = self.curr_mouse
            self.img = cv2.putText(self.img, 
                    self.kp_list[min(self.n-1,self.k[v//self.f][u//self.f])], 
                    (u+10, v), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, cv2.LINE_AA)
            cv2.imshow('Select Keypoints', self.img)
            k = cv2.waitKey(20) & 0xFF
            if k == 27:
                break
            elif k == 13:
                kp_n, kp_avg, kp_cov = self.kp_stats()
                if self.inspect_results(kp_n, kp_avg, kp_cov):
                    self.save_kp(kp_avg, kp_cov)
                    break
            elif k == ord('u'):
                self.undo()
            #elif k-48 >= 0 and k-48 < self.n: # Keypoint ID selection
            #    u, v = self.curr_mouse
            #    self.k[v//self.f][u//self.f] = k-48
            elif k == ord('w'):
                self.Rs = euler2R([-delta,0,0])[None,None,:,:] @ self.Rs
            elif k == ord('a'):
                self.Rs = euler2R([0,delta,0])[None,None,:,:] @ self.Rs
            elif k == ord('s'):
                self.Rs = euler2R([delta,0,0])[None,None,:,:] @ self.Rs
            elif k == ord('d'):
                self.Rs = euler2R([0,-delta,0])[None,None,:,:] @ self.Rs
            elif k == ord('i'):
                self.t[2] -= delta
            elif k == ord('o'):
                self.t[2] += delta

if __name__ == '__main__':
    parser = argparse.ArgumentParser("./manual_keypoints.py")
    parser.add_argument(
        '--ply_file',
        type=str,
        default="/media/nate/Elements/bop/bop_datasets/ycbv/models_fine/obj_000015.ply",
        help='Path to the input PLY mesh file.'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default="ycbv", choices=["ycbv", "tless"],
        help='"ycbv" or "tless"'
    )
    
    parser.add_argument('--inspect', dest='inspect', action='store_true',
        help='If set, inspect the keypoints already made for this ply file . '
             'Obviously, the keypoint file must exist '
             '(dirname(/path/to/blah.ply)/../kp_info/blah_kp_info.json). '
             'Note that if you save, it will overwrite the view pose.'
    )
    
    parser.add_argument(
        '--viz',
        type=str,
        default=None,
        help='Path to a directory containing ply files'
             ' instances in kp_config_file. Program will write a visualization of them all.'
    )
        
    parser.add_argument(
        '-r', type=int, default=5,
        help='Radius size in pixels of the rendered circles.'
    )

    args = parser.parse_args()
    setattr(args, "kp_config_file", f"./data/{args.dataset}_kp_config.csv")

    if args.viz is not None:
        config_data = pd.read_csv(args.kp_config_file)
        num_objects = config_data.shape[0]
        if args.dataset == "ycbv":
            rows, cols = 3, 7
            assert num_objects == 21
        else:
            rows, cols = 3, 10
            assert num_objects == 30
        img_combined = np.zeros((rows*IMG_WH, cols*IMG_WH, 3), dtype=np.uint8)
        for i in range(rows):
            for j in range(cols):
                # TODO Why does this memory leak and fill up RAM?!
                object_idx = i*cols + j
                file_stem = "obj_" + str(object_idx+1).zfill(6)
                ply_file = os.path.join(args.viz, file_stem + ".ply")
                gui = SelectionGui(ply_file=ply_file, 
                        kp_config_file=args.kp_config_file, r=args.r)
                img = gui.inspect_from_file(once=True)
                img_combined[i*IMG_WH:(i+1)*IMG_WH, j*IMG_WH:(j+1)*IMG_WH, :] = img
                cv2.imshow("kp_viz.png", img_combined)
                cv2.waitKey(1)

        print(f"Writing visualization image to ./assets/{args.dataset}_kp_viz.png")
        cv2.imwrite(f"./assets/{args.dataset}_kp_viz.png", img_combined)
        cv2.imshow(f"{args.dataset}_kp_viz.png", img_combined)
        cv2.waitKey(0)
    else:
        gui = SelectionGui(ply_file=args.ply_file, 
                kp_config_file=args.kp_config_file, r=args.r)
        
        if args.inspect: 
            gui.inspect_from_file()
        else:
            gui.run()
