# https://github.com/RainerKuemmerle/g2o/blob/master/g2o/examples/ba/ba_demo.cpp

import numpy as np
import g2o 

from collections import defaultdict
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--noise', dest='pixel_noise', type=float, default=1.,
    help='noise in image pixel space (default: 1.0)')
parser.add_argument('--outlier', dest='outlier_ratio', type=float, default=0.,
    help='probability of spuroius observation  (default: 0.0)')
parser.add_argument('--robust', dest='robust_kernel', action='store_true', help='use robust kernel')
parser.add_argument('--dense', action='store_true', help='use dense solver')
parser.add_argument('--seed', type=int, help='random seed', default=0)
args = parser.parse_args()


def euler2R(euler):
    # Assume euler = {gamma, beta, alpha} (degrees)
    g, b, a = np.deg2rad(euler).astype(np.float64)
    """
    Rx = np.array([[1, 0, 0],
                    [0, cos(r), -sin(r)],
                    [0, sin(r), cos(r)]], dtype=np.float64)
    Ry = np.array([[cos(p), 0, sin(p)],
                    [0, 1, 0],
                    [-sin(p), 0, cos(p)]], dtype=np.float64)
    Rz = np.array([[cos(y), -sin(y), 0],
                  [sin(y), cos(y), 0],
                  [0, 0, 1]], dtype=np.float64)
    R = np.matmul( np.matmul(Rz, Ry), Rx )
    """
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


def main():    
    optimizer = g2o.SparseOptimizer()
    solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
    solver = g2o.OptimizationAlgorithmLevenberg(solver)
    optimizer.set_algorithm(solver)

    focal_length = 320
    principal_point = (320, 240)
    #cam = g2o.CameraParameters(focal_length, principal_point, 0)
    #cam.set_id(0)
    #optimizer.add_parameter(cam)
    cam_k = np.array([focal_length, focal_length, 
            principal_point[0], principal_point[1]], dtype=np.float64)
    

    true_poses = []
    num_pose = 15
    for i in range(num_pose):
        # pose here means transform points from world coordinates to camera coordinates
        pose = g2o.SE3Quat(euler2R(np.random.normal(0,10, size=(3))),
                                   np.random.normal(0,.5, size=(3)))

        v_se3 = g2o.VertexSE3Expmap()
        v_se3.set_id(i)
        v_se3.set_estimate(pose)
        if i < 2:
            v_se3.set_fixed(True)
        optimizer.add_vertex(v_se3)
        true_poses.append(v_se3)

    num_object = 6
    num_points_per_object = 8
    true_object_points = np.random.uniform(-0.1, 0.1, 
            size=(num_object, num_points_per_object, 3))
    true_object_poses = []
    object_verts = []

    rmse_before = 0
    for i in range(num_object):
        # pose here means transform points from object frame to world
        pose = g2o.SE3Quat(euler2R(np.random.uniform(-180,180, size=(3))), 
                                   np.random.uniform(-.1,.1, size=(3)))
        true_object_poses.append(pose)
        v_se3 = g2o.VertexSE3Expmap()
        v_se3.set_id(i + num_pose)
        dx = g2o.SE3Quat(euler2R(np.random.normal(0,10,size=(3))), 
                                 np.random.normal(0,0.1,size=(3)))
        perturbed_pose = pose * dx

        v_se3.set_estimate(perturbed_pose)
        optimizer.add_vertex(v_se3)
        object_verts.append(v_se3)
        
        rmse_before += np.linalg.norm(dx.to_minimal_vector())
        
        # TODO Add prior on object pose
        '''
        edge = g2o.EdgeSE3Prior()
        edge.set_vertex(0, v_se3)
        edge.set_measurement(perturbed_pose.Isometry3d())
        edge.set_information(np.identity(6))
        if args.robust_kernel:
            edge.set_robust_kernel(g2o.RobustKernelHuber())
        edge.set_parameter_id(0, 0) # For caching parameter
        optimizer.add_edge(edge)
        '''

    rmse_before = rmse_before / num_object

    point_id = num_object + num_pose + 1
    
    cam_edges = np.zeros(num_pose, dtype=np.int)

    for o_i, object_pose in enumerate(object_verts):
        for i in range(num_points_per_object):
            point = object_pose.estimate() * true_object_points[o_i,i]
            visible = []
            for j, pose in enumerate(true_poses):
                xyz = pose.estimate() * point
                z = np.array([cam_k[0]*xyz[0]/xyz[2]+cam_k[2],
                              cam_k[1]*xyz[1]/xyz[2]+cam_k[3]])
                if 0 <= z[0] < 640 and 0 <= z[1] < 480:
                    print(f"object {o_i}, cam pose {j} is visible")
                    visible.append((j, z))
                    cam_edges[j] += 1
            if len(visible) < 1:
                continue

            for j, z in visible:
                if np.random.random() < args.outlier_ratio:
                    z = np.random.random(2) * [640, 480]
                z += np.random.randn(2) * args.pixel_noise

                edge = g2o.EdgeSE3ProjectFromObject(cam_k, true_object_points[o_i,i])
                edge.set_vertex(0, object_pose)
                edge.set_vertex(1, true_poses[j])
                edge.set_measurement(z)
                edge.set_information(np.identity(2))
                if args.robust_kernel:
                    edge.set_robust_kernel(g2o.RobustKernelHuber())

                #edge.set_parameter_id(0, 0)
                optimizer.add_edge(edge)

            point_id += 1
    
    num_img = 0
    for i in range(num_pose):
        if cam_edges[i] < 1:
            optimizer.remove_vertex(optimizer.vertex(i))
        else:
            num_img += 1

    print('num vertices:', len(optimizer.vertices()))
    print('num edges:', len(optimizer.edges()), f' ({num_img} images in view)')

    print('Performing full BA:')
    optimizer.initialize_optimization()
    optimizer.set_verbose(True)
    optimizer.optimize(10)

    rmse_after = 0
    for i in range(num_object):
        dx = true_object_poses[i].inverse() * object_verts[i].estimate()
        rmse_after += np.linalg.norm(dx.to_minimal_vector())
    rmse_after = rmse_after / num_object


    print('\nRMSE:')
    print('object pose RMSE before optimization:', rmse_before)
    print('object pose RMSE after  optimization:', rmse_after)
                    


if __name__ == '__main__':
    if args.seed > 0:
        np.random.seed(args.seed)

    main()
