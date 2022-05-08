
#ifndef G2O_OBJECT_SLAM_TYPES
#define G2O_OBJECT_SLAM_TYPES

#include <Eigen/Geometry>
#include <iostream>

#include "g2o/core/base_vertex.h"
#include "g2o/core/base_unary_edge.h"
#include "g2o/core/base_binary_edge.h"
#include "g2o/core/base_multi_edge.h"
#include "g2o/types/sba/sbacam.h"
#include "g2o/types/sba/g2o_types_sba_api.h"
#include "g2o/types/sba/types_six_dof_expmap.h"

namespace g2o {

/**
 * \brief Object edge. Constrains the object pose and camera pose by projecting the CAD model
 * keypoints from object frame into camera frame and comparing the UV coordinates.
 * Each measurement is a single UV coordinate from the keypoint network, so just make
 * a new edge for each keypoint.
 */
class EdgeSE3ProjectFromObject: 
        public  g2o::BaseBinaryEdge<2, Eigen::Vector2d, g2o::VertexSE3Expmap /* Object */,
        g2o::VertexSE3Expmap /* Camera */> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    EdgeSE3ProjectFromObject();
    
    EdgeSE3ProjectFromObject(const Eigen::Vector4d &cam_k_, 
                             const Eigen::Vector3d &p_inO_) {
        set_info(cam_k_, p_inO_);
    }

    inline void set_info(const Eigen::Vector4d &cam_k_, 
                         const Eigen::Vector3d &p_inO_) {
        cam_k = cam_k_;
        p_inO = p_inO_;
        set = true;
    }

    bool read(std::istream& is);

    bool write(std::ostream& os) const;

    void computeError();

    bool isDepthPositive();

    virtual void linearizeOplus();

private:
    bool set = false;
    Eigen::Vector4d cam_k; // fx fy cx cy
    Eigen::Vector3d p_inO; // The corresponding CAD model point in object frame
};

class EdgeSE3ProjectFromFixedObject: 
        public  g2o::BaseUnaryEdge<2, Eigen::Vector2d, g2o::VertexSE3Expmap /* Camera */> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    EdgeSE3ProjectFromFixedObject();
    
    EdgeSE3ProjectFromFixedObject(const Eigen::Vector4d &cam_k_, 
                                  const Eigen::Vector3d &p_inO_,
                                  const Eigen::Matrix<double,3,4> &T_OtoG_) {
        set_info(cam_k_, p_inO_, T_OtoG_);
    }

    inline void set_info(const Eigen::Vector4d &cam_k_, 
                         const Eigen::Vector3d &p_inO_,
                         const Eigen::Matrix<double,3,4> &T_OtoG_) {
        cam_k = cam_k_;
        p_inG = T_OtoG_.block<3,3>(0,0) * p_inO_ + T_OtoG_.block<3,1>(0,3);
        set = true;
    }

    bool read(std::istream& is);

    bool write(std::ostream& os) const;

    void computeError();

    bool isDepthPositive();

    virtual void linearizeOplus();

private:
    bool set = false;
    Eigen::Vector4d cam_k; // fx fy cx cy
    Eigen::Vector3d p_inG; // Object point in G
};

}
#endif
