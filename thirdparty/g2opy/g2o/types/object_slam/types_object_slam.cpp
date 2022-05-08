
#include "types_object_slam.h"
#include <iostream>

#include "g2o/core/factory.h"
#include "g2o/stuff/macros.h"

namespace g2o {

G2O_REGISTER_TYPE_GROUP(object_slam);

G2O_REGISTER_TYPE(EDGE_SE3_PROJECT_FROM_OBJECT, EdgeSE3ProjectFromObject);
G2O_REGISTER_TYPE(EDGE_SE3_PROJECT_FROM_FIXED_OBJECT, EdgeSE3ProjectFromFixedObject);

EdgeSE3ProjectFromObject::EdgeSE3ProjectFromObject()
        : BaseBinaryEdge<2, Eigen::Vector2d, g2o::VertexSE3Expmap, g2o::VertexSE3Expmap>(),
          cam_k(Eigen::Vector4d::Zero()), p_inO(Eigen::Vector3d::Zero()) {}

bool EdgeSE3ProjectFromObject::read(std::istream& is){
    for (int i=0; i<2; i++){
        is >> _measurement[i];
    }
    for (int i=0; i<2; i++)
        for (int j=i; j<2; j++) {
            is >> information()(i,j);
            if (i!=j)
                information()(j,i)=information()(i,j);
        }
    return true;
}

bool EdgeSE3ProjectFromObject::write(std::ostream& os) const {

    for (int i=0; i<2; i++){
        os << measurement()[i] << " ";
    }

    for (int i=0; i<2; i++)
        for (int j=i; j<2; j++){
            os << " " <<  information()(i,j);
        }
    return os.good();
}

void EdgeSE3ProjectFromObject::computeError()  {
    if (!set) {
        std::cerr << "Please call EdgeSE3ProjectFromObject::set_info before optimizing!\n";
        exit(EXIT_FAILURE);
    }
    // Maps points in world frame into camera frame
    const g2o::VertexSE3Expmap* vTcw = static_cast<const g2o::VertexSE3Expmap*>(_vertices[1]);
    // Maps points in object frame into world frame
    const g2o::VertexSE3Expmap* vTwo = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
    Eigen::Vector2d uv_meas(_measurement); // measured UV of keypoint
    Eigen::Vector3d p_inC = vTcw->estimate().map(vTwo->estimate().map(p_inO));
    Eigen::Vector2d uv_estim;
    uv_estim[0] = cam_k[0] * p_inC[0] / p_inC[2] + cam_k[2];
    uv_estim[1] = cam_k[1] * p_inC[1] / p_inC[2] + cam_k[3];
    _error = uv_meas - uv_estim;
}

bool EdgeSE3ProjectFromObject::isDepthPositive() {
    // Maps points in world frame into camera frame
    const g2o::VertexSE3Expmap* vTcw = static_cast<const g2o::VertexSE3Expmap*>(_vertices[1]);
    // Maps points in object frame into world frame
    const g2o::VertexSE3Expmap* vTwo = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
    return vTcw->estimate().map( vTwo->estimate().map(p_inO) )(2)>0.0;
}

void EdgeSE3ProjectFromObject::linearizeOplus() {
    g2o::VertexSE3Expmap *vTcw = static_cast<g2o::VertexSE3Expmap*>(_vertices[1]);
    g2o::SE3Quat Tcw(vTcw->estimate());
    g2o::VertexSE3Expmap *vTwo = static_cast<g2o::VertexSE3Expmap*>(_vertices[0]);
    g2o::SE3Quat Two(vTwo->estimate());
    Eigen::Vector3d xyz = Two.map(p_inO);
    Eigen::Vector3d xyz_trans = Tcw.map(xyz);

    Eigen::Matrix<double, 2, 3> projectJac_;
    projectJac_(0, 0) = cam_k[0] / xyz_trans[2];
    projectJac_(0, 1) = 0.f;
    projectJac_(0, 2) = -cam_k[0] * xyz_trans[0] / (xyz_trans[2] * xyz_trans[2]);
    projectJac_(1, 0) = 0.f;
    projectJac_(1, 1) = cam_k[1] / xyz_trans[2];
    projectJac_(1, 2) = -cam_k[1] * xyz_trans[1] / (xyz_trans[2] * xyz_trans[2]);
    Eigen::Matrix<double, 2, 3> projectJac = -projectJac_;

    double x = xyz[0];
    double y = xyz[1];
    double z = xyz[2];
    Eigen::Matrix<double,3,6> woSE3deriv;
    woSE3deriv << 0.f,  z,  -y,  1.f, 0.f, 0.f,
                  -z , 0.f,  x,  0.f, 1.f, 0.f,
                   y , -x , 0.f, 0.f, 0.f, 1.f;
    
    x = xyz_trans[0];
    y = xyz_trans[1];
    z = xyz_trans[2];
    Eigen::Matrix<double,3,6> cwSE3deriv;
    cwSE3deriv << 0.f,  z,  -y,  1.f, 0.f, 0.f,
                  -z , 0.f,  x,  0.f, 1.f, 0.f,
                   y , -x , 0.f, 0.f, 0.f, 1.f;

    // Jacobian of residual wrt object state
    _jacobianOplusXi = projectJac * Tcw.rotation().toRotationMatrix() * woSE3deriv;

    // Jacobian of residual wrt camera pose
    _jacobianOplusXj = projectJac * cwSE3deriv;

    // DEBUG to compare the analytical jacobians to numerical ones.
    /*
    Eigen::Matrix<double,2,6> analytic_i;
    Eigen::Matrix<double,2,6> analytic_j;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 6; j++) {
            analytic_i(i,j) = _jacobianOplusXi(i,j);
            analytic_j(i,j) = _jacobianOplusXj(i,j);
        }
    }
    BaseBinaryEdge::linearizeOplus(); // Populate jacobians with finite diff solution
    std::cout << "Analytic Jxi error: \n" << _jacobianOplusXi-analytic_i << "\n";
    std::cout << "Analytic Jxj error: \n" << _jacobianOplusXj-analytic_j << "\n\n\n";
    */
}

// ====================================================================================
EdgeSE3ProjectFromFixedObject::EdgeSE3ProjectFromFixedObject()
        : BaseUnaryEdge<2, Eigen::Vector2d, g2o::VertexSE3Expmap>(),
          cam_k(Eigen::Vector4d::Zero()), p_inG(Eigen::Vector3d::Zero()) {}

bool EdgeSE3ProjectFromFixedObject::read(std::istream& is){
    for (int i=0; i<2; i++){
        is >> _measurement[i];
    }
    for (int i=0; i<2; i++)
        for (int j=i; j<2; j++) {
            is >> information()(i,j);
            if (i!=j)
                information()(j,i)=information()(i,j);
        }
    return true;
}

bool EdgeSE3ProjectFromFixedObject::write(std::ostream& os) const {

    for (int i=0; i<2; i++){
        os << measurement()[i] << " ";
    }

    for (int i=0; i<2; i++)
        for (int j=i; j<2; j++){
            os << " " <<  information()(i,j);
        }
    return os.good();
}

void EdgeSE3ProjectFromFixedObject::computeError()  {
    if (!set) {
        std::cerr << "Please call EdgeSE3ProjectFromFixedObject::set_info before optimizing!\n";
        exit(EXIT_FAILURE);
    }
    // Maps points in world frame into camera frame
    const g2o::VertexSE3Expmap* vTcw = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
    Eigen::Vector2d uv_meas(_measurement); // measured UV of keypoint
    Eigen::Vector3d p_inC = vTcw->estimate().map(p_inG);
    Eigen::Vector2d uv_estim;
    uv_estim[0] = cam_k[0] * p_inC[0] / p_inC[2] + cam_k[2];
    uv_estim[1] = cam_k[1] * p_inC[1] / p_inC[2] + cam_k[3];
    _error = uv_meas - uv_estim;
}

bool EdgeSE3ProjectFromFixedObject::isDepthPositive() {
    // Maps points in world frame into camera frame
    const g2o::VertexSE3Expmap* vTcw = static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
    return vTcw->estimate().map(p_inG)(2) > 0.0;
}

void EdgeSE3ProjectFromFixedObject::linearizeOplus() {
    g2o::VertexSE3Expmap *vTcw = static_cast<g2o::VertexSE3Expmap*>(_vertices[0]);
    g2o::SE3Quat Tcw(vTcw->estimate());
    Eigen::Vector3d xyz_trans = Tcw.map(p_inG);

    Eigen::Matrix<double, 2, 3> projectJac_;
    projectJac_(0, 0) = cam_k[0] / xyz_trans[2];
    projectJac_(0, 1) = 0.f;
    projectJac_(0, 2) = -cam_k[0] * xyz_trans[0] / (xyz_trans[2] * xyz_trans[2]);
    projectJac_(1, 0) = 0.f;
    projectJac_(1, 1) = cam_k[1] / xyz_trans[2];
    projectJac_(1, 2) = -cam_k[1] * xyz_trans[1] / (xyz_trans[2] * xyz_trans[2]);
    Eigen::Matrix<double, 2, 3> projectJac = -projectJac_;

    double x = xyz_trans[0];
    double y = xyz_trans[1];
    double z = xyz_trans[2];
    Eigen::Matrix<double,3,6> cwSE3deriv;
    cwSE3deriv << 0.f,  z,  -y,  1.f, 0.f, 0.f,
                  -z , 0.f,  x,  0.f, 1.f, 0.f,
                   y , -x , 0.f, 0.f, 0.f, 1.f;

    // Jacobian of residual wrt camera pose
    _jacobianOplusXi = projectJac * cwSE3deriv;
}

}

