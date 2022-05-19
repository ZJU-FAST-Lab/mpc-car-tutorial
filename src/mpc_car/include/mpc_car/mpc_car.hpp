#pragma once
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Path.h>
#include <ros/ros.h>

#include <arc_spline/arc_spline.hpp>
#include <deque>
#include <iosqp/iosqp.hpp>

#include "lbfgs_raw.hpp"

namespace mpc_car {

static constexpr int n = 4;  // state x y phi v
static constexpr int m = 2;  // input a delta
typedef Eigen::Matrix<double, n, n> MatrixA;
typedef Eigen::Matrix<double, n, m> MatrixB;
typedef Eigen::Vector4d VectorG;
typedef Eigen::Vector4d VectorX;
typedef Eigen::Vector2d VectorU;

class MpcCar {
 private:
  ros::NodeHandle nh_;
  ros::Publisher ref_pub_, traj_pub_, traj_delay_pub_;
  bool init_ = false;

  double ll_;
  double dt_;
  double rho_;
  int N_;
  double rhoN_;

  double v_max_, a_max_, delta_max_, ddelta_max_;
  double delay_;

  arc_spline::ArcSpline s_;
  double desired_v_;

  osqp::IOSQP qpSolver_;

  std::vector<VectorX> predictState_;
  std::vector<Eigen::VectorXd> reference_states_;
  std::vector<VectorU> predictInput_;
  std::deque<VectorU> historyInput_;
  int history_length_;
  VectorX x0_observe_;

  MatrixA Ad_;
  MatrixB Bd_;
  VectorG gd_;
  // x_{k+1} = Ad * x_{k} + Bd * u_k + gd

  Eigen::SparseMatrix<double> P_, q_, A_, l_, u_;
  // Eigen::SparseMatrix<double> P0_, q0_;
  Eigen::SparseMatrix<double> Cx_, lx_, ux_;  // p, v constrains
  Eigen::SparseMatrix<double> Cu_, lu_, uu_;  // a delta vs constrains
  Eigen::SparseMatrix<double> Qx_;

  void linearization(const double& phi,
                     const double& v,
                     const double& delta) {
    // TODO: set values to Ad_, Bd_, gd_
    Ad_(0, 2) = -v * sin(phi) * dt_;
    Ad_(0, 3) = cos(phi) * dt_;
    Ad_(1, 2) = v * cos(phi) * dt_;
    Ad_(1, 3) = sin(phi) * dt_;
    Ad_(2, 3) = tan(delta) / ll_ * dt_;
    Bd_(2, 1) = v / ll_ / cos(delta) / cos(delta) * dt_;
    gd_(0) = v * sin(phi) * dt_ * phi;
    gd_(1) = -v * cos(phi) * dt_ * phi;
    gd_(2) = -v / ll_ / cos(delta) / cos(delta) * dt_ * delta;
    return;
  }

  void calLinPoint(const double& s0, double& phi, double& v, double& delta) {
    Eigen::Vector2d dxy = s_(s0, 1);
    Eigen::Vector2d ddxy = s_(s0, 2);
    double dx = dxy.x();
    double dy = dxy.y();
    double ddx = ddxy.x();
    double ddy = ddxy.y();
    double dphi = (ddy * dx - dy * ddx) / (dx * dx + dy * dy);
    phi = atan2(dy, dx);
    v = desired_v_;
    delta = atan2(ll_ * dphi, 1.0);
  }

  inline VectorX diff(const VectorX& state,
                      const VectorU& input) const {
    VectorX ds;
    double phi = state(2);
    double v = state(3);
    double a = input(0);
    double delta = input(1);
    ds(0) = v * cos(phi);
    ds(1) = v * sin(phi);
    ds(2) = v / ll_ * tan(delta);
    ds(3) = a;
    return ds;
  }

  inline void step(VectorX& state, const VectorU& input, const double dt) const {
    // Runge–Kutta
    VectorX k1 = diff(state, input);
    VectorX k2 = diff(state + k1 * dt / 2, input);
    VectorX k3 = diff(state + k2 * dt / 2, input);
    VectorX k4 = diff(state + k3 * dt, input);
    state = state + (k1 + k2 * 2 + k3 * 2 + k4) * dt / 6;
  }

  VectorX compensateDelay(const VectorX& x0) {
    VectorX x0_delay = x0;
    // TODO: compensate delay
    double dt = 0.001;
    for (double t = delay_; t > 0; t -= dt) {
      int i = std::ceil(t / dt_);
      VectorU input = historyInput_[history_length_ - i];
      step(x0_delay, input, dt);
    }
    return x0_delay;
  }

 public:
  MpcCar(ros::NodeHandle& nh) : nh_(nh) {
    // load map
    std::vector<double> track_points_x, track_points_y;
    nh.getParam("track_points_x", track_points_x);
    nh.getParam("track_points_y", track_points_y);
    nh.getParam("desired_v", desired_v_);
    s_.setWayPoints(track_points_x, track_points_y);
    // load parameters
    nh.getParam("ll", ll_);
    nh.getParam("dt", dt_);
    nh.getParam("rho", rho_);
    nh.getParam("N", N_);
    nh.getParam("rhoN", rhoN_);
    nh.getParam("v_max", v_max_);
    nh.getParam("a_max", a_max_);
    nh.getParam("delta_max", delta_max_);
    nh.getParam("ddelta_max", ddelta_max_);
    nh.getParam("delay", delay_);
    history_length_ = std::ceil(delay_ / dt_);

    ref_pub_ = nh.advertise<nav_msgs::Path>("reference_path", 1);
    traj_pub_ = nh.advertise<nav_msgs::Path>("traj", 1);
    traj_delay_pub_ = nh.advertise<nav_msgs::Path>("traj_delay", 1);

    // TODO: set initial value of Ad, Bd, gd
    Ad_.setIdentity();  // Ad for instance
    Bd_.setZero();
    Bd_(3, 0) = dt_;
    gd_.setZero();
    // set size of sparse matrices
    P_.resize(m * N_, m * N_);
    q_.resize(m * N_, 1);
    Qx_.resize(n * N_, n * N_);
    // stage cost
    Qx_.setIdentity();
    for (int i = 1; i < N_; ++i) {
      Qx_.coeffRef(i * n - 2, i * n - 2) = rho_;
      Qx_.coeffRef(i * n - 1, i * n - 1) = 0;
    }
    Qx_.coeffRef(N_ * n - 4, N_ * n - 4) = rhoN_;
    Qx_.coeffRef(N_ * n - 3, N_ * n - 3) = rhoN_;
    Qx_.coeffRef(N_ * n - 2, N_ * n - 2) = rhoN_ * rho_;
    int n_cons = 4;  // v a delta ddelta
    A_.resize(n_cons * N_, m * N_);
    l_.resize(n_cons * N_, 1);
    u_.resize(n_cons * N_, 1);
    // v constrains
    Cx_.resize(1 * N_, n * N_);
    lx_.resize(1 * N_, 1);
    ux_.resize(1 * N_, 1);
    // a delta constrains
    Cu_.resize(3 * N_, m * N_);
    lu_.resize(3 * N_, 1);
    uu_.resize(3 * N_, 1);
    // set lower and upper boundaries
    for (int i = 0; i < N_; ++i) {
      // TODO: set stage constraints of inputs (a, delta, ddelta)
      // -a_max <= a <= a_max for instance:
      Cu_.coeffRef(i * 3 + 0, i * m + 0) = 1;
      Cu_.coeffRef(i * 3 + 1, i * m + 1) = 1;
      Cu_.coeffRef(i * 3 + 2, i * m + 1) = 1;
      lu_.coeffRef(i * 3 + 0, 0) = -a_max_;
      uu_.coeffRef(i * 3 + 0, 0) = a_max_;
      lu_.coeffRef(i * 3 + 1, 0) = -delta_max_;
      uu_.coeffRef(i * 3 + 1, 0) = delta_max_;
      lu_.coeffRef(i * 3 + 2, 0) = -ddelta_max_ * dt_;
      uu_.coeffRef(i * 3 + 2, 0) = ddelta_max_ * dt_;
      if (i > 0) {
        Cu_.coeffRef(i * 3 + 2, (i - 1) * m + 1) = -1;
      }

      // TODO: set stage constraints of states (v)
      // -v_max <= v <= v_max
      Cx_.coeffRef(i, i * n + 3) = 1;
      lx_.coeffRef(i, 0) = -0.1;
      ux_.coeffRef(i, 0) = v_max_;
    }
    // set predict mats size
    predictState_.resize(N_);
    predictInput_.resize(N_);
    for (int i = 0; i < N_; ++i) {
      predictInput_[i].setZero();
    }
    for (int i = 0; i < history_length_; ++i) {
      historyInput_.emplace_back(0, 0);
    }
  }

  int solveQP(const VectorX& x0_observe) {
    x0_observe_ = x0_observe;
    historyInput_.pop_front();
    historyInput_.push_back(predictInput_.front());
    lu_.coeffRef(2, 0) = predictInput_.front()(1) - ddelta_max_ * dt_;
    uu_.coeffRef(2, 0) = predictInput_.front()(1) + ddelta_max_ * dt_;
    VectorX x0 = compensateDelay(x0_observe_);
    // set BB, AA, gg
    Eigen::MatrixXd BB, AA, gg;
    BB.setZero(n * N_, m * N_);
    AA.setZero(n * N_, n);
    gg.setZero(n * N_, 1);
    double s0 = s_.findS(x0.head(2));
    double phi, v, delta;
    double last_phi = x0(2);
    Eigen::SparseMatrix<double> qx;
    qx.resize(n * N_, 1);
    for (int i = 0; i < N_; ++i) {
      calLinPoint(s0, phi, v, delta);
      if (phi - last_phi > M_PI) {
        phi -= 2 * M_PI;
      } else if (phi - last_phi < -M_PI) {
        phi += 2 * M_PI;
      }
      last_phi = phi;
      if (init_) {
        double phii = predictState_[i](2);
        v = predictState_[i](3);
        delta = predictInput_[i](1);
        if (phii - last_phi > M_PI) {
          phii -= 2 * M_PI;
        } else if (phii - last_phi < -M_PI) {
          phii += 2 * M_PI;
        }
        last_phi = phii;
        linearization(phii, v, delta);
      } else {
        linearization(phi, v, delta);
      }
      // calculate big state-space matrices
      /* *                BB                AA
       * x1    /       B    0  ... 0 \    /   A \
       * x2    |      AB    B  ... 0 |    |  A2 |
       * x3  = |    A^2B   AB  ... 0 |u + | ... |x0 + gg
       * ...   |     ...  ...  ... 0 |    | ... |
       * xN    \A^(n-1)B  ...  ... B /    \ A^N /
       *
       *     X = BB * U + AA * x0 + gg
       * */
      if (i == 0) {
        BB.block(0, 0, n, m) = Bd_;
        AA.block(0, 0, n, n) = Ad_;
        gg.block(0, 0, n, 1) = gd_;
      } else {
        // TODO: set BB AA gg
        BB.block(n * i, 0, n, m * N_) = Ad_ * BB.block(n * (i - 1), 0, n, m * N_);
        BB.block(n * i, m * i, n, m) = Bd_;
        AA.block(n * i, 0, n, n) = Ad_ * AA.block(n * (i - 1), 0, n, n);
        gg.block(n * i, 0, n, 1) = Ad_ * gg.block(n * (i - 1), 0, n, 1) + gd_;
      }
      // TODO: set qx
      Eigen::Vector2d xy = s_(s0);  // reference (x_r, y_r)
      qx.coeffRef(i * n + 0, 0) = -xy.x();
      qx.coeffRef(i * n + 1, 0) = -xy.y();
      qx.coeffRef(i * n + 2, 0) = -rho_ * phi;
      // std::cout << "phi[" << i << "]: " << phi << std::endl;
      if (i == N_ - 1) {
        qx.coeffRef(i * n + 0, 0) *= rhoN_;
        qx.coeffRef(i * n + 1, 0) *= rhoN_;
        qx.coeffRef(i * n + 2, 0) *= rhoN_;
      }
      s0 += desired_v_ * dt_;
      s0 = s0 < s_.arcL() ? s0 : s_.arcL();
    }
    Eigen::SparseMatrix<double> BB_sparse = BB.sparseView();
    Eigen::SparseMatrix<double> AA_sparse = AA.sparseView();
    Eigen::SparseMatrix<double> gg_sparse = gg.sparseView();
    Eigen::SparseMatrix<double> x0_sparse = x0.sparseView();

    Eigen::SparseMatrix<double> Cx = Cx_ * BB_sparse;
    Eigen::SparseMatrix<double> lx = lx_ - Cx_ * AA_sparse * x0_sparse - Cx_ * gg_sparse;
    Eigen::SparseMatrix<double> ux = ux_ - Cx_ * AA_sparse * x0_sparse - Cx_ * gg_sparse;
    Eigen::SparseMatrix<double> A_T = A_.transpose();
    A_T.middleCols(0, Cx.rows()) = Cx.transpose();
    A_T.middleCols(Cx.rows(), Cu_.rows()) = Cu_.transpose();
    A_ = A_T.transpose();
    for (int i = 0; i < lx.rows(); ++i) {
      l_.coeffRef(i, 0) = lx.coeff(i, 0);
      u_.coeffRef(i, 0) = ux.coeff(i, 0);
    }
    for (int i = 0; i < lu_.rows(); ++i) {
      l_.coeffRef(i + lx.rows(), 0) = lu_.coeff(i, 0);
      u_.coeffRef(i + lx.rows(), 0) = uu_.coeff(i, 0);
    }
    Eigen::SparseMatrix<double> BBT_sparse = BB_sparse.transpose();
    P_ = BBT_sparse * Qx_ * BB_sparse;
    q_ = BBT_sparse * Qx_.transpose() * (AA_sparse * x0_sparse + gg_sparse) + BBT_sparse * qx;
    // osqp
    Eigen::VectorXd q_d = q_.toDense();
    Eigen::VectorXd l_d = l_.toDense();
    Eigen::VectorXd u_d = u_.toDense();
    qpSolver_.setMats(P_, q_d, A_, l_d, u_d);
    qpSolver_.solve();
    int ret = qpSolver_.getStatus();
    if (ret != 1) {
      ROS_ERROR("fail to solve QP!");
      return ret;
    }
    Eigen::VectorXd sol = qpSolver_.getPrimalSol();
    Eigen::MatrixXd solMat = Eigen::Map<const Eigen::MatrixXd>(sol.data(), m, N_);
    Eigen::VectorXd solState = BB * sol + AA * x0 + gg;
    Eigen::MatrixXd predictMat = Eigen::Map<const Eigen::MatrixXd>(solState.data(), n, N_);

    for (int i = 0; i < N_; ++i) {
      predictInput_[i] = solMat.col(i);
      predictState_[i] = predictMat.col(i);
    }
    init_ = true;
    return ret;
  }

  void getPredictXU(double t, VectorX& state, VectorU& input) {
    if (t <= dt_) {
      state = predictState_.front();
      input = predictInput_.front();
      return;
    }
    int horizon = std::floor(t / dt_);
    double dt = t - horizon * dt_;
    state = predictState_[horizon - 1];
    input = predictInput_[horizon - 1];
    double phi = state(2);
    double v = state(3);
    double a = input(0);
    double delta = input(1);
    state(0) += dt * v * cos(phi);
    state(1) += dt * v * sin(phi);
    state(2) += dt * v / ll_ * tan(delta);
    state(3) += dt * a;
  }

  void forward(const VectorX& xk,
               const VectorU& uk,
               VectorX& xk_1) {
    // const auto& x = xk(0);
    // const auto& y = xk(1);
    const auto& phi = xk(2);
    const auto& v = xk(3);
    const auto& a = uk(0);
    const auto& delta = uk(1);
    xk_1 = xk;
    xk_1(0) += v * std::cos(phi) * dt_;
    xk_1(1) += v * std::sin(phi) * dt_;
    xk_1(2) += v / ll_ * std::tan(delta) * dt_;
    xk_1(3) += a * dt_;
  }
  void backward(const VectorX& xk,
                const VectorU& uk,
                const VectorX& grad_xk_1,
                VectorX& grad_xk,
                VectorU& grad_uk) {
    // const auto& x = xk(0);
    // const auto& y = xk(1);
    const auto& phi = xk(2);
    const auto& v = xk(3);
    // const auto& a = uk(0);
    const auto& delta = uk(1);
    const auto& grad_x_1 = grad_xk_1(0);
    const auto& grad_y_1 = grad_xk_1(1);
    const auto& grad_phi_1 = grad_xk_1(2);
    const auto& grad_v_1 = grad_xk_1(3);
    // auto& grad_x = grad_xk(0);
    // auto& grad_y = grad_xk(1);
    auto& grad_phi = grad_xk(2);
    auto& grad_v = grad_xk(3);
    auto& grad_a = grad_uk(0);
    auto& grad_delta = grad_uk(1);
    grad_xk = grad_xk_1;
    grad_uk.setZero();
    grad_v += grad_x_1 * std::cos(phi) * dt_;
    grad_phi += grad_x_1 * v * (-std::sin(phi)) * dt_;
    grad_v += grad_y_1 * std::sin(phi) * dt_;
    grad_phi += grad_y_1 * v * std::cos(phi) * dt_;
    grad_v += grad_phi_1 * std::tan(delta) / ll_ * dt_;
    grad_delta += grad_phi_1 * v / ll_ / std::cos(delta) / std::cos(delta) * dt_;
    grad_a += grad_v_1 * dt_;
  }

  double box_constrant(const double& x,
                       const double& l,
                       const double& u,
                       double& grad) {
    double rho = 1e4;
    double lpen = l - x;
    double upen = x - u;
    if (lpen > 0) {
      double lpen2 = lpen * lpen;
      grad = -rho * 3 * lpen2;
      return rho * lpen2 * lpen;
    } else if (upen > 0) {
      double upen2 = upen * upen;
      grad = rho * 3 * upen2;
      return rho * upen2 * upen;
    } else {
      grad = 0;
      return 0;
    }
  }

  double stage_cost_gradient(const int& k,
                             const VectorX& x,
                             VectorX& grad_x) {
    const Eigen::Vector3d& x_r = reference_states_[k];
    Eigen::Vector3d dx = x.head(3) - x_r;
    grad_x.head(3) = 2 * dx;
    grad_x(3) = 0;
    double cost = dx.squaredNorm();
    // TODO: penalty constraints
    double grad_v = 0;
    cost += box_constrant(x(3), -0.1, v_max_, grad_v);
    grad_x(3) += grad_v;
    return cost;
  }

  static inline double objectiveFunc(void* ptrObj,
                                     const double* x,
                                     double* grad,
                                     const int n) {
    // std::cout << "\033[32m ************************************** \033[0m" << std::endl;
    MpcCar& obj = *(MpcCar*)ptrObj;
    Eigen::Map<const Eigen::MatrixXd> inputs(x, m, obj.N_);
    Eigen::Map<Eigen::MatrixXd> grad_inputs(grad, m, obj.N_);

    // forward propogate
    std::vector<VectorX> states(obj.N_ + 1);
    states[0] = obj.x0_observe_;
    VectorX xk_1 = obj.x0_observe_;
    for (int i = 0; i < obj.N_; ++i) {
      obj.forward(states[i], inputs.col(i), xk_1);
      states[i + 1] = xk_1;
    }
    // cost and gradient of states
    double total_cost = 0;
    VectorX grad_xk, grad_xk_1;
    VectorU grad_uk;
    grad_xk.setZero();
    for (int i = obj.N_ - 1; i >= 0; i--) {
      total_cost += obj.stage_cost_gradient(i, states[i + 1], grad_xk_1);
      grad_xk_1 = grad_xk_1 + grad_xk;
      obj.backward(states[i], inputs.col(i), grad_xk_1, grad_xk, grad_uk);
      grad_inputs.col(i) = grad_uk;
    }
    // cost and gradient of inputs
    for (int i = 0; i < obj.N_; ++i) {
      double a = inputs.col(i)(0);
      double delta = inputs.col(i)(1);
      double grad_a, grad_delta;
      total_cost += obj.box_constrant(a, -obj.a_max_, obj.a_max_, grad_a);
      grad_inputs.col(i)(0) += grad_a;
      total_cost += obj.box_constrant(delta, -obj.delta_max_, obj.delta_max_, grad_delta);
      grad_inputs.col(i)(1) += grad_delta;
    }
    for (int i = 0; i < obj.N_ - 1; ++i) {
      double delta_k = inputs.col(i)(1);
      double delta_k_1 = inputs.col(i + 1)(1);
      double ddelta = delta_k_1 - delta_k;
      double grad_ddelta;
      total_cost += obj.box_constrant(ddelta,
                                      -obj.ddelta_max_ * obj.dt_,
                                      obj.ddelta_max_ * obj.dt_,
                                      grad_ddelta);
      grad_inputs.col(i)(1) -= grad_ddelta;
      grad_inputs.col(i + 1)(1) += grad_ddelta;
    }
    return total_cost;
  }

  int solveNMPC(const VectorX& x0_observe) {
    historyInput_.pop_front();
    historyInput_.push_back(predictInput_.front());
    // x0_observe_ = x0_observe;
    x0_observe_ = compensateDelay(x0_observe);
    double s0 = s_.findS(x0_observe_.head(2));
    reference_states_.resize(N_);

    Eigen::Vector2d xy_r;
    double phi, last_phi = x0_observe_(2);
    for (int i = 0; i < N_; ++i) {
      s0 += desired_v_ * dt_;
      s0 = s0 < s_.arcL() ? s0 : s_.arcL();
      // calculate desired x,y.phi
      xy_r = s_(s0);
      Eigen::Vector2d dxy = s_(s0, 1);
      phi = std::atan2(dxy.y(), dxy.x());
      if (phi - last_phi > M_PI) {
        phi -= 2 * M_PI;
      } else if (phi - last_phi < -M_PI) {
        phi += 2 * M_PI;
      }
      last_phi = phi;
      reference_states_[i] = Eigen::Vector3d(xy_r.x(), xy_r.y(), phi);
    }
    double* x = new double[m * N_];
    Eigen::Map<Eigen::MatrixXd> inputs(x, m, N_);
    inputs.setZero();
    lbfgs::lbfgs_parameter_t lbfgs_params;
    lbfgs::lbfgs_load_default_parameters(&lbfgs_params);
    lbfgs_params.mem_size = 16;
    lbfgs_params.past = 3;
    lbfgs_params.g_epsilon = 0.0;
    lbfgs_params.min_step = 1e-32;
    lbfgs_params.delta = 1e-4;
    lbfgs_params.line_search_type = 0;
    double minObjective;
    auto ret = lbfgs::lbfgs_optimize(m * N_, x, &minObjective, &objectiveFunc, nullptr, nullptr, this, &lbfgs_params);
    std::cout << "\033[32m"
              << "ret: " << ret << "\033[0m" << std::endl;
    VectorX xk = x0_observe_, xk_1;
    for (int i = 0; i < N_; ++i) {
      predictInput_[i] = inputs.col(i);
      forward(xk, inputs.col(i), xk_1);
      predictState_[i] = xk_1;
      xk = xk_1;
    }
    return ret;
  }

  // visualization
  void visualization() {
    nav_msgs::Path msg;
    msg.header.frame_id = "world";
    msg.header.stamp = ros::Time::now();
    geometry_msgs::PoseStamped p;
    for (double s = 0; s < s_.arcL(); s += 0.01) {
      p.pose.position.x = s_(s).x();
      p.pose.position.y = s_(s).y();
      p.pose.position.z = 0.0;
      msg.poses.push_back(p);
    }
    ref_pub_.publish(msg);
    msg.poses.clear();
    for (int i = 0; i < N_; ++i) {
      p.pose.position.x = predictState_[i](0);
      p.pose.position.y = predictState_[i](1);
      p.pose.position.z = 0.0;
      msg.poses.push_back(p);
    }
    traj_pub_.publish(msg);
    msg.poses.clear();
    VectorX x0_delay = x0_observe_;
    double dt = 0.001;
    for (double t = delay_; t > 0; t -= dt) {
      int i = std::ceil(t / dt_);
      VectorU input = historyInput_[history_length_ - i];
      step(x0_delay, input, dt);
      p.pose.position.x = x0_delay(0);
      p.pose.position.y = x0_delay(1);
      p.pose.position.z = 0.0;
      msg.poses.push_back(p);
    }
    traj_delay_pub_.publish(msg);
  }
};

}  // namespace mpc_car