## How to Run
```
./install_tools.sh
catkin_make -j1
source devel/setup.zsh
roslaunch mpc_car simulation.launch
```
## How to Tune Parameters
```
./src/mpc_car/config/mpc_car.yaml -> mpc parameters
./src/car_simulator/config/car_simulator.yaml -> initial states (in simulation)
```

## Object and Constraints
> Implement MPC of tracking the reference trajectory in C++;
```
min  J = \sum_{i=0}^N (x-x_r)^2+ (y-y_r)^2 + rho * (phi-phi_r)^2

s.t. -0.1 <= v_k <= v_max
     |a_k| <= a_max
     |delta_k| <= delta_max
     |delta_{k+1} - delta_k| <= ddelta_max * dt
```

## NMPC or QP

```
./src/car_simulator/src/car_simulator_nodelet.cpp
```
>ret = mpcPtr_->solveNMPC(state_);

>ret = mpcPtr_->solveQP(state_);

QP is solved by OSQP and NMPC is solved by lbfgs.
