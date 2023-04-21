# Extended Position Based Dynamics in PyTorch
Based on the StVk Garment model, we develop an cloth simulation toy example using pytorch.

![XPBD + StVK + Bend Simulation.gif](./assets/XPBD%20%2B%20StVK%20%2B%20Bend%20Simulation.gif)

## Installation
This project is python and pytorch based and we've only tested it on ubuntu. Install dependecies with `pip install -r requirements.txt`

## Usage
- `simulate_global_stvk_bend` implements a global position based dynamics update (using backward euler). This global update is typically slower to converge on single node updates.
- `simulate_loss_stvk_bend` treats the cloth simulation as a optimization problem where energies are defined as optimization target for the network to reach equilibirum.
- `simulate_dynamic_stvk_bend` will combine the previous two types of simulation techniques and try to introduce more dynamics into the process (preserves momentum through position based dynamics)
- `visualize_open3d_animation` provides a cross-platform visualization of the simulation result (tested on windows too).

```shell
python simulate_loss_stvk_bend.py # will output tshirt-drop#loss.npz
python simulate_global_stvk_bend.py # will output tshirt-drop#global.npz
python simulate_dynamic_stvk_bend.py # will output tshirt-drop#dynamic.npz

python visualize_open3d_animation.py "tshirt-drop#loss.npz" 10  # will render the stvk loss simulation in 10 fps (in terms of simulation) (6 times slower)
python visualize_open3d_animation.py "tshirt-drop#global.npz" 10  # will render the global xpbd simulation in 10 fps (in terms of simulation) (6 times slower)
python visualize_open3d_animation.py "tshirt-drop#dynamic.npz" 10  # will render the dynamic xpbd + loss simulation in 10 fps (in terms of simulation) (6 times slower)
```