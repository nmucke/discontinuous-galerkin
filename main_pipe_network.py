import time
import numpy as np
import yaml
from discontinuous_galerkin.base.base_model import BaseModel
from discontinuous_galerkin.network.pipe_network import PipeNetwork
from discontinuous_galerkin.start_up_routines.start_up_1D import StartUp1D
import matplotlib.pyplot as plt
import pdb

from scipy.optimize import fsolve

from matplotlib.animation import FuncAnimation

NUM_PIPE_SECTIONS = 2

# load network config
CONFIG = 'configs/pipe_network.yml'
with open(CONFIG, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

SINGLE_PIPE_PDE_PARAMS = {
    'L': 2000,
    'd': 0.508,
    'c': 308.,
    'rho_ref': 52.67,
    'p_amb': 101325.,
    'e': 1e-8,
    'mu': 1.2e-5
}

DG_args = {}
for pipe_id in range(NUM_PIPE_SECTIONS):
    DG_args[pipe_id] = config['pipe_DG_args']

PDE_params = {}
for pipe_id in range(NUM_PIPE_SECTIONS):
    PDE_params[pipe_id] = SINGLE_PIPE_PDE_PARAMS




if __name__ == '__main__':

    network_structure = {}

    pipe_DG = PipeNetwork(
        network_structure=network_structure,
        pipe_PDE_params=PDE_params,
        pipe_DG_args=DG_args,        
    )
    pdb.set_trace()



    t_final = 5.0
    sol, t_vec = pipe_DG.solve(
        t=0, 
        t_final=t_final, 
    )

    x = np.linspace(0, 2000, 2000)

    u = np.zeros((len(x), len(t_vec)))
    rho = np.zeros((len(x), len(t_vec)))
    for t in range(sol.shape[-1]):
        rho[:, t] = pipe_DG.evaluate_solution(x, sol_nodal=sol[0, :, t])
        u[:, t] = pipe_DG.evaluate_solution(x, sol_nodal=sol[1, :, t])
    rho = rho / pipe_DG.A
    u = u / rho / pipe_DG.A

    print(rho[0, :])
    
    t_vec = np.arange(0, u.shape[1])

    plt.figure()
    plt.imshow(u, extent=[0, t_final, 2000, 0], aspect='auto')
    plt.colorbar()
    plt.show()

    plt.figure()
    plt.plot(x, u[:, 0], label='u', linewidth=2)
    plt.plot(x, u[:, -1], label='u', linewidth=2)
    plt.grid()
    plt.legend()
    plt.savefig('pipeflow.png')
    plt.show()

    fig, ax = plt.subplots()
    xdata, ydata = [], []
    ln, = ax.plot([], [], lw=3, animated=True)

    def init():
        ax.set_xlim(0, 2000)
        ax.set_ylim(u.min(), u.max())
        return ln,

    def update(frame):
        xdata.append(x)
        ydata.append(u[:, frame])
        ln.set_data(x, u[:, frame])
        return ln,

    ani = FuncAnimation(
        fig,
        update,
        frames=t_vec,
        init_func=init, 
        blit=True,
        interval=10,
        )
    ani.save('pipeflow.gif', fps=30)
    plt.show()