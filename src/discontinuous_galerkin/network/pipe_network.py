import pdb
from discontinuous_galerkin.network.single_pipe import SinglePipe

class PipeNetwork():
    """Pipe network model class."""

    def __init__(
        self, 
        network_structure: dict,
        pipe_PDE_params: dict,
        pipe_DG_args: dict,
        ) -> None: 
        super().__init__()

        self.network_structure = network_structure
        self.pipe_PDE_params = pipe_PDE_params
        self.pipe_DG_args = pipe_DG_args
        
        self.step_size = self.pipe_DG_args[0]['time_integrator_args']['step_size']

        self.pipe_ids = list(self.pipe_DG_args.keys())

        self.pipe_sections = self.create_pipe_sections()

        self.n_pipes = len(self.pipe_sections)

    
    def _get_boundary_conditions(self):
        """Get boundary conditions from the topology."""

        BCs = {}
        for pipe_name, pipe_args in self.topology.items():
            BCs[pipe_name] = pipe_args['BCs']

        return BCs

    def create_pipe_sections(self):
        """Create DG models for pipe sections from the pipe_section_args."""
        
        pipe_sections = {}
        for pipe_id in self.pipe_ids:
            pipe_sections[pipe_id] = SinglePipe(
                self.pipe_PDE_params[pipe_id],
                **self.pipe_DG_args[pipe_id]
                )

        return pipe_sections