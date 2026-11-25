import importlib.util
from pathlib import Path

import sdsl


module_path = Path(__file__).resolve().parent / "experiment_movement.py"
spec = importlib.util.spec_from_file_location("experiment_movement", module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_state_inside_dominant_belief_voxel_is_converged():
    voxels = [
        sdsl.Voxel_R3(
            sdsl.R3(0.0, 0.0, 0.0),
            sdsl.R3(2.0, 2.0, 1.0),
        ),
        sdsl.Voxel_R3(
            sdsl.R3(10.0, 10.0, 0.0),
            sdsl.R3(12.0, 12.0, 1.0),
        ),
    ]
    belief = [0.9, 0.1]
    assert module.state_is_in_dominant_belief_voxel(voxels, belief, 1.0, 1.0)


def test_state_outside_dominant_belief_voxel_is_not_converged():
    voxels = [
        sdsl.Voxel_R3(
            sdsl.R3(0.0, 0.0, 0.0),
            sdsl.R3(2.0, 2.0, 1.0),
        ),
        sdsl.Voxel_R3(
            sdsl.R3(10.0, 10.0, 0.0),
            sdsl.R3(12.0, 12.0, 1.0),
        ),
    ]
    belief = [0.9, 0.1]
    assert not module.state_is_in_dominant_belief_voxel(voxels, belief, 11.0, 11.0)
