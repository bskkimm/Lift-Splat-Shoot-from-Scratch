import ast
import json
import os
import subprocess
import sys
from pathlib import Path


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks/lss_e2e_walkthrough.ipynb"


def test_notebook_defines_complete_architecture_without_package_imports():
    notebook = json.loads(NOTEBOOK.read_text())
    code = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
    tree = ast.parse(code)
    definitions = [node.name for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))]
    expected = {"CameraEncoder", "BEVEncoder", "VoxelPool", "LSSSegmentation", "LSSPlanner",
                "trajectory_logits", "metric_trajectory_cells", "make_frustum", "camera_to_ego"}
    assert expected.issubset(definitions)
    assert len(definitions) == len(set(definitions)), "architecture definitions must not be shadowed"
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.module != "lss" and not (node.module or "").startswith("lss.")
        if isinstance(node, ast.Import):
            assert all(alias.name != "lss" and not alias.name.startswith("lss.") for alias in node.names)


def test_notebook_model_matches_package_outputs_and_input_gradients():
    code = r'''
import json
import torch
from pathlib import Path
from lss.models.segmentation import LSSSegmentation as PackageModel
notebook = json.loads(Path('notebooks/lss_e2e_walkthrough.ipynb').read_text())
namespace = {'__name__': 'notebook_parity'}
for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        exec(compile(''.join(cell['source']), '<notebook>', 'exec'), namespace)
kwargs = dict(x_bounds=(-8,8,1), y_bounds=(-8,8,1), z_bounds=(0,3,3), depth_bounds=(1,3,1), context_channels=8, pretrained=False)
torch.manual_seed(7)
notebook_model = namespace['LSSSegmentation'](**kwargs).eval()
package_model = PackageModel(**kwargs).eval()
package_model.load_state_dict(notebook_model.state_dict(), strict=True)
images = torch.randn(1,2,3,32,64, requires_grad=True)
other = images.detach().clone().requires_grad_()
k = torch.tensor([[10.,0.,32.],[0.,10.,16.],[0.,0.,1.]])[None,None].expand(1,2,3,3).clone()
extrinsics = torch.eye(4)[None,None].expand(1,2,4,4).clone()
out_notebook = notebook_model(images,k,extrinsics)
out_package = package_model(other,k,extrinsics)
torch.testing.assert_close(out_notebook, out_package, rtol=1e-5, atol=1e-6)
out_notebook.square().mean().backward()
out_package.square().mean().backward()
torch.testing.assert_close(images.grad, other.grad, rtol=1e-5, atol=1e-6)
assert images.grad is not None and torch.count_nonzero(images.grad) > 0
for (name, notebook_parameter), (other_name, package_parameter) in zip(notebook_model.named_parameters(), package_model.named_parameters()):
    assert name == other_name
    assert (notebook_parameter.grad is None) == (package_parameter.grad is None), name
    if notebook_parameter.grad is not None:
        torch.testing.assert_close(notebook_parameter.grad, package_parameter.grad, rtol=1e-5, atol=1e-6, msg=lambda msg: f'{name}: {msg}')
print('PARITY_COMPLETE')
'''
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")
    result = subprocess.run([sys.executable, "-c", code], cwd=NOTEBOOK.parents[1], env=env,
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr
    assert "PARITY_COMPLETE" in result.stdout


def test_notebook_training_step_matches_package_parameters_and_buffers():
    code = r'''
import json
from pathlib import Path
import torch
from lss.models.segmentation import LSSSegmentation as PackageModel
notebook = json.loads(Path('notebooks/lss_e2e_walkthrough.ipynb').read_text())
namespace = {'__name__': 'notebook_training_parity'}
for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        exec(compile(''.join(cell['source']), '<notebook>', 'exec'), namespace)
kwargs = dict(x_bounds=(-8,8,1), y_bounds=(-8,8,1), z_bounds=(0,3,3), depth_bounds=(1,3,1), context_channels=8, pretrained=False)
torch.manual_seed(81)
notebook_model = namespace['LSSSegmentation'](**kwargs).train()
package_model = PackageModel(**kwargs).train()
package_model.load_state_dict(notebook_model.state_dict(), strict=True)
notebook_optimizer = torch.optim.Adam(notebook_model.parameters(), lr=1e-3)
package_optimizer = torch.optim.Adam(package_model.parameters(), lr=1e-3)
images = torch.randn(1,2,3,32,64)
k = torch.tensor([[10.,0.,32.],[0.,10.,16.],[0.,0.,1.]])[None,None].expand(1,2,3,3).clone()
extrinsics = torch.eye(4)[None,None].expand(1,2,4,4).clone()
target = torch.ones(1,1,16,16)
torch.manual_seed(123)
notebook_logits = notebook_model(images, k, extrinsics)
notebook_loss = torch.nn.functional.binary_cross_entropy_with_logits(notebook_logits, target)
notebook_loss.backward()
torch.manual_seed(123)
package_logits = package_model(images, k, extrinsics)
package_loss = torch.nn.functional.binary_cross_entropy_with_logits(package_logits, target)
package_loss.backward()
torch.testing.assert_close(notebook_logits, package_logits, rtol=1e-5, atol=1e-6)
torch.testing.assert_close(notebook_loss, package_loss, rtol=1e-5, atol=1e-6)
for (name, p), (other_name, q) in zip(notebook_model.named_parameters(), package_model.named_parameters()):
    assert name == other_name
    assert (p.grad is None) == (q.grad is None), name
    if p.grad is not None:
        torch.testing.assert_close(p.grad, q.grad, rtol=1e-5, atol=1e-6, msg=lambda msg: f'{name}: {msg}')
notebook_optimizer.step()
package_optimizer.step()
for key, value in notebook_model.state_dict().items():
    other = package_model.state_dict()[key]
    if torch.is_tensor(value):
        torch.testing.assert_close(value, other, rtol=1e-5, atol=1e-6, msg=lambda msg: f'{key}: {msg}')
    else:
        assert value == other
print('TRAINING_PARITY_COMPLETE')
'''
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")
    result = subprocess.run([sys.executable, "-c", code], cwd=NOTEBOOK.parents[1], env=env,
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr
    assert "TRAINING_PARITY_COMPLETE" in result.stdout


def test_notebook_planner_matches_package_output_and_trajectory_gradients():
    code = r'''
import json
from pathlib import Path
import torch
from lss.models.planner import LSSPlanner as PackagePlanner
notebook = json.loads(Path('notebooks/lss_e2e_walkthrough.ipynb').read_text())
namespace = {'__name__': 'notebook_planner_parity'}
for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        exec(compile(''.join(cell['source']), '<notebook>', 'exec'), namespace)
kwargs = dict(x_bounds=(-8,8,1), y_bounds=(-8,8,1), z_bounds=(0,3,3), depth_bounds=(1,3,1), context_channels=8, pretrained=False)
torch.manual_seed(6)
notebook_model = namespace['LSSPlanner'](**kwargs).eval()
package_model = PackagePlanner(**kwargs).eval()
package_model.load_state_dict(notebook_model.state_dict(), strict=True)
images = torch.randn(1,2,3,32,64)
intrinsics = torch.tensor([[10.,0.,32.],[0.,10.,16.],[0.,0.,1.]])[None,None].expand(1,2,3,3).clone()
extrinsics = torch.eye(4)[None,None].expand(1,2,4,4).clone()
trajectories = torch.tensor([[[[1.,1.],[2.,1.]], [[1.,2.],[2.,2.]]]])
actual = notebook_model(images, intrinsics, extrinsics, trajectories)
expected = package_model(images, intrinsics, extrinsics, trajectories)
for key in ('cost_map', 'logits', 'probabilities'):
    torch.testing.assert_close(actual[key], expected[key], rtol=1e-5, atol=1e-6)
actual['logits'].sum().backward()
expected['logits'].sum().backward()
for (name, p), (other_name, q) in zip(notebook_model.named_parameters(), package_model.named_parameters()):
    assert name == other_name and (p.grad is None) == (q.grad is None), name
    if p.grad is not None:
        torch.testing.assert_close(p.grad, q.grad, rtol=1e-5, atol=1e-6, msg=lambda msg: f'{name}: {msg}')
print('PLANNER_PARITY_COMPLETE')
'''
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="", MPLBACKEND="Agg")
    result = subprocess.run([sys.executable, "-c", code], cwd=NOTEBOOK.parents[1], env=env,
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr
    assert "PLANNER_PARITY_COMPLETE" in result.stdout
