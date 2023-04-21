from __future__ import annotations
import torch
import mcubes
import numpy as np

from copy import copy
from typing import Callable
from pytorch3d.structures import Meshes
from pytorch3d.io import load_ply, load_obj
from torch.autograd.function import FunctionCtx, Function, once_differentiable
from pytorch3d.ops.laplacian_matrices import laplacian, cot_laplacian, norm_laplacian
from typing import Callable, Tuple, Union, Mapping, TypeVar, Union, Iterable, Callable, Dict, Tuple, Union, List, Dict

# these are generic type vars to tell mapping to accept any type vars when creating a type
KT = TypeVar("KT")  # key type
VT = TypeVar("VT")  # value type

# TODO: move this to engine implementation
# TODO: this is a special type just like Config
# ? However, dotdict is a general purpose data passing object, instead of just designed for config
# The only reason we defined those special variables are for type annotations
# If removed, all will still work flawlessly, just no editor annotation for output, type and meta


def return_dotdict(func: Callable):
    def inner(*args, **kwargs):
        return dotdict(func(*args, **kwargs))
    return inner


class dotdict(dict, Dict[KT, VT]):
    """
    This is the default data passing object used throughout the codebase
    Main function: dot access for dict values & dict like merging and updates

    a dictionary that supports dot notation 
    as well as dictionary access notation 
    usage: d = make_dotdict() or d = make_dotdict{'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """

    def update(self, dct: Dict = None, **kwargs):
        dct = copy(dct)  # avoid modifying the original dict, use super's copy to avoid recursion

        # Handle different arguments
        if dct is None:
            dct = kwargs
        elif isinstance(dct, Mapping):
            dct.update(kwargs)
        else:
            super().update(dct, **kwargs)
            return

        # Recursive updates
        for k, v in dct.items():
            if k in self:

                # Handle type conversions
                target_type = type(self[k])
                if not isinstance(v, target_type):
                    # NOTE: bool('False') will be True
                    if target_type == bool and isinstance(v, str):
                        dct[k] = v == 'True'
                    else:
                        dct[k] = target_type(v)

                if isinstance(v, dict):
                    self[k].update(v)  # recursion from here
                else:
                    self[k] = v
            else:
                if isinstance(v, dict):
                    self[k] = dotdict(v)  # recursion?
                else:
                    self[k] = v
        return self

    def __init__(self, *args, **kwargs):
        self.update(*args, **kwargs)

    copy = return_dotdict(dict.copy)
    fromkeys = return_dotdict(dict.fromkeys)

    # def __hash__(self):
    #     # return hash(''.join([str(self.values().__hash__())]))
    #     return super(dotdict, self).__hash__()

    # def __init__(self, *args, **kwargs):
    #     super(dotdict, self).__init__(*args, **kwargs)

    """
    Uncomment following lines and 
    comment out __getattr__ = dict.__getitem__ to get feature:
    
    returns empty numpy array for undefined keys, so that you can easily copy things around
    TODO: potential caveat, harder to trace where this is set to np.array([], dtype=np.float32)
    """

    def __getitem__(self, key):
        try:
            return dict.__getitem__(self, key)
        except KeyError as e:
            raise AttributeError(e)
    # MARK: Might encounter exception in newer version of pytorch
    # Traceback (most recent call last):
    #   File "/home/xuzhen/miniconda3/envs/torch/lib/python3.9/multiprocessing/queues.py", line 245, in _feed
    #     obj = _ForkingPickler.dumps(obj)
    #   File "/home/xuzhen/miniconda3/envs/torch/lib/python3.9/multiprocessing/reduction.py", line 51, in dumps
    #     cls(buf, protocol).dump(obj)
    # KeyError: '__getstate__'
    # MARK: Because you allow your __getattr__() implementation to raise the wrong kind of exception.
    # FIXME: not working typing hinting code
    __getattr__: Callable[..., 'torch.Tensor'] = __getitem__  # type: ignore # overidden dict.__getitem__
    __getattribute__: Callable[..., 'torch.Tensor']  # type: ignore
    # __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    # TODO: better ways to programmically define these special variables?

    @property
    def meta(self) -> dotdict:
        # Special variable used for storing cpu tensor in batch
        if 'meta' not in self:
            self.meta = dotdict()
        return self.__getitem__('meta')

    @meta.setter
    def meta(self, meta):
        self.__setitem__('meta', meta)

    @property
    def output(self) -> dotdict:  # late annotation needed for this
        # Special entry for storing output tensor in batch
        if 'output' not in self:
            self.output = dotdict()
        return self.__getitem__('output')

    @output.setter
    def output(self, output):
        self.__setitem__('output', output)

    @property
    def type(self) -> str:  # late annotation needed for this
        # Special entry for type based construction system
        return self.__getitem__('type')

    @type.setter
    def type(self, type):
        self.__setitem__('type', type)


class default_dotdict(dotdict):
    def __init__(self, default_type=object, *arg, **kwargs):
        super().__init__(*arg, **kwargs)
        dict.__setattr__(self, 'default_type', default_type)

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except (AttributeError, KeyError) as e:
            super().__setitem__(key, dict.__getattribute__(self, 'default_type')())
            return super().__getitem__(key)


context = dotdict()  # a global context object. Forgot why I did this. TODO: remove this


def export_dotdict(batch: dotdict, filename):
    batch = to_numpy(batch)
    np.savez_compressed(filename, **batch)


def to_numpy(batch, non_blocking=False) -> Union[List, Dict, np.ndarray]:  # almost always exporting, should block
    if isinstance(batch, (tuple, list)):
        batch = [to_numpy(b) for b in batch]
    elif isinstance(batch, dict):
        batch = dotdict({k: to_numpy(v) for k, v in batch.items()})
    elif isinstance(batch, torch.Tensor):
        batch = batch.detach().to('cpu', non_blocking=non_blocking).numpy()
    else:  # numpy and others
        batch = np.asarray(batch)
    return batch


class StVKMaterial(dotdict):
    '''
    This class stores parameters for the StVK material model
    copied from https://github.com/isantesteban/snug/blob/main/losses/material.py

    # Fabric material parameters
    thickness = 0.00047 # (m)
    bulk_density = 426  # (kg / m3)
    area_density = thickness * bulk_density

    material = Material(
        density=area_density, # Fabric density (kg / m2)
        thickness=thickness,  # Fabric thickness (m)
        young_modulus=0.7e5, 
        poisson_ratio=0.485,
        stretch_multiplier=1,
        bending_multiplier=50
    )

    '''

    def __init__(self,
                 density=426 * 0.00047,  # Fabric density (kg / m2)
                 thickness=0.00047,  # Fabric thickness (m)
                 young_modulus=0.7e5,
                 poisson_ratio=0.485,
                 bending_multiplier=50.0,
                 stretch_multiplier=1.0,
                 material_multiplier=1.0,
                 ):

        self.density = density
        self.thickness = thickness
        self.young_modulus = young_modulus
        self.poisson_ratio = poisson_ratio

        self.bending_multiplier = bending_multiplier
        self.stretch_multiplier = stretch_multiplier

        # Bending and stretching coefficients (ARCSim)
        self.A = young_modulus / (1.0 - poisson_ratio**2)
        self.stretch_coeff = self.A
        self.stretch_coeff *= stretch_multiplier * material_multiplier

        self.bending_coeff = self.A / 12.0 * (thickness ** 3)
        self.bending_coeff *= bending_multiplier * material_multiplier

        # Lamé coefficients
        self.lame_mu = 0.5 * self.stretch_coeff * (1.0 - self.poisson_ratio)
        self.lame_lambda = self.stretch_coeff * self.poisson_ratio


class Garment(dotdict):
    '''
    This class stores mesh and material information of the garment
    No batch dimension here
    '''

    def __init__(self, v: torch.Tensor, f: torch.Tensor, vm: torch.Tensor, fm: torch.Tensor, material: StVKMaterial = StVKMaterial()):
        self.material = material

        # Face attributes
        self.f = f
        self.f_connectivity_edges, self.f_connectivity = get_face_connectivity(f)  # Pairs of adjacent faces
        self.f_connected_faces = linear_gather(self.f, self.f_connectivity.view(-1), dim=-2).view(-1, 2, 3)  # E * 2, 3 -> E, 2, 3
        self.f_area = get_face_areas(v, f)

        # Vertex attributes
        self.v = v
        self.v_mass = get_vertex_mass(v, f, self.material.density, self.f_area)
        self.inv_mass = 1 / self.v_mass

        # Rest state of the cloth (computed in material space)
        self.vm = vm
        self.fm = fm
        tris_m = multi_gather_tris(vm, fm)
        # tris_m = multi_gather_tris(v, f)
        self.Dm = get_shape_matrix(tris_m)
        self.Dm_inv = torch_inverse_2x2(self.Dm)

# batch dimension applicable


def get_shape_matrix(tris: torch.Tensor):
    return torch.stack([tris[..., 0, :] - tris[..., 2, :],
                        tris[..., 1, :] - tris[..., 2, :],
                        ], dim=-1)


def deformation_gradient(tris: torch.Tensor, Dm_invs: torch.Tensor):
    Ds = get_shape_matrix(tris)
    return Ds @ Dm_invs


def get_matching_identity_matrix(F: torch.Tensor):
    # match shape of F
    shape = F.shape[:-2]
    I = torch.eye(F.shape[-1], dtype=F.dtype, device=F.device)
    for i in shape:
        I = I[None]
    I = I.expand(*shape, -1, -1)

    return I


def green_strain_tensor(F: torch.Tensor) -> torch.Tensor:
    I = get_matching_identity_matrix(F)
    return 0.5 * (F.mT @ F - I)


def stretch_energy_constraints(v: torch.Tensor, garment: Garment, **kwargs):
    '''
    v: B, V, 3

    Computes strech energy of the cloth for the vertex positions v
    Material model: Saint-Venant–Kirchhoff (StVK)
    Reference: ArcSim (physics.cpp)
    '''
    # XPBD step
    triangles = multi_gather_tris(v, garment.f)  # B, F, 3, 3
    inv_mass = multi_gather_tris(garment.inv_mass[None, ..., None], garment.f)[..., 0]  # B, F, 3

    def func(triangles, garment): return stretch_energy_components(triangles, garment)
    return xpbd_constraints(func, triangles, inv_mass, garment, **kwargs)


def bending_energy_constraints(v: torch.Tensor, garment: Garment, **kwargs):
    # XPBD step
    triangles = multi_gather_tris(v, garment.f_connected_faces.view(-1, 3)).view(v.shape[0], -1, 2 * 3, 3)  # B, E, 2 * 3, 3
    inv_mass = multi_gather_tris(garment.inv_mass[None, ..., None], garment.f_connected_faces.view(-1, 3))[..., 0].view(v.shape[0], -1, 2 * 3)  # B, E, 2 * 3

    def func(triangles, garment): return bending_energy_components(v, triangles.view(v.shape[0], -1, 2, 3, 3), garment)
    return xpbd_constraints(func, triangles, inv_mass, garment, **kwargs)


def xpbd_constraints(func: Callable[..., torch.Tensor], triangles: torch.Tensor, inv_mass: torch.Tensor, garment: Garment, accum_lambda: torch.Tensor = None, delta_t: float = 1, compliance: float = 0):
    triangles.requires_grad_()
    with torch.enable_grad():
        energy = func(triangles, garment)  # B, E
    grad = take_gradient(energy.sum() / energy.shape[0], triangles, create_graph=False, retain_graph=False)  # B, E, 2, 3, 3, gradient on all participating triangles
    grad = grad.detach()
    energy = energy.detach()

    grad_d2 = (grad ** 2).sum(dim=-1)  # B, E, 2, 3

    # preparation for lambda
    if accum_lambda is None:
        accum_lambda = torch.zeros_like(inv_mass)

    compliance_factor = compliance / delta_t ** 2
    delta_lambda = (-energy - compliance_factor * accum_lambda) / ((inv_mass * grad_d2).sum(dim=-1) + compliance_factor)
    delta_p = delta_lambda[..., None, None] * inv_mass[..., None] * grad  # B, F, 3, 3

    return energy, grad, grad_d2, delta_lambda, delta_p


def multi_gather_tris(v: torch.Tensor, f: torch.Tensor, dim=-2) -> torch.Tensor:
    # compute faces normals w.r.t the vertices (considering batch dimension)
    if v.ndim == (f.ndim + 1):
        f = f[None].expand(v.shape[0], *f.shape)
    # assert verts.shape[0] == faces.shape[0]
    shape = torch.tensor(v.shape)
    remainder = shape.flip(0)[:(len(shape) - dim - 1) % len(shape)]
    return multi_gather(v, f.view(*f.shape[:-2], -1), dim=dim).view(*f.shape, *remainder)  # B, F, 3, 3


def multi_gather(values: torch.Tensor, index: torch.Tensor, dim=-2):
    # Gather the value at the -2th dim of values, augment index shape on the back
    # Example: values: B, P, 3, index: B, N, -> B, N, 3

    # index will first be augmented to match the values' dimentionality at the back
    # take care of batch dimension of, and acts like a linear indexing in the target dimention
    # we assume that the values's second to last dimension is the dimension to be indexed on
    return values.gather(dim, multi_indexing(index, values.shape, dim))


def expand0(x: torch.Tensor, B: int):
    return x[None].expand(B, *x.shape)


def stretch_energy_components(triangles: torch.Tensor, garment: Garment):
    '''
    triangles: B, F, 3, 3

    Computes strech energy of the cloth for the vertex positions v
    Material model: Saint-Venant–Kirchhoff (StVK)
    Reference: ArcSim (physics.cpp)
    '''
    B = triangles.shape[0]

    Dm_invs = expand0(garment.Dm_inv, B)  # B, F, 2, 2

    F = deformation_gradient(triangles, Dm_invs)  # B, F, 3, 2
    G = green_strain_tensor(F)  # B, F, 2, 2

    # Energy
    mat = garment.material

    I = get_matching_identity_matrix(G)
    S = mat.lame_mu * G + 0.5 * mat.lame_lambda * torch_trace(G)[..., None, None] * I  # B, F, 2, 2

    energy_density = torch_trace(S.mT @ G)  # B, F
    energy = garment.f_area[None] * mat.thickness * energy_density  # B, F

    # return torch.sum(energy) / B
    return energy


def bending_energy_components(v: torch.Tensor, connected_triangles: torch.Tensor, garment: Garment) -> torch.Tensor:
    '''
    connected_triangles: B, E, 4, 3 (0, 1, 2, 3) -> edge verts and point verts?
                       : B, E, 2, 3, 3 -> triangles but no edge information?

    Computes the bending energy of the cloth for the vertex positions v
    Reference: ArcSim (physics.cpp)
    '''

    B = connected_triangles.shape[0]

    # Compute face normals
    # fn = face_normals(v, garment.f)  # B, F, 3
    # n0 = linear_gather(fn, garment.f_connectivity[:, 0], dim=-2)  # B, E, 3
    # n1 = linear_gather(fn, garment.f_connectivity[:, 1], dim=-2)  # B, E, 3
    n = torch.cross(connected_triangles[:, :, :, 1, :] - connected_triangles[:, :, :, 0, :], connected_triangles[:, :, :, 2, :] - connected_triangles[:, :, :, 1, :])
    n0 = n[:, :, 0]
    n1 = n[:, :, 1]

    # Compute edge length
    v0 = linear_gather(v, garment.f_connectivity_edges[:, 0], dim=-2)  # B, E, 3
    v1 = linear_gather(v, garment.f_connectivity_edges[:, 1], dim=-2)  # B, E, 3
    e = v1 - v0
    l = e.norm(dim=-1, keepdim=True)
    # e_norm = e / l

    # Compute area
    f_area = expand0(garment.f_area, B)
    a0 = linear_gather(f_area, garment.f_connectivity[:, 0], dim=-1)
    a1 = linear_gather(f_area, garment.f_connectivity[:, 1], dim=-1)
    a = a0 + a1

    # Compute dihedral angle between faces
    cos = (n0 * n1).sum(dim=-1)  # dot product, B, E
    crs = torch.cross(n0, n1)
    sin = (crs ** 2).sum(dim=-1) / crs.norm(dim=-1)  # B, E
    theta = torch.atan2(sin, cos)  # B, E

    # Compute bending coefficient according to material parameters,
    # triangle areas (a) and edge length (l)
    mat = garment.material
    scale = l[..., 0]**2 / (4 * a)

    # Bending energy
    energy = mat.bending_coeff * scale * (theta ** 2) / 2  # B, E

    return energy


def stretch_energy(v: torch.Tensor, garment: Garment):
    '''
    v: B, V, 3

    Computes strech energy of the cloth for the vertex positions v
    Material model: Saint-Venant–Kirchhoff (StVK)
    Reference: ArcSim (physics.cpp)
    '''
    B = v.shape[0]
    triangles = multi_gather_tris(v, garment.f)  # B, F, 3, 3

    Dm_invs = expand0(garment.Dm_inv, B)  # B, F, 2, 2

    F = deformation_gradient(triangles, Dm_invs)  # B, F, 3, 2
    G = green_strain_tensor(F)  # B, F, 2, 2

    # Energy
    mat = garment.material

    I = get_matching_identity_matrix(G)
    S = mat.lame_mu * G + 0.5 * mat.lame_lambda * torch_trace(G)[..., None, None] * I  # B, F, 2, 2

    energy_density = torch_trace(S.mT @ G)  # B, F
    energy = garment.f_area[None] * mat.thickness * energy_density  # B, F

    return torch.sum(energy) / B


def bending_energy(v: torch.Tensor, garment: Garment):
    '''
    v: B, V, 3

    Computes the bending energy of the cloth for the vertex positions v
    Reference: ArcSim (physics.cpp)
    '''

    B = v.shape[0]

    # Compute face normals
    fn = face_normals(v, garment.f)
    n0 = linear_gather(fn, garment.f_connectivity[:, 0], dim=-2)
    n1 = linear_gather(fn, garment.f_connectivity[:, 1], dim=-2)

    # Compute edge length
    v0 = linear_gather(v, garment.f_connectivity_edges[:, 0], dim=-2)
    v1 = linear_gather(v, garment.f_connectivity_edges[:, 1], dim=-2)
    e = v1 - v0
    l = e.norm(dim=-1, keepdim=True)
    e_norm = e / l

    # Compute area
    f_area = expand0(garment.f_area, B)
    a0 = linear_gather(f_area, garment.f_connectivity[:, 0], dim=-1)
    a1 = linear_gather(f_area, garment.f_connectivity[:, 1], dim=-1)
    a = a0 + a1

    # Compute dihedral angle between faces
    cos = (n0 * n1).sum(dim=-1)  # dot product, B, E
    sin = (e_norm * torch.cross(n0, n1)).sum(dim=-1)  # B, E
    theta = torch.atan2(sin, cos)  # B, E

    # Compute bending coefficient according to material parameters,
    # triangle areas (a) and edge length (l)
    mat = garment.material
    scale = l[..., 0]**2 / (4 * a)

    # Bending energy
    energy = mat.bending_coeff * scale * (theta ** 2) / 2  # B, E

    return torch.sum(energy) / B


def gravity_energy(x: torch.Tensor, mass: torch.Tensor, g=9.81):
    # S, V, 3
    # U = m * g * h
    U = g * mass * (x[:, :, 1] + 0)  # mgh

    return torch.sum(U) / x.shape[0]


def inertia_term(x_next: torch.Tensor,
                 x_curr: torch.Tensor,
                 v_curr: torch.Tensor,
                 mass: torch.Tensor,
                 delta_t
                 ):

    x_pred = x_curr + delta_t * v_curr
    x_diff = x_next - x_pred

    inertia = (x_diff ** 2).sum(dim=-1) * mass / (2 * delta_t ** 2)

    return inertia.sum() / x_next.shape[0]


def dynamic_term(x_next: torch.Tensor,
                 x_curr: torch.Tensor,
                 v_curr: torch.Tensor,
                 mass: torch.Tensor,
                 delta_t,
                 gravitational_acceleration=[0, -9.81, 0]):

    gravitational_acceleration = torch.tensor(gravitational_acceleration, device=x_next.device, dtype=x_next.dtype)

    x_pred = x_curr + delta_t * v_curr + delta_t * delta_t * gravitational_acceleration  # gravity added at last dimension for every point and timestep

    x_diff = x_next - x_pred
    dynamic = (x_diff ** 2).sum(dim=-1) * mass / (2 * delta_t ** 2)

    return dynamic.sum() / x_next.shape[0]


def inertia_term_sequence(x: torch.Tensor,
                          mass: torch.Tensor,
                          delta_t: float,
                          method: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float], torch.Tensor] = inertia_term,
                          compliance: float = 1.0,
                          ):
    """
    x: torch.Tensor of shape [batch_size, num_frames, num_vertices, 3]
    """
    B = x.shape[0]
    V = x.shape[-2]
    # Compute velocities
    x_next = x[:, 1:]  # B, T-1, V, 3
    x_curr = x[:, :-1]  # B, T-1, V, 3
    v_next = (x_next - x_curr) / delta_t
    zeros = torch.zeros([B, 1, V, 3], dtype=x.dtype, device=x.device)
    v_curr = torch.cat([zeros, v_next[:, :-1]], dim=1)  # B, T-1, V, 3

    # Flatten
    x_next = x_next.view(-1, V, 3)  # B * T-1, V, 3
    x_curr = x_curr.view(-1, V, 3)  # B * T-1, V, 3, should have been the correct x_curr: for 3 frame case, this holds
    v_curr = v_curr.view(-1, V, 3)  # B * T-1, V, 3, should have been the gt v_curr: for 3 frame case, this holds

    return compliance * method(x_next, x_curr, v_curr, mass, delta_t)

# @torch.jit.script


def torch_inverse_2x2(A: torch.Tensor, eps=torch.finfo(torch.float).eps):
    B = torch.zeros_like(A)
    # for readability
    a = A[..., 0, 0]
    b = A[..., 0, 1]
    c = A[..., 1, 0]
    d = A[..., 1, 1]
    # slightly slower but save 20% of space (??) by
    # storing determinat inplace
    det = B[..., 1, 1]
    det = (a * d - b * c)
    det = det + eps
    B[..., 0, 0] = d / det
    B[..., 0, 1] = -b / det
    B[..., 1, 0] = -c / det
    B[..., 1, 1] = a / det
    return B

# @torch.jit.script


def torch_trace(x: torch.Tensor):
    return x.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)


def meshes_attri_laplacian_smoothing(meshes: Meshes, attri: torch.Tensor, method: str = "uniform"):
    if meshes.isempty():
        return torch.tensor(
            [0.0], dtype=torch.float32, device=meshes.device, requires_grad=True
        )

    N = len(meshes)
    verts_packed = meshes.verts_packed()  # (sum(V_n), 3)
    faces_packed = meshes.faces_packed()  # (sum(F_n), 3)
    num_verts_per_mesh = meshes.num_verts_per_mesh()  # (N,)
    verts_packed_idx = meshes.verts_packed_to_mesh_idx()  # (sum(V_n),)
    weights = num_verts_per_mesh.gather(0, verts_packed_idx)  # (sum(V_n),)
    weights = 1.0 / weights.float()

    # We don't want to backprop through the computation of the Laplacian;
    # just treat it as a magic constant matrix that is used to transform
    # verts into normals
    with torch.no_grad():
        if method == "uniform":
            L = meshes.laplacian_packed()
        elif method in ["cot", "cotcurv"]:
            L, inv_areas = cot_laplacian(verts_packed, faces_packed)
            if method == "cot":
                norm_w = torch.sparse.sum(L, dim=1).to_dense().view(-1, 1)
                idx = norm_w > 0
                norm_w[idx] = 1.0 / norm_w[idx]
            else:
                L_sum = torch.sparse.sum(L, dim=1).to_dense().view(-1, 1)
                norm_w = 0.25 * inv_areas
        else:
            raise ValueError("Method should be one of {uniform, cot, cotcurv}")

    attri_packed = attri.reshape(-1, attri.shape[-1])

    if method == "uniform":
        loss = L.mm(attri_packed)
    elif method == "cot":
        loss = L.mm(attri_packed) * norm_w - attri_packed
    elif method == "cotcurv":
        # pyre-fixme[61]: `norm_w` may not be initialized here.
        loss = (L.mm(attri_packed) - L_sum * attri_packed) * norm_w
    loss = loss.norm(dim=1)

    loss = loss * weights
    return loss.sum() / N


def take_jacobian(func: Callable, input: torch.Tensor, create_graph=False, vectorize=True, strategy='reverse-mode'):
    return torch.autograd.functional.jacobian(func, input, create_graph=create_graph, vectorize=vectorize, strategy=strategy)


def take_gradient(output: torch.Tensor,
                  input: torch.Tensor,
                  d_out: torch.Tensor = None,
                  create_graph: bool = True,
                  retain_graph: bool = True,
                  is_grads_batched: bool = False,
                  ):
    if d_out is not None:
        d_output = d_out
    elif isinstance(output, torch.Tensor):
        d_output = torch.ones_like(output, requires_grad=False)
    else:
        d_output = [torch.ones_like(o, requires_grad=False) for o in output]
    grads = torch.autograd.grad(inputs=input,
                                outputs=output,
                                grad_outputs=d_output,
                                create_graph=create_graph,
                                retain_graph=retain_graph,
                                only_inputs=True,
                                is_grads_batched=is_grads_batched,
                                )
    if len(grads) == 1:
        return grads[0]  # return the gradient directly
    else:
        return grads  # to be expanded


class RegisterSDFGradient(Function):
    @staticmethod
    def forward(ctx: FunctionCtx, verts: torch.Tensor, decoder: Callable[[torch.Tensor], torch.Tensor], chunk_size=65536):
        ctx.save_for_backward(verts)
        ctx.decoder = decoder
        ctx.chunk_size = chunk_size
        return verts

    @staticmethod
    def backward(ctx: FunctionCtx, grad: torch.Tensor):
        chunk_size = ctx.chunk_size
        decoder: Callable[[torch.Tensor], torch.Tensor] = ctx.decoder
        verts: torch.Tensor = ctx.saved_tensors[0]

        verts = verts.detach().requires_grad_()  # should not affect the original verts
        with torch.enable_grad():
            sdf = torch.cat([decoder(verts[i:i + chunk_size]) for i in range(0, verts.shape[0], chunk_size)])
            norm = normalize(take_gradient(sdf, verts, create_graph=False, retain_graph=True))  # N, 3
            grad = -torch.einsum('ni,ni->n', norm, grad) * sdf.view(verts.shape[0])  # N
            loss = grad.sum()
        loss.backward(retain_graph=True)  # accumulate gradients into decorder parameters
        return None, None, None


register_sdf_gradient = RegisterSDFGradient.apply


def differentiable_marching_cubes(points: torch.Tensor, decoder: Callable[[torch.Tensor], torch.Tensor], chunk_size=65536):
    """
    Will use torchmcubes and return the corrsponding vertices of the marching cubes result
    currently no batch dimension supported

    TODO: use octree to make this faster

    points: [X, Y, Z]
    """

    sh = points.shape
    points = points.view(-1, 3)
    upper = points.max(dim=0, keepdim=True)[0]
    lower = points.min(dim=0, keepdim=True)[0]
    points = points.detach().requires_grad_(False)  # should not affect the original verts
    with torch.no_grad():
        sdf = np.concatenate(
            [
                decoder(points[i:i + chunk_size]).detach().to('cpu').numpy()
                for i in range(0, points.shape[0], chunk_size)
            ]
        )
        # MARK: GPU CPU SYNC
    # verts, faces = marching_cubes(-sdf.view(*sh[:-1]), 0.0)
    verts, faces = mcubes.marching_cubes(-sdf.reshape(*sh[:-1]), 0.0)
    verts = torch.from_numpy(verts.astype(np.float32)).to(points.device, non_blocking=True)
    faces = torch.from_numpy(faces.astype(np.int32)).to(points.device, non_blocking=True)
    verts = verts * (upper - lower) / (torch.tensor(sh[:-1], device=points.device) - 1) + lower

    verts = register_sdf_gradient(verts, decoder, chunk_size)  # literally a state switch, no other stuff
    return verts, faces


def multi_indexing(index: torch.Tensor, shape: torch.Size, dim=-2):
    # index will first be augmented to match the values' dimentionality at the back
    # then we will try to broatcast index's shape to values shape
    shape = list(shape)
    back_pad = len(shape) - index.ndim
    for _ in range(back_pad):
        index = index.unsqueeze(-1)
    expand_shape = shape
    expand_shape[dim] = -1
    return index.expand(*expand_shape)


def multi_gather(values: torch.Tensor, index: torch.Tensor, dim=-2):
    # Gather the value at the -2th dim of values, augment index shape on the back
    # Example: values: B, P, 3, index: B, N, -> B, N, 3

    # index will first be augmented to match the values' dimentionality at the back
    # take care of batch dimension of, and acts like a linear indexing in the target dimention
    # we assume that the values's second to last dimension is the dimension to be indexed on
    return values.gather(dim, multi_indexing(index, values.shape, dim))


def multi_scatter(target: torch.Tensor, index: torch.Tensor, values: torch.Tensor, dim=-2):
    # backward of multi_gather
    return target.scatter(dim, multi_indexing(index, values.shape, dim), values)


def multi_scatter_(target: torch.Tensor, index: torch.Tensor, values: torch.Tensor, dim=-2):
    # inplace version of multi_scatter
    return target.scatter_(dim, multi_indexing(index, values.shape, dim), values)


def multi_gather_tris(v: torch.Tensor, f: torch.Tensor, dim=-2) -> torch.Tensor:
    # compute faces normals w.r.t the vertices (considering batch dimension)
    if v.ndim == (f.ndim + 1):
        f = f[None].expand(v.shape[0], *f.shape)
    # assert verts.shape[0] == faces.shape[0]
    shape = torch.tensor(v.shape)
    remainder = shape.flip(0)[:(len(shape) - dim - 1) % len(shape)]
    return multi_gather(v, f.view(*f.shape[:-2], -1), dim=dim).view(*f.shape, *remainder)  # B, F, 3, 3


def linear_indexing(index: torch.Tensor, shape: torch.Size, dim=0):
    assert index.ndim == 1
    shape = list(shape)
    dim = dim if dim >= 0 else len(shape) + dim
    front_pad = dim
    back_pad = len(shape) - dim - 1
    for _ in range(front_pad):
        index = index.unsqueeze(0)
    for _ in range(back_pad):
        index = index.unsqueeze(-1)
    expand_shape = shape
    expand_shape[dim] = -1
    return index.expand(*expand_shape)


def linear_gather(values: torch.Tensor, index: torch.Tensor, dim=0):
    # only taking linea indices as input
    return values.gather(dim, linear_indexing(index, values.shape, dim))


def linear_scatter(target: torch.Tensor, index: torch.Tensor, values: torch.Tensor, dim=0):
    return target.scatter(dim, linear_indexing(index, values.shape, dim), values)


def linear_scatter_(target: torch.Tensor, index: torch.Tensor, values: torch.Tensor, dim=0):
    return target.scatter_(dim, linear_indexing(index, values.shape, dim), values)


def merge01(x: torch.Tensor):
    return x.reshape(-1, *x.shape[2:])


def scatter0(target: torch.Tensor, inds: torch.Tensor, value: torch.Tensor):
    return target.scatter(0, expand_at_the_back(target, inds), value)  # Surface, 3 -> B * S, 3


def gather0(target: torch.Tensor, inds: torch.Tensor):
    return target.gather(0, expand_at_the_back(target, inds))  # B * S, 3 -> Surface, 3


def expand_at_the_back(target: torch.Tensor, inds: torch.Tensor):
    for _ in range(target.ndim - 1):
        inds = inds.unsqueeze(-1)
    inds = inds.expand(-1, *target.shape[1:])
    return inds


def expand0(x: torch.Tensor, B: int):
    return x[None].expand(B, *x.shape)


def expand1(x: torch.Tensor, P: int):
    return x[:, None].expand(-1, P, *x.shape[1:])


def nonzero0(condition: torch.Tensor):
    # MARK: will cause gpu cpu sync
    # return those that are true in the provided tensor
    return condition.nonzero(as_tuple=True)[0]


def face_normals(v: torch.Tensor, f: torch.Tensor):
    # compute faces normals w.r.t the vertices (considering batch dimension)
    tris = multi_gather_tris(v, f)

    # Compute face normals
    v0, v1, v2 = torch.split(tris, split_size_or_sections=1, dim=-2)
    v0, v1, v2 = v0[..., 0, :], v1[..., 0, :], v2[..., 0, :]
    e1 = v1 - v0
    e2 = v2 - v1
    face_normals = torch.cross(e1, e2)

    face_normals = normalize(face_normals)

    return face_normals


def normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # channel last: normalization
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def normalize_sum(x: torch.Tensor, eps: float = 1e-8):
    return x / (x.sum(dim=-1, keepdim=True) + eps)


def get_face_connectivity(faces: torch.Tensor):
    '''
    Returns a list of adjacent face pairs
    '''

    he = triangle_to_halfedge(None, faces)
    twin = he.twin
    vert = he.vert
    edge = he.edge
    hedg = torch.arange(he.HE, device=faces.device)  # HE, :record half edge indices
    face = hedg // 3
    manifold = (twin >= 0).nonzero(as_tuple=True)[0]

    # NOTE: some repeated computation
    edge_manifold = edge[manifold]  # manifold edge indices

    args = edge_manifold.argsort()  # 00, 11, 22 ...
    inds = hedg[manifold][args]  # index of the valid half edges

    connected_faces = face[inds].view(-1, 2)
    edges_connecting_faces = vert[inds].view(-1, 2)

    return edges_connecting_faces, connected_faces


def get_edge_length(v, e):
    v0 = linear_gather(v, e[..., 0], dim=-2)
    v1 = linear_gather(v, e[..., 1], dim=-2)
    return torch.norm(v0 - v1, dim=-1)


def get_vertex_mass(v: torch.Tensor, f: torch.Tensor, density: float, areas=None):
    '''
    Computes the mass of each vertex according to triangle areas and fabric density
    '''
    if areas is None:
        areas = get_face_areas(v, f)
    triangle_masses = density * areas

    vertex_masses = torch.zeros(v.shape[:-1], device=v.device, dtype=v.dtype)
    vertex_masses[f[..., 0]] += triangle_masses / 3
    vertex_masses[f[..., 1]] += triangle_masses / 3
    vertex_masses[f[..., 2]] += triangle_masses / 3

    return vertex_masses


def get_face_areas(v: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    v0 = v[f[..., 0]]
    v1 = v[f[..., 1]]
    v2 = v[f[..., 2]]

    u = v2 - v0
    v = v1 - v0

    return torch.norm(torch.cross(u, v), dim=-1) / 2.0


def torch_unique_with_indices_and_inverse(x, dim=0):
    unique, inverse = torch.unique(x, return_inverse=True, dim=dim)
    perm = torch.arange(inverse.size(dim), dtype=inverse.dtype, device=inverse.device)
    indices, perm = inverse.flip([dim]), perm.flip([dim])
    return unique, indices.new_empty(unique.size(dim)).scatter_(dim, indices, perm), inverse


def unmerge_faces(faces: torch.Tensor, *args):
    # stack into pairs of (vertex index, texture index)
    stackable = [faces.reshape(-1)]
    # append multiple args to the correlated stack
    # this is usually UV coordinates (vt) and normals (vn)
    for arg in args:
        stackable.append(arg.reshape(-1))

    # unify them into rows of a numpy array
    stack = torch.column_stack(stackable)
    # find unique pairs: we're trying to avoid merging
    # vertices that have the same position but different
    # texture coordinates
    _, unique, inverse = torch_unique_with_indices_and_inverse(stack)

    # only take the unique pairs
    pairs = stack[unique]
    # try to maintain original vertex order
    order = pairs[:, 0].argsort()
    # apply the order to the pairs
    pairs = pairs[order]

    # we re-ordered the vertices to try to maintain
    # the original vertex order as much as possible
    # so to reconstruct the faces we need to remap
    remap = torch.zeros(len(order), dtype=torch.long, device=faces.device)
    remap[order] = torch.arange(len(order), device=faces.device)

    # the faces are just the inverse with the new order
    new_faces = remap[inverse].reshape((-1, 3))

    # the mask for vertices and masks for other args
    result = [new_faces]
    result.extend(pairs.T)

    return result


def merge_faces(faces, *args, n_verts=None):
    # TODO: batch this
    # remember device the faces are on
    device = faces.device
    # start with not altering faces at all
    result = [faces]
    # find the maximum index referenced by faces
    if n_verts is None:  # sometimes things get padded
        n_verts = faces.max() + 1
    # add a vertex mask which is just ordered
    result.append(torch.arange(n_verts, device=device))

    # now given the order is fixed do our best on the rest of the order
    for arg in args:
        # create a mask of the attribute-vertex mapping
        # note that these might conflict since we're not unmerging
        masks = torch.zeros((3, n_verts), dtype=torch.long, device=device)
        # set the mask using the unmodified face indexes
        for i, f, a in zip(range(3), faces.permute(*torch.arange(faces.ndim - 1, -1, -1)), arg.permute(*torch.arange(arg.ndim - 1, -1, -1))):
            masks[i][f] = a
        # find the most commonly occurring attribute (i.e. UV coordinate)
        # and use that index note that this is doing a float conversion
        # and then median before converting back to int: could also do this as
        # a column diff and sort but this seemed easier and is fast enough
        result.append(torch.median(masks, dim=0)[0].to(torch.long))

    return result


def halfedge_to_triangle(halfedge: dotdict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    # assuming the mesh has clean topology? except for boundary edges
    # assume no boundary edge for now!
    verts = halfedge.verts
    vert = halfedge.vert  # HE,

    HE = len(vert)
    hedg = torch.arange(HE, device=verts.device)
    next = hedg & ~3 | (hedg + 1) & 3

    e = torch.stack([vert, vert[next]], dim=-1)
    e01, e12, e20 = e[0::3], e[1::3], e[2::3]
    faces = torch.stack([e01[..., 0], e12[..., 0], e20[..., 0]], dim=-1)

    return verts, faces


def triangle_to_halfedge(verts: Union[torch.Tensor, None],
                         faces: torch.Tensor,
                         is_manifold: bool = False,
                         ):
    # assuming the mesh has clean topology? except for boundary edges
    # assume no boundary edge for now!
    F = len(faces)
    V = len(verts) if verts is not None else faces.max().item()
    HE = 3 * F

    # create halfedges
    v0, v1, v2 = faces.chunk(3, dim=-1)
    e01 = torch.cat([v0, v1], dim=-1)  # (sum(F_n), 2)
    e12 = torch.cat([v1, v2], dim=-1)  # (sum(F_n), 2)
    e20 = torch.cat([v2, v0], dim=-1)  # (sum(F_n), 2)

    # stores the vertex indices for each half edge
    e = torch.empty(HE, 2, device=faces.device, dtype=faces.dtype)
    e[0::3] = e01
    e[1::3] = e12
    e[2::3] = e20
    vert = e[..., 0]  # HE, :record starting half edge
    vert_next = e[..., 1]

    edges = torch.stack([torch.minimum(vert_next, vert), torch.maximum(vert_next, vert)], dim=-1)
    hash = V * edges[..., 0] + edges[..., 1]  # HE, 2, contains edge hash, should be unique
    _, edge, counts = hash.unique(sorted=False, return_inverse=True, return_counts=True)
    E = len(counts)

    hedg = torch.arange(HE, device=faces.device)  # HE, :record half edge indices

    if is_manifold:
        inds = edge.argsort()  # 00, 11, 22 ...
        twin = torch.empty_like(inds)
        twin[inds[0::2]] = inds[1::2]
        twin[inds[1::2]] = inds[0::2]

    else:
        # now we have edge indices, if it's a good mesh, each edge should have two half edges
        # in some non-manifold cases this would be broken so we need to first filter those non-manifold edges out
        manifold = counts == 2  # non-manifold mask
        manifold = manifold[edge]  # non-manifold half edge mask

        edge_manifold = edge[manifold]  # manifold edge indices

        args = edge_manifold.argsort()  # 00, 11, 22 ...
        inds = hedg[manifold][args]
        twin_manifold = torch.empty_like(inds)
        twin_manifold[args[0::2]] = inds[1::2]
        twin_manifold[args[1::2]] = inds[0::2]

        twin = torch.empty(HE, device=faces.device, dtype=torch.long)
        twin[manifold] = twin_manifold
        twin[~manifold] = -counts[edge][~manifold]  # non-manifold half edge mask, number of half edges stored in the twin

    # should return these values
    halfedge = dotdict()

    # geometric info
    halfedge.verts = verts  # V, 3

    # connectivity info
    halfedge.twin = twin  # HE,
    halfedge.vert = vert  # HE,
    halfedge.edge = edge  # HE,

    halfedge.HE = HE
    halfedge.E = E
    halfedge.F = F
    halfedge.V = V

    return halfedge


def get_edges(faces: torch.Tensor):
    V = faces.max()
    F = faces.shape[0]
    HE = F * 3

    # create halfedges
    v0, v1, v2 = faces.chunk(3, dim=-1)
    e01 = torch.cat([v0, v1], dim=-1)  # (sum(F_n), 2)
    e12 = torch.cat([v1, v2], dim=-1)  # (sum(F_n), 2)
    e20 = torch.cat([v2, v0], dim=-1)  # (sum(F_n), 2)

    # stores the vertex indices for each half edge
    e = torch.empty(HE, 2, device=faces.device, dtype=faces.dtype)
    e[0::3] = e01
    e[1::3] = e12
    e[2::3] = e20
    vert = e[..., 0]  # HE, :record starting half edge
    vert_next = e[..., 1]

    edges = torch.stack([torch.minimum(vert_next, vert), torch.maximum(vert_next, vert)], dim=-1)
    hash = V * edges[..., 0] + edges[..., 1]  # HE, 2, contains edge hash, should be unique
    u, i, c = hash.unique(sorted=False, return_inverse=True, return_counts=True)

    e = torch.stack([u // V, u % V], dim=1)
    return e, i, c


def adjacency(verts: torch.Tensor, edges: torch.Tensor):
    V = verts.shape[0]

    e0, e1 = edges.unbind(1)

    idx01 = torch.stack([e0, e1], dim=1)  # (E, 2)
    idx10 = torch.stack([e1, e0], dim=1)  # (E, 2)
    idx = torch.cat([idx01, idx10], dim=0).t()  # (2, 2*E)

    # First, we construct the adjacency matrix,
    # i.e. A[i, j] = 1 if (i,j) is an edge, or
    # A[e0, e1] = 1 &  A[e1, e0] = 1
    ones = torch.ones(idx.shape[1], dtype=torch.float32, device=verts.device)
    # pyre-fixme[16]: Module `sparse` has no attribute `FloatTensor`.
    A = torch.sparse.FloatTensor(idx, ones, (V, V))
    return A


def laplacian(verts: torch.Tensor, edges: torch.Tensor):
    """
    Computes the laplacian matrix.
    The definition of the laplacian is
    L[i, j] =    -1       , if i == j
    L[i, j] = 1 / deg(i)  , if (i, j) is an edge
    L[i, j] =    0        , otherwise
    where deg(i) is the degree of the i-th vertex in the graph.

    Args:
        verts: tensor of shape (V, 3) containing the vertices of the graph
        edges: tensor of shape (E, 2) containing the vertex indices of each edge
    Returns:
        L: Sparse FloatTensor of shape (V, V)
    """
    V = verts.shape[0]

    e0, e1 = edges.unbind(1)

    idx01 = torch.stack([e0, e1], dim=1)  # (E, 2)
    idx10 = torch.stack([e1, e0], dim=1)  # (E, 2)
    idx = torch.cat([idx01, idx10], dim=0).t()  # (2, 2*E)

    # First, we construct the adjacency matrix,
    # i.e. A[i, j] = 1 if (i,j) is an edge, or
    # A[e0, e1] = 1 &  A[e1, e0] = 1
    ones = torch.ones(idx.shape[1], dtype=torch.float32, device=verts.device)
    # pyre-fixme[16]: Module `sparse` has no attribute `FloatTensor`.
    A = torch.sparse.FloatTensor(idx, ones, (V, V))

    # the sum of i-th row of A gives the degree of the i-th vertex
    deg = torch.sparse.sum(A, dim=1).to_dense()

    # We construct the Laplacian matrix by adding the non diagonal values
    # i.e. L[i, j] = 1 ./ deg(i) if (i, j) is an edge
    deg0 = deg[e0]
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    deg0 = torch.where(deg0 > 0.0, 1.0 / deg0, deg0)
    deg1 = deg[e1]
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    deg1 = torch.where(deg1 > 0.0, 1.0 / deg1, deg1)
    val = torch.cat([deg0, deg1])
    # pyre-fixme[16]: Module `sparse` has no attribute `FloatTensor`.
    L = torch.sparse.FloatTensor(idx, val, (V, V))

    # Then we add the diagonal values L[i, i] = -1.
    idx = torch.arange(V, device=verts.device)
    idx = torch.stack([idx, idx], dim=0)
    ones = torch.ones(idx.shape[1], dtype=torch.float32, device=verts.device)
    # pyre-fixme[16]: Module `sparse` has no attribute `FloatTensor`.
    L -= torch.sparse.FloatTensor(idx, ones, (V, V))

    return L


def laplacian_smoothing(v: torch.Tensor, e: torch.Tensor, inds: torch.Tensor = None, alpha=0.33, iter=90):
    for i in range(iter):
        # 1st gaussian smoothing pass
        L = laplacian(v, e)
        vln = L @ v
        if inds is None:
            v += alpha * vln
        else:
            v[inds] += alpha * vln[inds]

        # 2nd gaussian smoothing pass
        L = laplacian(v, e)
        vln = L @ v
        if inds is None:
            v += -(alpha + 0.01) * vln
        else:
            v[inds] += -(alpha + 0.01) * vln[inds]
    return v


def load_mesh(filename: str, device='cuda', load_uv=False, load_aux=False):

    vm, fm = None, None
    if filename.endswith('.npz'):
        mesh = np.load(filename)
        v = torch.from_numpy(mesh['verts'])
        f = torch.from_numpy(mesh['faces'])

        if load_uv:
            vm = torch.from_numpy(mesh['uvs'])
            fm = torch.from_numpy(mesh['uvfaces'])
    else:
        if filename.endswith('.ply'):
            v, f = load_ply(filename)
        elif filename.endswith('.obj'):
            v, faces_attr, aux = load_obj(filename)
            f = faces_attr.verts_idx

            if load_uv:
                vm = aux.verts_uvs
                fm = faces_attr.textures_idx
        else:
            raise NotImplementedError(f'Unrecognized input format for: {filename}')

    v = v.to(device, non_blocking=True).contiguous()
    f = f.to(device, non_blocking=True).contiguous()

    if load_uv:
        vm = vm.to(device, non_blocking=True).contiguous()
        fm = fm.to(device, non_blocking=True).contiguous()

    if load_uv:
        if load_aux:
            return v, f, vm, fm, aux
        else:
            return v, f, vm, fm
    else:
        return v, f
