import torch
from tqdm import tqdm
from largesteps.optimize import AdamUniform
from largesteps.geometry import compute_matrix
from largesteps.parameterize import from_differential, to_differential
from utils import *


def main(
    delta_t=1 / 60,  # 1s for the simulation (physics time)
    S=15,  # 4s for the simulation (physics steps

    lr=4e-2,
    opt_iter=6000,
    ep_iter=20,
    lambda_smooth=1,
    input_file='tshirt.obj',
    output_file='tshirt-drop#{}.npz',
    device='cuda',
):

    v, f, vm, fm = load_mesh(input_file, load_uv=True, device=device)
    v = v + 1  # move up 1 meter
    V = len(v)

    # construct the actual garment
    material = StVKMaterial(material_multiplier=0.5)
    garment = Garment(v, f, vm, fm, material)

    v = v[None].expand(S, *v.shape).contiguous()  # S, V, 3
    f = f[None].expand(S, *f.shape).contiguous() + torch.arange(S, device=v.device)[..., None, None] * V  # S, F, 3

    M = compute_matrix(v.view(-1, 3), f.view(-1, 3), lambda_smooth)

    p = to_differential(M, v.view(-1, 3)).view(S, -1, 3)
    p = p.requires_grad_()
    o = AdamUniform([p], lr=lr)

    # stages = [1, 2, 3, 4, 5, 6, 10, 12, 20][::-1]

    pbar = tqdm(total=opt_iter)
    for i in range(opt_iter):
        # this part can also be considered as an optimization problem
        # solve constraints: with conditional gradient descent for optimization purpose
        q = from_differential(M, p.view(-1, 3)).view(S, -1, 3)  # S, V, 3

        q[:, 145] = garment.v[145]
        y = q[..., 1]
        q[..., 1] = y * (y > 0)

        energy_stretch = stretch_energy(q, garment)
        energy_bending = bending_energy(q, garment)
        energy_gravity = gravity_energy(q, garment.v_mass)
        energy_inertia = inertia_term_sequence(torch.cat([garment.v[None], q])[None], garment.v_mass, delta_t, method=inertia_term, compliance=1)  # fake batch dimension
        # stage_idx = int(i / opt_iter * len(stages))
        # stage_mul = stages[stage_idx]
        # energy_inertia = inertia_term_sequence(torch.cat([garment.v[None], q])[None][::stage_mul], garment.v_mass, delta_t * stage_mul, method=dynamic_term, dynamic_multiplier=1e5)  # fake batch dimension
        # this is actually the right formulation, but don't know why just won't work...: this term will only be valid when modeling just the dynamics
        # to obtain a balance between the dynamics and the other energy term, we need to reconsider the formulation
        # becomes a next level optimization problem that we'd have to solve in order to proceed to insert the loss into data terms
        # need to find a way to separate gravitational term and inertial term... otherwise the multiplier would just not work
        # energy_inertia = inertia_term_sequence(torch.cat([garment.v[None], q])[None], garment.v_mass, delta_t, method=dynamic_term, compliance=5)  # fake batch dimension
        # we can deal with this, but it might be too slow to converge? maybe data term would be too significant

        l = energy_stretch + energy_bending + energy_inertia + energy_gravity
        # l = energy_stretch + energy_bending + energy_inertia
        # l = energy_inertia + energy_gravity
        # l = energy_inertia

        o.zero_grad(set_to_none=True)
        l.backward()
        o.step()

        pbar.update(1)

        if i % ep_iter == 0:
            tqdm.write(f'Loss: {l.item()}')
            tqdm.write(f'Stretch energy: {energy_stretch.item()}')
            tqdm.write(f'Bending energy: {energy_bending.item()}')
            tqdm.write(f'Gravity energy: {energy_gravity.item()}')
            tqdm.write(f'Inertia energy: {energy_inertia.item()}')

    q = from_differential(M, p.detach().view(-1, 3)).view(S, -1, 3)
    q = torch.cat([garment.v[None], q])

    animated_mesh = dotdict()
    animated_mesh.animation = to_numpy(q, non_blocking=False)
    animated_mesh.faces = to_numpy(garment.f, non_blocking=False)
    export_dotdict(animated_mesh, filename=output_file.format('loss').replace('.ply', '.npz'))


if __name__ == '__main__':
    main()
