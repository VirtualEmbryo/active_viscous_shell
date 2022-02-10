import os
import numpy as np
from dolfin import (
    parameters,
    XDMFFile,
    Expression,
    Mesh,
    MeshFunction,
    near,
    AutoSubDomain,
    ds,
    assemble,
    dx,
)
from ufl import as_tensor, pi
from save_data import save_data
from active_shell import ActiveShell
import subprocess
import meshio
import configreader
import json

# System parameters

parameters["form_compiler"]["quadrature_degree"] = 4
parameters["allow_extrapolation"] = True


cwd = os.getcwd()
print(cwd)
# Create config object
C = configreader.Config()
config = C.read("config.conf")


output_dir = "output/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Opening JSON file
with open("../../paths.json") as json_file:
    paths = json.load(json_file)

geometry = "eighthsphere"
xdmf_name = geometry + ".xdmf"


# Simulation parameters
time = 0
Time = float(config["simulation"]["Time_max"])
dt = float(config["simulation"]["timestep"])
polymerization = int(config["simulation"]["polymerization"])

remeshing_frequency = float(
    config["remeshing"]["remeshing_frequency"]
)  # remeshing every n time steps


# Physical parameteres
inital_thickness = config["parameters"]["thickness"]
thick = Expression(inital_thickness, degree=4)
mu = float(config["parameters"]["viscosity"])
zeta = float(config["parameters"]["contractility_strength"])
basal = float(config["parameters"]["contractility_basal"])
gaussian_width = float(config["parameters"]["contractility_width"])
kd = float(config["parameters"]["depolymerization"])
vp = float(config["parameters"]["polymerization"])

Q_tensor = as_tensor([[1.0 / 6, 0.0], [0.0, 1.0 / 6]])
q_33 = -1.0 / 3

# Volume variation
dV = config["parameters"].get("volume_variation", "0")

subprocess.call(
    [
        paths["gmsh"],
        "-2",
        "-format",
        "msh2",
        "-v",
        "1",
        "../../" + geometry + ".geo",
        "-o",
        xdmf_name.replace("xdmf", "msh"),
    ]
)
msh = meshio.read(xdmf_name.replace(".xdmf", ".msh"))
meshio.write(
    xdmf_name,
    meshio.Mesh(points=msh.points, cells={"triangle": msh.cells_dict["triangle"]}),
)
mmesh = meshio.read(xdmf_name)
mesh = Mesh()
with XDMFFile(xdmf_name) as mesh_file:
    mesh_file.read(mesh)


def radius(problem):
    # Furrow radius
    boundary_subdomains = MeshFunction(
        "size_t", problem.mesh, problem.mesh.topology().dim() - 1
    )
    boundary_subdomains.set_all(0)
    boundary_y = lambda x, on_boundary: near(x[1], 0.0, 1.0e-3) and on_boundary
    AutoSubDomain(boundary_y).mark(boundary_subdomains, 1)
    dss = ds(subdomain_data=boundary_subdomains)
    return assemble((2.0 / pi) * dss(1)(domain=problem.mesh))


initial_volume = assemble(1.0 * dx(domain=mesh)) / 3
print("Initial volume:", initial_volume)


current_radius = 1.0


filename = output_dir + xdmf_name.replace(".xdmf", "_results.xdmf")
problem = ActiveShell(
    mesh,
    mmesh,
    thick=thick,
    mu=mu,
    basal=basal,
    zeta=zeta,
    gaussian_width=gaussian_width,
    kd=kd,
    vp=vp,
    Q_tensor=Q_tensor,
    q_33=q_33,
    dt=dt,
    vol_ini=initial_volume,
    paths=paths,
    dV=dV,
    fname=filename,
)

hdr = "time, current_volume , current_radius, membrane_total_dissipation, membrane_passive_dissipation, membrane_active_dissipation, membrane_polymerization_dissipation, bending_total_dissipation, bending_passive_dissipation, bending_active_dissipation, bending_polymerization, dissipation_shear, furrow_thickness"

f = open("output/Data.csv", "w")
np.savetxt(f, [], header=hdr)

problem.write(
    time,
    u=True,
    beta=True,
    phi=True,
    frame=True,
    epaisseur=True,
    activity=True,
    energies=True,
)
i = 0
radius_old = 1.0
d_radius = 1
while time < Time:
    i += 1
    print("Iteration {}. Time step : {}".format(i, dt))

    problem.initialize()

    niter, _ = problem.solve()

    problem.evolution(dt)
    current_radius = radius(problem)
    d_radius = abs(current_radius - radius_old)
    print("Variation in radius: {}".format(d_radius))
    radius_old = current_radius

    problem.write(
        time + dt,
        u=True,
        beta=True,
        phi=True,
        frame=True,
        epaisseur=True,
        activity=True,
        energies=True,
    )
    print(
        "rmin={}, rmax={}, hmin={}, hmax={}".format(
            problem.mesh.rmin(),
            problem.mesh.rmax(),
            problem.mesh.hmin(),
            problem.mesh.hmax(),
        )
    )

    save_data(f, time, problem)
    time += dt

    if i % remeshing_frequency == 0:
        if problem.mesh.rmin() < 1.5e-3:
            problem.mesh_refinement("hsiz")
            print("Uniform mesh!")
        else:
            problem.mesh_refinement("hausd")
            print("Hausdorff distance")

    if (
        current_radius < 0.06
    ):  # If the furrow radius is smaller than twice the thickness it means that it should have stopped dividing!
        problem.write(
            time + dt,
            u=True,
            beta=True,
            phi=True,
            frame=True,
            epaisseur=True,
            activity=True,
            energies=False,
        )
        break


f.close()

print("It ended at iteration {}, and Time {}".format(i, time))
