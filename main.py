import os, sys

# sys.path.append('../')
#
##cwd = os.getcwd()
#
# my_absolute_dirpath = os.path.abspath(os.path.dirname(__file__))
#
# module_path = os.path.abspath(os.path.join('../'))
#
# if module_path not in sys.path :
#    sys.path.append(module_path)

import numpy as np

# import matplotlib.pyplot as plt

from dolfin import *
from ufl import Index, unit_vector, shape, Jacobian, JacobianDeterminant, atan_2, Max
from save_data import *
from ActiveShell import *
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
dt_max = dt
dt_min = 1e-3 * dt_max
remeshing_frequency = float(
    config["remeshing"]["remeshing_frequency"]
)  # remeshing every n time steps

HyperOsmotic = int(config["simulation"]["HyperOsmotic"])
HypoOsmotic = int(config["simulation"]["HypoOsmotic"])

# Physical parameteres
inital_thickness = config["parameters"]["thickness"]
thick = Expression(inital_thickness, degree=4)
mu = float(config["parameters"]["viscosity"])
zeta = float(config["parameters"]["contractility_strength"])
basal = float(config["parameters"]["contractility_basal"])
gaussian_width = float(config["parameters"]["contractility_width"])
kd = float(config["parameters"]["depolymerization"])
vp = float(config["parameters"]["polymerization"])
q_11, q_12, q_22 = 1.0 / 6, 0.0, 1.0 / 6
Q_tensor = as_tensor([[1.0 / 6, 0.0], [0.0, 1.0 / 6]])
q_33 = -1.0 / 3

mesh_path = "/mesh/"

subprocess.call(
    [
        paths["gmsh"],
        "-2",
        "-format",
        "msh2",
        "../../" + geometry + ".geo",
        "-o",
        xdmf_name.replace("xdmf", "msh"),
    ]
)  # Hudson's
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

# print("cytokinesis-zeta_{}-kd_{}-vp_{}".format(zeta, kd, vp))
filename = output_dir + xdmf_name.replace(".xdmf", "_results.xdmf")
problem = NonlinearProblem_metric_from_mesh(
    mesh,
    mmesh,
    thick=thick,
    mu=mu,
    basal=basal,
    zeta=zeta,
    gaussian_width=gaussian_width,
    kd=kd,
    vp=vp,
    dt=dt,
    vol_ini=initial_volume,
    paths=paths,
    HyperOsmotic=HyperOsmotic,
    HypoOsmotic=HypoOsmotic,
    fname=filename,
)

# filename = 'output/data.csv'

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
    # problem.evolution(dt/2)
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


# r=open('output/final_radius.csv','w')
#
# np.savetxt(r, np.column_stack((zeta,radius(problem))),delimiter=',')
# r.close()


print("It ended at iteration {}, and Time {}".format(i, time))
