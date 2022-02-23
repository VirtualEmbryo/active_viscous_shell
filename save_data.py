import numpy as np
from dolfin import assemble, Constant, MeshFunction, AutoSubDomain, near, ds, dx, Function
from ufl import dot, pi


def save_data(filename, time, problem):
    # Volume

    current_volume = (
        2
        * assemble(dot(problem.phi0, problem.n0) * problem.dx(domain=problem.mesh))
        / pi
    )

    # Furrow radius
    boundary_subdomains = MeshFunction(
        "size_t", problem.mesh, problem.mesh.topology().dim() - 1
    )
    boundary_subdomains.set_all(0)
    boundary_y = lambda x, on_boundary: near(x[1], 0.0, 1.0e-3) and on_boundary
    AutoSubDomain(boundary_y).mark(boundary_subdomains, 1)
    dss = ds(subdomain_data=boundary_subdomains)
    current_radius = assemble((2.0 / pi) * dss(1)(domain=problem.mesh))
    Furrow_thickness = assemble(
        problem.thickness * (2.0 / (pi * current_radius)) * dss(1)(domain=problem.mesh)
    )
    thickness = assemble(problem.thickness*problem.dx(domain = problem.mesh))/assemble(Constant(1.0)*dx(domain = problem.mesh))

    #            furrow_radius = np.append(furrow_radius, current_radius )
    print("radius of the furrow:", current_radius)
    print("thickness of the furrow:", Furrow_thickness)

    # Dissipation
    membrane_total_dissipation = assemble(
        problem.psi_m * problem.dx(domain=problem.mesh)
    )
    membrane_passive_dissipation = assemble(
        problem.passive_membrane_energy * problem.dx(domain=problem.mesh)
    )
    membrane_active_dissipation = assemble(
        problem.active_membrane_energy * problem.dx(domain=problem.mesh)
    )
    membrane_polymerization_dissipation = assemble(
        problem.polymerization_membrane * problem.dx(domain=problem.mesh)
    )

    bending_total_dissipation = assemble(
        problem.psi_b * problem.dx(domain=problem.mesh)
    )
    bending_passive_dissipation = assemble(
        problem.passive_bending_energy * problem.dx(domain=problem.mesh)
    )
    bending_active_dissipation = assemble(
        problem.active_bending_energy * problem.dx(domain=problem.mesh)
    )
    bending_polymerization = assemble(
        problem.polymerization_bending * problem.dx(domain=problem.mesh)
    )

    dissipation_shear = assemble(problem.psi_s * problem.dx(domain=problem.mesh))

    Pressure = Function(problem.Q.sub(2).collapse())
    Pressure.interpolate(problem.q_.sub(2, True))
    Pressure_ = Pressure.vector()[0]
    print("Pressure = {}".format(Pressure_))

    data = np.column_stack(
        (
            time,
            current_volume,
            current_radius,
            membrane_total_dissipation,
            membrane_passive_dissipation,
            membrane_active_dissipation,
            membrane_polymerization_dissipation,
            bending_total_dissipation,
            bending_passive_dissipation,
            bending_active_dissipation,
            bending_polymerization,
            dissipation_shear,
            thickness,
            Pressure_,
        )
    )

    np.savetxt(filename, data, delimiter=";")
