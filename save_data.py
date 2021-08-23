import os, sys
import numpy as np
from dolfin import *
from ufl import Index, unit_vector, shape, Jacobian, JacobianDeterminant, atan_2, Max
import subprocess
import meshio

def save_data(filename, time, problem):
    # Volume
    
    current_volume = 2*assemble(dot(problem.phi0, problem.n0)*problem.dx(domain=problem.mesh))/pi
    if problem.geometry == "Hemisphere":
        current_radius = 0
    elif problem.geometry == "eighthsphere":
        # Furrow radius
        boundary_subdomains = MeshFunction("size_t", problem.mesh, problem.mesh.topology().dim() - 1)
        boundary_subdomains.set_all(0)
        boundary_y = lambda x, on_boundary: near(x[1], 0., 1.e-3) and on_boundary
        AutoSubDomain(boundary_y).mark(boundary_subdomains, 1)
        dss = ds(subdomain_data=boundary_subdomains)
        current_radius = assemble((2./pi) * dss(1)(domain = problem.mesh))
        Furrow_thickness = assemble(problem.thickness*(2./(pi*current_radius)) * dss(1)(domain = problem.mesh))

        #            furrow_radius = np.append(furrow_radius, current_radius )
        print("radius of the furrow:", current_radius)
        print("thickness of the furrow:", Furrow_thickness)


    # Dissipation
    membrane_total_dissipation  = assemble(problem.psi_m*problem.dx(domain=problem.mesh))
    membrane_passive_dissipation  = assemble(problem.passive_membrane_energy*problem.dx(domain=problem.mesh))
    membrane_active_dissipation = assemble(problem.active_membrane_energy*problem.dx(domain=problem.mesh))
    membrane_polymerization_dissipation = assemble(problem.polymerization_membrane*problem.dx(domain = problem.mesh))
    
    bending_total_dissipation   = assemble(problem.psi_b*problem.dx(domain=problem.mesh))
    bending_passive_dissipation   =  assemble(problem.passive_bending_energy*problem.dx(domain=problem.mesh))
    bending_active_dissipation = assemble(problem.active_bending_energy*problem.dx(domain=problem.mesh))
    bending_polymerization =  assemble(problem.polymerization_bending*problem.dx(domain = problem.mesh))
    
    dissipation_shear     = assemble(problem.psi_s*problem.dx(domain=problem.mesh))
    
#    furrow_indicator = K(degree = 0)
#
#    furrow_dissipation_m = assemble(problem.psi_m*furrow_indicator*problem.dx(domain=problem.mesh))
#    furrow_dissipation_b = assemble(problem.psi_b*furrow_indicator*problem.dx(domain=problem.mesh))
#
#    furrow_passive_dissipation_m =  assemble(problem.passive_membrane_energy*furrow_indicator*problem.dx(domain=problem.mesh))
#    furrow_passive_dissipation_b =  assemble(problem.passive_bending_energy*furrow_indicator*problem.dx(domain=problem.mesh))
#
#    furrow_polymerization_m =  assemble(furrow_indicator*problem.polymerization_membrane*problem.dx(domain = problem.mesh))
#    furrow_polymerization_b =  assemble(furrow_indicator*problem.polymerization_bending*problem.dx(domain = problem.mesh))

    Pressure = Function(problem.Q.sub(2).collapse())
    Pressure.interpolate(problem.q_.sub(2,True))
    Pressure_ = Pressure.vector()[0]
    print("Pressure = {}".format(Pressure_))


    data = np.column_stack((time, current_volume , current_radius, membrane_total_dissipation, membrane_passive_dissipation, membrane_active_dissipation, membrane_polymerization_dissipation, bending_total_dissipation, bending_passive_dissipation, bending_active_dissipation, bending_polymerization, dissipation_shear, Furrow_thickness, Pressure_))

    np.savetxt(filename, data,  delimiter=';')

#class K(UserExpression):
#    def eval(self, value, x):
#        tol = 1E-14
#        if x[1] <= 0.2 + tol:
#            value[0] = 1.
#        else:
#            value[0] = 0.
#
#
