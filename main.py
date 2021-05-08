import os, sys

import numpy as np
import matplotlib.pyplot as plt

from dolfin import *
from ufl import Index, unit_vector, shape, Jacobian, JacobianDeterminant, atan_2, Max
from mshr import *
from mpl_toolkits.mplot3d import Axes3D
from adapt_fix import adapt
import subprocess

import meshio

class CustomNonlinearProblem(NonlinearProblem):

    def __init__(self, L, a, bcs):
        NonlinearProblem.__init__(self)
        self.L = L
        self.a = a
        self.bcs = bcs

    def form(self, A, P, b, x):
        # this function is called before calling F or J
        assemble_system(self.a, self.L, A_tensor=A, b_tensor=b, bcs=self.bcs, x0=x)

    def F(self,b,x):
        pass

    def J(self,A,x):
        pass

class SurfaceNormal(UserExpression):
    def __init__ (self, mesh, mmesh, **kwargs):
        super(SurfaceNormal, self).__init__(**kwargs)
        self.mesh = mesh
        self.mmesh = mmesh
        self.mesh.init_cell_orientations(Expression(('x[0]', 'x[1]', 'x[2]'), degree = 0))


    def eval_cell(self, value, x, ufc_cell):
        nodes = self.mmesh.cells_dict["triangle"][ufc_cell.index, :]
        x = self.mesh.coordinates()[nodes, :]
        value[:] = np.cross(x[0, :] - x[1, :], x[0, :] - x[2, :])
        value /= np.linalg.norm(value)
        # For the spherical mesh the normal was not well oriented
        value[self.mesh.cell_orientations() == 1] *= -1

    def value_shape(self):
        return (3,)

boundary_x = lambda x, on_boundary: near(x[0], 0., 1.e-3) and on_boundary
boundary_y = lambda x, on_boundary: near(x[1], 0., 1.e-3) and on_boundary
boundary_z = lambda x, on_boundary: near(x[2], 0., 1.e-3) and on_boundary

def save_data(filename, problem):
    # Volume
    
    current_volume = 2*assemble(dot(problem.phi0, problem.n0)*problem.dx(domain=problem.mesh))/pi

    # Furrow radius
    boundary_subdomains = MeshFunction("size_t", problem.mesh, problem.mesh.topology().dim() - 1)
    boundary_subdomains.set_all(0)
    AutoSubDomain(boundary_y).mark(boundary_subdomains, 1)
    dss = ds(subdomain_data=boundary_subdomains)
    current_radius = assemble((2./pi) * dss(1)(domain = problem.mesh))
    #            furrow_radius = np.append(furrow_radius, current_radius )
    print("radius of the furrow:", current_radius)

    # Dissipation
    dissipation_membrane  = assemble(problem.psi_m*problem.dx(domain=problem.mesh))
    passive_dissipation_membrane  = assemble(problem.passive_membrane_energy*problem.dx(domain=problem.mesh))
    dissipation_bending   = assemble(problem.psi_b*problem.dx(domain=problem.mesh))
    passive_dissipation_bending   =  assemble(problem.passive_bending_energy*problem.dx(domain=problem.mesh))
    dissipation_shear     = assemble(problem.psi_s*problem.dx(domain=problem.mesh))
    polymerization_membrane = assemble(problem.polymerization_membrane*problem.dx(domain = problem.mesh))
    polymerization_bending =  assemble(problem.polymerization_bending*problem.dx(domain = problem.mesh))

    furrow_dissipation_m = assemble(problem.psi_m*furrow_indicator*problem.dx(domain=problem.mesh))
    furrow_dissipation_b = assemble(problem.psi_b*furrow_indicator*problem.dx(domain=problem.mesh))

    furrow_passive_dissipation_m =  assemble(problem.passive_membrane_energy*furrow_indicator*problem.dx(domain=problem.mesh))
    furrow_passive_dissipation_b =  assemble(problem.passive_bending_energy*furrow_indicator*problem.dx(domain=problem.mesh))

    furrow_polymerization_m =  assemble(furrow_indicator*problem.polymerization_membrane*problem.dx(domain = problem.mesh))
    furrow_polymerization_b =  assemble(furrow_indicator*problem.polymerization_bending*problem.dx(domain = problem.mesh))


    data = np.column_stack((time, current_volume , current_radius, dissipation_membrane, dissipation_bending, dissipation_shear,
    passive_dissipation_membrane, passive_dissipation_bending, polymerization_membrane, polymerization_bending, furrow_dissipation_m, furrow_dissipation_b, furrow_passive_dissipation_m, furrow_passive_dissipation_b, furrow_polymerization_m, furrow_polymerization_b ))

    np.savetxt(filename, data,  delimiter=';')


def local_frame(mesh, normal=None):
    t = Jacobian(mesh)
    J = JacobianDeterminant(mesh)
    if mesh.geometric_dimension()==2:
        tt1 = as_vector([t[0, 0], t[1, 0], 0])
        tt2 = as_vector([t[0, 1], t[1, 1], 0])
    else:
        tt1 = as_vector([t[0, 0], t[1, 0], t[2, 0]])
        tt2 = as_vector([t[0, 1], t[1, 1], t[2, 1]])
    # switch t1 and t2 if Jacobian is negative
    t1 = conditional(J<0, tt2, tt1)
    t2 = conditional(J<0, tt1, tt2)
    n = cross(t1, t2)
    if normal is not None:
        n = conditional(dot(normal, n) <0, -n, n)
    n /= sqrt(dot(n, n))
    ey = as_vector([0, 1, 0])
    ez =  as_vector([0, 0, 1])
    a1 = cross(ey, n)
    norm_a1 = sqrt(dot(a1, a1))
    a1 = conditional(lt(norm_a1, DOLFIN_EPS), ez, a1/norm_a1)

    a2 = cross(n, a1)
    a2 /= sqrt(dot(a2, a2))
    n = cross(a1, a2)
    n /= sqrt(dot(n, n))
    return a1, a2, n
    

class NonlinearProblem_metric_from_mesh:
    def __init__(self, mesh, mmesh, thick, mu, zeta, kd, vp, vol_ini, fname = None,
                 hypothesis="small strain", geometry = "eighthsphere", LE = "False"):
        self.mesh = mesh
        self.mmesh = mmesh
        self.thick = thick
        self.mu = mu
        self.zeta = zeta
        self.kd = kd
        self.vp = vp
        self.vol_ini = vol_ini
        self.hypothesis = hypothesis
        self.geometry = geometry
        self.set_solver()
        self.set_functions_space()
        self.thickness.interpolate(thick)
        self.initialize()
        self.LE = LE
        self.fname = fname
        
        if fname is not None:
            self.output_file = XDMFFile(fname)
            self.output_file.parameters["functions_share_mesh"] = True
            self.output_file.parameters["flush_output"] = True

    def write(self, i, u = True, beta = True, phi = True, frame = False, epaisseur = False, activity = False, energies = True):
        if u:
            u = self.q_.sub(0, True)
            u.rename("u", "u")
            self.output_file.write(u, i)
        if beta:
            beta = self.q_.sub(1, True)
            beta.rename("beta", "beta")
            self.output_file.write(beta, i)
        if phi:
            self.phi0.rename("phi0", "phi0")
            self.output_file.write(self.phi0, i)
        if frame:
            self.a1.rename("a1", "a1")
            self.output_file.write(self.a1, i)
            self.a2.rename("a2", "a2")
            self.output_file.write(self.a2, i)
            self.n0.rename("n0", "n0")
            self.output_file.write(self.n0, i)
            
            director = project(self.d, self.VT)
            director.rename("director", "director")
            self.output_file.write(director, i)
            
            mean_curvature = project(self.H, self.V_thickness)
            mean_curvature.rename("Meancurvature", "Meancurvature")
            self.output_file.write(mean_curvature, i)
            
        if epaisseur:
            self.thickness.rename("thickness", "thickness")
            self.output_file.write(self.thickness, i)

        if activity:
            self.Q_field = project(self.Q_field, self.V_thickness)
            self.Q_field.rename("activity", "activity")
            self.output_file.write(self.Q_field, i)

        if energies:
            psi_m = project(self.psi_m, self.V_thickness)
            psi_b = project(self.psi_b, self.V_thickness)
            psi_s = project(self.psi_s, self.V_thickness)
            
            psi_m.rename("psi_m", "psi_m")
            psi_b.rename("psi_b", "psi_b")
            psi_s.rename("psi_s", "psi_s")
            self.output_file.write(psi_m, i)
            self.output_file.write(psi_b, i)
            self.output_file.write(psi_s, i)

    def set_functions_space(self):
        P2 = FiniteElement("Lagrange", self.mesh.ufl_cell(), degree = 2)
        P1 = FiniteElement("Lagrange", self.mesh.ufl_cell(), degree = 2)
        CR1 = FiniteElement("CR", self.mesh.ufl_cell(), degree = 1)
        R = FiniteElement("Real", self.mesh.ufl_cell(), degree=0)
        if self.geometry == "Hemisphere":
            element = MixedElement([VectorElement(P2, dim=3), VectorElement(CR1, dim=2), R, VectorElement(R, dim=3)])
        elif self.geometry == "eighthsphere" :
            element = MixedElement([VectorElement(P2, dim=3), VectorElement(CR1, dim=2), R])

        self.Q = FunctionSpace(self.mesh, element)
        self.q_, self.q, self.q_t = Function(self.Q), TrialFunction(self.Q), TestFunction(self.Q)
  
#        self.u_t, self.beta_t, self.lbda_t, self.rigid_t = split(self.q_t)
        if self.geometry == "Hemisphere":
            self.u_, self.beta_, self.lbda_ , self.rigid_ = split(self.q_)
            self.u, self.beta, self.lbda, self.rigid = split(self.q)
        elif self.geometry == "eighthsphere":
            self.u_, self.beta_, self.lbda_= split(self.q_)
            self.u, self.beta, self.lbda = split(self.q)

            
        self.V_phi = self.Q.sub(0).collapse() #FunctionSpace(self.mesh, VectorElement("P", mesh.ufl_cell(), degree = 2, dim = 3))
        self.V_beta = self.Q.sub(1).collapse() #FunctionSpace(self.mesh, VectorElement("P", mesh.ufl_cell(), degree = 2, dim = 2))
        self.V_thickness = FunctionSpace(self.mesh, P1)
        self.V_alpha = FunctionSpace(self.mesh, "DG", 0)
        self.VT = VectorFunctionSpace(self.mesh, "DG", 0, dim = 3)
        self.V_normal = self.Q.sub(0).collapse()
        self.a1 = Function(self.VT)
        self.a2 = Function(self.VT)
        self.n0 = Function(self.VT)
#        self.d0 = Function(self.VT)
        
        self.thickness = Function(self.V_thickness)
        x = Expression(("x[0]", "x[1]", "x[2]"), degree=1)
        self.mesh.init_cell_orientations(x)
        self.n0.interpolate(x)
        
    def set_boundary_conditions(self):
        # Re-definition of the boundary for the new mesh
        boundary = lambda x, on_boundary: on_boundary
        boundary_x = lambda x, on_boundary: near(x[0], 0., 1.e-3) and on_boundary
        boundary_y = lambda x, on_boundary: near(x[1], 0., 1.e-3) and on_boundary
        boundary_z = lambda x, on_boundary: near(x[2], 0., 1.e-3) and on_boundary
        
        if self.geometry == "Hemisphere":
            bc_sphere_disp = DirichletBC(self.Q.sub(0).sub(2), Constant(0.), boundary)
            bc_sphere_rot = DirichletBC(self.Q.sub(1).sub(1), Constant(0.), boundary)
            
            self.bcs = [bc_sphere_disp, bc_sphere_rot]

        elif self.geometry == "eighthsphere":
            bc_sphere_disp_x = DirichletBC(self.Q.sub(0).sub(0), Constant(0.), boundary_x)
            bc_sphere_disp_y = DirichletBC(self.Q.sub(0).sub(1), Constant(0.), boundary_y)
            bc_sphere_disp_z = DirichletBC(self.Q.sub(0).sub(2), Constant(0.), boundary_z)
            
            bc_sphere_rot_y = DirichletBC(self.Q.sub(1), Constant((0., 0.)), boundary_y)
            bc_sphere_rot_x = DirichletBC(self.Q.sub(1).sub(1), Constant(0.), boundary_x)
            bc_sphere_rot_z = DirichletBC(self.Q.sub(1).sub(1), Constant(0.), boundary_z)
            
            self.bcs = [bc_sphere_disp_x, bc_sphere_disp_y, bc_sphere_disp_z, bc_sphere_rot_x, bc_sphere_rot_y, bc_sphere_rot_z]
    
        
    def boundary_conditions_n0(self):
        if self.geometry == "eighthsphere":
            boundary_x = lambda x, on_boundary: near(x[0], 0., 1.e-3) and on_boundary
            boundary_y = lambda x, on_boundary: near(x[1], 0., 1.e-3) and on_boundary
            boundary_z = lambda x, on_boundary: near(x[2], 0., 1.e-3) and on_boundary
            
            bc_n0_x = DirichletBC(self.V_normal.sub(0), Constant(0.), boundary_x)
            bc_n0_y = DirichletBC(self.V_normal.sub(1), Constant(0.), boundary_y)
            bc_n0_z = DirichletBC(self.V_normal.sub(2), Constant(0.), boundary_z)
            bcs_n0 = [bc_n0_x, bc_n0_y, bc_n0_z]
        
        elif self.geometry == "Hemisphere":
            boundary_z = lambda x, on_boundary: on_boundary
            bcs_n0 = [DirichletBC(self.V_normal.sub(2), Constant(0.), boundary_z)]

        return bcs_n0
        
    def set_solver(self):
        self.solver = PETScSNESSolver()
        self.solver.parameters["method"] = "newtonls"
        self.solver.parameters['maximum_iterations'] = 20
        self.solver.parameters['linear_solver'] = "mumps"
        self.solver.parameters['absolute_tolerance'] = 1E-6
        self.solver.parameters['relative_tolerance'] = 1E-6
#        self.solver = NewtonSolver()
#        self.solver.parameters['maximum_iterations'] = 50
#        #solver.parameters['linear_solver'] = "mumps"
#        self.solver.parameters['absolute_tolerance'] = 1E-6
#        self.solver.parameters['relative_tolerance'] = 1E-6


    def set_shape(self):
        x = SpatialCoordinate(self.mesh)
        initial_shape = x
        self.phi0 = project(initial_shape, self.V_phi)

    def set_local_frame(self):
        a1, a2, n = local_frame(self.mesh, self.n0)
        self.a1.assign(project(a1, self.VT, form_compiler_parameters={"representation":"uflacs",
                                                            "quadrature_degree": 4}))
        self.a2.assign(project(a2, self.VT, form_compiler_parameters={"representation":"uflacs",
                                                            "quadrature_degree": 4}))
                                                            
        
        self.n0.assign(project(n, self.V_normal, form_compiler_parameters={"representation":"uflacs",
                                                            "quadrature_degree": 4}))
        bcs = self.boundary_conditions_n0()
        for bc in bcs:
            bc.apply(self.n0.vector())


    def director(self, beta):
         return as_vector([sin(beta[1])*cos(beta[0]), -sin(beta[0]), cos(beta[1])*cos(beta[0])])
#         return as_vector([sin(beta[1])*cos(beta[0]), sin(beta[0])*sin(beta[1]), cos(beta[1])]) # Spherical coord


    def d_director(self, beta, beta_):
        """ linearized director """
        B0 = as_matrix([
                        [-sin(beta[0])*sin(beta[1]),  cos(beta[0])*cos(beta[1])],
                        [-cos(beta[0])             ,  0.                       ],
                        [-sin(beta[0])*cos(beta[1]), -cos(beta[0])*sin(beta[1])]
                        ])
                   
        return dot(B0, beta_)



    def set_director(self):
        n = self.n0
#        self.beta0 = project(as_vector([atan_2(-n[1], sqrt(n[0]**2 + n[2]**2)),
#                                      atan_2(n[0], n[2])]), VectorFunctionSpace(self.mesh, "P", 1, dim = 2))

        self.beta0 = project(as_vector([atan_2(-n[1], sqrt(n[0]**2 + n[2]**2)),
                                        atan_2(n[0], n[2]+DOLFIN_EPS)]), self.V_beta)
        # The director in the initial configuration is then written as ::
        self.d0 = self.director(self.beta0)


    def d1(self, u):
        return u.dx(0)*self.a1[0]+u.dx(1)*self.a1[1]+u.dx(2)*self.a1[2]
    def d2(self, u):
        return u.dx(0)*self.a2[0]+u.dx(1)*self.a2[1]+u.dx(2)*self.a2[2]
    def grad_(self, u):
        return as_tensor([self.d1(u), self.d2(u)])

    def set_fundamental_forms(self):
        self.a0_init = as_tensor([[dot(self.a1, self.a1), dot(self.a1, self.a2)],\
                                  [dot(self.a2, self.a1), dot(self.a2, self.a2)]])

        self.b0_init = - sym(as_tensor([[inner(as_vector(self.grad_(self.d0)[0,:]), self.a1), inner(as_vector(self.grad_(self.d0)[0,:]), self.a2)],\
                                        [inner(as_vector(self.grad_(self.d0)[1,:]), self.a1), inner(as_vector(self.grad_(self.d0)[1,:]), self.a2)]]))


    def set_kinematics_and_fundamental_forms(self):
        self.F = self.grad_(self.u_) + self.grad_(self.phi0)
        self.F0 = self.grad_(self.phi0)
        self.g_u = self.grad_(self.u_)

        self.d = self.director(self.beta_ + self.beta0)

        self.a0 = as_tensor([[dot(self.a1, self.a1), dot(self.a1, self.a2)],\
                            [dot(self.a2, self.a1), dot(self.a2, self.a2)]])

        self.b0 = - sym(as_tensor([[dot(as_vector(self.grad_(self.d0)[0,:]), self.a1), inner(as_vector(self.grad_(self.d0)[0,:]), self.a2)],\
                                   [inner(as_vector(self.grad_(self.d0)[1,:]), self.a1), dot(as_vector(self.grad_(self.d0)[1,:]), self.a2)]]))
                                   
        self.j0 = det(self.a0)

        self.a0_contra = inv(self.a0)
        self.H = 0.5*inner(self.a0_contra, self.b0)

    def membrane_deformation(self):
        if self.hypothesis == "finite strain":
            return 0.5*(self.F*self.F.T - self.a0)
        elif self.hypothesis == "small strain":
            return 0.5*(self.F0*self.F0.T - self.a0 + self.F0*self.g_u.T + self.g_u*self.F0.T)

    def bending_deformation(self):
        if self.hypothesis == "finite strain":
            return -0.5*(self.F*self.grad_(self.d).T + self.grad_(self.d)*self.F.T) - self.b0
        elif self.hypothesis == "small strain":
            dd = self.d_director(self.beta0, self.beta_)
            return -0.5*(self.F0*self.grad_(self.d0).T + self.grad_(self.d0)*self.F0.T +
                         self.g_u*self.grad_(self.d0).T + self.grad_(self.d0)*self.g_u.T +
                         self.F0*self.grad_(dd).T + self.grad_(dd)*self.F0.T) - self.b0


    def shear_deformation(self):
        if self.hypothesis == "finite strain":
            return self.F*self.d - self.grad_(self.phi0)*self.d0
        elif self.hypothesis == "small strain":
            return self.g_u*self.d0 + self.F0*self.d_director(self.beta0, self.beta_) #- self.grad_(self.phi0)*self.d0

    
    def set_thickness(self, dt):
        D_Delta= inner(self.a0_contra, self.membrane_deformation())
        
        if self.LE:
            normal_velocity = Function(self.V_normal)
            normal_velocity.assign(project(dot(outer(self.n0,self.n0),self.u_), self.V_normal))  # The mesh motion is simply the normal component of the displacement

            bcs = self.boundary_conditions_n0()
            for bc in bcs:
                bc.apply(normal_velocity.vector())

            a = inner(self.q_.sub(0, True) - normal_velocity, grad(self.thickness))
#            print("Shape of grad T: ", grad(self.thickness).ufl_shape)
            self.thickness.assign(project((self.thickness  + dt*(- a - self.thickness*D_Delta - self.kd*self.thickness + self.vp*(1-0.*self.H*self.thickness))),self.V_thickness))
        
        else:
            self.thickness.assign(project((self.thickness/dt + self.vp)/(1/dt + D_Delta + self.kd + self.vp*self.H),self.V_thickness))

   
   
   
    def set_energies(self):
        # Gaussian signal in the middle of the plate and uniform across one of the directions
        sig_q = 0.2;
        basal = 1.
    
        self.Q_Expression = Expression(('basal + (zeta - 1.)*exp(-0.5*(x[1]*x[1])/(sig_q*sig_q))'), sig_q = sig_q, basal = basal, zeta = self.zeta, degree = 2)

        self.Q_field = interpolate(self.Q_Expression, self.V_thickness)
        self.q_11, self.q_12, self.q_22, self.q_33 = 1.0/6, 0., 1./6, -1./3
        self.Q_tensor = as_tensor([[1./6, 0.0], [0.0, 1./6]])

        i, j, l, m = Index(), Index(), Index(), Index()
        A_ = as_tensor((0.5*self.a0_contra[i,j]*self.a0_contra[l,m]
                       + 0.25*(self.a0_contra[i,l]*self.a0_contra[j,m] + self.a0_contra[i,m]*self.a0_contra[j,l]))
                       ,[i,j,l,m])
        C_active = as_tensor(self.Q_tensor[l,m]*(self.H*(self.a0_contra[i,l]*self.a0_contra[j,m]+self.a0_contra[j,l]*self.a0_contra[i,m]) + self.a0_contra[i,j] * self.b0[m,l]-\
                0.75*(self.a0_contra[i,m] * self.b0[j,l] + self.a0_contra[i,l] * self.b0[j,m] + self.a0_contra[j,m]*self.b0[i,l] + self.a0_contra[j,l]*self.b0[i,m]))
                           +(self.b0[i,j]-4*self.H*self.a0_contra[i,j])*self.q_33
                            ,[i,j])
        B_ = 0.25*as_tensor(8*self.H*(0.5*self.a0_contra[i,j]*self.a0_contra[l,m] + 0.25*(self.a0_contra[i,l]*self.a0_contra[j,m] + self.a0_contra[i,m]*self.a0_contra[j,l])) - 2*self.a0_contra[i,j]*self.b0[l,m] - 4*self.a0_contra[l,m]*self.b0[i,j] - 3*(self.a0_contra[i,l]*self.b0[j,m] + self.a0_contra[j,m]*self.b0[i,l])
        ,[i,j,l,m])

        C_ = as_tensor(8*self.H*(0.5*self.a0_contra[i,j]*self.a0_contra[l,m] + 0.25*(self.a0_contra[i,l]*self.a0_contra[j,m] + self.a0_contra[i,m]*self.a0_contra[j,l])) - self.a0_contra[i,j]*self.b0[l,m] - self.a0_contra[l,m]*self.b0[i,j] - 0.5*(self.a0_contra[i,l]*self.b0[j,m]+self.a0_contra[j,l]*self.b0[i,m] + self.a0_contra[i,m]*self.b0[j,l]+self.a0_contra[j,m]*self.b0[i,l])
        ,[i,j,l,m])
        
        self.N = self.thickness*self.mu *as_tensor(A_[i,j,l,m]*self.membrane_deformation()[l,m] + self.a0_contra[i,j]*(self.kd - self.vp/self.thickness), [i,j])+ \
                (self.thickness**3/12.0)*self.mu*as_tensor(C_[i,j,l,m]*self.bending_deformation()[l,m],[i,j])+\
            self.Q_field*self.thickness*as_tensor((self.a0_contra[i,l]*self.a0_contra[j,m]*self.Q_tensor[l,m] - self.a0_contra[i,j]*self.q_33),[i,j])

        self.M = (self.thickness**3/3.0)*self.mu*as_tensor(A_[i,j,l,m]*self.bending_deformation()[l,m] + B_[i,j,l,m]*self.membrane_deformation()[l,m] + (self.H*self.a0_contra[i,j] - self.b0[i,j])*(self.kd-self.vp/self.thickness),[i,j])\
            + (self.thickness**3/12.)*C_active*self.Q_field

        self.T = self.thickness*self.mu*as_tensor(self.a0_contra[i,j]*self.shear_deformation()[j], [i])

        self.passive_membrane_energy = 0.5*self.thickness*self.mu* inner(as_tensor(A_[i,j,l,m]*self.membrane_deformation()[l,m], [i,j]), self.membrane_deformation())
        self.passive_bending_energy = 0.5*(self.thickness**3/3.0)*self.mu* inner(as_tensor(A_[i,j,l,m]*self.bending_deformation()[l,m], [i,j]), self.bending_deformation())
        self.polymerization_membrane = 0.5*self.thickness*self.mu* inner(as_tensor(self.a0_contra[i,j]*(self.kd - self.vp/self.thickness), [i,j]), self.membrane_deformation())
        self.polymerization_bending = 0.5*(self.thickness**3/3.0)*self.mu* inner(as_tensor((self.H*self.a0_contra[i,j] - self.b0[i,j])*(self.kd-self.vp/self.thickness), [i,j]), self.bending_deformation())

        self.psi_m = 0.5*inner(self.N, self.membrane_deformation())
        self.psi_b = 0.5*inner(self.M, self.bending_deformation())
        self.psi_s = 10*0.5*inner(self.T, self.shear_deformation()) # Shear penalisation should multiply by 1e2 or 1e3

    def total_energy(self):
        # Total Energy densities
        self.dx = Measure('dx', domain = self.mesh)

        # The total elastic energy and its first and second derivatives
        self.Pi = self.psi_m*sqrt(self.j0)*self.dx + self.psi_b*sqrt(self.j0)*self.dx + self.psi_s*sqrt(self.j0)*self.dx
        if self.geometry == "Hemisphere":
            self.Pi = self.Pi + dot(self.rigid_,self.u_)*sqrt(self.j0)*self.dx # Remotion of rigid motion
        self.Pi = self.Pi + self.lbda_*dot(self.u_, self.n0)*sqrt(self.j0)*self.dx # Volume conservation
        
        self.dPi = derivative(self.Pi, self.q_, self.q_t)# + self.lbda_*dot(self.u_t, self.n0)*sqrt(self.j0)*self.dx+ self.lbda_t*dot(self.u_, self.n0)*sqrt(self.j0)*self.dx
        self.J = derivative(self.dPi, self.q_, self.q)



    def initialize(self):
        self.set_shape()
        self.set_local_frame()
        self.set_boundary_conditions()
        self.set_director()
        self.set_fundamental_forms()
        self.set_kinematics_and_fundamental_forms()
        self.set_energies()
        self.total_energy()

    def evolution(self, dt):
    
        self.set_thickness(dt)
    
        displacement_mesh = Function(self.V_phi)
        displacement_mesh.interpolate(self.q_.sub(0, True))
        displacement_mesh.vector()[:] *= dt
#        self.phi0.assign(self.phi0 + displacement_mesh) # Lagrangian

        if self.LE:
            normal_displacement = Function(self.V_phi)

            normal_displacement.assign(project(inner(self.q_.sub(0, True),self.n0)*self.n0,self.V_phi))
#            normal_displacement.assign(project(dot(displacement_mesh, self.n0)*self.n0, self.V_normal))

            bcs = self.boundary_conditions_n0()
            for bc in bcs:
                bc.apply(normal_displacement.vector())
            normal_displacement.vector()[:] *= dt
            ALE.move(self.mesh, normal_displacement)

        else:
            ALE.move(self.mesh, displacement_mesh)
        self.initialize()


    def solve(self):
        problem = CustomNonlinearProblem(self.dPi, self.J, self.bcs)
        return self.solver.solve(problem, self.q_.vector())


    def mesh_refinement(self ):
        with XDMFFile("mesh.xdmf") as ffile:
             ffile.parameters["functions_share_mesh"] = True
             ffile.write(self.mesh)


        # Convert to Medit format
        os.system('meshio-convert --input-format xdmf --output-format medit mesh.xdmf mesh.mesh')

        dist = 0.001 # controls the optimized mesh resolution (see https://www.mmgtools.org/mmg-remesher-try-mmg/mmg-remesher-options/mmg-remesher-option-hausd)
         # call mmgs with mesh optimization and Hausdorff distance
        os.system('bin/mmgs_O3 -in mesh.mesh -out mesh_optimized.mesh -hausd {}'.format(dist)) # Hudson's
#        os.system('/opt/mmg/bin/mmgs_O3 -in mesh.mesh -out mesh_optimized.mesh -hausd {} -optim'.format(dist)) # Jeremy's

        # Convert back to .msh format using Gmsh
        os.system(' /Applications/Gmsh.app/Contents/MacOS/gmsh mesh_optimized.mesh -3 -o mesh_optimized.msh')
#        os.system('gmsh mesh_optimized.mesh -3 -o mesh_optimized.msh') # Jeremy's

        fname = "mesh.xdmf"
        fname_out = "mesh_optimized.xdmf"

        # read back with meshio to remove cells and cell data and convert to xdmf
        mmesh = meshio.read(fname_out.replace(".xdmf", ".msh"))
        #mmesh.remove_lower_dimensional_cells()
        #meshio.write(fname_out, meshio.Mesh(mmesh.points, mmesh.cells))
        meshio.write(fname_out,
                        meshio.Mesh(points = mmesh.points,
                               cells = {'triangle': mmesh.cells_dict['triangle']}))
        self.mmesh = meshio.read(fname_out)

        # read in with FEniCS
        new_mesh = Mesh()
        with XDMFFile(fname_out) as ffile:
            ffile.read(new_mesh)
        self.mesh = new_mesh
        self.adapt_and_interpolate()
        
    def mesh_refinement_hsiz(self ):
        with XDMFFile("mesh.xdmf") as ffile:
             ffile.parameters["functions_share_mesh"] = True
             ffile.write(self.mesh)


        # Convert to Medit format
        os.system('meshio-convert --input-format xdmf --output-format medit mesh.xdmf mesh.mesh')

        dist = 0.02 # controls the optimized mesh resolution (see https://www.mmgtools.org/mmg-remesher-try-mmg/mmg-remesher-options/mmg-remesher-option-hausd)
         # call mmgs with mesh optimization and Hausdorff distance
        os.system('bin/mmgs_O3 -in mesh.mesh -out mesh_optimized.mesh -hsiz {}'.format(dist)) # Hudson's
#        os.system('/opt/mmg/bin/mmgs_O3 -in mesh.mesh -out mesh_optimized.mesh -hausd {} -optim'.format(dist)) # Jeremy's

        # Convert back to .msh format using Gmsh
        os.system(' /Applications/Gmsh.app/Contents/MacOS/gmsh mesh_optimized.mesh -3 -o mesh_optimized.msh')
#        os.system('gmsh mesh_optimized.mesh -3 -o mesh_optimized.msh') # Jeremy's

        fname = "mesh.xdmf"
        fname_out = "mesh_optimized.xdmf"

        # read back with meshio to remove cells and cell data and convert to xdmf
        mmesh = meshio.read(fname_out.replace(".xdmf", ".msh"))
        #mmesh.remove_lower_dimensional_cells()
        #meshio.write(fname_out, meshio.Mesh(mmesh.points, mmesh.cells))
        meshio.write(fname_out,
                        meshio.Mesh(points = mmesh.points,
                               cells = {'triangle': mmesh.cells_dict['triangle']}))
        self.mmesh = meshio.read(fname_out)

        # read in with FEniCS
        new_mesh = Mesh()
        with XDMFFile(fname_out) as ffile:
            ffile.read(new_mesh)
        self.mesh = new_mesh
        self.adapt_and_interpolate()

    def adapt_and_interpolate(self):

        self.q_ = Function(adapt(self.q_._cpp_object, self.mesh))
        self.Q = self.q_.function_space()
        self.V_phi = FunctionSpace(adapt(self.V_phi._cpp_object, self.mesh))
        self.V_beta = FunctionSpace(adapt(self.V_beta._cpp_object, self.mesh))
        self.V_thickness = FunctionSpace(adapt(self.V_thickness._cpp_object, self.mesh))
        self.V_alpha = FunctionSpace(adapt(self.V_alpha._cpp_object, self.mesh))
        self.VT = FunctionSpace(adapt(self.VT._cpp_object, self.mesh))
        self.V_normal = FunctionSpace(adapt(self.V_normal._cpp_object, self.mesh))

        # interpolate
        self.phi0 = interpolate(self.phi0, self.V_phi)
        self.q_ = interpolate(self.q_, self.Q)
        self.u_ = interpolate(self.q_.sub(0),self.Q.sub(0).collapse())
        self.beta_ = interpolate(self.q_.sub(1),self.Q.sub(1).collapse())
        
        
        self.thickness = interpolate(self.thickness, self.V_thickness)
        self.Q_field = interpolate(self.Q_field, self.V_thickness)


        self.q_t, self.q = TrialFunction(self.Q), TestFunction(self.Q)
        if self.geometry == "Hemisphere":
            self.u_, self.beta_, self.lbda_ , self.rigid_ = split(self.q_)
        elif self.geometry == "eighthsphere":
            self.u_, self.beta_, self.lbda_  = split(self.q_)


        self.set_local_frame()
        self.set_boundary_conditions()
        self.set_director()
        self.set_kinematics_and_fundamental_forms()
        self.set_energies()
        self.total_energy()
        self.set_solver()




# System parameters

mu = 10
# t = Constant(0.03)
thick = Expression('0.02', degree = 4)

kd = 4
vp = 0.08
q_11, q_12, q_22 = 1.0/6, 0., 1./6
Q_tensor = as_tensor([[1./6, 0.0], [0.0, 1./6]])
q_33 = -1./3
zeta = 20.

output_dir = "output/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)



parameters["form_compiler"]["quadrature_degree"] = 4
parameters['allow_extrapolation'] = True # TODO: Bug Workaround?!?

geometry = "eighthsphere"

xdmf_name = geometry + ".xdmf"

#fname_msh = xdmf_name.replace("xdmf", "msh")

#subprocess.call(["gmsh", "-2", "-format", "msh2", "Hemisphere.geo", "-o", xdmf_name.replace("xdmf", "msh")]) # Jeremy's

subprocess.call(["/Applications/Gmsh.app/Contents/MacOS/gmsh", "-2", "-format", "msh2", geometry + ".geo", "-o", xdmf_name.replace("xdmf", "msh")]) # Hudson's

msh = meshio.read(xdmf_name.replace(".xdmf", ".msh"))

meshio.write(xdmf_name, meshio.Mesh(points = msh.points,
                                      cells = {'triangle': msh.cells_dict['triangle']}))

mmesh = meshio.read(xdmf_name)
mesh = Mesh()


with XDMFFile(xdmf_name) as mesh_file:
    mesh_file.read(mesh)


# initial_volume = assemble(Constant('1')*dx(domain=mesh))
initial_volume = assemble(1.*dx(domain=mesh))/3

print("Initial volume:", initial_volume)

LE = False

#print("cytokinesis-zeta_{}-kd_{}-vp_{}".format(zeta, kd, vp))
filename = output_dir + xdmf_name.replace(".xdmf", "_results.xdmf")
problem = NonlinearProblem_metric_from_mesh(mesh, mmesh, thick = thick, mu = mu, zeta = zeta,
                                            kd = kd, vp = vp, vol_ini = initial_volume,
                                            fname = filename, hypothesis="small strain", geometry = geometry, LE = LE)





time = 0
Time = 20
dt = 50E-3
dt_max = dt
dt_min = 1e-3*dt_max
i = 0

if LE:
    remeshing_frequency = 60 # remeshing every n time steps
else:
    remeshing_frequency = 2 # remeshing every n time steps


class K(UserExpression):
    def eval(self, value, x):
        tol = 1E-14
        if x[1] <= 0.2 + tol:
            value[0] = 1.
        else:
            value[0] = 0.
furrow_indicator = K(degree = 0)




current_radius = 1.



#filename = 'output/data.csv'
hdr = 'Time;Volume;FurrowRadius;Dissipation_Membrane;Dissipation_Bending;Dissipation_Shear;Passive_Dissipation_Membrane;Passive_Dissipation_Bending;Polymerization_Membrane;Polymerization_Bending;Furrow_Dissipation_Membrane;Furrow_Dissipation_Bending;Furrow_passive_dissipation_m;Furrow_passive_dissipation_b;Furrow_polymerization_m;Furrow_polymerization_b'

f=open('output/Data.csv','w')
np.savetxt(f,[], header= hdr)

problem.write(time, u = True, beta = True, phi = True, frame = True, epaisseur = True, activity = True, energies = True)
while time < Time:

    if dt < dt_min:# or current_radius < 0.05 : # If the furrow radius is smaller than twice the thickness it means that it should have stopped dividing!
        problem.write(time+dt, u = True, beta = True, phi = True, frame = True, epaisseur = True, activity = True, energies = False)
        break
        # raise ValueError("Reached minimal time step")
    try:
        i += 1
        print("Iteration {}. Time step : {}".format(i, dt))
        # problem.evolution(dt/2)
        niter, _ = problem.solve()
        converged = True
    except:
        # problem.evolution(-dt/2)
        dt /= 2
        converged = False
    else:   # execute if no convergence issues
        problem.evolution(dt)
        if i % 2 == 0:
            problem.write(time + dt, u = True, beta = True, phi = True, frame = True, epaisseur = True, activity = True, energies = True)
        
            print("rmin={}, rmax={}, hmin={}, hmax={}".format(
                problem.mesh.rmin(), problem.mesh.rmax(), problem.mesh.hmin(), problem.mesh.hmax()))

        save_data(f, problem)
        time +=dt


        # adaptive time step
        if niter < 5:
            dt = min(1.25*dt, dt_max)
        elif niter > 8:
            dt *= 0.75
    try:    # always test for mesh refinement (every so time step or when failure)
        if i % remeshing_frequency == 0 or not(converged):
            if problem.mesh.rmin() < 1e-3:
                problem.mesh_refinement_hsiz()
                print("Uniform mesh!")

            else:
                problem.mesh_refinement()
                print("From here it should be a new problem...")
    except:
        break
        
f.close()


print("It ended at iteration {}, and Time {}".format(i, time))
