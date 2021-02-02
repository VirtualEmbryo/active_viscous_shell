import os, sys

import numpy as np
import matplotlib.pyplot as plt

from dolfin import *
from ufl import Index, unit_vector, shape, Jacobian
from mshr import *
from mpl_toolkits.mplot3d import Axes3D
# from adapt_fix import adapt
import subprocess

import meshio

class CustomNonlinearProblem(NonlinearProblem):

    def __init__(self, L, a, bcs):
        NonlinearProblem.__init__(self)
        self.L = L
        self.a = a
        self.bcs = bcs

    def F(self, b, x):
        assemble(self.L, tensor=b)
        for bc in self.bcs:
            bc.apply(b, x)

    def J(self, A, x):
        assemble(self.a, tensor=A)
        for bc in self.bcs:
            bc.apply(A, x)

class SurfaceNormal(UserExpression):
    def __init__ (self, mesh, mmesh, **kwargs):
        super(SurfaceNormal, self).__init__(**kwargs)
        self.mesh = mesh
        self.mmesh = mmesh
        
        
    def eval_cell(self, value, x, ufc_cell):
        nodes = self.mmesh.cells_dict["triangle"][ufc_cell.index, :]
        x = self.mesh.coordinates()[nodes, :]
        value[:] = np.cross(x[0, :] - x[1, :], x[0, :] - x[2, :])
        value /= np.linalg.norm(value)
        # For the spherical mesh the normal was not well oriented
        self.mesh.init_cell_orientations(Expression(('x[0]', '-x[1]', 'x[2]'), degree = 0))
        value[self.mesh.cell_orientations() == 1] *= -1

    def value_shape(self):
        return (3,)
                

class NonlinearProblem_metric_from_mesh:
    def __init__(self, mesh,mmesh, thick, mu, zeta, kd, vp, vol_ini, fname = None):
        self.mesh = mesh
        self.mmesh = mmesh
        self.thick = thick
        self.mu = mu
        self.zeta = zeta
        self.kd = kd
        self.vp = vp
        self.vol_ini = vol_ini
        self.set_solver()
        self.set_functions_space()
        self.thickness = project(thick, self.V_thickness)
        self.initialize()
        
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
        if epaisseur:
            self.thickness.rename("thickness", "thickness")
            self.output_file.write(self.thickness, i)
            
        if activity:
            self.Q_field.rename("activity", "activity")
            self.output_file.write(self.Q_field, i)
            
        if energies:
            self.psi_m
            self.psi_b
            self.psi_s
            
    def set_functions_space(self):
        P2 = FiniteElement("Lagrange", self.mesh.ufl_cell(), degree = 2)
        bubble = FiniteElement("B", self.mesh.ufl_cell(), degree = 3)
        enriched = P2 + bubble
        R = FiniteElement("Real", self.mesh.ufl_cell(), degree=0)
        element = MixedElement([VectorElement(enriched, dim=3), VectorElement(P2, dim=2), R])
        
        element = MixedElement([VectorElement(enriched, dim=3), VectorElement(P2, dim=2), R, VectorElement(R, dim=3)])
        
        self.Q = FunctionSpace(self.mesh, element)
        self.q_, self.q, self.q_t = Function(self.Q), TrialFunction(self.Q), TestFunction(self.Q)
        self.u_, self.beta_, self.lbda_ , self.rigid_ = split(self.q_)
        self.u, self.beta, self.lbda, self.rigid = split(self.q)
        self.u_t, self.beta_t, self.lbda_t, self.rigid_t = split(self.q_t)
        
        self.V_phi =  FunctionSpace(self.mesh, VectorElement("P", mesh.ufl_cell(), degree = 2, dim = 3))
        self.V_beta = FunctionSpace(self.mesh, VectorElement("P", mesh.ufl_cell(), degree = 2, dim = 2))
        self.V_thickness = FunctionSpace(self.mesh, P2)
        self.V_alpha = FunctionSpace(self.mesh, "DG", 0)
        self.VT = VectorFunctionSpace(self.mesh, "DG", 0, dim = 3)
        
    def set_solver(self):
#        self.solver = PETScSNESSolver()
#        self.solver.parameters["method"] = "newtonls"
#        self.solver.parameters['maximum_iterations'] = 50
#        self.solver.parameters['linear_solver'] = "mumps"
#        self.solver.parameters['absolute_tolerance'] = 1E-6
#        self.solver.parameters['relative_tolerance'] = 1E-6

        self.solver = NewtonSolver()
        self.solver.parameters['maximum_iterations'] = 50
#        solver.parameters['linear_solver'] = "mumps"
        self.solver.parameters['absolute_tolerance'] = 1E-6
        self.solver.parameters['relative_tolerance'] = 1E-6


#         prm = self.solver.parameters
#         prm['maximum_iterations'] = 50
#         prm['report'] = True
#         prm['relative_tolerance'] = 1e-4
#         prm['absolute_tolerance'] = 1e-4
#         prm['linear_solver'] = 'mumps'
        
    def set_shape(self):
        x = SpatialCoordinate(self.mesh)
        initial_shape = x
        self.phi0 = project(initial_shape, self.V_phi)
            
    def set_local_frame(self):
        normal = SurfaceNormal(mesh = self.mesh, mmesh = self.mmesh)
        a1, a2 = self.local_frame(normal)
        self.a1 = project(a1, self.VT)
        self.a2 = project(a2, self.VT)
        self.n0 = project(normal, self.VT)
        
    def local_frame(mesh, normal):
        ey = as_vector([0, 1, 0])
        ez =  as_vector([0, 0, 1])
        a1 = cross(ey, normal)
        norm_a1 = sqrt(dot(a1, a1))
        a1 = conditional(lt(norm_a1,0.01), ez, a1/norm_a1)
        a2 = cross(normal, a1)
        a2 /= sqrt(dot(a2, a2))
        return a1, a2
    
    def director(self, beta):
         return as_vector([sin(beta[1])*cos(beta[0]), -sin(beta[0]), cos(beta[1])*cos(beta[0])])

#    def director(self, beta):
#        return as_vector([sin(beta[1])*cos(beta[0]), sin(beta[0])*sin(beta[1]), cos(beta[1])]) # Spherical coord


    def set_director(self):
        beta0_expression = Expression(["atan2(-n[1], sqrt(pow(n[0],2) + pow(n[2],2)))",
                                        "atan2(n[0], n[2])"], n = self.n0, degree=4)
#        beta0_expression = Expression(["atan2(n[1], n[0])",
#                               "atan2(sqrt(pow(n[0],2) + pow(n[1],2)), n[2])"], n = self.n0, degree=4) # spherical coord for the semicylinder


        self.beta0 = project(beta0_expression, self.V_beta)

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
        # Kinematics
#         grad_ = lambda u : as_tensor([self.d1(u), self.d2(u)])
#         print("Shape of grad_u: ", (self.grad_(self.u_).ufl_shape))


        self.F = self.grad_(self.u_) + self.grad_(self.phi0)
#         print("Shape of F ", (self.F.ufl_shape))

        
        self.d = self.director(self.beta_ + self.beta0)
    
        self.a0 = as_tensor([[dot(self.a1, self.a1), dot(self.a1, self.a2)],\
                            [dot(self.a2, self.a1), dot(self.a2, self.a2)]])
        
#         print("Shape of metric tensor: ", (self.a0.ufl_shape))

        
        self.b0 = - sym(as_tensor([[inner(as_vector(self.grad_(self.d0)[0,:]), self.a1), inner(as_vector(self.grad_(self.d0)[0,:]), self.a2)],\
                                  [inner(as_vector(self.grad_(self.d0)[1,:]), self.a1), inner(as_vector(self.grad_(self.d0)[1,:]), self.a2)]]))

#         print("Shape of curvature tensor: ", (self.b0.ufl_shape))

        self.j0 = det(self.a0)

        self.a0_contra = inv(self.a0)
        self.H = 0.5*inner(self.a0_contra, self.b0)
                
    def membrane_deformation(self):
        return 0.5*(self.F*self.F.T - self.a0)
    
    def bending_deformation(self):
        return -0.5*(self.F*self.grad_(self.d).T + self.grad_(self.d)*self.F.T) - self.b0
    
    def shear_deformation(self):
        return self.F*self.d - self.grad_(self.phi0)*self.d0
        
    def set_thickness(self, dt):
        D_Delta= inner(self.a0_contra, self.membrane_deformation())
#         self.thickness = project(-dt*(dot(self.u_,self.phi0.dx(0))*self.thickness.dx(0)+dot(self.u_,self.phi0.dx(1))*\
#                                       self.thickness.dx(1)) + self.thickness*(1-dt*D_Delta) - dt*self.thickness*self.kd + \
#                                  dt*self.vp*(1-self.H*self.thickness/2),self.V_thickness)
        self.thickness = project((self.thickness/dt + self.vp)/(1/dt + D_Delta + self.kd + self.vp*self.H/2),self.V_thickness)

    
    def set_energies(self):
        # Gaussian signal in the middle of the plate and uniform across one of the directions
        sig_q = 1./2 ;
        Q_Expression = Expression(('exp(-0.5*(x[0]*x[0])/(sig_q*sig_q))/(sig_q*sqrt(2.*pi))'), sig_q = sig_q, degree = 2)
        #Q_Expression = Expression(('1.'), sig_q= sig_q,degree = 2)
        self.Q_field = project(Q_Expression, self.V_thickness)
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


        self.N = self.thickness*self.mu *as_tensor(A_[i,j,l,m]*self.membrane_deformation()[l,m] + self.a0_contra[i,j]*(self.kd - self.vp/self.thickness), [i,j])+ \
            self.Q_field*self.zeta*self.thickness*as_tensor((self.a0_contra[i,l]*self.a0_contra[j,m]*self.Q_tensor[l,m] - self.a0_contra[i,j]*self.q_33),[i,j])
       
        self.M = (self.thickness**3/3.0)*self.mu*as_tensor(A_[i,j,l,m]*self.bending_deformation()[l,m] + (self.H*self.a0_contra[i,j] - self.b0[i,j])*(self.kd-self.vp/self.thickness),[i,j])\
            - (self.thickness**3/12.)*self.zeta*C_active*self.Q_field
       
        self.T = self.thickness*self.mu*as_tensor(self.a0_contra[i,j]*self.shear_deformation()[j], [i])
    
    
        self.psi_m = 0.5*inner(self.N, self.membrane_deformation())
        self.psi_b = 0.5*inner(self.M, self.bending_deformation())
        self.psi_s = 1*0.5*inner(self.T, self.shear_deformation())
        
    def total_energy(self):
        # Total Energy densities
        self.dx = Measure('dx', domain = self.mesh)
        self.dx_h = self.dx(metadata={'quadrature_degree': 2})
        self.h = CellDiameter(self.mesh)
#         alpha = self.thickness**2/self.h**2
        alpha = project(self.thickness**2/self.h**2, self.V_alpha)

        u_1, u_2, u_3 = split(self.u_)
        volume = assemble(1.*self.dx(domain=self.mesh))/3

       
        Pi_PSRI = (1.0 - alpha)*self.psi_s*sqrt(self.j0)*self.dx_h + (1.0 - alpha)*self.psi_m*sqrt(self.j0)*self.dx_h + \
                    self.psi_b*sqrt(self.j0)*self.dx + alpha*self.psi_m*sqrt(self.j0)*self.dx + alpha*self.psi_s*sqrt(self.j0)*self.dx
                     
                    

        # The total elastic energy and its first and second derivatives
        self.Pi = Pi_PSRI
        self.Pi = self.Pi + dot(self.rigid_,self.u_)*sqrt(self.j0)*self.dx # Remotion of rigid motion
        self.Pi = self.Pi + self.lbda_*dot(self.u_, self.n0)*sqrt(self.j0)*self.dx # Volume conservation
        self.dPi = derivative(self.Pi, self.q_, self.q_t)# + self.lbda_*dot(self.u_t, self.n0)*sqrt(self.j0)*self.dx+ self.lbda_t*dot(self.u_, self.n0)*sqrt(self.j0)*self.dx
        self.J = derivative(self.dPi, self.q_, self.q)
        
        
    
    def initialize(self):
        self.set_shape()
        self.set_local_frame()
        self.set_director()
        self.set_fundamental_forms()
        self.set_kinematics_and_fundamental_forms()
        self.set_energies()
        self.total_energy()
    def evolution(self, dt):
        self.set_thickness(dt)
        self.phi0 = project(self.u_*dt + self.phi0 , self.V_phi) # Lagrangian
#        print("Shape of metric tensor: ", (self.u_).ufl_shape)

#        self.phi0 = project((dot(self.u_ - dot(self.u_,self.n0)*self.n0, (self.d1(self.phi0) +self.d2(self.phi0))))*dt + self.phi0 , self.V_phi)
        
        displacement_mesh = project(self.u_*dt, self.V_phi)
#        displacement_mesh = project((dot(self.u_, self.n0)*self.n0)*dt , self.V_phi)

        ALE.move(self.mesh, displacement_mesh)

        self.set_local_frame()
#        self.set_director()
        self.set_kinematics_and_fundamental_forms()
        self.set_energies()
        self.total_energy()

            
    def solve(self, bcs):
        problem = CustomNonlinearProblem(self.dPi, self.J, bcs)
        return self.solver.solve(problem, self.q_.vector())
    
    
    def mesh_refinement(self ):
        with XDMFFile("mesh.xdmf") as ffile:
             ffile.parameters["functions_share_mesh"] = True
             ffile.write(self.mesh)
         

        # Convert to Medit format
        os.system('meshio-convert --input-format xdmf --output-format medit mesh.xdmf mesh.mesh')

        # call mmgs with mesh optimization and Hausdorff distance
        os.system('bin/mmgs_O3 -in mesh.mesh -out mesh_optimized.mesh -hausd 0.01')

        # Convert back to .msh format using Gmsh
        os.system(' /Applications/Gmsh.app/Contents/MacOS/gmsh mesh_optimized.mesh -3 -o mesh_optimized.msh')

        fname = "mesh.xdmf"
        dist = 0.001 # controls the optimized mesh resolution (see https://www.mmgtools.org/mmg-remesher-try-mmg/mmg-remesher-options/mmg-remesher-option-hausd)
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

        # adapt
#        self.Q = FunctionSpace(adapt(self.Q._cpp_object, self.mesh))
        self.q_ = Function(adapt(self.q_._cpp_object, self.mesh))
        self.Q = self.q_.function_space()
        self.V_phi = FunctionSpace(adapt(self.V_phi._cpp_object, self.mesh))
        self.V_beta = FunctionSpace(adapt(self.V_beta._cpp_object, self.mesh))
        self.V_thickness = FunctionSpace(adapt(self.V_thickness._cpp_object, self.mesh))
        self.V_alpha = FunctionSpace(adapt(self.V_alpha._cpp_object, self.mesh))
        self.VT = FunctionSpace(adapt(self.VT._cpp_object, self.mesh))
        
        # interpolate
        
        self.phi0 = interpolate(self.phi0, self.V_phi)
        self.q_ = interpolate(self.q_, self.Q)
        self.u_ = interpolate(self.q_.sub(0),self.Q.sub(0).collapse())
        self.beta_ = interpolate(self.q_.sub(1),self.Q.sub(1).collapse())
        self.thickness = interpolate(self.thickness, self.V_thickness)
        self.Q_field = interpolate(self.Q_field, self.V_thickness)

#        self.q = interpolate(self.q, self.Q._cpp_object)
#        self.q_t = interpolate(self.q_t, self.Q._cpp_object)
#        self.q = TestFunction(adapt(self.q._cpp_object, self.mesh))
#        self.q_t = TrialFunction(adapt(self.q_t, self.mesh))

        self.q_t, self.q = TrialFunction(self.Q), TestFunction(self.Q)
        self.u_, self.beta_ = split(self.q_)
    
        self.set_local_frame()
        self.set_director()
        self.set_fundamental_forms()
        self.set_kinematics_and_fundamental_forms()
        self.set_energies()
        self.total_energy()
        self.set_solver()

        
        
        
L = 1.
mu = 1.0E6
# t = Constant(0.03)
thick = Expression('0.03', degree = 4)

kd = 1.0*100.0e-3
vp = 1.0*3.0e-3
q_11, q_12, q_22 = 1.0/6, 0., 1./6
Q_tensor = as_tensor([[1./6, 0.0], [0.0, 1./6]])
q_33 = -1./3
zeta = 1.0e5

parameters["form_compiler"]["quadrature_degree"] = 4
parameters['allow_extrapolation'] = True # TODO: Bug Workaround?!?


xdmf_name = "Hemisphere.xdmf"

fname_msh = xdmf_name.replace("xdmf", "msh")
subprocess.call(["/Applications/Gmsh.app/Contents/MacOS/gmsh", "-2", "-format", "msh2", "Hemisphere.geo",
                 "-o", fname_msh])

msh = meshio.read(xdmf_name.replace(".xdmf", ".msh"))

meshio.write(xdmf_name, meshio.Mesh(points = msh.points,
                                      cells = {'triangle': msh.cells_dict['triangle']}))

mmesh = meshio.read(xdmf_name)
mesh = Mesh()
#global_normal = Expression(("x[0]", "x[1]", "x[2]"), degree=0)
#mesh.init_cell_orientations(global_normal)

with XDMFFile(xdmf_name) as mesh_file:
    mesh_file.read(mesh)

print(len(mesh.coordinates()))
    

# initial_volume = assemble(Constant('1')*dx(domain=mesh))
initial_volume = assemble(1.*dx(domain=mesh))/3

print("Initial volume:", initial_volume)



filename = xdmf_name.replace(".xdmf", "_results.xdmf")

problem = NonlinearProblem_metric_from_mesh(mesh, mmesh, thick = thick, mu = mu, zeta = zeta, kd = kd, vp = vp, vol_ini = initial_volume, fname = filename)

boundary = lambda x, on_boundary: on_boundary
bc_sphere_disp = DirichletBC(problem.Q.sub(0).sub(2), project(Constant(0.), problem.Q.sub(0).sub(2).collapse()), boundary)
bcs = [bc_sphere_disp]


time = 0
Time = 5
dt = 1.E-0
i = 1
while (time < Time):
    (niter,cond) = problem.solve(bcs)
#     print("Converged in {} newton steps".format(niter))
    problem.evolution(dt)
    problem.write(time, u = False, beta = False, phi = False, frame = True, epaisseur = True, activity = True, energies = False)
    time +=dt
    i+=1
    current_volume = assemble(1.*problem.dx(domain=problem.mesh))/3

    print("Current volume:", current_volume)

problem.mesh_refinement()
print("From here it should be a new problem...")

bc_sphere_disp = DirichletBC(problem.Q.sub(0).sub(2), project(Constant(0.), problem.Q.sub(0).sub(2).collapse()), boundary)
bcs = [bc_sphere_disp]
problem.write(time, u = False, beta = False, phi = False, frame = True, epaisseur = True, activity = True, energies = False)
(niter,cond) = problem.solve(bcs)

print("The end")
