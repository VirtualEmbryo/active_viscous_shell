# We can visualize the shell shape and its normal with this
# utility function::

def plot_shell(y,n=None):
    y_0, y_1, y_2 = y.split(deepcopy=True)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(y_0.compute_vertex_values(),
                    y_1.compute_vertex_values(),
                    y_2.compute_vertex_values(),
                    triangles=y.function_space().mesh().cells(),
                    linewidth=1, antialiased=True, shade = False)
    if n:
        n_0, n_1, n_2 = n.split(deepcopy=True)
        ax.quiver(y_0.compute_vertex_values(),
              y_1.compute_vertex_values(),
              y_2.compute_vertex_values(),
              n_0.compute_vertex_values(),
              n_1.compute_vertex_values(),
              n_2.compute_vertex_values(),
              length = .2, color = "r")
    ax.view_init(elev=45, azim=120)
    ax.set_xlim(-L, L)
    ax.set_ylim(-L, L)
    ax.set_zlim(-.5, .1)
    plt.xlabel(r"$x_0$")
    plt.ylabel(r"$x_1$")
    plt.xticks([-1,0,1])
    plt.yticks([0,3])
    return ax




def update_geometry(u_, beta_, phi0, beta0, thickness, Thickness_dynamic):
    # Kinematics
    F = grad(u_) + grad(phi0)
    d = director(beta_ + beta0)
    # Initial metric and curvature
    a0 = grad(phi0).T*grad(phi0)
    b0 = -0.5*(grad(phi0).T*grad(d0) + grad(d0).T*grad(phi0))
    a0_contra = inv(a0)
    j0 = det(a0)
    
    # The membrane, bending, and shear strain measures of the Naghdi model
    e = lambda F: 0.5*(F.T*F - a0)
    k = lambda F, d: -0.5*(F.T*grad(d) + grad(d).T*F) - b0
    gamma = lambda F, d: F.T*d - grad(phi0).T*d0
    if Thickness_dynamic:
        D_Delta= inner(a0_contra,e(F))
        thickness = project(thickness*(1-dt*D_Delta),V_thickness)
    
    # Contravariant Hooke's tensor
    i, j, l, m = Index(), Index(), Index(), Index()
    A_ = as_tensor((0.5*a0_contra[i,j]*a0_contra[l,m]
                    + 0.25*(a0_contra[i,l]*a0_contra[j,m] + a0_contra[i,m]*a0_contra[j,l]))
                    ,[i,j,l,m])
    # Stress
    N = thickness*mu*as_tensor(A_[i,j,l,m]*e(F)[l,m], [i,j])
    M = (thickness**3/3.0)*mu*as_tensor(A_[i,j,l,m]*k(F,d)[l,m],[i,j])
    T = thickness*mu*as_tensor(a0_contra[i,j]*gamma(F,d)[j], [i])
    # Energy densities
    psi_m = 0.5*inner(N, e(F))
    psi_b = 0.5*inner(M, k(F,d))
    psi_s = 100*0.5*inner(T, gamma(F,d))
    # Total Energy densities
    dx_h = dx(metadata={'quadrature_degree': 2})
    h = CellDiameter(mesh)
    alpha = project(t**2/h**2, FunctionSpace(mesh,'DG',0))
    u_1, u_2, u_3 = split(u_)
    # External work
    Force = 100.
    # W_Force = Force*u_3*thickness (If multiply by the thickness we need to rescale the force)
    W_Force = Force*u_3

    Pi_PSRI = (psi_b*sqrt(j0)*dx + alpha*psi_m*sqrt(j0)*dx + alpha*psi_s*sqrt(j0)*dx +
               (1.0 - alpha)*psi_s*sqrt(j0)*dx_h + (1.0 - alpha)*psi_m*sqrt(j0)*dx_h)+W_Force*sqrt(j0)*dx

    # The total elastic energy and its first and second derivatives
    Pi = Pi_PSRI
    dPi = derivative(Pi, q_, q_t)
    J = derivative(dPi, q_, q)
    #
    return dPi, J, thickness

def save_plots (ii):
    if (ii%10==0):
        # 2D displacement from a contour plot
        print(displacement[ii])
        plt.figure()
        # c=plot(phi[2], cmap='RdGy', mode = 'color',vmin=-.2, vmax=0)
        c=plot(phi[2], cmap='RdGy', mode = 'color',vmin=-.5, vmax=.0)
        plt.colorbar(c)
        plt.savefig("output/phi2D/phi2"+str(ii).zfill(4)+".png")
        plt.close()
        
        # Vertical displacement in a line passing in the middle of the plate
        plt.figure()
        tol = 0.001 # avoid hitting points outside the domain
        yy = np.linspace(-1 + tol, 1 - tol, 101)
        points = [(y_,0) for y_ in yy] # 2D points
        w_line = np.array([phi(point)[2] for point in points])
        plt.plot(yy, 1*w_line, linewidth=2) # magnify w
        plt.ylim(-0.50, 0.01)
        # plt.close()
        # plot_shell(phi)
        plt.savefig("output/phi/phi"+str(ii).zfill(4)+".png")
        plt.close()
        
        #3D geometry evolution
        plt.figure()
        plot_shell(phi)
        plt.savefig("output/phi3D/phi3D"+str(ii).zfill(4)+".png")
        plt.close()
        
        # Evolution of the thickness of the plate
        plt.figure()
        #plot(mesh)
        c=plot(100*thickness, cmap='RdGy', mode = 'color',vmin=4.2, vmax=5)
        plt.colorbar(c)
        plt.savefig("output/thick/t"+str(ii).zfill(4)+".png")
        plt.close()

def update_mesh(phi,u_):
    vm = inner(normal(phi),u_)*normal(phi)
    return vm

# ======================================================
# Clamped viscous shell plate under uniform force
# ======================================================
#
#
# This demo program solves the nonlinear Naghdi shell equations for a
# semi-cylindrical shell loaded by a point force. This problem is a standard
# reference for testing shell finite element formulations, see [1].
# The numerical locking issue is cured using enriched finite
# element including cubic bubble shape functions and Partial Selective
# Reduced Integration [2].
#


import os, sys

import numpy as np
import matplotlib.pyplot as plt

from dolfin import *
from ufl import Index
from mshr import *
from mpl_toolkits.mplot3d import Axes3D



class NonlinearProblemPointSource(NonlinearProblem):

    def __init__(self, L, a, bcs):
        NonlinearProblem.__init__(self)
        self.L = L
        self.a = a
        self.bcs = bcs
        self.P = 0.0

    def F(self, b, x):
        assemble(self.L, tensor=b)
        #point_source = PointSource(self.bcs[0].function_space().sub(0).sub(2), Point(0.0, 0.0), self.P)
        #point_source.apply(b)
        for bc in self.bcs:
            bc.apply(b, x)
    
    def J(self, A, x):
        assemble(self.a, tensor=A)
        for bc in self.bcs:
            bc.apply(A, x)
            
parameters["form_compiler"]["quadrature_degree"] = 4

output_dir = "output/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
output_dir_phi = "output/phi"
if not os.path.exists(output_dir_phi):
    os.makedirs(output_dir_phi)

output_dir_phi2 = "output/phi2D"
if not os.path.exists(output_dir_phi2):
    os.makedirs(output_dir_phi2)

output_dir_phi3 = "output/phi3D"
if not os.path.exists(output_dir_phi3):
    os.makedirs(output_dir_phi3)
    
output_dir_t = "output/thick"
if not os.path.exists(output_dir_t):
    os.makedirs(output_dir_t)
    
# We consider a semi-cylindrical shell of radius :math:`\rho` and axis length
# :math:`L`. The shell is made of a linear elastic isotropic homogeneous
# material with Young modulus :math:`E` and Poisson ratio :math:`\nu`. The
# (uniform) shell thickness is denoted by :math:`t`.
# The Lamé moduli :math:`\lambda`, :math:`\mu` are introduced to write later
# the 2D constitutive equation in plane-stress::

L = 1.
E, nu = 2.0685E7, 0.3
mu = 1.0E7
#E/(2.0*(1.0 + nu))
t = Constant(0.03)

# The midplane of the initial (stress-free) configuration
# :math:`{\mit \Phi_0}` of the shell is given in the form of an analytical
# expression
#
# .. math:: \phi_0:x\in\omega\subset R^2 \to \phi_0(x) \in {\mit \Phi_0} \subset \mathcal R^3
#
# in terms of the curvilinear coordinates :math:`x`. In the specific case
# we adopt the cylindrical coordinates :math:`x_0` and :math:`x_1`
# representing the angular and axial coordinates, respectively.
# Hence we mesh the two-dimensional domain
# :math:`\omega \equiv [0,L_y] \times [-\pi/2,\pi/2]`. ::

P1, P2 = Point(-L, -L), Point(L, L)
ndiv = 11
mesh = generate_mesh(Rectangle(P1, P2), ndiv)
plot(mesh); plt.xlabel(r"$x_0$"); plt.ylabel(r"$x_1$")
plt.savefig("output/mesh.png")


#    Discretisation of the parametric domain.
#
# We provide the analytical expression of the initial shape as an
# ``Expression`` that we represent on a suitable ``FunctionSpace`` (here
# :math:`P_2`, but other are choices are possible)::

# initial_shape = Expression(('r*sin(x[0])','x[1]','r*cos(x[0])'), r=rho, degree = 4)
initial_shape = Expression(('x[0]','x[1]','r'), r=0, degree = 4)
V_phi =  FunctionSpace(mesh, VectorElement("P", triangle, degree = 2, dim = 3))
phi0 = project(initial_shape, V_phi)




## Discretisation of the thickness. I need to find the right way to discretize to see if I can avoid the FCC stuff
PP2 = FiniteElement("Lagrange", triangle, degree = 2)
TT = FiniteElement("CG", mesh.ufl_cell(), 2)

V_thickness =  FunctionSpace(mesh, PP2)
thick = Expression('0.05', degree = 4)
thickness = project( thick, V_thickness)

Thickness_dynamic = True  # True if we consider the variation of T over time




# Given the midplane, we define the corresponding unit normal as below and
# project on a suitable function space (here :math:`P_1` but other choices
# are possible)::

def normal(y):
    n = cross(y.dx(0), y.dx(1))
    return n/sqrt(inner(n,n))

V_normal = FunctionSpace(mesh, VectorElement("P", triangle, degree = 1, dim = 3))
n0 = project(normal(phi0), V_normal)

# The kinematics of the Nadghi shell model is defined by the following
# vector fields :
#
# - :math:`\phi`: the position of the midplane, or the displacement from the reference configuration :
# .     math:`u = \phi - \phi_0`:
# - :math:`d`: the director, a unit vector giving the orientation of the microstructure
#
# We parametrize the director field by two angles, which correspond to spherical coordinates,
# so as to explicitly resolve the unit norm constraint (see [3])::

def director(beta):
    return as_vector([sin(beta[1])*cos(beta[0]), -sin(beta[0]), cos(beta[1])*cos(beta[0])])

# We assume that in the initial configuration the director coincides with
# the normal. Hence, we can define the angles :math:`\beta`: for the initial
# configuration as follows: ::

beta0_expression = Expression(["atan2(-n[1], sqrt(pow(n[0],2) + pow(n[2],2)))",
                               "atan2(n[0],n[2])"], n = n0, degree=4)

V_beta = FunctionSpace(mesh, VectorElement("P", triangle, degree = 2, dim = 2))
beta0 = project(beta0_expression, V_beta)

# The director in the initial configuration is then written as ::

d0 = director(beta0)

# plot_shell(phi0, project(d0, V_normal))
# plt.savefig("output/initial_configuration.png")


# In our 5-parameter Naghdi shell model the configuration of the shell is
# assigned by
#
# - the 3-component vector field :math:`u`: representing the displacement
#   with respect to the initial configuration :math:`\phi_0`:
#
# - the 2-component vector field :math:`\beta`: representing the angle variation
#   of the director :math:`d`: with respect to the initial configuration
#
# Following [1], we use a :math:`[P_2 + B_3]` element for :math:`u` and a :math:`[CG_2]^2`
# element for :math:`beta`, and collect them in the state vector
# :math:`q = (u, \beta)`::

P2 = FiniteElement("Lagrange", triangle, degree = 2)
bubble = FiniteElement("B", triangle, degree = 3)
enriched = P2 + bubble

element = MixedElement([VectorElement(enriched, dim=3), VectorElement(P2, dim=2)])

Q = FunctionSpace(mesh, element)

# Then, we define :py:class:`Function`, :py:class:`TrialFunction` and :py:class:`TestFunction` objects
# to express the variational forms and we split them into each individual component function::
    
q_, q, q_t = Function(Q), TrialFunction(Q), TestFunction(Q)
u_, beta_ = split(q_)


# Shear and membrane locking is treated using the partial reduced
# selective integration proposed in Arnold and Brezzi [2]. In this approach
# shear and membrane energy are splitted as a sum of two contributions
# weighted by a factor :math:`\alpha`. One of the two contributions is
# integrated with a reduced integration. While [1] suggests a 1-point
# reduced integration, we observed that this leads to spurious modes in
# the present case. We use then :math:`2\times 2`-points Gauss integration
# for a portion :math:`1-\alpha` of the energy, whilst the rest is
# integrated with a :math:`4\times 4` scheme. We further refine the
# approach of [1] by adopting an optimized weighting factor
# :math:`\alpha=(t/h)^2`, where :math:`h` is the mesh size. ::

dx_h = dx(metadata={'quadrature_degree': 2})
h = CellDiameter(mesh)
alpha = project(t**2/h**2, FunctionSpace(mesh,'DG',0))


# Here we initialize the Energy, to define the problem
# The computation of kinetics, metrics, Forces are all hidden here
# Would need to expand this function if we want to access forces, for instance

dPi, J, thickness = update_geometry(u_, beta_, phi0, beta0, thickness, False)
# We do not want to update the thickness in the initialization


# The boundary conditions prescribe a full clamping on the boundaries,

whole_boundary = lambda x, on_boundary: on_boundary
leftright_boundary = lambda x, on_boundary: near(abs(x[0]), L, 1.e-6)  and on_boundary # left-right clamp

bc_clamped = DirichletBC(Q, project(q_, Q), whole_boundary)
bcs = [bc_clamped]


# defining a custom :py:class:`NonlinearProblem`::
problem = NonlinearProblemPointSource(dPi, J, bcs)

# We use a standard Newton solver and setup the files for the writing the
# results to disk::

solver = NewtonSolver()
solver.parameters['error_on_nonconvergence'] = False
solver.parameters['maximum_iterations'] = 50
#solver.parameters['linear_solver'] = "mumps"
solver.parameters['absolute_tolerance'] = 1E-6
solver.parameters['relative_tolerance'] = 1E-6
output_dir = "output/"
#file_phi = File(output_dir + "configuration.pvd")
#file_energy = File(output_dir + "energy.pvd")

# Finally, we can solve the quasi-static problem, incrementally increasing

q_.assign(project(Constant((0,0,0,0,0)), Q))


# Begin of the time loop
ii=0
time = 0.0
dt = 1.E-0  # time step
loop_size = 600
Time = loop_size*dt
displacement=np.zeros(loop_size+1)
displacement_0=phi0(0.0,L/2)[2]
while (time < Time):
    print(min(thickness.vector()))
    (niter,cond) = solver.solve(problem, q_.vector())
    
    # Update geometry
    phi = project(u_*dt + phi0 , V_phi)
    displacement[ii] = phi(0.0, 0)[2] - displacement_0
    phi0 = project(phi,V_phi)
    # phi.rename("phi", "phi")
    # file_phi << (phi, time)
    
    # Redefinition of the problem
    dPi, J, thickness = update_geometry(u_, beta_, phi0, beta0, thickness, Thickness_dynamic)
    problem = NonlinearProblemPointSource(dPi, J, bcs)
    save_plots (ii)
    
    ii+=1;
    time += dt

# We can plot the final configuration of the shell
plot_shell(phi)
#plt.figure()
#c=plot(phi[2], cmap='RdGy', mode = 'color',vmin=-.5, vmax=0)
#plt.colorbar(c)
       

# References
# ----------
#
# [1] K. Sze, X. Liu, and S. Lo. Popular benchmark problems for geometric
# nonlinear analysis of shells. Finite Elements in Analysis and Design,
# 40(11):1551 – 1569, 2004.
#
# [2] D. Arnold and F.Brezzi, Mathematics of Computation, 66(217): 1-14, 1997.
# https://www.ima.umn.edu/~arnold//papers/shellelt.pdf
#
# [3] P. Betsch, A. Menzel, and E. Stein. On the parametrization of finite
# rotations in computational mechanics: A classification of concepts with
# application to smooth shells. Computer Methods in Applied Mechanics and
# Engineering, 155(3):273 – 305, 1998.
#
# [4] P. G. Ciarlet. An introduction to differential geometry with
# applications to elasticity. Journal of Elasticity, 78-79(1-3):1–215, 2005.
