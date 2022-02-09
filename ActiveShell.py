import os, sys
import numpy as np
from dolfin import *
from ufl import Index, unit_vector, shape, Jacobian, JacobianDeterminant, atan_2, Max
import subprocess
import meshio
from mesh_adapt import mesh_adapt

SHEAR_PENALTY = Constant(100.0)


class CustomNonlinearProblem(NonlinearProblem):
    def __init__(self, L, a, bcs):
        NonlinearProblem.__init__(self)
        self.L = L
        self.a = a
        self.bcs = bcs

    def form(self, A, P, b, x):
        # this function is called before calling F or J
        assemble_system(self.a, self.L, A_tensor=A, b_tensor=b, bcs=self.bcs, x0=x)

    def F(self, b, x):
        pass

    def J(self, A, x):
        pass


class NonlinearProblem_metric_from_mesh:
    def __init__(
        self,
        mesh,
        mmesh,
        thick,
        mu,
        basal,
        zeta,
        gaussian_width,
        kd,
        vp,
        vol_ini,
        dt,
        HyperOsmotic,
        HypoOsmotic,
        paths,
        time=0,
        fname=None,
    ):
        self.mesh = mesh
        self.mmesh = mmesh
        self.thick = thick
        self.mu = mu
        self.basal = basal
        self.zeta = zeta
        self.gaussian_width = gaussian_width
        self.kd = kd
        self.vp = vp
        self.vol_ini = vol_ini
        self.time = time
        self.dt = dt
        self.set_solver()
        self.set_functions_space()
        self.thickness.interpolate(thick)
        self.dV = 0
        self.paths = paths
        self.HyperOsmotic = HyperOsmotic
        self.HypoOsmotic = HypoOsmotic

        self.initialize()
        self.fname = fname

        if fname is not None:
            self.output_file = XDMFFile(fname)
            self.output_file.parameters["functions_share_mesh"] = True
            self.output_file.parameters["flush_output"] = True

    def write(
        self,
        i,
        u=True,
        beta=True,
        phi=True,
        frame=False,
        epaisseur=False,
        activity=False,
        energies=True,
    ):
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

            # mean_curvature = project(self.H, self.V_thickness)
            # mean_curvature.rename("Meancurvature", "Meancurvature")
            # self.output_file.write(mean_curvature, i)

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

            D = project(
                self.passive_bending_energy
                / (self.passive_bending_energy + self.passive_membrane_energy),
                self.V_thickness,
            )
            D.rename("D", "D")
            self.output_file.write(D, i)

            polymerization_membrane = project(
                self.polymerization_membrane, self.V_thickness
            )
            polymerization_membrane.rename("Polymerization", "Polymerization")
            self.output_file.write(polymerization_membrane, i)

            active_membrane_energy = project(
                self.active_membrane_energy, self.V_thickness
            )
            active_membrane_energy.rename("Contractility", "Contractility")
            self.output_file.write(active_membrane_energy, i)

    def set_functions_space(self):
        P2 = FiniteElement("Lagrange", self.mesh.ufl_cell(), degree=2)
        P1 = FiniteElement("Lagrange", self.mesh.ufl_cell(), degree=2)
        CR1 = FiniteElement("CR", self.mesh.ufl_cell(), degree=1)
        R = FiniteElement("Real", self.mesh.ufl_cell(), degree=0)

        element = MixedElement([VectorElement(P2, dim=3), VectorElement(CR1, dim=2), R])

        self.Q = FunctionSpace(self.mesh, element)
        self.q_, self.q, self.q_t = (
            Function(self.Q),
            TrialFunction(self.Q),
            TestFunction(self.Q),
        )

        self.u_, self.beta_, self.lbda_ = split(self.q_)
        self.u, self.beta, self.lbda = split(self.q)

        self.V_phi = self.Q.sub(0).collapse()
        self.V_beta = self.Q.sub(1).collapse()
        self.V_thickness = FunctionSpace(self.mesh, P1)
        self.V_alpha = FunctionSpace(self.mesh, "DG", 0)
        self.VT = VectorFunctionSpace(self.mesh, "DG", 0, dim=3)
        self.V_normal = self.Q.sub(0).collapse()
        self.a1 = Function(self.VT)
        self.a2 = Function(self.VT)
        self.n0 = Function(self.VT)

        self.thickness = Function(self.V_thickness)
        x = Expression(("x[0]", "x[1]", "x[2]"), degree=1)
        self.mesh.init_cell_orientations(x)
        self.n0.interpolate(x)

    def set_boundary_conditions(self):
        # Re-definition of the boundary for the new mesh
        boundary = lambda x, on_boundary: on_boundary
        boundary_x = lambda x, on_boundary: near(x[0], 0.0, 1.0e-3) and on_boundary
        boundary_y = lambda x, on_boundary: near(x[1], 0.0, 1.0e-3) and on_boundary
        boundary_z = lambda x, on_boundary: near(x[2], 0.0, 1.0e-3) and on_boundary

        bc_sphere_disp_x = DirichletBC(self.Q.sub(0).sub(0), Constant(0.0), boundary_x)
        bc_sphere_disp_y = DirichletBC(self.Q.sub(0).sub(1), Constant(0.0), boundary_y)
        bc_sphere_disp_z = DirichletBC(self.Q.sub(0).sub(2), Constant(0.0), boundary_z)

        bc_sphere_rot_y = DirichletBC(self.Q.sub(1), Constant((0.0, 0.0)), boundary_y)
        bc_sphere_rot_x = DirichletBC(self.Q.sub(1).sub(1), Constant(0.0), boundary_x)
        bc_sphere_rot_z = DirichletBC(self.Q.sub(1).sub(1), Constant(0.0), boundary_z)

        self.bcs = [
            bc_sphere_disp_x,
            bc_sphere_disp_y,
            bc_sphere_disp_z,
            bc_sphere_rot_x,
            bc_sphere_rot_y,
            bc_sphere_rot_z,
        ]

    def boundary_conditions_n0(self):
        boundary_x = lambda x, on_boundary: near(x[0], 0.0, 1.0e-3) and on_boundary
        boundary_y = lambda x, on_boundary: near(x[1], 0.0, 1.0e-3) and on_boundary
        boundary_z = lambda x, on_boundary: near(x[2], 0.0, 1.0e-3) and on_boundary

        bc_n0_x = DirichletBC(self.V_normal.sub(0), Constant(0.0), boundary_x)
        bc_n0_y = DirichletBC(self.V_normal.sub(1), Constant(0.0), boundary_y)
        bc_n0_z = DirichletBC(self.V_normal.sub(2), Constant(0.0), boundary_z)
        bcs_n0 = [bc_n0_x, bc_n0_y, bc_n0_z]

        return bcs_n0

    def set_solver(self):
        self.solver = PETScSNESSolver()
        self.solver.parameters["method"] = "newtonls"
        self.solver.parameters["maximum_iterations"] = 20
        self.solver.parameters["linear_solver"] = "lu"
        self.solver.parameters["absolute_tolerance"] = 1e-6
        self.solver.parameters["relative_tolerance"] = 1e-6

    def set_shape(self):
        x = SpatialCoordinate(self.mesh)
        initial_shape = x
        self.phi0 = project(initial_shape, self.V_phi)

    def set_local_frame(self):
        a1, a2, n = local_frame(self.mesh, self.n0)
        self.a1.assign(
            project(
                a1,
                self.VT,
                form_compiler_parameters={
                    "representation": "uflacs",
                    "quadrature_degree": 4,
                },
            )
        )
        self.a2.assign(
            project(
                a2,
                self.VT,
                form_compiler_parameters={
                    "representation": "uflacs",
                    "quadrature_degree": 4,
                },
            )
        )

        self.n0.assign(
            project(
                n,
                self.V_normal,
                form_compiler_parameters={
                    "representation": "uflacs",
                    "quadrature_degree": 4,
                },
            )
        )
        bcs = self.boundary_conditions_n0()
        for bc in bcs:
            bc.apply(self.n0.vector())

    def director(self, beta):
        return as_vector(
            [sin(beta[1]) * cos(beta[0]), -sin(beta[0]), cos(beta[1]) * cos(beta[0])]
        )

    def d_director(self, beta, beta_):
        """ linearized director """
        B0 = as_matrix(
            [
                [-sin(beta[0]) * sin(beta[1]), cos(beta[0]) * cos(beta[1])],
                [-cos(beta[0]), 0.0],
                [-sin(beta[0]) * cos(beta[1]), -cos(beta[0]) * sin(beta[1])],
            ]
        )

        return dot(B0, beta_)

    def set_director(self):
        n = self.n0

        self.beta0 = project(
            as_vector(
                [
                    atan_2(-n[1] - DOLFIN_EPS, sqrt(n[0] ** 2 + n[2] ** 2)),
                    atan_2(n[0] + DOLFIN_EPS, n[2] + DOLFIN_EPS),
                ]
            ),
            self.V_beta,
        )
        # The director in the initial configuration is then written as ::
        self.d0 = self.director(self.beta0)

    def d1(self, u):
        return u.dx(0) * self.a1[0] + u.dx(1) * self.a1[1] + u.dx(2) * self.a1[2]

    def d2(self, u):
        return u.dx(0) * self.a2[0] + u.dx(1) * self.a2[1] + u.dx(2) * self.a2[2]

    def grad_(self, u):
        return as_tensor([self.d1(u), self.d2(u)])

    def set_fundamental_forms(self):
        self.a0_init = as_tensor(
            [
                [dot(self.a1, self.a1), dot(self.a1, self.a2)],
                [dot(self.a2, self.a1), dot(self.a2, self.a2)],
            ]
        )

        self.b0_init = -sym(
            as_tensor(
                [
                    [
                        inner(as_vector(self.grad_(self.d0)[0, :]), self.a1),
                        inner(as_vector(self.grad_(self.d0)[0, :]), self.a2),
                    ],
                    [
                        inner(as_vector(self.grad_(self.d0)[1, :]), self.a1),
                        inner(as_vector(self.grad_(self.d0)[1, :]), self.a2),
                    ],
                ]
            )
        )

    def set_kinematics_and_fundamental_forms(self):
        self.F = self.grad_(self.u_) + self.grad_(self.phi0)
        self.F0 = self.grad_(self.phi0)
        self.g_u = self.grad_(self.u_)

        self.d = self.director(self.beta_ * self.dt + self.beta0)

        self.a0 = as_tensor(
            [
                [dot(self.a1, self.a1), dot(self.a1, self.a2)],
                [dot(self.a2, self.a1), dot(self.a2, self.a2)],
            ]
        )

        self.b0 = -sym(
            as_tensor(
                [
                    [
                        dot(as_vector(self.grad_(self.d0)[0, :]), self.a1),
                        inner(as_vector(self.grad_(self.d0)[0, :]), self.a2),
                    ],
                    [
                        inner(as_vector(self.grad_(self.d0)[1, :]), self.a1),
                        dot(as_vector(self.grad_(self.d0)[1, :]), self.a2),
                    ],
                ]
            )
        )

        self.j0 = det(self.a0)

        self.a0_contra = inv(self.a0)
        self.H = 0.5 * inner(self.a0_contra, self.b0)

    def membrane_deformation(self):
        return 0.5 * (self.F0 * self.g_u.T + self.g_u * self.F0.T)

    def bending_deformation(self):
        dd = self.d_director(self.beta0, self.beta_)
        return -0.5 * (
            self.g_u * self.grad_(self.d0).T
            + self.grad_(self.d0) * self.g_u.T
            + self.F0 * self.grad_(dd).T
            + self.grad_(dd) * self.F0.T
        )

    def shear_deformation(self):
        return self.g_u * self.d0 + self.F0 * self.d_director(self.beta0, self.beta_)

    def set_thickness(self, dt):
        D_Delta = inner(self.a0_contra, self.membrane_deformation())

        self.thickness.assign(
            project(
                (self.thickness / dt + self.vp)
                / (1 / dt + D_Delta + self.kd + 1.0 * self.vp * self.H),
                self.V_thickness,
            )
        )

    def set_energies(self):
        # Gaussian signal in the middle of the plate and uniform across one of the directions

        self.Q_Expression = Expression(
            ("basal + (zeta - basal)*exp(-0.5*(x[1]*x[1])/(sig_q*sig_q))"),
            sig_q=self.gaussian_width,
            basal=self.basal,
            zeta=self.zeta,
            degree=2,
        )

        self.Q_field = interpolate(self.Q_Expression, self.V_thickness)
        self.q_11, self.q_12, self.q_22, self.q_33 = 1.0 / 6, 0.0, 1.0 / 6, -1.0 / 3
        self.Q_tensor = as_tensor([[1.0 / 6, 0.0], [0.0, 1.0 / 6]])

        i, j, l, m = Index(), Index(), Index(), Index()
        A_ = as_tensor(
            (
                0.5 * self.a0_contra[i, j] * self.a0_contra[l, m]
                + 0.25
                * (
                    self.a0_contra[i, l] * self.a0_contra[j, m]
                    + self.a0_contra[i, m] * self.a0_contra[j, l]
                )
            ),
            [i, j, l, m],
        )
        C_active = as_tensor(
            self.Q_tensor[l, m]
            * (
                self.H
                * (
                    self.a0_contra[i, l] * self.a0_contra[j, m]
                    + self.a0_contra[j, l] * self.a0_contra[i, m]
                )
                + self.a0_contra[i, j] * self.b0[m, l]
                - 0.75
                * (
                    self.a0_contra[i, m] * self.b0[j, l]
                    + self.a0_contra[i, l] * self.b0[j, m]
                    + self.a0_contra[j, m] * self.b0[i, l]
                    + self.a0_contra[j, l] * self.b0[i, m]
                )
            )
            + (self.b0[i, j] - 4 * self.H * self.a0_contra[i, j]) * self.q_33,
            [i, j],
        )

        Q_alphabeta = as_tensor(
            (
                self.a0_contra[i, l] * self.a0_contra[j, m] * self.Q_tensor[l, m]
                - self.a0_contra[i, j] * self.q_33
            ),
            [i, j],
        )

        self.N_passive = (
            4.0
            * self.thickness
            * self.mu
            * as_tensor(A_[i, j, l, m] * self.membrane_deformation()[l, m], [i, j])
        )
        self.N_polymerization = (
            2.0
            * self.thickness
            * self.mu
            * self.a0_contra
            * (self.kd - self.vp / self.thickness)
        )
        self.N_active = self.Q_field * self.thickness * Q_alphabeta

        self.N = self.N_active + self.N_passive + self.N_polymerization

        self.M = (
            (self.thickness ** 3 / 3.0)
            * self.mu
            * as_tensor(A_[i, j, l, m] * self.bending_deformation()[l, m], [i, j])
        )

        self.T_shear = (
            self.thickness
            * self.mu
            * as_tensor(self.a0_contra[i, j] * self.shear_deformation()[j], [i])
        )

        self.passive_membrane_energy = inner(
            self.N_passive, self.membrane_deformation()
        )

        self.active_membrane_energy = inner(self.N_active, self.membrane_deformation())

        self.passive_bending_energy = inner(self.M, self.bending_deformation())

        # does not appear in M...
        self.active_bending_energy = (self.thickness ** 3 / 12.0) * inner(
            C_active * self.Q_field, self.bending_deformation()
        )

        self.polymerization_membrane = inner(
            self.N_polymerization, self.membrane_deformation()
        )

        # does not appear in M...
        self.polymerization_bending = (
            (self.thickness ** 3 / 3.0)
            * self.mu
            * inner(
                as_tensor(
                    (self.H * self.a0_contra[i, j] - self.b0[i, j])
                    * (self.kd - self.vp / self.thickness),
                    [i, j],
                ),
                self.bending_deformation(),
            )
        )

        self.psi_m = inner(self.N, self.membrane_deformation())
        self.psi_b = inner(self.M, self.bending_deformation())
        self.psi_s = SHEAR_PENALTY * inner(self.T_shear, self.shear_deformation())

    def total_energy(self):
        # Total Energy densities
        self.dx = Measure("dx", domain=self.mesh)

        # The total elastic energy and its first and second derivatives
        self.Pi = (self.psi_m + self.psi_b + self.psi_s) * sqrt(self.j0) * self.dx

        if self.HyperOsmotic == 1:
            a, b, c, d = 0.2, 5.0, 4 * np.pi / 3, -1.0  # Hypershock
            self.dV = Expression(
                ("a*b*exp(-b*(d+t))/pow(1.+exp(-b*(d+t)),2)"),
                t=self.time,
                a=a,
                b=b,
                c=c,
                d=d,
                element=self.Q.sub(2).collapse().ufl_element(),
            )
        if self.HypoOsmotic == 1:
            a, b, c, d = -1.0, 4.0, 4 * np.pi / 3, -2.0  # Hyposhock
            self.dV = Expression(
                ("a*b*exp(-b*(d+t))/pow(1.+exp(-b*(d+t)),2)"),
                t=self.time,
                a=a,
                b=b,
                c=c,
                d=d,
                element=self.Q.sub(2).collapse().ufl_element(),
            )

        self.Pi = (
            self.Pi
            + self.lbda_ * (dot(self.u_, self.n0) + self.dV) * sqrt(self.j0) * self.dx
        )

        self.dPi = derivative(self.Pi, self.q_, self.q_t)
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
        self.time += dt
        self.set_thickness(dt)

        # Current displacement U*dt
        displacement_mesh = Function(self.V_phi)
        displacement_mesh.interpolate(self.q_.sub(0, True))
        displacement_mesh.vector()[:] *= dt

        # Update mesh position with current displacement
        ALE.move(self.mesh, displacement_mesh)
        # Reinitialize quantities on new mesh
        self.initialize()

    def solve(self):
        problem = CustomNonlinearProblem(self.dPi, self.J, self.bcs)
        return self.solver.solve(problem, self.q_.vector())

    def mesh_refinement(self, control_type):
        with XDMFFile("mesh.xdmf") as ffile:
            ffile.parameters["functions_share_mesh"] = True
            ffile.write(self.mesh)

        fname_out = mesh_adapt(self.paths["mmg"], self.paths["gmsh"], control_type)

        self.mmesh = meshio.read(fname_out)

        # read in with FEniCS
        new_mesh = Mesh()
        with XDMFFile(fname_out) as ffile:
            ffile.read(new_mesh)
        self.mesh = new_mesh
        self.adapt_and_interpolate()

    def adapt_and_interpolate(self):

        P2 = FiniteElement("Lagrange", self.mesh.ufl_cell(), degree=2)
        P1 = FiniteElement("Lagrange", self.mesh.ufl_cell(), degree=2)
        CR1 = FiniteElement("CR", self.mesh.ufl_cell(), degree=1)
        R = FiniteElement("Real", self.mesh.ufl_cell(), degree=0)

        element = MixedElement([VectorElement(P2, dim=3), VectorElement(CR1, dim=2), R])

        self.Q = FunctionSpace(self.mesh, element)
        self.V_phi = self.Q.sub(0).collapse()
        self.V_beta = self.Q.sub(1).collapse()
        self.V_thickness = FunctionSpace(self.mesh, P1)
        self.V_alpha = FunctionSpace(self.mesh, "DG", 0)
        self.VT = VectorFunctionSpace(self.mesh, "DG", 0, dim=3)
        self.V_normal = self.Q.sub(0).collapse()

        # interpolate
        self.phi0 = interpolate(self.phi0, self.V_phi)
        self.beta0 = interpolate(self.beta0, self.V_beta)

        self.q_ = interpolate(self.q_, self.Q)
        self.u_ = interpolate(self.q_.sub(0), self.Q.sub(0).collapse())
        self.beta_ = interpolate(self.q_.sub(1), self.Q.sub(1).collapse())

        self.thickness = interpolate(self.thickness, self.V_thickness)
        self.Q_field = interpolate(self.Q_field, self.V_thickness)

        self.q_t, self.q = TrialFunction(self.Q), TestFunction(self.Q)
        self.u_, self.beta_, self.lbda_ = split(self.q_)

        self.set_local_frame()
        self.set_boundary_conditions()
        self.set_director()
        self.set_kinematics_and_fundamental_forms()
        self.set_energies()
        self.total_energy()
        self.set_solver()


def local_frame(mesh, normal=None):
    t = Jacobian(mesh)
    J = JacobianDeterminant(mesh)
    if mesh.geometric_dimension() == 2:
        tt1 = as_vector([t[0, 0], t[1, 0], 0])
        tt2 = as_vector([t[0, 1], t[1, 1], 0])
    else:
        tt1 = as_vector([t[0, 0], t[1, 0], t[2, 0]])
        tt2 = as_vector([t[0, 1], t[1, 1], t[2, 1]])
    # switch t1 and t2 if Jacobian is negative
    t1 = conditional(J < 0, tt2, tt1)
    t2 = conditional(J < 0, tt1, tt2)
    n = cross(t1, t2)
    if normal is not None:
        n = conditional(dot(normal, n) < 0, -n, n)
    n /= sqrt(dot(n, n))
    ey = as_vector([0, 1, 0])
    ez = as_vector([0, 0, 1])
    a1 = cross(ey, n)
    norm_a1 = sqrt(dot(a1, a1))
    a1 = conditional(lt(norm_a1, DOLFIN_EPS), ez, a1 / norm_a1)

    a2 = cross(n, a1)
    a2 /= sqrt(dot(a2, a2))
    n = cross(a1, a2)
    n /= sqrt(dot(n, n))
    return a1, a2, n
