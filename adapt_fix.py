#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 13:11:35 2019

@author: bleyerj
"""
from dolfin import *

cpp_code = """
    #include<pybind11/pybind11.h>
    #include<dolfin/adaptivity/adapt.h>
    #include<dolfin/mesh/Mesh.h>
    #include<dolfin/mesh/MeshFunction.h>
    #include<dolfin/fem/DirichletBC.h>
    #include<dolfin/function/FunctionSpace.h>
    #include<dolfin/function/Function.h>
    #include<dolfin/fem/Form.h>
    #include<dolfin/fem/LinearVariationalProblem.h>

    namespace py = pybind11;

    PYBIND11_MODULE(SIGNATURE, m) {
       m.def("adapt", (std::shared_ptr<dolfin::MeshFunction<std::size_t>> (*)(const dolfin::MeshFunction<std::size_t>&,
          std::shared_ptr<const dolfin::Mesh>)) &dolfin::adapt,
          py::arg("mesh_function"), py::arg("adapted_mesh"));
       m.def("adapt", (std::shared_ptr<dolfin::DirichletBC> (*)(const dolfin::DirichletBC&,
       std::shared_ptr<const dolfin::Mesh>, const dolfin::FunctionSpace&)) &dolfin::adapt,
        py::arg("bc"), py::arg("adapted_mesh"), py::arg("S"));
       m.def("adapt", (std::shared_ptr<dolfin::FunctionSpace> (*)(const dolfin::FunctionSpace&,
       std::shared_ptr<const dolfin::Mesh>)) &dolfin::adapt,
        py::arg("space"), py::arg("adapted_mesh"));
       m.def("adapt", (std::shared_ptr<dolfin::Form> (*)(const dolfin::Form&,
       std::shared_ptr<const dolfin::Mesh>, bool)) &dolfin::adapt,
        py::arg("form"), py::arg("adapted_mesh"), py::arg("adapt_coefficients")=true);
       m.def("adapt", (std::shared_ptr<dolfin::Function> (*)(const dolfin::Function&,
       std::shared_ptr<const dolfin::Mesh>, bool)) &dolfin::adapt,
       py::arg("function"), py::arg("adapted_mesh"), py::arg("interpolate")=true);
       m.def("adapt", (std::shared_ptr<dolfin::LinearVariationalProblem> (*)(const dolfin::LinearVariationalProblem&,
       std::shared_ptr<const dolfin::Mesh>)) &dolfin::adapt, py::arg("problem"), py::arg("adapted_mesh"));
    }
    """

adapt = compile_cpp_code(cpp_code).adapt
