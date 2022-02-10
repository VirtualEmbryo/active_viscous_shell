import meshio
import os


def mesh_adapt(mmg_path, gmsh_path, control_type):
    print("Remeshing....")
    # Convert to Medit format
    os.system(
        "meshio convert --input-format xdmf --output-format medit mesh.xdmf mesh.mesh"
    )

    dist = 0.002  # controls the optimized mesh resolution (see https://www.mmgtools.org/mmg-remesher-try-mmg/mmg-remesher-options/mmg-remesher-option-hausd)
    # call mmgs with mesh optimization and Hausdorff distance
    if control_type == "hausd":
        os.system(
            mmg_path
            + " -in mesh.mesh -v 0 -out mesh_optimized.mesh -hausd {}".format(dist)
        )
    #             os.system('/opt/mmg/bin/mmgs_O3 -in mesh.mesh -out mesh_optimized.mesh -hausd {} -optim'.format(dist)) # Jeremy's
    elif control_type == "hsiz":
        dist = 0.02
        os.system(
            mmg_path
            + " -in mesh.mesh -v 0 -out mesh_optimized.mesh -hsiz {}".format(dist)
        )
    # Convert back to .msh format using Gmsh
    os.system(
        gmsh_path + " mesh_optimized.mesh -3 -format msh2 -v 1 -o mesh_optimized.msh"
    )

    fname_out = "mesh_optimized.xdmf"

    # read back with meshio to remove cells and cell data and convert to xdmf
    mmesh = meshio.read(fname_out.replace(".xdmf", ".msh"))
    # mmesh.remove_lower_dimensional_cells()
    # meshio.write(fname_out, meshio.Mesh(mmesh.points, mmesh.cells))
    meshio.write(
        fname_out,
        meshio.Mesh(
            points=mmesh.points, cells={"triangle": mmesh.cells_dict["triangle"]}
        ),
    )
    return fname_out
