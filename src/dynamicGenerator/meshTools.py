import gmsh


def generate_mesh_from_stp(stp_file_path, output_mesh_file='data/stl/output.stl', mesh_size=1.0, range=1.0,gui=False):
    """
    generate mesh from stp
    参数:
        stp_file_path (str): STP file path
        output_mesh_file (str): output path
        mesh_size (float): global mesh size
        range (float): MeshSizeMin=mesh_size-range, MeshSizeMax=mesh_size+range
    """

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)  # 启用终端输出
    
    try:
        gmsh.model.occ.importShapes(stp_file_path)
        gmsh.model.occ.synchronize()
        
        entities = gmsh.model.getEntities()
        
        gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size-range)
        gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size+range)
        

        gmsh.model.mesh.generate(3) # 3D mesh
        
        gmsh.model.mesh.optimize("Netgen")
        
        gmsh.write(output_mesh_file)
        print(f"mesh saved: {output_mesh_file}")
        
        if gui:
            gmsh.fltk.run()
    
    except Exception as e:
        print(f"mesh generate error: {str(e)}")
    
    finally:
        gmsh.finalize()