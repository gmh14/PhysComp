import pypgo
import numpy as np
import trimesh
import os
import tetgen

def get_surface_vf(faces):
    # get surface faces
    org_triangles = np.vstack(
        [
            faces[:, [1, 2, 3]],
            faces[:, [0, 3, 2]],
            faces[:, [0, 1, 3]],
            faces[:, [0, 2, 1]],
        ]
    )
    # Sort each triangle's vertices to avoid duplicates due to ordering
    triangles = np.sort(org_triangles, axis=1)
    unique_triangles, tri_idx, counts = np.unique(
        triangles, axis=0, return_index=True, return_counts=True
    )
    once_tri_id = counts == 1
    surface_triangles = unique_triangles[once_tri_id]
    surface_vertices = np.unique(surface_triangles)
    vertex_mapping = {vertex_id: i for i,
                      vertex_id in enumerate(surface_vertices)}
    mapped_triangles = np.vectorize(vertex_mapping.get)(
        org_triangles[tri_idx][once_tri_id]
    )
    return surface_vertices, mapped_triangles


def matching(save_folder):
    density = 1000
    E = 500000 # e547b70f8bb549e3ad5cea9a4eed800d 
    nu = 0.45
    
    tetmesh = pypgo.create_tetmesh_from_file(os.path.join(save_folder, "reconstruct_res.veg"))
    tet_vtx = pypgo.get_tetmesh_vertex_positions(tetmesh)
    tet_elem = pypgo.get_tetmesh_element_indices(tetmesh)
    
    tetmesh = pypgo.create_tetmesh(tet_vtx.flatten(), tet_elem.flatten(), E, nu, density)
    pypgo.save_tetmesh_to_file(tetmesh, os.path.join(save_folder, "inverse_opt_init.veg"))
    
    surf_vid, surf_fid = get_surface_vf(tet_elem)
    surfmesh_init = trimesh.Trimesh(
        vertices=tet_vtx[surf_vid], faces=surf_fid)
    surfmesh_init.export(os.path.join(save_folder, "inverse_opt_init.obj"))
    
    # get fixed vtx
    os.makedirs(save_folder, exist_ok=True)

    min_y = np.min(tet_vtx[:, 2])
    fixed_vertices = np.where(tet_vtx[:, 2] < min_y + 5e-3)[0]
    fixed_vtx_file = os.path.join(save_folder, "fixed.txt")
    np.savetxt(fixed_vtx_file, fixed_vertices, fmt='%d')
    
    # static eq w/o opt
    tetmesh_static_eq = pypgo.run_quasi_static_sim(os.path.join(
        save_folder, "inverse_opt_init.veg"), fixed_vtx_file, os.path.join(save_folder, "static_eq_debug.veg"))
    tetmesh_static_eq_vtx = pypgo.get_tetmesh_vertex_positions(
        tetmesh_static_eq)
    surfmesh_static_eq = trimesh.Trimesh(
        vertices=tetmesh_static_eq_vtx[surf_vid], faces=surf_fid)
    surfmesh_static_eq.export(os.path.join(
        save_folder, "static_eq_debug.obj"))
    
    ## inverse opt
    step_size = 0.2
    verbose = 0
    pypgo.inverse_plasticity_opt(os.path.join(save_folder, "inverse_opt_init.veg"), fixed_vtx_file, save_folder, step_size, verbose)
    
    # save surface meshes
    tetmesh_opt = pypgo.create_tetmesh_from_file(os.path.join(save_folder, "opt.veg"))
    tetmesh_opt_vtx = pypgo.get_tetmesh_vertex_positions(tetmesh_opt)
    tetmesh_opt_elem = pypgo.get_tetmesh_element_indices(tetmesh_opt)
    surf_vid, surf_fid = get_surface_vf(tetmesh_opt_elem)
    surfmesh_opt = trimesh.Trimesh(
        vertices=tetmesh_opt_vtx[surf_vid], faces=surf_fid)
    surfmesh_opt.export(os.path.join(save_folder, "opt.obj"))
    
    tetmesh_opt_rest = pypgo.create_tetmesh_from_file(os.path.join(save_folder, "opt_rest.veg"))
    tetmesh_opt_rest_vtx = pypgo.get_tetmesh_vertex_positions(tetmesh_opt_rest)
    tetmesh_opt_rest_elem = pypgo.get_tetmesh_element_indices(tetmesh_opt_rest)
    surf_vid, surf_fid = get_surface_vf(tetmesh_opt_rest_elem)
    surfmesh_opt_rest = trimesh.Trimesh(
        vertices=tetmesh_opt_rest_vtx[surf_vid], faces=surf_fid)
    surfmesh_opt_rest.export(os.path.join(save_folder, "opt_rest.obj"))


def stabilize(save_folder):
    density = 1000
    E = 100000000 # unicorn
    nu = 0.45
    
    tetmesh = pypgo.create_tetmesh_from_file(os.path.join(save_folder, "reconstruct_res.veg"))
    tet_vtx = pypgo.get_tetmesh_vertex_positions(tetmesh)
    tet_elem = pypgo.get_tetmesh_element_indices(tetmesh)
    
    tetmesh = pypgo.create_tetmesh(tet_vtx.flatten(), tet_elem.flatten(), E, nu, density)
    pypgo.save_tetmesh_to_file(tetmesh, os.path.join(save_folder, "make_it_stand_init.veg"))
    
    surf_vid, surf_fid = get_surface_vf(tet_elem)
    surfmesh_init = trimesh.Trimesh(
        vertices=tet_vtx[surf_vid], faces=surf_fid)
    surfmesh_init.export(os.path.join(save_folder, "make_it_stand_init.obj"))
    
    #### flatten base
    ### only for unicorn
    surfmesh_init = trimesh.load(os.path.join(save_folder, "make_it_stand_init.obj"))
    # rotate surfmesh around x axis by -90 degree
    surfmesh_init.apply_transform(trimesh.transformations.rotation_matrix(
        np.radians(-90), [1, 0, 0]))
    surfmesh_init.export(os.path.join(save_folder, "make_it_stand_init.obj"))
    
    init_trimesh = trimesh.load_mesh(os.path.join(save_folder, "make_it_stand_init.obj"))
    init_trimeshgeo = pypgo.create_trimeshgeo(init_trimesh.vertices.flatten(), init_trimesh.faces.flatten())
    min_z = pypgo.stablity_preprocess(init_trimeshgeo, os.path.join(save_folder, "make_it_stand_init_flattened.obj"))
    
    surfmesh_flattened = trimesh.load_mesh(os.path.join(
        save_folder, "make_it_stand_init_flattened.obj"))
    tetgen_init = tetgen.TetGen(
        surfmesh_flattened.vertices, surfmesh_flattened.faces)
    
    tet_vtx, tet_elem = tetgen_init.tetrahedralize(minratio=1.1, mindihedral=2, quality=False)
    
    tetmesh = pypgo.create_tetmesh(tet_vtx.flatten(), tet_elem.flatten(), E, nu, density)
    pypgo.save_tetmesh_to_file(tetmesh, os.path.join(save_folder, "make_it_stand_init_flattened.veg"))
    
    # get fixed vtx
    os.makedirs(save_folder, exist_ok=True)

    min_y = np.min(tet_vtx[:, 2])
    # fixed_vertices = np.where(tet_vtx[:, 2] < min_y + 5e-3)[0]
    fixed_vertices = np.where(tet_vtx[:, 2] < min_y + 1e-3)[0]
    fixed_vtx_file = os.path.join(save_folder, "fixed.txt")
    np.savetxt(fixed_vtx_file, fixed_vertices, fmt='%d')
    
    verbose = 0
    pypgo.stability_opt(os.path.join(save_folder, "make_it_stand_init_flattened.veg"), fixed_vtx_file, save_folder, verbose)
    
    # save surface meshes
    tetmesh_opt = pypgo.create_tetmesh_from_file(os.path.join(save_folder, "stand_opt.veg"))
    tetmesh_opt_vtx = pypgo.get_tetmesh_vertex_positions(tetmesh_opt)
    tetmesh_opt_elem = pypgo.get_tetmesh_element_indices(tetmesh_opt)
    surf_vid, surf_fid = get_surface_vf(tetmesh_opt_elem)
    surfmesh_opt = trimesh.Trimesh(
        vertices=tetmesh_opt_vtx[surf_vid], faces=surf_fid)
    surfmesh_opt.export(os.path.join(save_folder, "stand_opt.obj"))
    
    tetmesh_opt_rest = pypgo.create_tetmesh_from_file(os.path.join(save_folder, "stand_opt_rest.veg"))
    tetmesh_opt_rest_vtx = pypgo.get_tetmesh_vertex_positions(tetmesh_opt_rest)
    tetmesh_opt_rest_elem = pypgo.get_tetmesh_element_indices(tetmesh_opt_rest)
    surf_vid, surf_fid = get_surface_vf(tetmesh_opt_rest_elem)
    surfmesh_opt_rest = trimesh.Trimesh(
        vertices=tetmesh_opt_rest_vtx[surf_vid], faces=surf_fid)
    surfmesh_opt_rest.export(os.path.join(save_folder, "stand_opt_rest.obj"))
    
    ## scale the mesh for rigid-ipc
    scale = 50
    surfmesh_opt_rest = trimesh.load_mesh(os.path.join(save_folder, "stand_opt_rest.obj"))
    surfmesh_opt_rest.vertices *= scale
    surfmesh_opt_rest.export(os.path.join(save_folder, "stand_opt_rest_scaled.obj"))
    surfmesh_flattened = trimesh.load_mesh(os.path.join(save_folder, "make_it_stand_init_flattened.obj"))
    surfmesh_flattened.vertices *= scale
    surfmesh_flattened.export(os.path.join(save_folder, "make_it_stand_init_flattened_scaled.obj"))
    
    
if __name__ == "__main__":

    # matching
    mesh_name = "e547b70f8bb549e3ad5cea9a4eed800d"
    save_folder = os.path.join("examples", "neurips2024", mesh_name)
    matching(save_folder)
    
    # stability
    mesh_name = "unicorn"
    save_folder = os.path.join("examples", "neurips2024", mesh_name)
    stabilize(save_folder)
    