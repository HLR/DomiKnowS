"""Physical mesh measurements; no task names, goals, or reference plans."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def native_mesh(handle):
    """Read/copy/free all three Coppelia mesh buffers exactly once.

    Pinned PyRep 8f420be's simGetShapeMesh overwrites its normals pointer with
    indices, reads past the index buffer, then frees an integer as a pointer.
    Do not call that binding, even when only vertices/triangles are needed.
    """
    from pyrep.backend import sim
    ffi=sim.ffi
    vertices=ffi.new('float **');indices=ffi.new('int **');normals=ffi.new('float **')
    vertex_count=ffi.new('int *');index_count=ffi.new('int *')
    try:
        result=sim.lib.simGetShapeMesh(handle,vertices,vertex_count,indices,index_count,normals)
        if result<0:raise RuntimeError('native mesh query failed')
        v=np.array([vertices[0][i] for i in range(vertex_count[0])],dtype=float).reshape(-1,3)
        t=np.array([indices[0][i] for i in range(index_count[0])],dtype=int).reshape(-1,3)
        return v,t
    finally:
        for pointer in (vertices,indices,normals):
            if pointer[0]!=ffi.NULL:sim.simReleaseBuffer(ffi.cast('char *',pointer[0]))


def world_vertices(vertices, transform):
    points=np.asarray(vertices,dtype=float).reshape(-1,3)
    if not len(points) or not np.isfinite(points).all():raise ValueError('missing physical mesh')
    transform=np.asarray(transform)
    return points@transform[:3,:3].T+transform[:3,3]


def symmetry_axis(vertices, tolerance=.0005):
    """Require both 45 and 90 degree invariance; a square is not a cylinder."""
    points=np.asarray(vertices,dtype=float).reshape(-1,3)
    points=points-(points.max(axis=0)+points.min(axis=0))/2
    tree=cKDTree(points)
    for axis in range(3):
        errors=[]
        for angle in (np.pi/4,np.pi/2):
            vector=np.zeros(3);vector[axis]=angle
            errors.append(np.max(tree.query(Rotation.from_rotvec(vector).apply(points))[0]))
        if max(errors)<=tolerance:return axis
    return None


def separating_support_contact(*, initial_bottom, bottom, support, initial_center, center,
                               initial_tip, tip, tolerance=.003):
    """Allow existing shallow support contact only during upward separation.

    Never allows a new collision, deeper penetration, or lateral scraping.
    The caller must first establish measured support at the initial pose.
    """
    return bool(abs(initial_bottom-support)<=tolerance and
                bottom>=initial_bottom-.001 and
                np.asarray(center)[2]>=np.asarray(initial_center)[2]-.001 and
                np.asarray(tip)[2]>=np.asarray(initial_tip)[2]-.001 and
                np.linalg.norm((np.asarray(center)-initial_center)[:2])<=.002)
