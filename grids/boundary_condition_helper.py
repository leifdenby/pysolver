import numpy as np
import matplotlib.pyplot as plot
from grids import boundary_conditions as BCs

"""
    This module contains helpers to easily construct the strings needed for setting boundary conditions.

    The main work is done by the 'make_bc_string' method, which may then used for higher level operations, see
    e.g. 'make_2d_all_reflective_bc_string' as an example.
"""



def make_2d_all_transmissive_bc_string(var_name, boundary_thickness):
    s = []
    for t in range(1,boundary_thickness+1):
        for (i, j) in [(1,0), (-1,0), (0,-1), (0,1)]:
            n = i != 0 and -i or -j
            s.append(make_bc_string(n, (i,j), t, var_name=var_name))
    return "\n".join(s)

def make_2d_all_periodic_bc_string(var_name, boundary_thickness):
    s = []
    for t in range(1,boundary_thickness+1):
        for (i, j) in [(1,0), (-1,0), (0,-1), (0,1)]:
            n = i != 0 and i or j
            s.append(make_bc_string(n, (i,j), t, var_name=var_name))
    return "\n".join(s)

def make_1d_fixed_gradient_bc(normal, edge_pos, boundary_thickness, var_name, gradient_times_spacing):
    if gradient_times_spacing != 0.0 and boundary_thickness > 1:
        raise Exception("Error: None-zero gradient has not been implemented for boundaries thicker than one cell")
    else:
        return make_bc_string(normal=normal, edge_pos=edge_pos, boundary_thickness=boundary_thickness, var_name=var_name, coefficients = (-gradient_times_spacing, None))

def make_bc_string(normal, edge_pos, boundary_thickness, apply_to_edges = False, var_name="Q", coefficients = (None, None)):
    """
        Constructs the strings for correctly indexing an array of arbitrary
        dimension (though likely you wont need more than 3 dimensions).

        The following arguments are passed to 'make_bc_slices' internally, the
        documentation is reproduced here:

            The edge_pos is a tuple discribing the position of the edge we are
            currently working on. E.g. for a 2D domain (-1,0) would be the
            leftmost edge, if x is in the horizontal direction, (0,1) would be
            the topmost edge.

            The normal direction is expected to be either 1 or -1, pointing in
            the direction of the axis that the position vector defines.
            Depending the relative sign of the position and the normal the
            indexing will wrap around (useful for creating cyclic boundary
            conditions).

            boundary_thickness is fairly selfexplanatory

            It may sometimes be the case that the boundary conditions should be
            applied all the way to the domain edge, e.g. for a moving wall
            boundary condition. This can be set by 'apply_to_edges'.

        Optionally the variable name can be set through 'var_name' and
        coefficients used in constructing the string may be passed in as a
        two-tuple as 'coefficients'. They will be combined in the strings to
        give e.g.  'Q[1:-2,-1,1:-2] = a +(b)*Q[1:-2,-2,1:-2]'
    """
    
    if normal != 0:
        (a, b) = (coefficients[0] and "%s +" % str(coefficients[0]) or "", coefficients[1] and "(%s)*" % str(coefficients[1]) or "")
    elif normal == 0:
        a = str(coefficients[0])

    slices = make_bc_slices(normal, edge_pos, boundary_thickness, apply_to_edges)

    def slice_to_str(slice):
        if slice.start is not None and slice.stop is not None:
            return "%d:%d" % (slice.start, slice.stop)
        elif slice.stop is not None:
            return str(slice.stop)
        else:
            return ":"

    s = []
    for bc_slice in slices:
        ghost_slice_str = ",".join([slice_to_str(ss) for ss in bc_slice['i_ghost']])
        if normal != 0:
            internal_slice_str = ",".join([slice_to_str(ss) for ss in bc_slice['i_internal']])
            s.append("%s[%s] = %s%s%s[%s]" % (var_name, ghost_slice_str, a, b, var_name, internal_slice_str))
        elif normal == 0:
            s.append("%s[%s] = %s" % (var_name, ghost_slice_str, a))

    return "\n".join(s)

def make_bc_slices(normal, edge_pos, boundary_thickness, apply_to_edges = False):
    """
        Constructs boundary condition slices for correctly indexing an array of
        arbitrary dimension (though likely you wont need more than 3
        dimensions).

        The edge_pos is a tuple discribing the position of the edge we are
        currently working on. E.g. for a 2D domain (-1,0) would be the leftmost
        edge, if x is in the horizontal direction, (0,1) would be the topmost
        edge.

        The normal direction is expected to be either 1 or -1, pointing in the
        direction of the axis that the position vector defines.  Depending the
        relative sign of the position and the normal the indexing will wrap
        around (useful for creating cyclic boundary conditions).

        boundary_thickness is fairly selfexplanatory

        It may sometimes be the case that the boundary conditions should be
        applied all the way to the domain edge, e.g. for a moving wall boundary
        condition. This can be set by 'apply_to_edges'.
    """
    def get_ranges(pos, n = 0):
        def l(p):
            # p is either 0, each which case it represents that the range of indecies for this direction is requested
            # or p is non-zeros, in which case it represents the distance from the boundary for which we are requesting an index
            if p != 0:
                # if the normal direction and the position have the same sign then we need to do some wrapping
                if n != 0:
                    # requesting the index of the cell from which data is taken
                    wrap = n * p > 0
                    if wrap:
                        # indexing must wrap-around the end of the domain, used for cyclic BCs
                        if p < 0:
                            return p-boundary_thickness
                        elif p > 0:
                            return p+boundary_thickness-1
                    else:
                        if p < 0:
                            return boundary_thickness-p-1
                        elif p > 0:
                            return -p-boundary_thickness
                else:
                    # just requesting the index of the boundary cell
                    if p > 0:
                        return -boundary_thickness-1+p
                    elif p < 0:
                        return boundary_thickness+p
            else:
                # not requesting the index for a single cell row, return ranges of cells in the plane of the boundary
                if apply_to_edges:
                    return slice(None)
                else:
                    return slice(boundary_thickness, -boundary_thickness)

        return map(lambda p: l(p), pos)

    if list(edge_pos).count(0) != len(edge_pos)-1:
        raise Exception("Only one of the position indexes should be non-zero")
    if abs(normal) != 1 and normal != 0:
        raise Exception("The normal should be either 1, -1 or 0")
    if not (-1 in edge_pos or 1 in edge_pos):
        raise Exception("The edge position should be a tuple of either -1, 0 or 1, e.g. (-1,0) for the leftmost boundary in 2D")

    s = []
    for r in range(1,boundary_thickness+1):
        # create a local position vector which represents the relative distance between the ghost cells we are currently
        # interested in and the boundary
        pos = map(lambda t: t == 0 and t or t*r, edge_pos)
        if normal != 0:
            s.append({'i_ghost':get_ranges(pos), 'i_internal':get_ranges(pos, normal), 'row':r-1})
        elif normal == 0:
            s.append({'i_ghost':get_ranges(pos), 'i_internal':None, 'row':r-1})
    return s

def applyCellCenteredBCs(Q, all_boundary_conditions, grid, num_ghost_cells = None):
    if num_ghost_cells is None:
        num_ghost_cells = grid.num_ghost_cells

    for component, boundary_conditions in all_boundary_conditions.items():
        c = component
        for i, bc in enumerate(boundary_conditions):
            axis = i / 2
            side = -1 if i % 2 == 0 else 1

            if isinstance(bc, BCs.Neumann) or isinstance(bc, BCs.Dirichlet):
                # normal is facing in
                normal = side * -1
            elif isinstance(bc, BCs.Periodic):
                normal = side
            else:
                raise Exception("Primitive boundary condition type not understood")

            edge_pos = grid.edges[i]
            if isinstance(bc, BCs.MovingWall):
                apply_to_edges = True
            else:
                apply_to_edges = False
            slices = make_bc_slices(normal=normal, edge_pos=edge_pos,
                        boundary_thickness=num_ghost_cells, apply_to_edges=apply_to_edges)

            if isinstance(bc, BCs.Periodic):
                for bc_slice in slices:
                    Q[...,c][bc_slice['i_ghost']] = Q[...,c][bc_slice['i_internal']]
            elif isinstance(bc, BCs.Neumann):
                for bc_slice in slices:
                    row = bc_slice['row'] #  distance from interface
                    dx = grid.getGridSpacing()[axis]
                    Q[...,c][bc_slice['i_ghost']] = Q[...,c][bc_slice['i_internal']] - (row*2 + 1)*dx*bc.slope*normal
            elif isinstance(bc, BCs.Dirichlet):
                for bc_slice in slices:
                    Q[...,c][bc_slice['i_ghost']] = 2*bc.fixed_value - Q[...,c][bc_slice['i_internal']]
            else:
                raise Exception("Primitive boundary condition type not understood")


            

def test():
    from grids import grid2d
    edges = grid2d.edges

    assert sliceFromEdge(edges[0]) == (0, slice(None, None, None))

if __name__ == "__main__":
    test()
