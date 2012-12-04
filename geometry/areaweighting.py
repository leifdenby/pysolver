import numpy as np
import unittest


def calc_volumefraction(vertices, phi):
    signs_at_vertices = calc_phi_sign(phi, vertices)

    # find which edges have been intersected
    intersected_edges, left_vertices, right_vertices = find_intersected_edges(vertices, phi)

    num_vertices_inside = sum(s == -1.0 for s in signs_at_vertices)

    if num_vertices_inside == 2:
        # check that we only have two intersected edges
        if np.sum(intersected_edges) != 2:
            raise Exception("The volumefraction calculation currently only supports intersection of cells with a single shape")
        else:
            # since two vertices are inside and we have two intersected edge, these edges must be opposite
            # get length of edge along intersected edge from vertices that are inside
            lengths = []
            for l, r in zip(left_vertices, right_vertices):
                if calc_phi_sign(phi, [l]) < 0.0:
                    lengths.append(np.abs(calc_intersection(l, r, phi)))
                else:
                    lengths.append(np.abs(calc_intersection(r, l, phi)))


            # find an edge that wasn't sliced so that we may calculate dy
            dn = np.abs(left_vertices[0] - left_vertices[1])

            if left_vertices[0][1] == right_vertices[0][1]:
                # same y-coordinate
                n = 0
                dx = dn[1]
            else:
                n = 1
                dx = dn[0]

            return 0.5*dx*(lengths[1][n] + lengths[0][n])/(dx*dx)

    elif num_vertices_inside in [1,3]:
        # find index of vertex which is "on its own"
        if num_vertices_inside == 1:
            i = np.nonzero(signs_at_vertices == -1.0)[0][0]
        else:
            i = np.nonzero(signs_at_vertices == 1.0)[0][0]

        # find neighbouring vertices
        i_r = i + 1 if i < 3 else 0
        i_l = i - 1 if i > 0 else 3

        l1 = np.abs(calc_intersection(vertices[i], vertices[i_l], phi))
        l2 = np.abs(calc_intersection(vertices[i], vertices[i_r], phi))

        if vertices[i][1] == vertices[i_r][1]:
            # same y-coordinate
            n_r = 0
            n_l = 1
        else:
            n_r = 1
            n_l = 0

        dx = abs(vertices[i_r][n_r] - vertices[i][n_r])
        dy = abs(vertices[i][n_l] - vertices[i_l][n_l])

        if num_vertices_inside == 1:
            # we've been calculating from a vertex inside, just use the lengths
            return 0.5*l1[n_l]*l2[n_r]/(dx*dy)
        else:
            # we've been calculating from a vertex outside, need to subtract the area from dx*dy
            return (dx*dy - 0.5*l1[n_l]*l2[n_r])/(dx*dy)

    elif num_vertices_inside == 4:
        return 1.0
    else:
        return 0.0

def find_intersected_edges(vertices, phi):
    """
    Creates a list of same length as vertices, each entry denotes whether
    a specific edge has been intersected by the curve described by the
    sign-distance function phi.

    The ordering is that the first element in the returned list corresponds
    to the edge between the first two vertices. The last element of the list
    corresponds to the edge between the last and first vertex.
    """
    
    s_l = calc_phi_sign(phi, vertices)
    s_r = rotate(s_l)
    sliced_edges = np.nonzero(s_l == s_r)[0]
    return (s_l == s_r, vertices[sliced_edges], rotate(vertices, reverse=True)[sliced_edges] )

def rotate(l, reverse = False):
    """
    return a rotated copy, the last element is put at the beginning
    """
    n_list = list(l)
    if reverse:
        n_list.insert(len(n_list), n_list.pop(0))
    else:
        n_list.insert(0, n_list.pop())
    return np.array(n_list)

def calc_intersection(p1, p2, phi):
    """
    Calculates a first-order intersection between the line joining p1 and p2
    and curve described by sign-distance function phi

    The intersection is a vector originating at p1
    """
    dist1 = abs(phi(p1))
    dist2 = abs(phi(p2))

    if dist1 == 0.0:
        return p1-p1
    elif dist2 == 0.0:
        return p2-p1
    else:
        return (p2 - p1)/(1.0 + dist2/dist1)

def calc_phi_sign(phi, positions):
    return np.array([1.0 if phi(pos) > 0.0 else -1.0 for pos in positions])

def circle_function(center_pos, radius):
    [x0, y0] = center_pos
    def get_dist(pos):
        [x, y] = pos
        return np.sqrt((x0-x)**2.0 + (y0-y)**2.0) - radius
    return get_dist

def line_function(line_spec = None, x0 = None, y0 = None):
    if line_spec is not None:
        (a, b) = line_spec
        def calc_dist(pos):
            (x, y) = pos
            return y - a*x - b
    elif y0 is not None:
        def calc_dist(pos):
            [x, y] = pos
            return y - y0
    elif x0 is not None:
        def calc_dist(pos):
            [x, y] = pos
            return x - x0
    else:
        raise Exception("Pass in either x0, y0 or (a, b) to the line sign-distance function")
    return calc_dist

class Tests(unittest.TestCase):
    def test_bubbleshape(self):
        vertices = np.array([[ 400., 250.], [ 400., 250.], [ 410.,  260.], [ 410., 250.]])
        phi = circle_function(center_pos=(500.0, 300.0), radius=100.0)
        print calc_volumefraction(vertices=vertices, phi=phi)
        self.assertTrue(False)
    
    def test_intersection(self):
        vertices = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])

        #phi = line_function(x0 = 0.3)
        #self.assertEqual(list(find_intersected_edges(vertices, phi)[0]), [False, True, False, True])

        #phi = line_function(y0 = 0.3)
        #self.assertEqual(list(find_intersected_edges(vertices, phi)[0]), [True, False, True, False])

    def test_volumes(self):
        vertices = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])
        
        #phi = line_function(x0 = 0.3)
        #self.assertEqual(calc_volumefraction(vertices, phi), 0.3)

        #phi = line_function(y0 = 0.6)
        #self.assertEqual(calc_volumefraction(vertices, phi), 0.6)

        #phi = line_function(line_spec = (1.0, 0.0))
        #self.assertEqual(calc_volumefraction(vertices, phi), 0.5)
        
        #phi = line_function(line_spec = (-1.0, 1.0))
        #self.assertEqual(calc_volumefraction(vertices, phi), 0.5)
        
        #phi = line_function(line_spec = (1.0, 0.5))
        #self.assertEqual(calc_volumefraction(vertices, phi), 0.875)
        
        vertices += np.array([1.0, 1.0])
        
        #phi = line_function(line_spec = (1.0, 0.0))
        #self.assertEqual(calc_volumefraction(vertices, phi), 0.5)



if __name__ == "__main__":
    unittest.main()
