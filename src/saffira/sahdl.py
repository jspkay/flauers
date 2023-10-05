import numpy as np
import migen as mg
from . import utils


def get_pe_mapping(n1, n2, n3, T):
    class ProcessingElement:
        def __init__(self, x, y):
            self.x = x
            self.y = y

            """
            flowing is a dictionary of lists that indicates how the PE in position (x,y) connects to the neighbors.
            flowing[1] indicates the PE to which a value is forwarded
            flowing[-1] indicates the PE from which the value is read

            for each direction d in {-1, 1} we have three values: a, b, c. The indexes are connected through the type
            LineType which hold an index for each line. 

            """
            self.flowing = {
                i: [None for j in utils.LineType] for i in [-1, 1]
            }

        def __repr__(self):
            return f"PE{self.x, self.y}"

    P = T[0:2, :]  # space projection matrix
    flow_dirs = [  # Computed from the Uniform Recurrent Equations system
        P @ np.array([1, 0, 0]),  # for b (same order of LineType: b first, a second, and c last)
        P @ np.array([0, 1, 0]),  # for a
        P @ np.array([0, 0, 1]),  # for c
    ]
    print(flow_dirs)

    # PEs is a map of the coordinates of the processing elements. Specifically, the key is a tuple (x, y) that
    # represents the physical coordinate of that PE
    PEs = {}
    for i in range(1, n1+1):
        for j in range(1, n2+1):
            for k in range(1, n3+1):
                # Here we iterate over the whole iteration-space and compute the corresponding physical coordinate
                s = P @ np.array([i, j, k])

                pe = PEs.get(tuple(s))  # we check whether that PE is already in our map
                if pe is not None:
                    continue  # if that's the case, we skip it

                current_pe = ProcessingElement(s[0], s[1])  # otherwise, we create a new one
                PEs[tuple(s)] = current_pe  # and we put it in our map

                # then, we want to find the PEs from which a value comes from (-1)
                # and the PEs to which forward a value(1)
                for direction in [-1, 1]:
                    for line in utils.LineType:  # and we need to do this for the three lines a, b and c
                        # we compute the position associated with that direction and that line
                        new_pos = s + direction * flow_dirs[line.value - 1]
                        pe = PEs.get(tuple(new_pos))  # and check whether we already put it in the map
                        if pe is not None:  # if that's the case, we update the current_pe and the pe it is connected to
                            current_pe.flowing[direction][line.value - 1] = (new_pos[0], new_pos[1])
                            pe.flowing[-1 * direction][line.value - 1] = (s[0], s[1])

    """
    a_index = utils.LineType.a.value - 1
    b_index = utils.LineType.b.value - 1
    c_index = utils.LineType.c.value - 1
    for p, el in PEs.items():
        print(f"{p} ->\n"
              f"\ta {el.flowing[-1][a_index]} -> {el} -> {el.flowing[1][a_index]}\n"
              f"\tb {el.flowing[-1][b_index]} -> {el} -> {el.flowing[1][b_index]}\n"
              f"\tc {el.flowing[-1][c_index]} -> {el} -> {el.flowing[1][c_index]}")
    # """
    return PEs


class PE(mg.Module):
    """
    Processing Element

    For each clock cycle, the computation:
        c_out = c_in + (a_in * b_in)
    furthermore, the values a_in and b_in are forwarded in a_out and b_out
    respectively
    """

    def __init__(self, in_depth, mac_depth):
        # inputs
        self.a_in = mg.Signal(in_depth)
        self.b_in = mg.Signal(in_depth)
        self.c_in = mg.Signal(in_depth)

        # outputs
        self.a_out = mg.Signal(in_depth)
        self.b_out = mg.Signal(in_depth)
        self.c_out = mg.Signal(in_depth)

        ###

        self.sync += [
            self.a_out.eq(self.a_in),
            self.b_out.eq(self.b_in),
            ]

        self.sync += self.c_out.eq(self.a_in * self.b_in + self.c_in)


class Sahdl(mg.Module):

    N1 = 0
    N2 = 0
    N3 = 0

    T = 0

    in_depth = None
    mac_depth = None

    PEs_mapping = None

    def __init__(self,
                 n1: int, n2: int, n3: int,
                 T: np.ndarray,
                 in_depth: int = 8,
                 mac_depth: int = 32,
                 ):

        # IO signals
        self.a_inputs = None # these three will be arrays of signals
        self.b_inputs = None
        self.c_inputs = None

        ###

        #input lists
        inputs = [
            [] for l in utils.LineType
        ]
        # The physical positions (x,y) will be needed for ordering the array
        input_positions = [
            [] for l in utils.LineType
        ]

        self.in_depth = in_depth
        self.mac_depth = mac_depth

        PEs_mapping = get_pe_mapping(n1, n2, n3, T)
        self.PEs_mapping = PEs_mapping

        self.PEs = {p: PE(in_depth, mac_depth) for p in PEs_mapping.keys()}
        PEs = self.PEs

        """
        # TODO it would be possible to use the special migen.Instance instead of just having all the signals in the same
        # module, but in order to do that, it is necessary to instantiate a couple of signals to connect the different 
        # PEs 
        pe = mg.Instance(
            "PE",
            i_a_in = ,
            i_b_in = ,
            i_c_in = ,
            o_a_out = ,
            o_b_out = ,
            o_c_out = ,
        )
        # """

        for p, el in PEs_mapping.items():

            # connections on line a
            p_a = el.flowing[-1][utils.LineType.a.value-1]
            if p_a is not None:
                PEs[p].a_out.eq( PEs[p_a].a_in )
            else:  # if this is None, then the associated signal is an input
                inputs[utils.LineType.a.value-1].append(PEs[p].a_in)
                input_positions[utils.LineType.a.value-1].append(p)

            # connections on line b
            p_b = el.flowing[-1][utils.LineType.b.value-1]
            if p_b is not None:
                PEs[p].b_out.eq( PEs[p_b].b_in )
            else:  # external input
                inputs[utils.LineType.b.value-1].append(PEs[p].b_in)
                input_positions[utils.LineType.b.value - 1].append(p)

            # connections on line c
            p_c = el.flowing[-1][utils.LineType.c.value - 1]
            if p_c is not None:
                PEs[p].c_out.eq(PEs[p_c].c_in)
            else:  # external input
                inputs[utils.LineType.c.value-1].append(PEs[p].c_in)
                input_positions[utils.LineType.c.value - 1].append(p)

            # TODO external outputs


        self.submodules += list( self.PEs.values() )  # signals
        self.a_inputs = mg.Array( inputs[utils.LineType.a.value-1] )
        self.b_inputs = mg.Array( inputs[utils.LineType.b.value-1] )
        self.c_inputs = mg.Array( inputs[utils.LineType.c.value-1] )

        self.io = set([*self.a_inputs, *self.b_inputs, *self.c_inputs])
