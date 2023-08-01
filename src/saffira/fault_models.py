import logging
import numpy as np

from .utils import LineType


class Fault:
    # Intrinsic parameters of a fault
    x = None
    y = None
    t_start = None
    t_stop = None

    # Iteration vectors bounds for actual injection. Computed in self.transform
    iteration_start = [None]*3
    iteration_stop = [None]*3

    def _parse_time(self, time_string) -> tuple[int, int]:
        result = [0, 0]

        if time_string == "inf":
            result[0] = 0
            result[1] = 1000000

        return result[0], result[1]

    # TODO: implement this shit
    def __init__(self,
                 line: LineType,
                 x: int,
                 y: int,
                 time: str,
                 bit: int,
                 polarity: int,
                 msb: str = "first",
                 mode: str = "input"):
        """
        This class defines a fault occuring in the systolic array!

        Parameters
        ----------
        line -> which line to inject between a, b and c
        x -> processing element x coordinate
        y -> processing element y coordinate
        time -> string describing the time
        bit -> position of the bit to inject
        polarity -> 1 or 0, value of the new bit
        msb -> string between "first" or "last": "first" means that the msb has index 0
        mode -> string between "input" and "output": determines whether to inject the input or the output of
                a specific PE
        """

        logging.debug(f"[Fault] initializing a new Fault")

        logging.debug(f"[Fault] injection line is {line}")
        self.line = line

        self.x = x  # This has to be an int!
        self.y = y  # This has to be an int!

        # This should be a string, so that it is possible to parse it
        self.t_start, self.t_stop = self._parse_time(time)
        logging.debug(f"[Fault] start_time: {self.t_start}, stop_time: {self.t_stop}")
        # Some examples for the time:
        # ">4" -> means every CC after the 4 (excluded)
        # ">=4" -> every CC after the 4 (included)
        # ">3<=4" -> may mean only for CC 3 and 4
        # ">3<4" -> only for CC 3 (4 is excluded)
        # "inf" -> for the whole time!

        self.bit = bit
        self.polarity = polarity

        assert msb == "first" or msb == "last", f'msb can be either "first" or "last"! Value  {msb} not valid!'
        self.should_reverse_bits = msb == "first"

        assert mode == "input" or mode == "output", f'mode can be either "input" or "output"! Value  {mode} not valid!'
        self.mode = mode

        logging.debug(f"[Fault] bit: {bit}, polarity: {polarity}, msb: {msb}")

    def transform(self, transformation_matrix):
        logging.debug(f"[Fault] starting transformation of fault")
        t_inv = np.linalg.inv(transformation_matrix)  # Compute the inverse transformation
        s_start = np.array([self.x, self.y, self.t_start])  # start time
        s_stop = np.array([self.x, self.y, self.t_stop])  # until stop time

        # In general, we need only two nu vectors, to take into account every time t such that t_start < t < t_stop
        logging.debug(f"[Fault] transforming from s_start {s_start}")
        nu = t_inv @ s_start
        for i in range(3):
            self.iteration_start[i] = int(nu[i])
        if self.mode == "input": # TODO Check whethter self.mode=="output" is correctly handled
            self.iteration_start[self.line.value-1] -= 1
        logging.debug(f"[Fault] the transformed value is {self.iteration_start}")

        logging.debug(f"[Fault] transforming from s_stop {s_stop}")
        nu = t_inv @ s_stop
        for i in range(3):
            self.iteration_stop[i] = int(nu[i])
        if self.mode == "input":
            self.iteration_stop[self.line.value-1] -= 1
        logging.debug(f"[Fault] the transformed value is {self.iteration_stop}")

        logging.debug(f"[TransformedFault] t_inv: \n{t_inv}")

        logging.debug(
            f"[Fault] The injection of PE {self.x, self.y} between times {self.t_start} and {self.t_stop} "
            f"will happen between {self.iteration_start} and {self.iteration_stop}")

    def __repr__(self):
        str = f"Fault @ {self.line} PE{self.x, self.y} - t:[{self.t_start}-{self.t_stop}]/"\
            f"bit {self.bit} -> {self.polarity}"


class StuckAt(Fault):

    def __init__(self,
                 line: str,
                 *args,
                 **kwargs):
        l = None
        if line == "a":
            l = LineType.a
        elif line == "b":
            l = LineType.b
        elif line == "c":
            l = LineType.c
        else:
            raise Exception(f"line {line} not available. Please use either 'a', 'b', or 'c'.")

        """ kwargs = {
            "line": l,
            "x": kwargs["x"],
            "y": y,
            "time": "inf",
            "bit": bit,
            "polarity": polarity
        } """

        logging.debug(f"[StuckAt] Initializing stuckAt fault on line {line}")

        kwargs["line"] = l
        kwargs["time"] = "inf"
        super().__init__(*args, **kwargs)

    def __repr__(self):
        str = f"StuckAt @ {self.line} PE{self.x, self.y} / "\
            f"bit {self.bit} -> {self.polarity}"

        return str