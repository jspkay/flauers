class Fault:

    # TODO: implement this shit
    def __init__(self, parameters: dict):
        self.x = parameters["x"]  # This has to be an int!
        self.y = parameters["y"]  # This has to be an int!
        self.t = parameters["t"]  # This should be a string, so that it is possibile to parse it
        # Some examples for the time:
        # ">4" -> means every CC after the 4 (excluded)
        # ">=4" -> every CC after the 4 (included)
        # ">3<=4" -> may mean only for CC 3 and 4
        # ">3<4" -> only for CC 3 (4 is excluded)


class StuckAt(Fault):

    def __init__(self, parameters: dict):
        super().__init__(parameters)
