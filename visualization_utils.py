class systolic_state:
    last_time = 0
    color = "white"
    systolic_time = 0

    def update_time(self, t):
        self.last_time = t
        self.systolic_time += 1

    def invert_color(self):
        self.color = "red" if self.color == "white" else "white"
