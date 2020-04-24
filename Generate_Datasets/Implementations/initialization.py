import numpy as np

class dataset_characterisitcs:
    def __init__(self, dataset, a, b):
        if dataset == "zara1":
            self.left_side = {"x_min": 0, "x_max": 0, "y_min": 4, "y_max": 12}
            self.top = {"x_min": 7.5, "x_max": 15, "y_min": 12, "y_max": 12}
            self.right_side = {"x_min": 15, "x_max": 15, "y_min": 6, "y_max": 12}
            self.bottom = {}
            self.total_x_min = min(self.left_side["x_min"], self.top["x_min"], self.right_side["x_min"])
            self.total_x_max = max(self.left_side["x_max"], self.top["x_max"], self.right_side["x_max"])
            self.total_y_min = min(self.left_side["y_min"], self.top["y_min"], self.right_side["y_min"])
            self.total_y_max = max(self.left_side["y_max"], self.top["y_max"], self.right_side["y_max"])

        elif dataset == "zara2":
            self.left_side = {"x_min": 0, "x_max": 0, "y_min": 3.9, "y_max": 11.5}
            self.top = {"x_min": 7.8, "x_max": 14, "y_min": 12, "y_max": 12}
            self.right_side = {"x_min": 14, "x_max": 14, "y_min": 5.5, "y_max": 12}
            self.bottom = {}
            self.total_x_min = min(self.left_side["x_min"], self.top["x_min"], self.right_side["x_min"])
            self.total_x_max = max(self.left_side["x_max"], self.top["x_max"], self.right_side["x_max"])
            self.total_y_min = min(self.left_side["y_min"], self.top["y_min"], self.right_side["y_min"])
            self.total_y_max = max(self.left_side["y_max"], self.top["y_max"], self.right_side["y_max"])

        elif dataset == "univ":
            self.left_side = {"x_min": -2.5, "x_max": -2.5, "y_min": 0, "y_max": 8}
            self.top = {"x_min": 12.5, "x_max": 15, "y_min": 14, "y_max": 14}
            self.right_side = {"x_min": 15, "x_max": 15, "y_min": 0, "y_max": 14}
            self.bottom = {"x_min": 5.0, "x_max": 15, "y_min": 0, "y_max": 0}
            self.total_x_min = min(self.left_side["x_min"], self.top["x_min"], self.right_side["x_min"], self.bottom["x_min"])
            self.total_x_max = max(self.left_side["x_max"], self.top["x_max"], self.right_side["x_max"], self.bottom["x_max"])
            self.total_y_min = min(self.left_side["y_min"], self.top["y_min"], self.right_side["y_min"], self.bottom["y_min"])
            self.total_y_max = max(self.left_side["y_max"], self.top["y_max"], self.right_side["y_max"], self.bottom["y_max"])

        elif dataset == "hotel":
            self.left_side = {"x_min": 0, "x_max": 0, "y_min": 6, "y_max": 12}
            self.top = {"x_min": 0, "x_max": 8, "y_min": 13, "y_max": 13}
            self.right_side = {"x_min": 14, "x_max": 14, "y_min": 6, "y_max": 12}
            self.bottom = {}
            self.total_x_min = min(self.left_side["x_min"], self.top["x_min"], self.right_side["x_min"])
            self.total_x_max = max(self.left_side["x_max"], self.top["x_max"], self.right_side["x_max"])
            self.total_y_min = min(self.left_side["y_min"], self.top["y_min"], self.right_side["y_min"])
            self.total_y_max = max(self.left_side["y_max"], self.top["y_max"], self.right_side["y_max"])

        elif dataset == "eth":
            self.left_side = {"x_min": 0, "x_max": 0, "y_min": 22, "y_max": 25}
            self.right_side = {"x_min": 22, "x_max": 22, "y_min": 22, "y_max": 25}
            self.top = {"x_min": 4.5, "x_max": 15, "y_min": 25, "y_max": 25}
            self.bottom = {"x_min": 4.5, "x_max": 15, "y_min": 0, "y_max": 0}
            self.total_x_min = min(self.bottom["x_min"], self.top["x_min"], self.left_side["x_min"], self.right_side["x_min"])
            self.total_x_max = max(self.bottom["x_max"], self.top["x_max"], self.left_side["x_max"], self.right_side["x_max"])
            self.total_y_min = min(self.bottom["y_min"], self.top["y_min"], self.left_side["y_min"], self.right_side["y_min"])
            self.total_y_max = max(self.bottom["y_max"], self.top["y_max"], self.left_side["y_max"], self.right_side["y_max"])

        elif dataset == "square":
            self.left_side = {"x_min": -a/2, "x_max": -a/2, "y_min": -a/2, "y_max": a/2}
            self.top = {"x_min": -a/2, "x_max": a/2, "y_min": a/2, "y_max": a/2}
            self.right_side = {"x_min": a/2, "x_max": a/2, "y_min": -a/2, "y_max": a/2}
            self.bottom = {"x_min": -a/2, "x_max": a/2, "y_min": -a/2, "y_max": -a/2}
            self.total_x_min = min(self.left_side["x_min"], self.top["x_min"], self.right_side["x_min"],self.bottom["x_min"])
            self.total_x_max = max(self.left_side["x_max"], self.top["x_max"], self.right_side["x_max"],self.bottom["x_max"])
            self.total_y_min = min(self.left_side["y_min"], self.top["y_min"], self.right_side["y_min"],self.bottom["y_min"])
            self.total_y_max = max(self.left_side["y_max"], self.top["y_max"], self.right_side["y_max"],self.bottom["y_max"])

        elif dataset == "rectangle":
            self.left_side = {"x_min": -a / 2, "x_max": -a / 2, "y_min": -b / 2, "y_max": b / 2}
            self.top = {"x_min": -a / 2, "x_max": a / 2, "y_min": b / 2, "y_max": b / 2}
            self.right_side = {"x_min": a / 2, "x_max": a / 2, "y_min": -b / 2, "y_max": b / 2}
            self.bottom = {"x_min": -a / 2, "x_max": a / 2, "y_min": -b / 2, "y_max": -b / 2}
            self.total_x_min = min(self.left_side["x_min"], self.top["x_min"], self.right_side["x_min"],self.bottom["x_min"])
            self.total_x_max = max(self.left_side["x_max"], self.top["x_max"], self.right_side["x_max"],self.bottom["x_max"])
            self.total_y_min = min(self.left_side["y_min"], self.top["y_min"], self.right_side["y_min"],self.bottom["y_min"])
            self.total_y_max = max(self.left_side["y_max"], self.top["y_max"], self.right_side["y_max"],self.bottom["y_max"])

        else:
            raise ValueError("Please choose valid dataset. Given dataset: " + str(dataset) + " not known!")

class random_initialization(dataset_characterisitcs):
    def __init__(self, dataset, v_max, v_min, a, b, tau):
        self.v_max = v_max
        self.v_min = v_min
        self.tau = tau
        super(random_initialization,self).__init__(dataset, a, b)

    def choose_starting_points(self, side):
        # Left Side
        if side == 1:
            x = np.random.uniform(self.left_side["x_min"], self.left_side["x_max"])
            y = np.random.uniform(self.left_side["y_min"], self.left_side["y_max"])
        # Bottom
        elif side == 2:
            x = np.random.uniform(self.bottom["x_min"], self.bottom["x_max"])
            y = np.random.uniform(self.bottom["y_min"], self.bottom["y_max"])
        # Right Side
        elif side == 3:
            x = np.random.uniform(self.right_side["x_min"], self.right_side["x_max"])
            y = np.random.uniform(self.right_side["y_min"], self.right_side["y_max"])
        # Top
        elif side == 4:
            x = np.random.uniform(self.top["x_min"], self.top["x_max"])
            y = np.random.uniform(self.top["y_min"], self.top["y_max"])
        else:
            raise ValueError("Invalid number for sides!")

        return x, y

    def get_initial_vel(self, side):
        # Left Side
        if side == 1:
            v_x = np.random.uniform(self.v_min, self.v_max)
            v_y = np.random.uniform(-self.v_max, self.v_max)
        # Bottom
        elif side == 2:
            v_x = np.random.uniform(-self.v_max, self.v_max)
            v_y = np.random.uniform(self.v_min, self.v_max)
        # Right Side
        elif side == 3:
            v_x = np.random.uniform(-self.v_max, -self.v_min)
            v_y = np.random.uniform(-self.v_max, self.v_max)
        # Top
        elif side == 4:
            v_x = np.random.uniform(-self.v_max, self.v_max)
            v_y = np.random.uniform(-self.v_max, -self.v_min)
        else:
            raise ValueError("Invalid number for sides!")

        return v_x, v_y

    def initialize_state(self, i):
        # select input & output side
        sides = []
        if not not self.left_side:
            sides.append(1)
        if not not self.bottom:
            sides.append(2)
        if not not self.right_side:
            sides.append(3)
        if not not self.top:
            sides.append(4)

        sides = np.random.permutation(sides)
        input = sides[0]
        output = sides[1]

        # defining sides of square
        x, y = self.choose_starting_points(input)
        #v_x, v_y = self.get_initial_vel(input)
        d_x, d_y = self.choose_starting_points(output)

        # get initial velocity in direction of destination
        distance = np.array([d_x, d_y]) - np.array([x, y])
        v =  np.random.uniform(self.v_min, self.v_max) * (distance/np.linalg.norm(distance))
        v_x = v[0]
        v_y = v[1]

        return np.array([[x, y, v_x, v_y, d_x, d_y, self.tau]])

class meeting(dataset_characterisitcs):
    def __init__(self, dataset, v_max, v_min, a, b, tau):
        self.v_max = v_max
        self.v_min = v_min
        self.tau = tau
        super(meeting,self).__init__(dataset, a, b)

    def choose_starting_points(self, agent):
        # Left Side
        if agent % 4 == 1:
            if self.left_side["x_max"] != self.left_side["x_min"]:
                x = (self.left_side["x_max"] + self.left_side["x_min"])/2
            else:
                x = self.left_side["x_max"]
            if self.left_side["y_max"] != self.left_side["y_min"]:
                y = (self.left_side["y_max"] + self.left_side["y_min"])/2
            else:
                y = self.left_side["y_max"]
        # Right Side
        elif agent % 4 == 2:
            if self.right_side["x_max"] != self.right_side["x_min"]:
                x = (self.right_side["x_max"] + self.right_side["x_min"])/2
            else:
                x = self.right_side["x_max"]
            if self.right_side["y_max"] != self.right_side["y_min"]:
                y = (self.right_side["y_max"] + self.right_side["y_min"])/2
            else:
                y = self.right_side["y_max"]
        # Top
        elif agent % 4 == 3:
            if self.top["x_max"] != self.top["x_min"]:
                x = (self.top["x_max"] + self.top["x_min"])/2
            else:
                x = self.top["x_max"]
            if self.top["y_max"] != self.top["y_min"]:
                y = (self.top["y_max"] + self.top["y_min"])/2
            else:
                y = self.top["y_max"]
        # Bottom
        elif agent % 4 == 0:
            if self.bottom["x_max"] != self.bottom["x_min"]:
                x = (self.bottom["x_max"] + self.bottom["x_min"])/2
            else:
                x = self.bottom["x_max"]
            if self.bottom["y_max"] != self.bottom["y_min"]:
                y = (self.bottom["y_max"] + self.bottom["y_min"])/2
            else:
                y = self.bottom["y_max"]
        else:
            raise ValueError("Invalid number for sides!")

        return x, y

    def get_initial_vel(self, agent):
        # Left Side
        if agent % 4 == 1:
            v_x = 0.8       # np.random.uniform(self.v_min, self.v_max)
            v_y = 0.001
        # Right Side
        elif agent % 4 == 2:
            v_x = 0.8       # np.random.uniform(-self.v_max, -self.v_min)
            v_y = 0.001
        # Top
        elif agent % 4 == 3:
            v_x = 0.001
            v_y = 0.8       # np.random.uniform(-self.v_max, -self.v_min)
        # Bottom
        elif agent % 4 == 0:
            v_x = 0.001
            v_y = 0.8       # np.random.uniform(self.v_min, self.v_max)
        else:
            raise ValueError("Invalid number for sides!")

        return v_x, v_y

    def initialize_state(self, agent):
        # select input & output side
        input = agent

        if agent % 4 == 1:
            output = 2
        elif agent % 4 == 2:
            output = 1
        elif agent % 4 == 3:
            output = 0
        elif agent % 4 == 0:
            output = 3
        else:
            raise ValueError("Invalid number for sides!")

        # defining sides of square
        x, y = self.choose_starting_points(input)
        v_x, v_y = self.get_initial_vel(input)
        d_x, d_y = self.choose_starting_points(output)

        if agent % 4 == 0:
            y += 0          #0.2
            d_y += 0        # 0.2
        elif agent % 4 == 1:
            y -= 0          # 0.2
            d_y -= 0        # 0.2
        elif agent % 4 == 2:
            x += 0          #0.2
            d_x += 0        #0.2
        elif agent % 4 == 3:
           x -= 0           #0.2
           d_x -= 0         #0.2
        else:
            raise ValueError("Invalid number for sides!")

        return np.array([[x, y, v_x, v_y, d_x, d_y, self.tau]])