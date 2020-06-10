import numpy as np

# ============================================= Description =============================================
# Classes that define the initial states of agents in the dataset.
# The agents either enter the scenario randomly at one side of the environment and leave it at another randomly
# selected side (class random_initialization) or the agents meet at the center of the scenario (class meeting)
# =======================================================================================================

class dataset_characterisitcs:
    """
    Defines the spatial configurations for different scenarios - x_min & x_max/y_min & y_max determine the entry and exit areas of the agents for the respective sides of the scenario
    """
    def __init__(self, scenario, a, b):
        if scenario == "zara1":
            self.left_side = {"x_min": 0, "x_max": 0, "y_min": 4, "y_max": 12}
            self.top = {"x_min": 7.5, "x_max": 15, "y_min": 12, "y_max": 12}
            self.right_side = {"x_min": 15, "x_max": 15, "y_min": 6, "y_max": 12}
            self.bottom = {}
            self.total_x_min = min(self.left_side["x_min"], self.top["x_min"], self.right_side["x_min"])
            self.total_x_max = max(self.left_side["x_max"], self.top["x_max"], self.right_side["x_max"])
            self.total_y_min = min(self.left_side["y_min"], self.top["y_min"], self.right_side["y_min"])
            self.total_y_max = max(self.left_side["y_max"], self.top["y_max"], self.right_side["y_max"])

        elif scenario == "zara2":
            self.left_side = {"x_min": 0, "x_max": 0, "y_min": 3.9, "y_max": 11.5}
            self.top = {"x_min": 7.8, "x_max": 14, "y_min": 12, "y_max": 12}
            self.right_side = {"x_min": 14, "x_max": 14, "y_min": 5.5, "y_max": 12}
            self.bottom = {}
            self.total_x_min = min(self.left_side["x_min"], self.top["x_min"], self.right_side["x_min"])
            self.total_x_max = max(self.left_side["x_max"], self.top["x_max"], self.right_side["x_max"])
            self.total_y_min = min(self.left_side["y_min"], self.top["y_min"], self.right_side["y_min"])
            self.total_y_max = max(self.left_side["y_max"], self.top["y_max"], self.right_side["y_max"])

        elif scenario == "univ":
            self.left_side = {"x_min": -2.5, "x_max": -2.5, "y_min": 0, "y_max": 8}
            self.top = {"x_min": 12.5, "x_max": 15, "y_min": 14, "y_max": 14}
            self.right_side = {"x_min": 15, "x_max": 15, "y_min": 0, "y_max": 14}
            self.bottom = {"x_min": 5.0, "x_max": 15, "y_min": 0, "y_max": 0}
            self.total_x_min = min(self.left_side["x_min"], self.top["x_min"], self.right_side["x_min"], self.bottom["x_min"])
            self.total_x_max = max(self.left_side["x_max"], self.top["x_max"], self.right_side["x_max"], self.bottom["x_max"])
            self.total_y_min = min(self.left_side["y_min"], self.top["y_min"], self.right_side["y_min"], self.bottom["y_min"])
            self.total_y_max = max(self.left_side["y_max"], self.top["y_max"], self.right_side["y_max"], self.bottom["y_max"])

        elif scenario == "hotel":
            self.left_side = {"x_min": 0, "x_max": 0, "y_min": 6, "y_max": 12}
            self.top = {"x_min": 0, "x_max": 8, "y_min": 13, "y_max": 13}
            self.right_side = {"x_min": 14, "x_max": 14, "y_min": 6, "y_max": 12}
            self.bottom = {}
            self.total_x_min = min(self.left_side["x_min"], self.top["x_min"], self.right_side["x_min"])
            self.total_x_max = max(self.left_side["x_max"], self.top["x_max"], self.right_side["x_max"])
            self.total_y_min = min(self.left_side["y_min"], self.top["y_min"], self.right_side["y_min"])
            self.total_y_max = max(self.left_side["y_max"], self.top["y_max"], self.right_side["y_max"])

        elif scenario == "eth":
            self.left_side = {"x_min": 0, "x_max": 0, "y_min": 22, "y_max": 25}
            self.right_side = {"x_min": 22, "x_max": 22, "y_min": 22, "y_max": 25}
            self.top = {"x_min": 4.5, "x_max": 15, "y_min": 25, "y_max": 25}
            self.bottom = {"x_min": 4.5, "x_max": 15, "y_min": 0, "y_max": 0}
            self.total_x_min = min(self.bottom["x_min"], self.top["x_min"], self.left_side["x_min"], self.right_side["x_min"])
            self.total_x_max = max(self.bottom["x_max"], self.top["x_max"], self.left_side["x_max"], self.right_side["x_max"])
            self.total_y_min = min(self.bottom["y_min"], self.top["y_min"], self.left_side["y_min"], self.right_side["y_min"])
            self.total_y_max = max(self.bottom["y_max"], self.top["y_max"], self.left_side["y_max"], self.right_side["y_max"])

        elif scenario == "square":
            self.left_side = {"x_min": -a/2, "x_max": -a/2, "y_min": -a/2, "y_max": a/2}
            self.top = {"x_min": -a/2, "x_max": a/2, "y_min": a/2, "y_max": a/2}
            self.right_side = {"x_min": a/2, "x_max": a/2, "y_min": -a/2, "y_max": a/2}
            self.bottom = {"x_min": -a/2, "x_max": a/2, "y_min": -a/2, "y_max": -a/2}
            self.total_x_min = min(self.left_side["x_min"], self.top["x_min"], self.right_side["x_min"],self.bottom["x_min"])
            self.total_x_max = max(self.left_side["x_max"], self.top["x_max"], self.right_side["x_max"],self.bottom["x_max"])
            self.total_y_min = min(self.left_side["y_min"], self.top["y_min"], self.right_side["y_min"],self.bottom["y_min"])
            self.total_y_max = max(self.left_side["y_max"], self.top["y_max"], self.right_side["y_max"],self.bottom["y_max"])

        elif scenario == "rectangle":
            self.left_side = {"x_min": -a / 2, "x_max": -a / 2, "y_min": -b / 2, "y_max": b / 2}
            self.top = {"x_min": -a / 2, "x_max": a / 2, "y_min": b / 2, "y_max": b / 2}
            self.right_side = {"x_min": a / 2, "x_max": a / 2, "y_min": -b / 2, "y_max": b / 2}
            self.bottom = {"x_min": -a / 2, "x_max": a / 2, "y_min": -b / 2, "y_max": -b / 2}
            self.total_x_min = min(self.left_side["x_min"], self.top["x_min"], self.right_side["x_min"],self.bottom["x_min"])
            self.total_x_max = max(self.left_side["x_max"], self.top["x_max"], self.right_side["x_max"],self.bottom["x_max"])
            self.total_y_min = min(self.left_side["y_min"], self.top["y_min"], self.right_side["y_min"],self.bottom["y_min"])
            self.total_y_max = max(self.left_side["y_max"], self.top["y_max"], self.right_side["y_max"],self.bottom["y_max"])

        else:
            raise ValueError("Please choose valid dataset. Given dataset: " + str(scenario) + " not known!\nPlease choose either zara1, zara2, univ, hotel, eth, square or rectangle!")


class random_initialization(dataset_characterisitcs):
    """
    Agents enter the scenario at one randomly selected side and leave it at another randomly selected side.
    The location where they enter and leave the scenario on the respective sides is randomly defined according to a uniform distribution.
    The direction of the initial velocity is chosen according to the direction of the goal of each agent. The magnitude is randomly defined using
    the uniform distribution U[v_min, v_max].
    """
    def __init__(self, scenario, v_max, v_min, a, b, tau):
        self.v_max = v_max
        self.v_min = v_min
        self.tau = tau
        super(random_initialization,self).__init__(scenario, a, b)


    def choose_starting_points(self, side):
        """
        Define specific location on side
        """
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


    def initialize_state(self, i=0):
        """
        Define the initial state of an agent
        :return: initial state of agent
        """
        # select entry & exit side
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

        # Specifying particular entry and exit location for the respective sides
        x, y = self.choose_starting_points(input)
        d_x, d_y = self.choose_starting_points(output)

        # get initial velocity in direction of destination
        distance = np.array([d_x, d_y]) - np.array([x, y])
        v =  np.random.uniform(self.v_min, self.v_max) * (distance/np.linalg.norm(distance))
        v_x = v[0]
        v_y = v[1]

        return np.array([[x, y, v_x, v_y, d_x, d_y, self.tau]])


class meeting(dataset_characterisitcs):
    """
    This class represents a scenario in which agents meet at the center of a square.
    """
    def __init__(self, scenario, v_max, v_min, a, b, tau):
        self.v_max = v_max
        self.v_min = v_min
        self.tau = tau
        super(meeting,self).__init__(scenario, a, b)

    def choose_starting_points(self, agent):
        """
        Define specific location on entry/exit side.
        """
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


    def initialize_state(self, agent):
        """
        Define the initial state of an agent for the meeting scenario
        :return: initial state of agent
        """
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
        d_x, d_y = self.choose_starting_points(output)

        # get initial velocity in direction of destination
        distance = np.array([d_x, d_y]) - np.array([x, y])
        v =  0.8 * (distance/np.linalg.norm(distance))
        v_x = v[0]
        v_y = v[1]

        # Define small offset >= 0 for location on sides such that the agents do not exactly crash into each other
        offset = 0

        if agent % 4 == 0:
            y += offset
            d_y += offset
        elif agent % 4 == 1:
            y -= offset
            d_y -= offset
        elif agent % 4 == 2:
            x += offset
            d_x += offset
        elif agent % 4 == 3:
           x -= offset
           d_x -= offset
        else:
            raise ValueError("Invalid number for sides!")

        return np.array([[x, y, v_x, v_y, d_x, d_y, self.tau]])