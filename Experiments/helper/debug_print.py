class debug_print():
    def __init__(self, debug):
        self.debug = debug

    def print(self, string):
        if self.debug:
            print(string)