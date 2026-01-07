import kociemba


class CubeSolver:
    def __init__(self):
        pass

    def solve(self, cube_string):
        try:
            solution = kociemba.solve(cube_string)
            return solution.split(), None
        except Exception as e:
            return None, str(e)
