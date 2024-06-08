from modules.vectorization.FEN import FEN


class PositionSet:

    def __init__(self, name: str, FEN_positions: list[FEN], color: str = "black"):
        self.name = name
        self.FEN_positions = FEN_positions
        self.color = color
