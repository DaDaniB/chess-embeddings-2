class FEN:
    def __init__(self, FEN: str):
        splitted_FEN = FEN.split()

        self.FEN: str = FEN
        self.piece_data: str = splitted_FEN[0]
        self.active_color_data: str = splitted_FEN[1]
        self.castling_right_data: str = splitted_FEN[2]
        self.enpassant_square_data: str = splitted_FEN[3]

    def __str__(self):
        return self.FEN

    def __repr__(self):
        return f"'{str(self.FEN)}'"
