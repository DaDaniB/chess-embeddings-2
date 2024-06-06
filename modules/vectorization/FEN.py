class FEN:
    def __init__(self, FEN):

        splitted_FEN = FEN.split()
        if len(splitted_FEN) != 6:
            raise ValueError(
                f"Given fen does not have 6 parts (Piece position data, active color data, castling right data, en passant square data), halfmove clock, fullmove number: {FEN}"
            )

        self.piece_data = splitted_FEN[0]
        self.active_color_data = splitted_FEN[1]
        self.castling_right_data = splitted_FEN[2]
        self.enpassant_square_data = splitted_FEN[3]
