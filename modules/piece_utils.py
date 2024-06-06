class PieceUtils:

    PIECE_MAPPINGS = [
        ("p", 0),
        ("n", 1),
        ("b", 2),
        ("r", 3),
        ("q", 4),
        ("k", 5),
        ("P", 6),
        ("N", 7),
        ("B", 8),
        ("R", 9),
        ("Q", 10),
        ("K", 11),
    ]

    @staticmethod
    def FEN_piece_char_to_index(FEN_piece_char: str) -> int:
        for mapping in PieceUtils.PIECE_MAPPINGS:
            if FEN_piece_char == mapping[0]:
                return mapping[1]

        raise ValueError(f"No mapping was found for FEN_piece_char: ${FEN_piece_char}")

    @staticmethod
    def piece_index_to_FEN_piece_char(index: int) -> str:
        for mapping in PieceUtils.PIECE_MAPPINGS:
            if index == mapping[1]:
                return mapping[0]

        raise ValueError(f"No mapping was found for index: ${index}")
