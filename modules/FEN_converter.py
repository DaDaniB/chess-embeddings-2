from modules.vectorization.FEN import FEN
from modules.vectorization import vec_to_FEN, FEN_to_vec


class FENConverter:

    @staticmethod
    def to_vector(FEN: FEN):
        return FEN_to_vec.FEN_to_vec(FEN)

    @staticmethod
    def to_vector_with_info(FEN: FEN):
        return FEN_to_vec.FEN_to_vec_with_info(FEN)

    @staticmethod
    def to_bitboards(FEN: FEN):
        return FEN_to_vec.FEN_to_bitboards(FEN)

    @staticmethod
    def to_bitboards_with_info(FEN: FEN):
        return FEN_to_vec.FEN_to_bitboards_with_info(FEN)

    @staticmethod
    def vector_to_FEN(vec):
        return vec_to_FEN.vec_to_FEN(vec)

    @staticmethod
    def vector_with_info_to_FEN(vec):
        return vec_to_FEN.vec_with_info_to_FEN(vec)

    @staticmethod
    def bitboards_to_FEN(bitboards):
        return vec_to_FEN.bitboards_to_fen(bitboards)

    @staticmethod
    def bitboards_with_info_to_FEN(bitboards):
        return vec_to_FEN.bitboards_with_info_to_FEN(bitboards)
