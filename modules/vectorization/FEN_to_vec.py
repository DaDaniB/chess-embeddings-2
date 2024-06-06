import numpy as np

from modules.vectorization.FEN import FEN
from modules import bitboard_utils, constants


def FEN_to_vec(FEN: FEN):
    piece_boards = bitboard_utils.get_piece_boards(FEN.piece_data)
    return piece_boards_to_vector(piece_boards)


def FEN_to_vec_with_info(FEN: FEN):
    piece_boards = bitboard_utils.get_piece_boards(FEN.piece_data)
    piece_vector = piece_boards_to_vector(piece_boards)

    active_color_vec = FEN_to_active_color_vec(FEN)
    castling_right_vec = FEN_to_castling_right_vec(FEN)
    enpassant_vec = FEN_to_enpassant_square_vec(FEN)

    return np.concatenate(
        (piece_vector, active_color_vec, castling_right_vec, enpassant_vec)
    )


def FEN_to_bitboards(FEN: FEN):
    return bitboard_utils.get_piece_boards(FEN.piece_data)


def FEN_to_bitboards_with_info(FEN: FEN):
    piece_boards = bitboard_utils.get_piece_boards(FEN.piece_data)
    info_board = bitboard_utils.get_info_board(
        FEN_to_active_color_vec(FEN),
        FEN_to_castling_right_vec(FEN),
        FEN_to_enpassant_square_vec(FEN),
    )
    return bitboard_utils.add_info_to_piece_boards(piece_boards, info_board)


def piece_boards_to_vector(bit_boards):
    """
    takes the 8x8x12 array and transforms it to a 1d vector of length 768 (64 * black pawn ..... 64 * white king)
    """
    return np.array(bit_boards).transpose(2, 0, 1).flatten()


def FEN_to_active_color_vec(FEN: FEN):
    active_color_FEN = FEN.active_color_data
    if active_color_FEN == "w":
        return [False]
    elif active_color_FEN == "b":
        return [True]
    else:
        raise ValueError("active color vec wrong ")


def FEN_to_castling_right_vec(FEN: FEN):
    castling_right_FEN = FEN.castling_right_data
    castling_right_vec = []

    if "K" in castling_right_FEN:
        castling_right_vec.append(True)
    else:
        castling_right_vec.append(False)

    if "Q" in castling_right_FEN:
        castling_right_vec.append(True)
    else:
        castling_right_vec.append(False)

    if "k" in castling_right_FEN:
        castling_right_vec.append(True)
    else:
        castling_right_vec.append(False)

    if "q" in castling_right_FEN:
        castling_right_vec.append(True)
    else:
        castling_right_vec.append(False)

    return castling_right_vec


def FEN_to_enpassant_square_vec(FEN: FEN):
    enpassant_FEN = FEN.enpassant_square_data
    if enpassant_FEN == "-":
        return np.zeros(constants.ENPASSANT_VEC_LENGTH)

    if len(enpassant_FEN) != 2:
        raise ValueError("enPassant FEN is wrong size")

    char_coord_part = enpassant_FEN[0]
    num_coord_part = enpassant_FEN[1]

    char_coord = ord(char_coord_part[0]) - ord("a") + 1
    if char_coord < 1 or char_coord > 8:
        raise ValueError("enpassant FEN is out of bounds")

    num_coord = int(num_coord_part)
    enpassant_coord = (num_coord - 1) * 8 + char_coord

    enpassant_coord_bit_string = "{0:06b}".format(enpassant_coord)
    enpassant_coord_bit_vec = np.array(
        list(map(int, enpassant_coord_bit_string)), dtype=constants.DATA_TYPE
    )

    return enpassant_coord_bit_vec
