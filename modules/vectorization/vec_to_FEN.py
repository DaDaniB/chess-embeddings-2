import math

from modules import constants, bitboard_utils
from modules.piece_utils import PieceUtils


def vec_to_FEN(vec) -> str:
    if len(vec) != constants.VECTOR_LENGTH:
        raise ValueError("given vector has not the right size")

    bitboards = bitboard_utils.get_piece_bitboards_from_vec(vec)
    return piece_bitboards_to_FEN(bitboards)


def vec_with_info_to_FEN(vec):
    piece_vec = vec[0 : constants.VECTOR_LENGTH]
    active_player_vec = vec[constants.VECTOR_LENGTH]
    castling_right_vec = vec[769:773]
    enpassant_vec = vec[773:779]

    piece_FEN = vec_to_FEN(piece_vec)
    active_player_FEN = get_active_color_FEN_from_vec(active_player_vec)
    castling_right_FEN = get_castling_rights_FEN_from_vec(castling_right_vec)
    enpassant_FEN = get_enpassant_FEN_from_vec(enpassant_vec)

    return f"{piece_FEN} {active_player_FEN} {castling_right_FEN} {enpassant_FEN}"


def bitboards_to_fen(bitboards):
    return piece_bitboards_to_FEN(bitboards)


def bitboards_with_info_to_FEN(bitboards):
    piece_bitboards, info_board = bitboard_utils.seperate_info_from_piece_boards(
        bitboards
    )
    piece_FEN = piece_bitboards_to_FEN(piece_bitboards)
    info_board_FEN = info_board_to_FEN(info_board)
    return f"{piece_FEN} {info_board_FEN}"


def piece_bitboards_to_FEN(bitboards):

    FEN_str = ""
    whitespace_counter = 0
    for row in range(constants.BOARD_SIDE_LENGTH):
        for column in range(constants.BOARD_SIDE_LENGTH):
            contained_piece = get_piece_index_at_position(bitboards, row, column)

            if contained_piece == -1:
                whitespace_counter += 1
            else:
                if whitespace_counter != 0:
                    FEN_str += str(whitespace_counter)
                    whitespace_counter = 0
                FEN_str += PieceUtils.piece_index_to_FEN_piece_char(contained_piece)

        if whitespace_counter != 0:
            FEN_str += str(whitespace_counter)
            whitespace_counter = 0
        FEN_str += "/"
    return FEN_str[:-1]  # removing last "/"


def info_board_to_FEN(info_board):
    active_player_vec = info_board[0][0]
    castling_right_vec = info_board[1][:4]
    enpassant_vec = info_board[2][: constants.ENPASSANT_VEC_LENGTH]

    active_player_FEN = get_active_color_FEN_from_vec(active_player_vec)
    castling_right_FEN = get_castling_rights_FEN_from_vec(castling_right_vec)
    enpassant_FEN = get_enpassant_FEN_from_vec(enpassant_vec)

    return f"{active_player_FEN} {castling_right_FEN} {enpassant_FEN}"


def get_piece_index_at_position(bitboards, row, column):

    for index in range(len(bitboards[row][column])):
        if bitboards[row][column][index]:
            return index
    return -1


def get_active_color_FEN_from_vec(vec):
    if vec == False:
        return "w"
    else:
        return "b"


def get_castling_rights_FEN_from_vec(castling_right_vec):
    castling_right_string = ""

    if castling_right_vec[0] == True:
        castling_right_string += "K"

    if castling_right_vec[1] == True:
        castling_right_string += "Q"

    if castling_right_vec[2] == True:
        castling_right_string += "k"

    if castling_right_vec[3] == True:
        castling_right_string += "q"

    return castling_right_string


def get_enpassant_FEN_from_vec(vec):

    enpassant_string = "".join(map(str, vec.astype(int)))
    enpassant_int = int(enpassant_string, 2)

    if enpassant_int == 0:
        return "-"

    enpassant_int -= 1

    enpassant_fen_string = ""
    enpassant_fen_string += chr((enpassant_int % 8) + (ord("a")))
    enpassant_fen_string += str(math.floor(enpassant_int / 8) + 1)

    return enpassant_fen_string
