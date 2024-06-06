import numpy as np
from modules import constants
from modules.piece_utils import PieceUtils
from modules.vectorization.FEN import FEN


def init_empty_piece_bitboards():
    return np.zeros(
        (
            constants.BOARD_SIDE_LENGTH,
            constants.BOARD_SIDE_LENGTH,
            constants.DIFFERENT_PIECES,
        ),
        dtype=constants.DATA_TYPE,
    )


def get_piece_bitboards_from_vec(vec):
    piece_bitboards = init_empty_piece_bitboards()
    piece_vecs = split_to_piece_vecs(vec)  # 12 * 64 long vectors

    for piece_index, piece_vec in enumerate(piece_vecs):
        for field_index, contains_piece in enumerate(piece_vec):

            row = field_index // constants.BOARD_SIDE_LENGTH
            column = field_index % constants.BOARD_SIDE_LENGTH
            piece_bitboards[row][column][piece_index] = contains_piece

    return piece_bitboards


def split_to_piece_vecs(vec):
    splitted_vecs = []
    for i in range(constants.DIFFERENT_PIECES):
        splitted_vecs.append(
            vec[(constants.BOARD_SQUARES * i) : (constants.BOARD_SQUARES * (i + 1))]
        )

    return splitted_vecs


def get_piece_boards(piece_data_FEN: str):
    piece_boards = init_empty_piece_bitboards()

    column = 0
    row = 0
    for char in piece_data_FEN:
        if char == "/":
            row += 1
            column = 0
            continue

        if char.isnumeric():
            column += int(char)
        else:
            piece_boards[row][column][PieceUtils.FEN_piece_char_to_index(char)] = True
            column += 1

    return piece_boards


def get_info_board(active_color_vec, castling_right_vec, en_passant_square_vec):
    info_board = np.zeros(
        (constants.BOARD_SIDE_LENGTH, constants.BOARD_SIDE_LENGTH),
        dtype=constants.DATA_TYPE,
    )

    info_board[0][: len(active_color_vec)] = active_color_vec
    info_board[1][: len(castling_right_vec)] = castling_right_vec
    info_board[2][: len(en_passant_square_vec)] = en_passant_square_vec

    return info_board


def add_info_to_piece_boards(piece_boards, info_board):
    transposed_piece_boards = np.transpose(piece_boards, (2, 0, 1))
    transposed_piece_boards_with_info = np.append(
        transposed_piece_boards, info_board[np.newaxis, :, :], axis=0
    )

    return np.transpose(transposed_piece_boards_with_info, (1, 2, 0))


def seperate_info_from_piece_boards(bitboards):
    transposed = np.transpose(bitboards, (2, 0, 1))
    info_board = transposed[-1]
    piece_boards = np.transpose(transposed[:-1], (1, 2, 0))
    return (piece_boards, info_board)
