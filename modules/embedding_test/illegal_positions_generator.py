import numpy as np
import random

from modules import constants
from modules.piece_utils import PieceUtils
from modules import bitboard_utils
from modules.FEN_converter import FENConverter
from modules.vectorization import vec_to_FEN
from modules.vectorization.FEN import FEN


class Coordinate:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        if isinstance(other, Coordinate):
            return self.x == other.x and self.y == other.y
        return False


class Illegal_Positions_Generator:

    @staticmethod
    def generate_FENs(amount: int, pieces_in_boundaries=False):
        FENs = []
        for i in range(amount):
            FENs.append(Illegal_Positions_Generator.generate_FEN(pieces_in_boundaries))
        return FENs

    @staticmethod
    def generate_FEN(pieces_in_boundaries=False):
        piece_position_vector = Illegal_Positions_Generator.generate_position(
            pieces_in_boundaries
        )
        random_additional_info = Illegal_Positions_Generator.generate_random_info()

        piece_position_FEN = vec_to_FEN.bitboards_to_fen(piece_position_vector)
        info_FEN = vec_to_FEN.info_vec_to_FEN(random_additional_info)
        return FEN(f"{piece_position_FEN} {info_FEN}")

    @staticmethod
    def generate_position(pieces_in_boundaries=False):
        bitboards = bitboard_utils.init_empty_piece_bitboards()
        bitboards = np.transpose(bitboards, (2, 0, 1))
        occupied_coordinates = [Coordinate]

        available_pieces = [
            ("p", 8),
            ("n", 2),
            ("b", 2),
            ("r", 2),
            ("q", 1),
            ("k", 1),
            ("P", 8),
            ("N", 2),
            ("B", 2),
            ("R", 2),
            ("Q", 1),
            ("K", 1),
        ]

        if pieces_in_boundaries:
            while len(available_pieces) > 0:
                random_piece = Illegal_Positions_Generator.get_random_piece(
                    available_pieces
                )
                available_pieces.remove(random_piece)
                random_piece_index = PieceUtils.FEN_piece_char_to_index(random_piece[0])
                random_bitboard = Illegal_Positions_Generator.generate_random_bitboard(
                    random.randint(0, random_piece[1]), occupied_coordinates
                )
                bitboards[random_piece_index] = random_bitboard
        else:
            piece_distribution = Illegal_Positions_Generator.generate_distribution()
            while len(available_pieces) > 0:
                random_piece = Illegal_Positions_Generator.get_random_piece(
                    available_pieces
                )
                available_pieces.remove(random_piece)
                random_piece_index = PieceUtils.FEN_piece_char_to_index(random_piece[0])
                random_bitboard = Illegal_Positions_Generator.generate_random_bitboard(
                    piece_distribution[random_piece_index], occupied_coordinates
                )
                bitboards[random_piece_index] = random_bitboard

        return np.transpose(bitboards[:-1], (1, 2, 0))

    @staticmethod
    def get_random_piece(available_pieces: list[str]):
        return available_pieces[random.randint(0, len(available_pieces) - 1)]

    @staticmethod
    def generate_distribution():
        sum_of_all_pieces = random.randint(0, 32)
        distribution = np.random.multinomial(
            sum_of_all_pieces,
            np.ones(constants.DIFFERENT_PIECES) / constants.DIFFERENT_PIECES,
        )
        return distribution

    @staticmethod
    def generate_random_bitboard(
        num_pieces: int, occupied_coordinates: list[Coordinate]
    ):
        bit_board = np.zeros(
            (constants.BOARD_SIDE_LENGTH, constants.BOARD_SIDE_LENGTH),
            dtype=constants.DATA_TYPE,
        )

        for i in range(num_pieces):
            coordinate = (
                Illegal_Positions_Generator.generate_random_unoccupied_coordinate(
                    occupied_coordinates
                )
            )
            bit_board[coordinate.x][coordinate.y] = True
            occupied_coordinates.append(coordinate)

        return bit_board

    @staticmethod
    def generate_random_unoccupied_coordinate(
        occupied_coordinates: list[Coordinate],
    ) -> Coordinate:
        MAX_TRIES = 1000
        breaker = 0
        while True:
            coordinate = Illegal_Positions_Generator.generate_random_coordinate()
            if coordinate not in occupied_coordinates:
                return coordinate

            breaker += 1
            if breaker > MAX_TRIES:
                raise ValueError(
                    f"No unoccupied coordinate found after ${MAX_TRIES} tries in {occupied_coordinates}"
                )

    @staticmethod
    def generate_random_coordinate() -> Coordinate:
        x = random.randint(0, constants.BOARD_SIDE_LENGTH - 1)
        y = random.randint(0, constants.BOARD_SIDE_LENGTH - 1)
        return Coordinate(x, y)

    @staticmethod
    def generate_random_info():
        return np.array(
            [
                random.choice([True, False])
                for _ in range(
                    constants.VECTOR_WITH_INFO_LENGTH - constants.VECTOR_LENGTH
                )
            ]
        )
