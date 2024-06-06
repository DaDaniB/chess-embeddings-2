import numpy as np
import chess
import chess.pgn
from chess.pgn import Game

from modules.vectorization.FEN import FEN


class PGNReader:

    @staticmethod
    def read_unique_positions_from_file(PGN_file: str, num_positions: int):
        read_positions = set()

        with open(PGN_file) as file:
            while True:

                game = chess.pgn.read_game(file)
                if game is None:
                    break

                positions_of_game = PGNReader.get_FEN_positions_of_game(game)
                read_positions.update(positions_of_game)

                if len(read_positions) > num_positions:
                    break

        return np.array(list(read_positions)[:num_positions])

    @staticmethod
    def get_FEN_positions_of_game(game: Game) -> list[FEN]:
        positions_of_game = []

        board = game.board()
        for move in game.mainline_moves():
            board.push(move)
            positions_of_game.append(FEN(board.fen()))

        return np.array(positions_of_game)
