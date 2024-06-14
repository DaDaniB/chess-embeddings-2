import numpy as np
import chess
import chess.pgn
from chess.pgn import Game
import io

from modules.vectorization.FEN import FEN


class PGNReader:

    @staticmethod
    def read_unique_positions_from_file_fast(
        PGN_file: str, num_positions: int, from_move: int = None, chunk_size=1024 * 1024
    ):
        read_positions = set()
        with open(PGN_file, "r") as file:
            buffer = ""
            while True:
                chunk = file.read(chunk_size)
                if not chunk:
                    break
                buffer += chunk

                # games = buffer.split("\n\n\n")
                splitter = "[Event"
                games = [splitter + text for text in buffer.split(splitter) if text]
                buffer = games.pop()

                for game_text in games:
                    game_io = io.StringIO(game_text)
                    game = chess.pgn.read_game(game_io)
                    if game is not None:
                        positions_of_game = PGNReader.get_FEN_positions_of_game(
                            game, from_move
                        )
                        read_positions.update(positions_of_game)

                if len(read_positions) > num_positions:
                    break
            return np.array(list(read_positions)[:num_positions])

    @staticmethod
    def read_unique_positions_from_file(
        PGN_file: str, num_positions: int, from_move: int = None
    ):
        read_positions = set()

        with open(PGN_file) as file:
            while True:

                game = chess.pgn.read_game(file)
                if game is None:
                    break

                positions_of_game = PGNReader.get_FEN_positions_of_game(game, from_move)
                read_positions.update(positions_of_game)

                if len(read_positions) > num_positions:
                    break

        return np.array(list(read_positions)[:num_positions])

    @staticmethod
    def get_FEN_positions_of_game(game: Game, from_move: int = None) -> list[FEN]:
        positions_of_game = []

        board = game.board()
        for index, move in enumerate(game.mainline_moves()):
            board.push(move)

            if from_move is not None and index < from_move:
                continue

            positions_of_game.append(FEN(board.fen()))

        return np.array(positions_of_game)

    @staticmethod
    def get_games_of_PGN_file(PGN_file, number_of_games=None):
        read_games = []
        with open(PGN_file) as file:
            while True:
                game = chess.pgn.read_game(file)
                if game is None:
                    break

                read_games.append(game)

                if number_of_games is not None and len(read_games) >= number_of_games:
                    break
        return read_games
