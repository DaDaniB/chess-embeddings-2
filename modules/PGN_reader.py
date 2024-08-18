import numpy as np
import chess
import chess.pgn
from chess.pgn import Game
import io
import random

from modules.vectorization.FEN import FEN


class PGNReader:

    @staticmethod
    def extrace_unique_positions_from_file(
        PGN_file: str, TXT_file: str, after_position: int, num_positions: int
    ):
        positions_to_skip = set()

        extracted_games = 0
        with open(TXT_file, "w") as output:
            with open(PGN_file, "r") as input:
                while True:

                    game = chess.pgn.read_game(input)
                    if game is None:
                        break

                    positions_of_game = PGNReader.get_FEN_positions_of_game(game)

                    if len(positions_to_skip) < after_position:
                        positions_to_skip.update(positions_of_game)
                    else:
                        for position in positions_of_game:
                            output.write(position.FEN + "\n")
                        extracted_games += len(positions_of_game)

                    if extracted_games > num_positions:
                        break

    @staticmethod
    def extract_random_lines(input_txt, output_txt, num_lines):
        with open(input_txt, "r") as input:
            input_lines = input.readlines()

        if num_lines > len(input_lines):
            raise ValueError(
                "Number of lines to read is greater than lines in inputfile"
            )

        random_input_lines = random.sample(input_lines, num_lines)
        with open(output_txt, "w") as output:
            for line in random_input_lines:
                output.write(line.strip() + "\n")

    @staticmethod
    def extract_unique_games_to_txt(PGN_file: str, TXT_file: str, num_positions):
        read_positions = PGNReader.read_unique_positions_from_file(
            PGN_file, num_positions
        )
        with open(TXT_file, "w") as file:
            for position in read_positions:
                file.write(position.FEN + "\n")

    @staticmethod
    def read_positions_from_txt(TXT_file: str, num_positions: int = None):
        read_positions = set()
        with open(TXT_file, "r") as file:
            for line in file:
                if num_positions is not None and len(read_positions) >= num_positions:
                    break
                line = line.strip()
                read_positions.add(FEN(line))

        return np.array(list(read_positions))

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
    def read_very_unique_positions_from_file(
        PGN_file: str, num_positions: int, from_move: int = None
    ):
        read_positions = set()

        with open(PGN_file) as file:
            while True:

                game = chess.pgn.read_game(file)
                if game is None:
                    break

                position_of_game = PGNReader.get_single_FEN_positions_of_game(game, 0)
                if position_of_game is not None:
                    read_positions.add(position_of_game)

                if len(read_positions) > num_positions:
                    break

        return np.array(list(read_positions)[:num_positions])

    @staticmethod
    def get_single_FEN_positions_of_game(game: Game, from_move: int = 0) -> FEN:

        game_length = sum(1 for move in game.mainline_moves())
        if from_move > game_length:
            return None
        random_position_index = random.randint(from_move, game_length)

        board = game.board()
        for index, move in enumerate(game.mainline_moves()):
            board.push(move)

            if index == random_position_index:
                return FEN(board.fen())

        return None

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
