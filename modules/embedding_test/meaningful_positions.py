import random
import copy
import chess
import tensorflow as tf

from modules.PGN_reader import PGNReader
from modules.autoencoder.base_autoencoder import BaseAutoEncoder
from modules.vectorization.FEN import FEN

BASE_POSITION_MOVE_RANGE = (
    20  # the base position is in range of 0 - 20 moves into the game
)


class PositionTriplet:

    def __init__(self, base_position: FEN, played_position: FEN, random_position: FEN):
        self.base_position = base_position
        self.played_position = played_position
        self.random_position = random_position


def compare_played_and_random_moves(
    autoencoder: BaseAutoEncoder,
    PGN_file,
    number_of_positions: int,
    number_of_moves: int,
    output_file,
):

    position_triplets = get_position_triplets(
        PGN_file, number_of_positions, number_of_moves
    )
    compare_postition_distances(autoencoder, position_triplets, output_file)


def get_position_triplets(PGN_file, number_of_positions, number_of_moves):
    position_triplets = []
    games = PGNReader.get_games_of_PGN_file(PGN_file, number_of_positions)

    for game in games:
        position_triplet = get_base_and_played_and_random_position(
            game, number_of_moves
        )
        if position_triplet is None:
            continue

        position_triplets.append(position_triplet)
    return position_triplets


def get_base_and_played_and_random_position(game, number_of_moves):
    base_position_move_number = random.randint(0, BASE_POSITION_MOVE_RANGE)

    base_position = get_position_after_n_moves(game, base_position_move_number)
    if base_position is None:
        return None

    played_position = get_position_after_n_moves(
        game, base_position_move_number + number_of_moves
    )
    if played_position is None:
        return None

    random_position = play_n_random_moves_from_position(base_position, number_of_moves)
    if random_position is None:
        return None

    return PositionTriplet(
        FEN(base_position), FEN(played_position), FEN(random_position)
    )


def get_position_after_n_moves(game, moves_into_game):
    board = game.board()

    for move_number, move in enumerate(game.mainline_moves()):
        board.push(move)
        if move_number == moves_into_game:
            return board.fen()

    return None


def play_n_random_moves_from_position(fen: str, number_of_moves):
    random_moves_board = chess.Board(fen)

    for i in range(number_of_moves):
        possible_moves = list(random_moves_board.legal_moves)
        if possible_moves is None or len(possible_moves) <= 0:
            return None

        move = random.choice(possible_moves)
        random_moves_board.push(move)

    return random_moves_board.fen()


def compare_postition_distances(
    autoencoder: BaseAutoEncoder, position_triplets, output_file
):

    cummulative_dist_with_played_position = 0
    cummulative_dist_with_random_position = 0

    for position_triplet in position_triplets:

        base_position_encoded = autoencoder.encode_FEN_position_tensor(
            position_triplet.base_position
        )
        played_position_encoded = autoencoder.encode_FEN_position_tensor(
            position_triplet.played_position
        )
        random_position_encoded = autoencoder.encode_FEN_position_tensor(
            position_triplet.random_position
        )

        dist_with_played_position = tf.norm(
            base_position_encoded - played_position_encoded, ord="euclidean"
        )
        dist_with_random_position = tf.norm(
            base_position_encoded - random_position_encoded, ord="euclidean"
        )

        cummulative_dist_with_played_position += dist_with_played_position
        cummulative_dist_with_random_position += dist_with_random_position

    avg_dist_with_played_position = cummulative_dist_with_played_position / len(
        position_triplets
    )
    avg_dist_with_random_position = cummulative_dist_with_random_position / len(
        position_triplets
    )

    result_text = f"avg dist with played position: {avg_dist_with_played_position} and avg dist with random position: {avg_dist_with_random_position}"
    print(result_text)
    output_file.write(result_text)
