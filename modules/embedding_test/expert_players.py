import chess
import numpy as np

from modules.PGN_reader import PGNReader
from modules.embedding_test.position_set import PositionSet


def get_expert_player_positions(
    player_a_PGN_file, player_b_PGN_file, num_positions, from_move
):

    player_a_positions = PGNReader.read_unique_positions_from_file(
        player_a_PGN_file, num_positions, from_move
    )
    player_b_positions = PGNReader.read_unique_positions_from_file(
        player_b_PGN_file, num_positions, from_move
    )

    return [
        PositionSet(player_a_PGN_file, player_a_positions, "red"),
        PositionSet(player_b_PGN_file, player_b_positions, "green"),
    ]
