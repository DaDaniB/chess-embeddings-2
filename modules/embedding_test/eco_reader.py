import csv
import chess
import numpy as np
from modules.embedding_test.position_set import PositionSet
from modules.vectorization.FEN import FEN


def convert_eco_file_to_position_sets(eco_file_src: str):
    eco_positions = [
        ["A", "red"],
        ["B", "green"],
        ["C", "blue"],
        ["D", "yellow"],
        ["E", "pink"],
    ]
    eco_positions_sets = [[] for _ in range(5)]
    illegal_positions = 0

    with open(eco_file_src, "r") as eco_csv:

        reader = csv.reader(eco_csv)

        next(reader)  # skip header row
        next(reader)

        for row in reader:
            eco_category_as_index = get_eco_category_as_index(row[0])

            try:
                position = get_position_after_moves(row[2])
                eco_positions_sets[eco_category_as_index].append(position)
            except Exception as data_error:
                illegal_positions += 1

    print("illegal positions: " + str(illegal_positions))
    position_sets = []
    for index, eco_position_set in enumerate(eco_positions_sets):
        position_sets.append(
            PositionSet(
                name=eco_positions[index][0],
                FEN_positions=eco_position_set,
                color=eco_positions[index][1],
            )
        )

    return np.array(position_sets)


def get_eco_category_as_index(eco_category):
    eco_start_char = eco_category[0]
    return ord(eco_start_char) - ord("A")


def get_position_after_moves(moves) -> FEN:
    moves = moves.split()

    board = chess.Board()
    for move in moves:
        board.push_san(move)

    return FEN(board.fen())
