import pandas
import numpy as np
import chess

from modules.embedding_test.position_set import PositionSet
from modules.vectorization.FEN import FEN


def get_before_mate_and_mate_sets(
    tactics_file_src: str, tactic: str, num_positions: int
):
    before_mate_positions = []
    mate_positions = []

    df = pandas.read_csv(tactics_file_src)
    for index, row in df.iterrows():

        themes = row["Themes"]
        if themes is None or themes == " " or str(themes) == "nan":
            continue

        if contains_tactic(themes, tactic) and contains_tactic(themes, "mateIn1"):
            before_mate_position = row["FEN"]
            moves = row["Moves"].split(" ")

            enemy_move = moves[0]
            mate_move = moves[1]

            board = chess.Board(before_mate_position)
            board.push_uci(enemy_move)
            board.push_uci(mate_move)

            before_mate_positions.append(FEN(before_mate_position))
            mate_positions.append(FEN(board.fen()))

        if len(before_mate_positions) > num_positions:
            break

    return [
        PositionSet("before mate", before_mate_positions, "red"),
        PositionSet("mate", mate_positions, "green"),
    ]


def contains_tactic(themes: str, tactic: str):
    themes = themes.split(" ")
    for theme in themes:

        if tactic in theme:
            return True

    return False


def convert_tactics_file_to_position_sets(
    tactics_file_src: str, tactics: list[list[str]], num_positions: int
):

    tactics_positions_sets = [[] for _ in range(len(tactics))]
    df = pandas.read_csv(tactics_file_src)

    for index, row in df.iterrows():

        themes = row["Themes"]
        if themes is None or themes == " " or str(themes) == "nan":
            continue

        tactics_category_as_index = get_tactics_category_as_index(tactics, themes)

        if (
            tactics_category_as_index == -1
            or len(tactics_positions_sets[tactics_category_as_index]) >= num_positions
        ):
            continue

        tactics_positions_sets[tactics_category_as_index].append(FEN(row["FEN"]))

        amount_all_positions = 0
        for tactic_position_set in tactics_positions_sets:
            amount_all_positions += len(tactic_position_set)

        if amount_all_positions >= len(tactics) * num_positions:
            break

    position_sets = []
    for index, tactics_positions_set in enumerate(tactics_positions_sets):
        position_sets.append(
            PositionSet(
                name=tactics[index][0],
                FEN_positions=tactics_positions_set,
                color=tactics[index][1],
            )
        )

    return np.array(position_sets)


def get_tactics_category_as_index(tactics, themes):
    themes = themes.split(" ")
    for theme in themes:

        for index, tactics_category in enumerate(tactics):
            category = tactics_category[0]
            if theme == tactics_category[0]:
                return index

    return -1
