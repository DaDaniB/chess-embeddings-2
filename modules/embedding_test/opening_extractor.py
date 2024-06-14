import chess.pgn
from modules.embedding_test.position_set import PositionSet
from modules.vectorization.FEN import FEN
import io


def extract_opening_position(
    PGN_file, openings: list[list[str]], max_moves_after_opening, num_positions
):
    colors = ["red", "green", "blue", "yellow", "cyan", "pink", "grey", "black"]

    openings_positions = []
    for opening in openings:
        openings_positions.append(
            get_opening_positions(
                PGN_file, opening, max_moves_after_opening, num_positions
            )
        )

    openings_positions_sets = []
    for index, opening_positions in enumerate(openings_positions):
        openings_positions_sets.append(
            PositionSet(" ".join(openings[index]), opening_positions, colors[index])
        )
    return openings_positions_sets


def get_opening_positions(
    PGN_file, opening: list[str], max_moves_after_opening, num_positions
):
    unique_opening_positions = set()

    with open(PGN_file) as file:
        while len(unique_opening_positions) < num_positions:

            game = chess.pgn.read_game(file)
            if game is None:
                print(
                    f"no more games to extract openings from, found unique opening positions {len(unique_opening_positions)}"
                )
                break

            positions_after_opening = get_positions_after_opening(
                game, opening, max_moves_after_opening
            )
            if positions_after_opening:
                unique_opening_positions.update(positions_after_opening)

        return unique_opening_positions


def get_opening_positions_fast(
    PGN_file,
    opening: list[str],
    max_moves_after_opening,
    num_positions,
    chunk_size=1024 * 1024 * 10,
):
    unique_opening_positions = set()

    with open(PGN_file) as file:
        buffer = ""
        while len(unique_opening_positions) < num_positions:

            chunk = file.read(chunk_size)
            if not chunk:
                break
            buffer += chunk

            splitter = "[Event"
            games = [splitter + text for text in buffer.split(splitter) if text]
            buffer = games.pop()

            for game_text in games:
                game_io = io.StringIO(game_text)
                game = chess.pgn.read_game(game_io)
                if game is not None:
                    positions_after_opening = get_positions_after_opening(
                        game, opening, max_moves_after_opening
                    )
                    if positions_after_opening:
                        unique_opening_positions.update(positions_after_opening)

        return unique_opening_positions


def get_positions_after_opening(
    game, opening: list[str], max_moves_after_opening
) -> list[FEN]:
    board = game.board()

    for move_number, move in enumerate(game.mainline_moves()):
        if move_number < len(opening):
            if board.san(move) != opening[move_number]:
                return []
        else:
            break

        board.push(move)

    return get_positions_from_to(
        game, len(opening), len(opening) + max_moves_after_opening
    )


def get_positions_from_to(game, from_move, to_move) -> list[FEN]:
    positions: FEN = []
    board = game.board()

    for move_number, move in enumerate(game.mainline_moves()):

        if move_number < from_move:
            board.push(move)
            continue

        if move_number > to_move:
            return positions

        board.push(move)
        positions.append(FEN(board.fen()))
