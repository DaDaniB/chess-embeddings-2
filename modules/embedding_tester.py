import time

from modules.autoencoder.base_autoencoder import BaseAutoEncoder
from modules.embedding_test.illegal_positions_generator import (
    Illegal_Positions_Generator,
)
from modules.PGN_reader import PGNReader
from modules.embedding_test.position_set import PositionSet
from modules.embedding_test.position_distance_calculator import (
    PositionDistanceCalculator,
)
from modules.embedding_test import (
    eco_reader,
    meaningful_positions,
    opening_extractor,
    tactics_reader,
    expert_players,
)
from modules.visualization.TSNE_visualizer import TSNEVisualizer

TEST_AMOUNT = 5000
VISUALIZATION_AMOUNT = 1000


TEST_END_TEXT = "######################################\n\n"


class EmbeddingTester:

    def __init__(self, autoencoder: BaseAutoEncoder, PGN_file: str):
        self.autoencoder = autoencoder
        self.PGN_file = PGN_file
        self.test_amount = TEST_AMOUNT

    def test_all(self):
        results_file_name = "output_" + self.autoencoder.__class__.__name__ + ".txt"
        with open(results_file_name, "w") as output_file:

            self.wrap_test_with_print(
                self.test_legal_vs_illegal_positions,
                "######## test legal vs illegal #############\n",
                output_file,
            )

            self.wrap_test_with_print(
                self.test_meaningful_positions,
                "######## test meaningful positions #########\n",
                output_file,
            )

            self.wrap_test_with_print(
                self.test_opening_positions,
                "######## test opening positions ############\n",
                output_file,
            )

            self.wrap_test_with_print(
                self.test_eco,
                "######## test eco ##########################\n",
                output_file,
            )

            self.wrap_test_with_print(
                self.test_tactics,
                "######## test tactics ######################\n",
                output_file,
            )

            self.wrap_test_with_print(
                self.test_before_mate_and_mate,
                "######## test mate #########################\n",
                output_file,
            )

            self.wrap_test_with_print(
                self.test_random_and_mates,
                "######## test mate #########################\n",
                output_file,
            )

            self.wrap_test_with_print(
                self.test_expert_players,
                "######## expert players ####################\n",
                output_file,
            )

    def test_legal_vs_illegal_positions(self, output_file):

        illegal_position_FENs = Illegal_Positions_Generator.generate_FENs(
            self.test_amount, True
        )

        ##### test
        white_king_count = 0
        for fen in illegal_position_FENs:
            if "K" in fen.piece_data:
                white_king_count += 1
        print(f"counted white king: {white_king_count} times")

        legal_position_FENs = PGNReader.read_very_unique_positions_from_file(
            self.PGN_file, self.test_amount
        )

        position_sets = [
            PositionSet("illegal random positions", illegal_position_FENs, "red"),
            PositionSet("legal positions", legal_position_FENs, "green"),
        ]
        self.compare_sets(position_sets, "legal vs illegal test", output_file)

    def test_meaningful_positions(self, output_file):
        for i in range(10):
            position_sets = meaningful_positions.compare_played_and_random_moves(
                self.autoencoder,
                self.PGN_file,
                self.test_amount / 10,
                i + 1,
                output_file,
            )
            self.compare_sets(position_sets, f"meaningful_positions{i}", output_file)

    def test_opening_positions(self, output_file):
        MOVES_AFTER_OPENING = 10

        opening_sets = opening_extractor.extract_opening_position(
            self.PGN_file,
            [["e4"], ["d4"]],
            MOVES_AFTER_OPENING,
            self.test_amount,
        )
        self.compare_sets(opening_sets, "opening_test", output_file)

        opening_sets = opening_extractor.extract_opening_position(
            self.PGN_file,
            [["e4", "e5", "Nf3", "Nc6", "Bc4"], ["e4", "e5", "Nf3", "Nc6", "Bb5"]],
            MOVES_AFTER_OPENING,
            self.test_amount,
        )
        self.compare_sets(opening_sets, "long_opening_test", output_file)

        # opening_sets = opening_extractor.extract_opening_position(
        #     self.PGN_file,
        #     [
        #         [
        #             "d4",
        #             "Nf6",
        #             "c4",
        #             "g6",
        #             "Nc3",
        #             "d5",
        #             "cxd5",
        #             "Nxd5",
        #             "e4",
        #             "Nxc3",
        #             "bxc3",
        #             "Bg7",
        #         ],
        #         ["e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6", "Nc3", "g6"],
        #     ],
        #     MOVES_AFTER_OPENING,
        #     self.test_amount,
        # )
        # self.compare_sets(opening_sets, "opening_test", output_file)

    def test_eco(self, output_file):
        ECO_FILE = "./data/openings_sheet.csv"
        eco_sets = eco_reader.convert_eco_file_to_position_sets(ECO_FILE)
        self.compare_sets(eco_sets, "eco_test", output_file)

    def test_tactics(self, output_file):
        TACTICS_FILE = "./data/lichess_db_puzzle.csv.zst"
        tactics = [
            ["anastasiaMate", "red"],
            ["arabianMate", "green"],
            ["backRankMate", "blue"],
            ["smotheredMate", "yellow"],
        ]
        tactics_sets = tactics_reader.convert_tactics_file_to_position_sets(
            TACTICS_FILE, tactics, self.test_amount
        )
        self.compare_sets(tactics_sets, "tactics_test", output_file)

    def test_before_mate_and_mate(self, output_file):
        TACTICS_FILE = "./data/lichess_db_puzzle.csv.zst"
        tactic = "backRankMate"

        mate_sets = tactics_reader.get_before_mate_and_mate_sets(
            TACTICS_FILE, tactic, 2000
        )
        self.compare_sets(mate_sets, "mate_test", output_file)

    def test_random_and_mates(self, output_file):
        TACTICS_FILE = "./data/lichess_db_puzzle.csv.zst"

        sets = tactics_reader.get_random_and_mates(TACTICS_FILE, self.PGN_file, 10000)
        self.compare_sets(sets, "mate_random_test", output_file, 10000)

    def test_expert_players(self, output_file):
        FROM_MOVE = 20
        CARLSEN_FILE = "./data/expert-players/Carlsen/Carlsen.pgn"
        TAL_FILE = "./data/expert-players/Tal/Tal.pgn"

        expert_player_position_sets = expert_players.get_expert_player_positions(
            CARLSEN_FILE, TAL_FILE, self.test_amount, FROM_MOVE
        )
        self.compare_sets(
            expert_player_position_sets,
            "expert_player_test",
            output_file,
            self.test_amount,
        )

    def compare_sets(
        self,
        position_sets: list[PositionSet],
        tsne_file_name: str,
        output_file,
        num_visualize: int = VISUALIZATION_AMOUNT,
    ):
        PositionDistanceCalculator.calculate_distances(
            self.autoencoder, position_sets, output_file
        )
        TSNEVisualizer.visualize_tsne(
            self.autoencoder,
            position_sets,
            visualization_file_name=tsne_file_name,
            num_points_to_visualize=num_visualize,
        )

    def wrap_test_with_print(self, test, test_start_text, output_file):
        print(test_start_text)
        output_file.write(test_start_text)
        start_time = time.time()
        print(f"start time {str(start_time)}")

        test(output_file)

        print(f"end time {str(time.time() - start_time)}")
        print(TEST_END_TEXT)
        output_file.write(TEST_END_TEXT)
