import os
from dotenv import load_dotenv

from modules.PGN_reader import PGNReader
from modules.FEN_converter import FENConverter
from modules.vectorization.FEN import FEN
from modules.embedding_test.illegal_positions_generator import (
    Illegal_Positions_Generator,
)
from modules.embedding_tester import EmbeddingTester

from modules.autoencoder.autoencoder_noencode import Autoencoder_NoEncode
from modules.autoencoder.custom_autoencoder.autoencoder_linear_16 import (
    Autoencoder_Linear16,
)

load_dotenv()
PGN_FILE = os.getenv("PGN_FILE")
TEST_FEN = "r1bqkb1r/pp2pp1p/2np1np1/8/3NP3/2NB4/PPP2PPP/R1BQ1RK1 b kq - 1 7"

# vector = FENConverter.to_bitboards_with_info(FEN(TEST_FEN))
# print(TEST_FEN)
# of = FENConverter.bitboards_with_info_to_FEN(vector)
# print(of)

autoencoder = Autoencoder_NoEncode()
autoencoder.compile(optimizer="adam", loss="mean_squared_error")

# autoencoder = Autoencoder_Linear16()

# ############# compile and train
# autoencoder.compile(optimizer="adam", loss="mean_squared_error")
# autoencoder.train(PGN_FILE, 1000, chunk_size=10)

tester = EmbeddingTester(autoencoder, PGN_FILE)
tester.test_all()
