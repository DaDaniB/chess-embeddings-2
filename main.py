import os
from dotenv import load_dotenv
import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam

from modules.PGN_reader import PGNReader
from modules.FEN_converter import FENConverter
from modules.vectorization.FEN import FEN
from modules.embedding_test.illegal_positions_generator import (
    Illegal_Positions_Generator,
)
from modules.embedding_tester import EmbeddingTester

from modules.autoencoder.autoencoder_noencode import Autoencoder_NoEncode
from modules.autoencoder.paper.autoencoder_linear_16 import Autoencoder_Linear16
from modules.autoencoder.paper.autoencoder_linear_32 import Autoencoder_Linear32
from modules.autoencoder.paper.autoencoder_linear_64 import Autoencoder_Linear64
from modules.autoencoder.paper.autoencoder_linear_128 import Autoencoder_Linear128
from modules.autoencoder.paper.autoencoder_linear_128_4layers import (
    Autoencoder_Linear128_4layers,
)
from modules.autoencoder.paper.autoencoder_linear_128_multiple import (
    Autoencoder_Linear128_multiple,
)
from modules.autoencoder.paper.autoencoder_convolutional_normal import (
    Autoencoder_Convolutional_normal,
)

from modules.autoencoder.paper.autoencoder_convolutional_normal_nocomp import (
    Autoencoder_Convolutional_normal_nocomp,
)
from modules.autoencoder.paper.autoencoder_convolutional_normal_safe import (
    Autoencoder_Convolutional_normal_safe,
)
from modules.autoencoder.paper.autoencoder_convolutional_normal_double_filters import (
    Autoencoder_Convolutional_normal_double_filters,
)
from modules.autoencoder.paper.autoencoder_convolutional_simple import (
    Autoencoder_Convolutional_Simple,
)


###### ENV ########################################################################################################
load_dotenv()
PGN_FILE = os.getenv("PGN_FILE")
TXT_FILE = "./data/extracted.txt"

TEST_UNSEEN = "./data/test_unseen.txt"
TEST_SEEN = "./data/test_seen.txt"
OUTPUT_TEST = "./test_output.txt"


###### AUTOENCODER ################################################################################################
# autoencoder = Autoencoder_NoEncode()
# autoencoder = Autoencoder_Linear16()
# autoencoder = Autoencoder_Linear32()
# autoencoder = Autoencoder_Linear64()
# autoencoder = Autoencoder_Linear128()
# autoencoder = Autoencoder_Linear128_4layers()
# autoencoder = Autoencoder_Linear128_multiple()
autoencoder = Autoencoder_Convolutional_normal()
# autoencoder = Autoencoder_Convolutional_normal_nocomp()  # bad
# autoencoder = Autoencoder_Convolutional_normal_safe()  # very bad
# autoencoder = Autoencoder_Convolutional_normal_double_filters()
# autoencoder = Autoencoder_Convolutional_Simple()


###### init ########################################################################################################
autoencoder.initialize()


###### load weights ################################################################################################

# autoencoder.load_weights(
#     "./models/Autoencoder_Convolutional_Simple128nPositions500000.h5"
# )
# autoencoder.load_weights(
#     "./models/Autoencoder_Convolutional_normal_nocompnPositions3000000.h5"
# )
# autoencoder.load_weights(
#     "./models/Autoencoder_Convolutional_normalnPositions3000000.h5"
# )
# autoencoder.load_weights(
#     "./models/Autoencoder_Convolutional_normal_safenPositions3000000.h5"
# )
# autoencoder.load_weights(
#     "./models/Autoencoder_Convolutional_normal_double_filtersnPositions3000000.h5"
# )


# autoencoder.load_weights("./models/Autoencoder_Linear128_4layersnPositions3000000.h5")
# autoencoder.load_weights("./models/Autoencoder_Linear128_multiplenPositions3000000.h5")
# autoencoder.load_weights("./models/Autoencoder_Linear128nPositions3000000.h5")
# autoencoder.load_weights("./models/Autoencoder_Linear64nPositions3000000.h5")
# autoencoder.load_weights("./models/Autoencoder_Linear32nPositions3000000.h5")
# autoencoder.load_weights("./models/Autoencoder_Linear16nPositions3000000.h5")


###### train ########################################################################################################
autoencoder.train(TXT_file=TXT_FILE, num_positions=3000020, epochs=12)

# encoded = autoencoder.encode_FEN_position(
#     FEN("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - ")
# )

autoencoder.test_encode_decode(
    FEN("rnbqkb1r/pp1pp1pp/2p2n2/5p2/2PP4/5N2/PP2PPPP/RNBQKB1R w KQkq - 1 4")
)
autoencoder.test_encode_decode(
    FEN("rn1qk2r/ppp2ppp/5n2/2b1Nb2/2B5/2NP4/PPP2PPP/R1BQ1RK1 b kq - 0 8")
)
autoencoder.test_encode_decode(FEN("R5k1/6p1/8/8/3K2p1/8/8/8 b - - 1 42"))
autoencoder.test_encode_decode(
    FEN("2kr3r/ppb5/2p1p1p1/5p2/2B3P1/2P4P/PP3R2/R5K1 b - - 4 25")
)
autoencoder.test_encode_decode(FEN("8/p1k5/1p6/2P3p1/4K1B1/2P4P/r7/8 b - - 0 38"))


###### test #########################################################################################################
# with open(OUTPUT_TEST, "w") as test_output:
#     autoencoder.test_on_txt_file(TEST_SEEN, test_output)
#     autoencoder.test_on_txt_file(TEST_UNSEEN, test_output)
# tester = EmbeddingTester(autoencoder, PGN_FILE)
# tester.test_all()


###### extract games to txt #########################################################################################
# PGNReader.extract_unique_games_to_txt(PGN_FILE, TXT_FILE, 3000000)
# PGNReader.extrace_unique_positions_from_file(PGN_FILE, TEST_UNSEEN, 3000000, 600000)
# PGNReader.extract_random_lines(TXT_FILE, TEST_SEEN, 600000)
