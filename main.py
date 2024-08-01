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


###### AUTOENCODER ################################################################################################
# autoencoder = Autoencoder_NoEncode()
# autoencoder = Autoencoder_Linear16()
# autoencoder = Autoencoder_Linear32()
# autoencoder = Autoencoder_Linear64()
# autoencoder = Autoencoder_Linear128()
# autoencoder = Autoencoder_Linear128_4layers()
# autoencoder = Autoencoder_Linear128_multiple()
# autoencoder = Autoencoder_Convolutional_normal()
# autoencoder = Autoencoder_Convolutional_normal_nocomp()
# autoencoder = Autoencoder_Convolutional_normal_safe()
autoencoder = Autoencoder_Convolutional_normal_double_filters()
# autoencoder = Autoencoder_Convolutional_Simple()


###### compile #####################################################################################################
optimizer = Adam(learning_rate=1e-4, clipnorm=1.0)
# optimizer = SGD(learning_rate=0.02)
autoencoder.compile(optimizer=optimizer, loss="mean_squared_error")


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
autoencoder.load_weights(
    "./models/Autoencoder_Convolutional_normal_double_filtersnPositions3000000.h5"
)


# autoencoder.load_weights("./models/Autoencoder_Linear128_4layersnPositions3000000.h5")
# autoencoder.load_weights("./models/Autoencoder_Linear128_multiplenPositions3000000.h5")
# autoencoder.load_weights("./models/Autoencoder_Linear128nPositions3000000.h5")
# autoencoder.load_weights("./models/Autoencoder_Linear64nPositions3000000.h5")
# autoencoder.load_weights("./models/Autoencoder_Linear32nPositions3000000.h5")
# autoencoder.load_weights("./models/Autoencoder_Linear16nPositions3000000.h5")


###### train ########################################################################################################
# autoencoder.train(TXT_file=TXT_FILE, num_positions=3000000, epochs=4)

# encoded = autoencoder.encode_FEN_position(
#     FEN("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - ")
# )

autoencoder.test_encode_decode(
    FEN("rnbqkb1r/pp1pp1pp/2p2n2/5p2/2PP4/5N2/PP2PPPP/RNBQKB1R w KQkq - 1 4")
)


###### test #########################################################################################################

tester = EmbeddingTester(autoencoder, PGN_FILE)
tester.test_all()


###### extract games to txt #########################################################################################
# PGNReader.extract_unique_games_to_txt(PGN_FILE, TXT_FILE, 3000000)
