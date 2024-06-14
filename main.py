import os
from dotenv import load_dotenv
import tensorflow as tf

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
from modules.autoencoder.custom_autoencoder.autoencoder_linear_32 import (
    Autoencoder_Linear32,
)
from modules.autoencoder.custom_autoencoder.autoencoder_linear_64 import (
    Autoencoder_Linear64,
)
from modules.autoencoder.custom_autoencoder.autoencoder_linear_128 import (
    Autoencoder_Linear128,
)
from modules.autoencoder.custom_autoencoder.autoencoder_linear_16_2layers import (
    Autoencoder_Linear16_2layers,
)
from modules.autoencoder.custom_autoencoder.autoencoder_linear_32_2layers import (
    Autoencoder_Linear32_2layers,
)
from modules.autoencoder.custom_autoencoder.autoencoder_linear_64_2layers import (
    Autoencoder_Linear64_2layers,
)
from modules.autoencoder.custom_autoencoder.autoencoder_linear_128_2layers import (
    Autoencoder_Linear128_2layers,
)
from modules.autoencoder.custom_autoencoder.convolutional.autoencoder_convolutional_16F_5k import (
    Autoencoder_Convolutional_16F_5K,
)
from modules.autoencoder.custom_autoencoder.convolutional.autoencoder_convolutional_128F_5k import (
    Autoencoder_Convolutional_128F_5K,
)
from modules.autoencoder.custom_autoencoder.convolutional.autoencoder_convolutional_128F_5k_1linear import (
    Autoencoder_Convolutional_128F_5K_1linear,
)
from modules.autoencoder.custom_autoencoder.autoencoder_linear_128_2layers import (
    Autoencoder_Linear128_2layers,
)
from modules.autoencoder.custom_autoencoder.autoencoder_linear_128_3layers import (
    Autoencoder_Linear128_3layers,
)
from modules.autoencoder.custom_autoencoder.autoencoder_linear_128_4layers import (
    Autoencoder_Linear128_4layers,
)
from modules.autoencoder.custom_autoencoder.autoencoder_linear_128_multiple import (
    Autoencoder_Linear128_multiple,
)
from modules.autoencoder.custom_autoencoder.convolutional.autoencoder_convolutional import (
    Autoencoder_Convolutional,
)
from modules.autoencoder.custom_autoencoder.convolutional.autoencoder_convolutional_gpt import (
    Autoencoder_Convolutional_GPT,
)
from modules.autoencoder.custom_autoencoder.convolutional.autoencoder_convolutional_gpt2 import (
    Autoencoder_Convolutional_GPT2,
)

###### ENV ######################
load_dotenv()
PGN_FILE = os.getenv("PGN_FILE")


###### AUTOENCODER ##############
# autoencoder = Autoencoder_NoEncode()
# autoencoder = Autoencoder_Linear16()
# autoencoder = Autoencoder_Linear32()
# autoencoder = Autoencoder_Linear64()
# autoencoder = Autoencoder_Linear128()
# autoencoder = Autoencoder_Linear128_2layers()
# autoencoder = Autoencoder_Linear128_3layers()
# autoencoder = Autoencoder_Linear128_4layers()
# autoencoder = Autoencoder_Linear128_multiple()

# autoencoder = Autoencoder_Linear16_2layers()
# autoencoder = Autoencoder_Linear32_2layers()
# autoencoder = Autoencoder_Linear64_2layers()
# autoencoder = Autoencoder_Linear128_2layers()
# autoencoder = Autoencoder_Convolutional_16F_5K()
# autoencoder = Autoencoder_Convolutional_128F_5K()
# autoencoder = Autoencoder_Convolutional_128F_5K_1linear()
# autoencoder = Autoencoder_Convolutional()
autoencoder = Autoencoder_Convolutional_GPT()
# autoencoder = Autoencoder_Convolutional_GPT2()

###### compile ##################
autoencoder.compile(optimizer="adam", loss="mean_squared_error")

###### load weights #############
# autoencoder.load_weights("./models/Autoencoder_Linear16nPositions1000000.h5")
# autoencoder.load_weights("./models/Autoencoder_Linear128nPositions1000000.h5")
autoencoder.load_weights("./models/Autoencoder_Convolutional_GPTnPositions1000000.h5")


###### train ####################
# autoencoder.train(PGN_FILE, 2000000, chunk_size=20000, epochs=4)

tester = EmbeddingTester(autoencoder, PGN_FILE)
tester.test_all()
