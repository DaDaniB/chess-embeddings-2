import os
from dotenv import load_dotenv
from modules.embedding_tester import EmbeddingTester

from modules.autoencoder.base_autoencoder import BaseAutoEncoder
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

load_dotenv()
PGN_FILE = os.getenv("PGN_FILE")


def test_autoencoder(autoencoder: BaseAutoEncoder):
    embedding_tester = EmbeddingTester(autoencoder, PGN_FILE)
    autoencoder.initialize()
    autoencoder.load_default_weights()
    embedding_tester.test_all()


test_autoencoder(Autoencoder_Linear16())
test_autoencoder(Autoencoder_Linear32())
test_autoencoder(Autoencoder_Linear64())
test_autoencoder(Autoencoder_Linear128())
test_autoencoder(Autoencoder_Linear128_4layers())
test_autoencoder(Autoencoder_Linear128_multiple())
test_autoencoder(Autoencoder_Convolutional_normal())
test_autoencoder(Autoencoder_Convolutional_normal_nocomp())
test_autoencoder(Autoencoder_Convolutional_normal_double_filters())
