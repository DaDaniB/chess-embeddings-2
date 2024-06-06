import os
from dotenv import load_dotenv

from modules.PGN_reader import PGNReader
from modules.FEN_converter import FENConverter
from modules.vectorization.FEN import FEN

load_dotenv()
PGN_FILE = os.getenv("PGN_FILE")
TEST_FEN = "r1bqkb1r/pp2pp1p/2np1np1/8/3NP3/2NB4/PPP2PPP/R1BQ1RK1 b kq - 1 7"

print("################# to vector ")
vector = FENConverter.to_vector(FEN(TEST_FEN))
converted = FENConverter.vector_to_FEN(vector)
if converted not in TEST_FEN:
    raise ValueError("error")
print(TEST_FEN)
print(converted)

print("################# to vector with info")
vector = FENConverter.to_vector_with_info(FEN(TEST_FEN))
converted = FENConverter.vector_with_info_to_FEN(vector)
if converted not in TEST_FEN:
    raise ValueError("error")
print(TEST_FEN)
print(converted)

print("################# to bitboards")
vector = FENConverter.to_bitboards(FEN(TEST_FEN))
converted = FENConverter.bitboards_to_FEN(vector)
if converted not in TEST_FEN:
    raise ValueError("error")
print(TEST_FEN)
print(converted)

print("################# to bitboards with info")
vector = FENConverter.to_bitboards_with_info(FEN(TEST_FEN))
converted = FENConverter.bitboards_with_info_to_FEN(vector)
if converted not in TEST_FEN:
    raise ValueError("error")
print(TEST_FEN)
print(converted)
