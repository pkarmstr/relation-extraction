import unittest
from feature_generator import Featurizer
from feature_functions import *

__author__ = 'keelan'

class RelTester(unittest.TestCase):

    def setUp(self):
        self.feats = Featurizer("resources/rel-devset.gold", [relation_type])

    def test_build_feats(self):
        self.feats.build_features()

    def test_lazy_dicts(self):
        assert RAW_SENTENCES["NYT20001017.1908.0279"][0]
        assert POS_SENTENCES["NYT20001017.1908.0279"][0]
        assert SYNTAX_PARSE_SENTENCES["NYT20001017.1908.0279"][0]

if __name__ == "__main__":
    unittest.main()