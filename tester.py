import unittest
from feature_generator import Featurizer
from feature_functions import *
from file_reader import FeatureRow

__author__ = 'keelan'

class RelTester(unittest.TestCase):

    def setUp(self):
        self.feats = Featurizer("resources/cleaned-dev.gold", [relation_type])

    def test_build_feats(self):
        self.feats.build_features()

    def test_lazy_dicts(self):
        assert RAW_SENTENCES["NYT20001017.1908.0279"][0]
        assert POS_SENTENCES["NYT20001017.1908.0279"][0]
        assert SYNTAX_PARSE_SENTENCES["NYT20001017.1908.0279"][0]

    #################
    ##JULIA'S TESTS##
    #################

    def test_path_enclosed_tree(self):
        ####doing some tests
        line1 ="EMP-ORG.Member-of-Group.reverse NYT20001019.2136.0319 12 4 5 ORG Republican 12 6 7 PER candidate Republican candidate".rstrip().split()
        line2 = "no_rel NYT20001019.2136.0319 12 4 5 ORG Republican 12 19 20 GPE Yemen Republican Yemen".rstrip().split()
        line3 = "PHYS.Located NYT20001019.2136.0319 25 14 15 FAC preserve 25 16 17 GPE Alaska preserve Alaska".rstrip().split()
        line4 = "ART.User-or-Owner NYT20001019.2136.0319 27 10 12 GPE ``_we 27 16 17 VEH cars ``_we cars".rstrip().split()
        line5 = "no_rel NYT20001020.2025.0304 14 6 7 GPE Palestinian 14 19 20 PER terrorists Palestinian terrorists".rstrip().split()
        line6 = "PHYS.Located NYT20001017.1908.0279 7 4 6 PER Murray_Schwartz 7 7 8 GPE Wilmington Murray_Schwartz Wilmington".rstrip().split()
        line7 = "EMP-ORG.Subsidiary.reverse NYT20001017.1908.0279 8 9 10 ORG which 8 12 13 ORG company which company".rstrip().split()
        line8 = "no_rel NYT20001017.1908.0279 9 4 5 ORG many 9 47 48 FAC headquarters many headquarters".rstrip().split()
        line9 = "no_rel NYT20001017.1908.0279 9 9 10 ORG companies 9 29 30 GPE Delaware companies Delaware".rstrip().split()
        ##testing Anya's trees
        line10 = "no_rel APW20001007.0339.0149 3 25 26 GPE Yugoslavia 3 35 37 PER Al_Gore Yugoslavia Al_Gore".rstrip().split()
        line11 = "no_rel APW20001023.2100.0686 5 9 10 PER their 5 31 34 ORG Greenwood_Village_police their Greenwood_Village_police".rstrip().split()
        line12 = "no_rel APW20001023.2100.0686 5 0 2 PER Michele_Roy 5 19 20 FAC bedroom Michele_Roy bedroom".rstrip().split()
        fr1 = FeatureRow(*line1)
        fr2 = FeatureRow(*line2)
        fr3 = FeatureRow(*line3)
        fr4 = FeatureRow(*line4)
        fr5 = FeatureRow(*line5)
        fr6 = FeatureRow(*line6)
        fr7 = FeatureRow(*line7)
        fr8 = FeatureRow(*line8)
        fr9 = FeatureRow(*line9)
        fr10 = FeatureRow(*line10)
        fr11 = FeatureRow(*line11)
        fr12 = FeatureRow(*line12)
        path_enclosed_tree(fr9).draw()



if __name__ == "__main__":
    unittest.main()