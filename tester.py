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

    def test_bow_mentions(self):
        line1 ="PHYS.Located NYT20001017.1908.0279 7 4 6 PER Murray_Schwartz 7 7 8 GPE Wilmington Murray_Schwartz Wilmington".rstrip().split()
        fr1 = FeatureRow(*line1)
        #print bow_mention1(fr1)
        #print bow_mention2(fr1)
        self.assertTrue(bow_mention1(fr1).split("=")[1] =="['Murray', 'Schwartz']")
        self.assertTrue(bow_mention2(fr1).split("=")[1] =="['Wilmington']")


    def test_first_word_inbetween(self):
        line1 ="PHYS.Located NYT20001017.1908.0279 7 4 6 PER Murray_Schwartz 7 7 8 GPE Wilmington Murray_Schwartz Wilmington".rstrip().split()
        fr1 = FeatureRow(*line1)
        #print first_word_inbetween(fr1)
        self.assertTrue(first_word_inbetween(fr1).endswith("[u'in']"))

    def test_last_word_inbetween(self):
        line1 = "no_rel NYT20001019.2136.0319 12 4 5 ORG Republican 12 19 20 GPE Yemen Republican Yemen".rstrip().split()
        fr1 = FeatureRow(*line1)
        self.assertTrue(last_word_inbetween(fr1).endswith("[u'in']"))

    def test_other_words_inbetween(self):
        line1 = "no_rel NYT20001019.2136.0319 12 4 5 ORG Republican 12 19 20 GPE Yemen Republican Yemen".rstrip().split()
        fr1 = FeatureRow(*line1)
        #print other_words_inbetween(fr1)
        #self.assertTrue(other_words_inbetween(fr1).split("=")[1] ==
                        #"[u'candidate', u'on', u'the', u'death', u'penalty', u',', u'the', u'bombing', u'of', u'a', u'Navy', u'ship']")

    def test_first_word_before_m1(self):
        line1 = "no_rel NYT20001019.2136.0319 12 4 5 ORG Republican 12 19 20 GPE Yemen Republican Yemen".rstrip().split()
        fr1 = FeatureRow(*line1)
        #print first_word_before_m1(fr1)
        self.assertTrue(first_word_before_m1(fr1).endswith("[u'the']"))

    def test_first_word_before_m2(self):
        line1 = "no_rel NYT20001019.2136.0319 12 4 5 ORG Republican 12 19 20 GPE Yemen Republican Yemen".rstrip().split()
        fr1 = FeatureRow(*line1)
        #print first_word_before_m2(fr1)
        self.assertTrue(first_word_before_m2(fr1).endswith("[u'in']"))

    def test_second_word_before_m1(self):
        line1 = "no_rel NYT20001019.2136.0319 12 4 5 ORG Republican 12 19 20 GPE Yemen Republican Yemen".rstrip().split()
        fr1 = FeatureRow(*line1)
        #print second_word_before_m1(fr1)
        self.assertTrue(second_word_before_m1(fr1).endswith("[u'grilled']"))

    def test_second_word_before_m2(self):
        line1 = "no_rel NYT20001019.2136.0319 12 4 5 ORG Republican 12 19 20 GPE Yemen Republican Yemen".rstrip().split()
        fr1 = FeatureRow(*line1)
        #print second_word_before_m2(fr1)
        self.assertTrue(second_word_before_m2(fr1).endswith("[u'ship']"))

    def test_head_words(self):
        line1 = "no_rel NYT20001019.2136.0319 12 4 5 ORG Republican 12 19 20 GPE Yemen Republican Yemen".rstrip().split()
        line3 ="EMP-ORG.Member-of-Group.reverse NYT20001019.2136.0319 12 4 5 ORG Republican 12 6 7 PER candidate Republican candidate".rstrip().split()
        line2 = "no_rel APW20001023.2100.0686 5 9 10 PER their 5 31 34 ORG Greenwood_Village_police their Greenwood_Village_police".rstrip().split()
        fr1 = FeatureRow(*line1)
        fr2 = FeatureRow(*line2)
        fr3 = FeatureRow(*line3)
        self.assertTrue(head_word_of_m1(fr1).endswith("[u'candidate']"))
        self.assertTrue(head_word_of_m2(fr1).endswith("[u'Yemen']"))
        self.assertTrue(head_word_of_m2(fr2).endswith("[u'police']"))
        self.assertTrue(same_head(fr3).endswith("True"))

    def test_first_phrase_head_inbetween(self):
        line1 = "no_rel NYT20001019.2136.0319 12 4 5 ORG Republican 12 19 20 GPE Yemen Republican Yemen".rstrip().split()
        fr1 = FeatureRow(*line1)
        line2 = "no_rel APW20001023.2100.0686 5 9 10 PER their 5 31 34 ORG Greenwood_Village_police their Greenwood_Village_police".rstrip().split()
        fr2 = FeatureRow(*line2)
        self.assertTrue(first_phrase_head_inbetween(fr1).split("=")[1]=="[u'penalty']")
        self.assertTrue(first_phrase_head_inbetween(fr2).split("=")[1]=="[u'Sunday']")

    def test_last_phrase_head_inbetween(self):
        line1 = "no_rel NYT20001019.2136.0319 12 4 5 ORG Republican 12 19 20 GPE Yemen Republican Yemen".rstrip().split()
        fr1 = FeatureRow(*line1)
        line2 = "no_rel APW20001023.2100.0686 5 9 10 PER their 5 31 34 ORG Greenwood_Village_police their Greenwood_Village_police".rstrip().split()
        fr2 = FeatureRow(*line2)
        self.assertTrue(last_phrase_head_inbetween(fr1).split("=")[1]=="[u'ship']")
        self.assertTrue(last_phrase_head_inbetween(fr2).split("=")[1]=="[u'wife']")



    #def test_path_enclosed_tree(self):
    #    ###doing some tests
    #    line1 ="EMP-ORG.Member-of-Group.reverse NYT20001019.2136.0319 12 4 5 ORG Republican 12 6 7 PER candidate Republican candidate".rstrip().split()
    #    line2 = "no_rel NYT20001019.2136.0319 12 4 5 ORG Republican 12 19 20 GPE Yemen Republican Yemen".rstrip().split()
    #    line3 = "PHYS.Located NYT20001019.2136.0319 25 14 15 FAC preserve 25 16 17 GPE Alaska preserve Alaska".rstrip().split()
    #    line4 = "ART.User-or-Owner NYT20001019.2136.0319 27 10 12 GPE ``_we 27 16 17 VEH cars ``_we cars".rstrip().split()
    #    line5 = "no_rel NYT20001020.2025.0304 14 6 7 GPE Palestinian 14 19 20 PER terrorists Palestinian terrorists".rstrip().split()
    #    line6 = "PHYS.Located NYT20001017.1908.0279 7 4 6 PER Murray_Schwartz 7 7 8 GPE Wilmington Murray_Schwartz Wilmington".rstrip().split()
    #    line7 = "EMP-ORG.Subsidiary.reverse NYT20001017.1908.0279 8 9 10 ORG which 8 12 13 ORG company which company".rstrip().split()
    #    line8 = "no_rel NYT20001017.1908.0279 9 4 5 ORG many 9 47 48 FAC headquarters many headquarters".rstrip().split()
    #    line9 = "no_rel NYT20001017.1908.0279 9 9 10 ORG companies 9 29 30 GPE Delaware companies Delaware".rstrip().split()
    #    line99= "no_rel APW20001002.0615.0146 3 11 12 PER ex-dictator 3 12 13 PER Suharto ex-dictator Suharto".rstrip().split()
    #    ##testing Anya's augmented trees
    #    line10 = "no_rel APW20001007.0339.0149 3 0 1 GPE WASHINGTON 3 7 10 PER George_W._Bush WASHINGTON George_W._Bush".rstrip().split()
    #    line11 = "no_rel APW20001023.2100.0686 5 9 10 PER their 5 31 34 ORG Greenwood_Village_police their Greenwood_Village_police".rstrip().split()
    #    line12 = "no_rel APW20001023.2100.0686 5 0 2 PER Michele_Roy 5 19 20 FAC bedroom Michele_Roy bedroom".rstrip().split()
    #    line3a="no_rel APW20001120.1450.0376 3 9 10 ORG Taliban 3 15 16 PER Shiite Taliban Shiite".rstrip().split()
    #    line4a="no_rel APW20001120.1450.0376 4 20 21 PER Islamic 4 28 29 PER Muslims Islamic Muslims".rstrip().split()
    #    line7a = "GPE-AFF.Citizen-or-Resident.reverse APW20001120.1450.0376 7 27 28 GPE country 7 33 34 PER groups country groups".rstrip().split()
    #    line33a="no_rel NYT20001123.1511.0062 33 22 23 ORG network 33 34 35 ORG its network its".rstrip().split()
    #    line34a="no_rel NYT20001123.1511.0062 34 21 22 PER columnist 34 23 25 PER ``_You columnist ``_You".rstrip().split()
    #
    #    #fr1 = FeatureRow(*line1)
    #    #fr2 = FeatureRow(*line2)
    #    #fr3 = FeatureRow(*line3)
    #    #fr4 = FeatureRow(*line4)
    #    #fr5 = FeatureRow(*line5)
    #    #fr6 = FeatureRow(*line6)
    #    #fr7 = FeatureRow(*line7)
    #    #fr8 = FeatureRow(*line8)
    #    fr9 = FeatureRow(*line9)
    #    #fr10 = FeatureRow(*line10)
    #    #fr11 = FeatureRow(*line11)
    #    #fr12 = FeatureRow(*line12)
    #    #fr3a=FeatureRow(*line3a)
    #    #fr4a=FeatureRow(*line4a)
    #    #fr7a=FeatureRow(*line7a)
    #    #fr33a=FeatureRow(*line33a)
    #    #fr34a=FeatureRow(*line34a)
    #    fr99 = FeatureRow(*line99)
    #    path_enclosed_tree(fr9).draw()



if __name__ == "__main__":
    unittest.main()