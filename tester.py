import unittest
from feature_generator import Featurizer
from feature_functions import *
from feature_functions import _head_of_m2_, _head_of_m1_
from file_reader import FeatureRow, RAW_SENTENCES

__author__ = 'keelan'

class RelTester(unittest.TestCase):

    #def setUp(self):
    #    self.feats = Featurizer("resources/cleaned-dev.gold", [relation_type])
    #
    #def test_build_feats(self):
    #    self.feats.build_features()
    #
    #def test_lazy_dicts(self):
    #    assert RAW_SENTENCES["NYT20001017.1908.0279"][0]
    #    assert POS_SENTENCES["NYT20001017.1908.0279"][0]
    #    assert SYNTAX_PARSE_SENTENCES["NYT20001017.1908.0279"][0]

    #################
    ##JULIA'S TESTS##
    #################

    def test_tokens(self):
        line1 = "no_rel APW20001006.0338.0184 4 20 21 PER their 4 23 24 PER they their they".rstrip().split()
        line2 ="PER-SOC.Family APW20001002.0615.0146 11 17 18 PER his 11 19 20 PER father his father".rstrip().split()
        fr1 = FeatureRow(*line1)
        fr2 = FeatureRow(*line2)
        #print i_token(fr1)
        #print j_token(fr1)
        #print i_token(fr2)
        #print j_token(fr2)


    def test_bow_mentions(self):
        line1 ="PER-SOC.Family APW20001002.0615.0146 11 17 18 PER his 11 19 20 PER father his father".rstrip().split()
        line2 = "EMP-ORG.Employ-Undetermined APW20001006.0338.0184 4 12 13 PER their 4 13 14 ORG campaigns their campaigns".rstrip().split()
        fr1 = FeatureRow(*line1)
        fr2 = FeatureRow(*line2)
        #print bow_mention1(fr1)
        #print bow_mention1(fr2)
        #print bow_mention2(fr1)
        #print bow_mention2(fr2)



    def test_first_word_in_between(self):
        line1 ="PHYS.Located NYT20001017.1908.0279 7 4 6 PER Murray_Schwartz 7 7 8 GPE Wilmington Murray_Schwartz Wilmington".rstrip().split()
        fr1 = FeatureRow(*line1)
        #print first_word_in_between(fr1)
        self.assertTrue(first_word_in_between(fr1).endswith("[u'in']"))

    def test_last_word_in_between(self):
        line1 = "no_rel NYT20001019.2136.0319 12 4 5 ORG Republican 12 19 20 GPE Yemen Republican Yemen".rstrip().split()
        fr1 = FeatureRow(*line1)
        #print last_word_in_between(fr1)
        self.assertTrue(last_word_in_between(fr1).endswith("[u'in']"))


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
        line1 = "PER-SOC.Family.reverse APW20001209.0634.0301 7 4 5 PER his 7 5 6 PER wife his wife".rstrip().split()
        line3 ="EMP-ORG.Member-of-Group.reverse NYT20001019.2136.0319 12 4 5 ORG Republican 12 6 7 PER candidate Republican candidate".rstrip().split()
        line2 = "no_rel APW20001023.2100.0686 5 9 10 PER their 5 31 34 ORG Greenwood_Village_police their Greenwood_Village_police".rstrip().split()
        fr1 = FeatureRow(*line1)
        fr2 = FeatureRow(*line2)
        fr3 = FeatureRow(*line3)
        #print head_of_m1_coref(fr1)
        #print head_of_m2_coref(fr1)
        #print head_of_m1_coref(fr2)
        #print head_of_m2_coref(fr2)
        #print POS_SENTENCES[fr1.article][int(fr1.i_sentence)]
        #print bow_mention1(fr1)
        #print "What the head of where it occurs is:", _head_of_m1_(fr1), "Head of the NP where it occurs or head of antecedent: ", head_of_m1_coref(fr1)
        #print "What the head of where it occurs is:", _head_of_m2_(fr2), "Head of the NP where it occurs or head of antecedent: ", head_of_m2_coref(fr2)
        self.assertTrue(same_head(fr3).endswith("True"))

    def test_first_phrase_head_in_between(self):
        line1 = "no_rel NYT20001019.2136.0319 12 4 5 ORG Republican 12 19 20 GPE Yemen Republican Yemen".rstrip().split()
        fr1 = FeatureRow(*line1)
        line2 = "no_rel APW20001023.2100.0686 5 9 10 PER their 5 31 34 ORG Greenwood_Village_police their Greenwood_Village_police".rstrip().split()
        fr2 = FeatureRow(*line2)

        self.assertTrue(first_np_head_in_between(fr1).split("=")[1]=="[u'penalty']")
        self.assertTrue(first_np_head_in_between(fr2).split("=")[1]=="[u'Sunday']")

    def test_last_phrase_head_in_between(self):
        line1 = "no_rel NYT20001019.2136.0319 12 4 5 ORG Republican 12 19 20 GPE Yemen Republican Yemen".rstrip().split()
        fr1 = FeatureRow(*line1)
        line2 = "no_rel APW20001023.2100.0686 5 9 10 PER their 5 31 34 ORG Greenwood_Village_police their Greenwood_Village_police".rstrip().split()
        fr2 = FeatureRow(*line2)
        self.assertTrue(last_np_head_in_between(fr1).split("=")[1]=="[u'ship']")
        self.assertTrue(last_np_head_in_between(fr2).split("=")[1]=="[u'wife']")

    def test_heads_in_between(self):
        line1 = "no_rel NYT20001019.2136.0319 12 4 5 ORG Republican 12 19 20 GPE Yemen Republican Yemen".rstrip().split()
        fr1 = FeatureRow(*line1)
        line2 = "no_rel APW20001023.2100.0686 5 9 10 PER their 5 31 34 ORG Greenwood_Village_police their Greenwood_Village_police".rstrip().split()
        fr2 = FeatureRow(*line2)
        #print boh_np_tree(fr1)
        #print boh_tree(fr2)

    def test_first_phrase_head_before_m1(self):
        line1 = "no_rel NYT20001019.2136.0319 12 4 5 ORG Republican 12 19 20 GPE Yemen Republican Yemen".rstrip().split()
        fr1 = FeatureRow(*line1)
        line2 = "no_rel APW20001023.2100.0686 5 9 10 PER their 5 31 34 ORG Greenwood_Village_police their Greenwood_Village_police".rstrip().split()
        fr2 = FeatureRow(*line2)
        self.assertTrue(first_np_head_before_m1(fr1).split("=")[1]=="[u'host']")
        self.assertTrue(first_np_head_before_m1(fr2).split("=")[1]=="[u'dispute']")

    def test_second_phrase_head_before_m1(self):
        line1 = "no_rel NYT20001019.2136.0319 12 4 5 ORG Republican 12 19 20 GPE Yemen Republican Yemen".rstrip().split()
        fr1 = FeatureRow(*line1)
        line2 = "no_rel APW20001023.2100.0686 5 9 10 PER their 5 31 34 ORG Greenwood_Village_police their Greenwood_Village_police".rstrip().split()
        fr2 = FeatureRow(*line2)
        self.assertTrue(second_np_head_before_m1(fr1).split("=")[1]=="[None]")
        self.assertTrue(second_np_head_before_m1(fr2).split("=")[1]=="[u'Roy']")

    def test_second_phrase_head_before_m2(self):
        line1 = "no_rel NYT20001019.2136.0319 12 4 5 ORG Republican 12 19 20 GPE Yemen Republican Yemen".rstrip().split()
        fr1 = FeatureRow(*line1)
        line3="no_rel NYT20001017.1908.0279 9 9 10 ORG companies 9 29 30 GPE Delaware companies Delaware".rstrip().split()
        fr2 = FeatureRow(*line3)
        self.assertTrue(second_np_head_before_m2(fr1).split("=")[1]=="[u'bombing']")
        self.assertTrue(second_np_head_before_m2(fr2).split("=")[1]=="[u'ruling']")

    def test_no_phrase_in_between(self):
        line1 = "no_rel NYT20001019.2136.0319 12 4 5 ORG Republican 12 19 20 GPE Yemen Republican Yemen".rstrip().split()
        line2 = "EMP-ORG.Member-of-Group.reverse NYT20001019.2136.0319 12 4 5 ORG Republican 12 6 7 PER candidate Republican candidate".rstrip().split()
        fr1 = FeatureRow(*line1)
        fr2 = FeatureRow(*line2)
        #print no_phrase_in_between(fr2)
        self.assertTrue(no_phrase_in_between(fr2).endswith("True"))

        #print i_token(fr1), j_token(fr1)
        #print i_entity_type(fr1), j_entity_type(fr1)
        #print first_word_in_between(fr1)
        #print last_word_in_between(fr1)
        #print first_word_before_m1(fr1)
        #print first_word_before_m2(fr1)
        #print second_word_before_m1(fr1)
        #print second_word_before_m2(fr1)
        #print bow_mention1(fr1)
        #print bow_mention2(fr1)
        #print no_phrase_in_between(fr1)
        #print no_words_in_between(fr1)
        #print head_of_m1_coref(fr1)
        #print head_of_m2_coref(fr1)
        #print same_head(fr1)
        #print first_head_before_m1(fr1)
        #print first_np_head_before_m1(fr1)
        #print first_np_head_in_between(fr1)
        #print first_head_in_between(fr1)
        #print second_np_head_before_m1(fr1)
        #print second_head_before_m1(fr1)
        #print last_np_head_in_between(fr1)
        #print last_head_in_between(fr1)
        #print second_np_head_before_m2(fr1)
        #print second_head_before_m2(fr1)
        #print bow_tree(fr1)
        #print boh_tree(fr1)
        #print boh_np_tree(fr1)
        #print lp_tree(fr1)
        #print lp_head_tree(fr1)
        #path_enclosed_tree(fr1).draw()
        #path_enclosed_tree_augmented(fr1).draw()

    def test_path_phrase_labels(self):
        line1 = "no_rel APW20001001.2021.0521 15 2 3 GPE government 15 35 36 ORG government government government".rstrip().split()
        fr1 = FeatureRow(*line1)
        #print lp_tree(fr1)
        #print lp_head_tree(fr1)






    #def test_path_enclosed_tree(self):
    #    ###SAMPLES FOR TESTING PATH AND CHUNKING FEATURES

    #    line1 ="EMP-ORG.Member-of-Group.reverse NYT20001019.2136.0319 12 4 5 ORG Republican 12 6 7 PER candidate Republican candidate".rstrip().split()
        #line2 = "no_rel NYT20001019.2136.0319 12 4 5 ORG Republican 12 19 20 GPE Yemen Republican Yemen".rstrip().split()
    #    line3 = "PHYS.Located NYT20001019.2136.0319 25 14 15 FAC preserve 25 16 17 GPE Alaska preserve Alaska".rstrip().split()
    #    line4 = "ART.User-or-Owner NYT20001019.2136.0319 27 10 12 GPE ``_we 27 16 17 VEH cars ``_we cars".rstrip().split()
    #    line5 = "no_rel NYT20001020.2025.0304 14 6 7 GPE Palestinian 14 19 20 PER terrorists Palestinian terrorists".rstrip().split()
     #   line6 = "PHYS.Located NYT20001017.1908.0279 7 4 6 PER Murray_Schwartz 7 7 8 GPE Wilmington Murray_Schwartz Wilmington".rstrip().split()
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
    #    line1 = "no_rel APW20001001.2021.0521 3 17 19 PER Bashar_Assad 3 38 39 GPE territories Bashar_Assad territories".rstrip().split()
    #    line1 = "no_rel APW20001001.2021.0521 15 2 3 GPE government 15 35 36 ORG government government government".rstrip().split()
    #    line1 = "no_rel APW20001002.0615.0146 3 2 3 GPE Indonesia 3 20 21 PER prosecutors Indonesia prosecutors".rstrip().split()

        ##testing Anya's augmented trees:
        #augtree1="(ROOT (S (S (S (NP (E-PER (NNP Michele) (NNP Roy))) (VP (VBD was) (RB not) (VP (VBN hurt) (PP (IN during) (NP (NP (DT the) (NN dispute)) (PP (IN at) (NP (PRP$ their) (NN home))) (NP-TMP (RB early) (NNP Sunday))))))) (, ,) (CC but) (S (NP (NNP Roy)) (VP (VP (VBD admitted) (S (VP (VBG pulling) (NP (DT a) (NN bedroom) (NN door)) (PP (IN off) (NP (NP (PRP$ its) (NNS hinges))(CC and) (JJ damaging) (NP (DT another)))) (PP (IN after) (NP (NP (PRP$ his) (NN wife)) (VP (VBN called) (NP (E-ORG (NNP Greenwood) (NNP Village) (NN police))))))))) (CC and) (VP (VBD hung) (PRT (RP up)) (PP (IN without) (NP (NN speaking))))))) (, ,) (NP (DT the) (NN report)) (VP (VBD said)) (. .)))"
        #augtree2 ="(ROOT (S (NP (NP (NP (E-GPE (NNP WASHINGTON))) (PRN (-LRB- -LRB-) (NP (NNP AP)) (-RRB- -RRB-))) (SBAR (S (NP (CD __) (NNPS Republicans)) (VP (VBP give) (NP (E-PER (NNP George) (NNP W.) (NNP Bush))) (NP (NP (NN credit)) (PP (S (VP (VBG promoting) (NP (DT a) (JJ Russian) (NN role)) (PP (IN in) (S (VP (VBG smoothing) (NP (DT the) (NN transition)) (PP (IN from) (NP (NN despot))) (PP(TO to) (NP (NP (NN democrat)) (PP (IN in) (NP (NNP Yugoslavia)))))))))))))))) (VP (VBD _) (NP (NP (DT an) (NN idea)) (VP (VBN dismissed) (PP (IN in) (NP (NN debate))) (PP (IN as) (ADJP (JJ risky))) (PP (IN by) (NP (NNP Al) (NNP Gore))))) (SBAR (RB even) (IN as) (S (NP (PRP$ his) (NN boss)) (VP (VBD was) (VP (VBG trying) (S (VP (TO to) (VP (VB get) (S (NP (NNP Moscow)) (VP (TO to) (VP (VB step) (PP (IN in))))))))))))) (. .)))"
        #augtree3 = "(ROOT (NP (NP (NP (E-GPE (NNP KABUL)) (, ,) (E-GPE (NNP Afghanistan))) (PRN (-LRB- -LRB-) (NP (NNP AP)) (-RRB- -RRB-))) (NP (NP (CD _)) (SBAR (S (NP (NP (DT The) (NN ruling) (E-ORG (NNP Taliban)) (E-ORG (NN militia))) (PP (IN on) (NP (NNP Monday)))) (VP (VP (VBD released) (NP (CD 137) (E-PER (JJ Shiite)) (E-PER (JJ Muslim)) (E-PER (NNS prisoners))) (SBAR (S (NP (E-ORG (PRP it))) (VP (VBD had) (VP (VBN held) (PP (IN for) (NP (QP (RB nearly) (CD two)) (NNS years)))))))) (CC and) (VP (VBD urged) (NP (DT the) (E-ORG (NN opposition))) (S (VP (TO to) (VP (VB follow) (NP (NP (NP (NN suit) (CC and) (NN release)) (E-ORG (NN government)) (E-PER (NNS prisoners))) (SBAR (S (NP (E-ORG (PRP it))) (VP (VBZ is) (VP (VBG holding)))))))))))))) (. .)))"
        #augtree4= "(ROOT (S (NP (DT The) (VBN freed) (E-PER (NNS men))) (PRN (, ,) (S (NP (DT all)) (VP (VBD said) (S (VP (TO to) (VP (VB be) (NP (NP (E-PER (NNS fighters))) (VP (VBG belonging) (PP (TO to) (NP (DT the) (E-ORG (NN opposition)) (E-ORG (NN alliance))))))))))) (, ,)) (VP (VBD were) (VP (VBN released) (ADVP (RB ahead)) (PP (IN of) (NP (NP (DT the) (E-PER (JJ Islamic)) (JJ holy) (NN month)) (PP (IN of) (NP (NNP Ramadan))))) (, ,) (SBAR (WHADVP (WRB when)) (S (NP (JJ devout) (E-PER (NNPS Muslims))) (ADVP (RB fast) (PP (IN from) (NP (NN sunrise)))) (VP (TO to) (VP (VB sunset))))))) (. .)))"
        #augtree7="(ROOT (S (NP (NP (DT The) (E-ORG (NN opposition)) (E-ORG (NN alliance))) (, ,) (SBAR (WHNP (E-ORG (WDT which))) (S (VP (VP (VBZ controls) (NP (NP (QP (RB barely) (CD five)) (NN percent)) (PP (IN of) (NP (E-GPE (NNP Afghanistan)))))) (CC and) (VP (VBZ is) (VP (VBG fighting) (NP (NP (DT a) (NN war)) (PP (IN against) (NP (DT the) (JJ dominant) (E-ORG (NNP Taliban)))))))))) (, ,)) (VP (VBZ is) (VP (ADVP (RB mostly)) (VBN made) (ADVP (IN up) (PP (IN of) (NP (NP (DT the) (E-GPE (NN country)) (POS 's)) (NN minority)))) (NP (ADJP (JJ ethnic) (CC and) (JJ religious)) (E-PER (NNS groups))))) (. .)))"
        #augtree33="(ROOT (SBARQ (SBAR (NP (NN __)) (IN If) (S (NP (E-ORG (NNP CBS))) (VP (VBZ shows) (NP (NP (DT the) (JJ first) (NN episode)) (PP (IN of) (`` ``) (NP (NNP Survivor) (NNP II)) ('' ''))) (PP (IN after) (NP (NP (E-ORG (PRP$ its)) (NN broadcast)) (PP (IN of) (NP (NNP Super) (NNP Bowl) (NNP XXXV)))))))) (, ,) (SQ (MD will) (NP (DT the) (E-ORG (NN network))) (VP (ADVP (RB someday)) (VBP show) (NP (NP (DT the) (JJ first) (NN episode)) (PP (IN of) (NP (`` ``) (NP (NNP Survivor) (NNP XXXV)) ('' '') (PP (IN after) (NP (NP (E-ORG (PRP$ its)) (NN broadcast)) (PP (IN of) (NP (NNP Super) (NNP Bowl) (NNP LXVIII)))))))))) (. ?)))"
        #augtree34="(ROOT (S (NP (NN __)) (VP (MD Will) (S (ADJP (JJ cheerful)) (SBAR (S (NP (NP (NN chorus) (E-PER (NNS members))) (VP (VBN dressed) (PP (IN in) (NP (JJ V-neck) (NNS sweaters))))) (VP (VBP interrupt) (NP (E-PER (PRP$ their)) (NNS serenades)) (PP (TO to) (NP (JJ online) (NN shopping))) (ADVP (RB long) (RB enough) (S (VP (TO to) (VP (VB tell) (NP (NP (DT an) (NN advertising) (E-PER (NN columnist))) (, ,) (S (`` ``) (NP (PRP You)) (VP (VBP ask) (NP (NP (DT a) (NN lot)) (PP (IN of) (NP (NP (NNS questions)) (PP (IN for) (NP (E-PER (NN someone))))))) (PP (IN from) (NP (E-GPE (NNP Brooklyn))))) ('' '')))))))))))) (. ?)))"
        #s_tree = ParentedTree.parse(augtree4)
        #print "printing leaves corresponding to indices"
        #print s_tree.leaves()[int(fr.i_offset_begin)] #checking indices first...
        #print s_tree.leaves()[int(fr.j_offset_begin)]
    #
        fr1 = FeatureRow(*line1)
        fr2 = FeatureRow(*line6)
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
        #path_enclosed_tree(fr1).draw()
        #path_enclosed_tree(fr1).draw()
        #path_enclosed_tree_augmented(fr1).draw()
        #path_enclosed_tree(fr2).draw()
        #path_enclosed_tree_augmented(fr2).draw()




if __name__ == "__main__":
    unittest.main()