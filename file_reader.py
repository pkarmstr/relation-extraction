__author__ = 'keelan'

import os
import re
from collections import namedtuple
from nltk.tree import Tree
from build_raw_sentences import pos_split
from corenlp import parse_parser_xml_results


def tree_reader():
    d = {}
    trees = BracketParseCorpusReader("parsed_sentences/", ".*")
    for name in trees.fileids():
        d_name = re.sub(r"\.tree", "", name)
        d[d_name] = list(trees.parsed_sents(name))

    return d

def pos_reader():
    d = {}
    for f in os.listdir("pos_sentences/"):
        name = re.sub(r"\.pos", "", f)
        with open(os.path.join("pos_sentences", f), "r") as f_in:
            sentences = []
            for line in f_in:
                line = line.rstrip()
                if line != "":
                    pairs = [pos_split(p) for p in line.split()]
                    sentences.append(pairs)
            d[name] = sentences
    return d

def raw_reader():
    d = {}
    for f in os.listdir("raw_sentences/"):
        with open(os.path.join("raw_sentences", f), "r") as f_in:
            sentences = []
            for line in f_in:
                line = line.rstrip()
                if line != "":
                    tokens = line.split()[1:-1] #dont want <s> tags
                    sentences.append(tokens)
            d[f] = sentences
    return d

def pronoun_reader():
    ls = []
    with open("resources/pronouns.txt", "r") as f_in:
        for line in f_in:
            ls.append(line.rstrip())
    return ls

def noncontent_reader():
    ls = []
    with open("resources/noncontentwords.txt", "r") as f_in:
        for line in f_in:
            ls.append(line.rstrip())
    return set(ls)

def dcoref_opener(file_path, fname):
    with open(os.path.join(file_path, fname+".raw.xml"), "r") as f_in:
        nlp = parse_parser_xml_results(f_in.read())
        all_coref_groups = []
        for group in nlp["coref"]:
            chain = set()
            for pair in group:
                for i in pair:
                    chain.add((i[0], i[1], i[3], i[4]))
            all_coref_groups.append(chain)
    return all_coref_groups

def stanford_raw_opener(file_path, f_name):
    with open(os.path.join(file_path, f_name+".xml"), "r") as f_in:
        nlp = parse_parser_xml_results(f_in.read())
        all_sentences = []
        for s in nlp["sentences"]:
            all_sentences.append(s["text"])
    return all_sentences

def stanford_pos_opener(file_path, f_name):
    with open(os.path.join(file_path, f_name+".xml"), "r") as f_in:
        nlp = parse_parser_xml_results(f_in.read())
        all_sentences = []
        for s in nlp["sentences"]:
            sentence = []
            for w in s["words"]:
                sentence.append([w[0], w[1]["PartOfSpeech"]])
            all_sentences.append(sentence)
    return all_sentences

def stanford_tree_opener(file_path, f_name):
    with open(os.path.join(file_path, f_name+".xml"), "r") as f_in:
        nlp = parse_parser_xml_results(f_in.read())
        all_sentences = []
        for s in nlp["sentences"]:
            all_sentences.append(Tree.parse(s["parsetree"]))
    return all_sentences

class LazyDict:

    def __init__(self, file_path, opener):
        self.d = {}
        self.file_path = file_path
        self.opener = opener

    def __getitem__(self, item):
        try:
            return self.d[item]
        except KeyError:
            self.d[item] = self.opener(self.file_path, item)
            return self.d[item]

FeatureRow = namedtuple("FeatureRow", ["article", "sentence", "offset_begin",
                                        "offset_end", "entity_type", "token",
                                        "sentence_ref", "offset_begin_ref",
                                        "offset_end_ref", "entity_type_ref",
                                        "token_ref", "i_cleaned", "j_cleaned",
                                        "is_referent"])

###########################
# `final` data structures #
###########################
basedir = "/home/keelan/stanford-full-pipeline"

#RAW_DICTIONARY = raw_reader()
#POS_DICTIONARY = pos_reader()
#TREES_DICTIONARY = tree_reader()
PRONOUN_LIST = pronoun_reader()
NONCONTENT_SET = noncontent_reader()
COREF_DICTIONARY = LazyDict(basedir, dcoref_opener)
RAW_DICTIONARY = LazyDict(basedir, stanford_raw_opener)
POS_DICTIONARY = LazyDict(basedir, stanford_pos_opener)
TREES_DICTIONARY = LazyDict(basedir, stanford_tree_opener)

TITLE_SET=set(["chairman", "Chairman", "director", "Director", "president",
              "President", "manager", "Manager", "executive", "CEO", "Officer",
              "officer", "consultant","Chief", "CFO", "COO", "CTO", "CMO",
              "founder", "shareholder", "researcher", "professor",
              "principal", "Principal","minister","Minister","prime","Prime","chief", "Chief",
              "prosecutor","Prosecutor","queen","Queen","leader","Leader","secretary","Secretary","ex-Leader",
              "ex-leader","coach","Coach","composer","Composer","head","Head","governor","Governor","judge",
              "Judge","democrat","Democrat","republican","Republican","senator","Senator","congressman",
              "Congressman","congresswoman", "Congresswoman","analyst","Analyst","sen","Sen","Rep","rep","MP",
              "mp","justice","Justice","co-chairwoman","co-chair","co-chairman","Mr.","mr.","Mr","mr","Ms.","ms.",
              "Mrs.","mrs.",])