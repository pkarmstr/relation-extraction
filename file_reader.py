__author__ = 'keelan'

import os
import re
from collections import namedtuple
from nltk.tree import ParentedTree
from corenlp import parse_parser_xml_results

def pos_split(string):
    if not string.startswith("_"):
        return string.split("_")
    else:
        return [string[:-2], string[-1:]]

def feature_list_reader(file_path):
    ls = []
    with open(file_path, "r") as f_in:
        for line in f_in:
            ls.append(eval(line.rstrip()))

    return ls

class LazyDict:

    def __init__(self, file_path, opener):
        self.d = {}
        self.file_path = file_path
        self.opener = opener

    def __getitem__(self, item):
        try:
            return self.d[item]
        except KeyError:
            res = self.opener(self.file_path, item)
            self.d[item] = res
            return res

class SuperLazyDict:

    def __init__(self, lazy_dict, reader):
        self.lazy_dict = lazy_dict
        self.reader = reader
        self.d = {}

    def __getitem__(self, item):
        try:
            return self.d[item]
        except KeyError:
            res = self.reader(self.lazy_dict[item])
            self.d[item] = res
            return res

def stanford_raw_reader(nlp):
    all_sentences = []
    for s in nlp["sentences"]:
        all_sentences.append(s["text"])
    return all_sentences

def stanford_pos_reader(nlp):
    all_sentences = []
    for s in nlp["sentences"]:
        sentence = []
        for w in s["words"]:
            sentence.append([w[0], w[1]["PartOfSpeech"]])
        all_sentences.append(sentence)
    return all_sentences

def stanford_tree_reader(nlp):
    all_sentences = []
    for s in nlp["sentences"]:
        all_sentences.append(ParentedTree.parse(s["parsetree"]))
    return all_sentences

def stanford_general_opener(file_path, f_name):
    with open(os.path.join(file_path, f_name+".xml"), "r") as f_in:
        return parse_parser_xml_results(f_in.read())


FeatureRow = namedtuple("FeatureRow", ["relation_type", "article", "i_sentence",
                                       "i_offset_begin","i_offset_end",
                                       "i_entity_type", "i_token",
                                       "j_sentence", "j_offset_begin",
                                       "j_offset_end", "j_entity_type",
                                       "j_token", "i_cleaned", "j_cleaned"])

###########################
# `final` data structures #
###########################
basedir = "stanford-full-pipeline"

all_stanford = LazyDict(basedir, stanford_general_opener)
RAW_SENTENCES = SuperLazyDict(all_stanford, stanford_raw_reader)
POS_SENTENCES = SuperLazyDict(all_stanford, stanford_pos_reader)
SYNTAX_PARSE_SENTENCES = SuperLazyDict(all_stanford, stanford_tree_reader())

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