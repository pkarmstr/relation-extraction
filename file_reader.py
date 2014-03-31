__author__ = 'keelan'

import os
import re
from collections import defaultdict,namedtuple
from nltk.tree import Tree,ParentedTree
from corenlp import parse_parser_xml_results
from tree_converter import *

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

class AutoVivification(dict):
    """
    Implementation of perl's autovivification feature.
    It allows one to populate all the levels of a nested dict at the same time.
    (http://stackoverflow.com/questions/651794/whats-the-best-way-to-initialize-a-dict-of-dicts-in-python)
    """
    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value

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

def stanford_nonparented_tree_reader(nlp):
    all_sentences = []
    for s in nlp["sentences"]:
        all_sentences.append(Tree(s["parsetree"]))
    return all_sentences

def stanford_general_opener(file_path, f_name):
    with open(os.path.join(file_path, f_name+".head.coref.raw.xml"), "r") as f_in:
        return parse_parser_xml_results(f_in.read())

def gather_entities():
    """
    Loops through the training, dev, and test data and creates a dictionary that contains entity types
    that correspond to offset indices.

    Format of the dictionary:
    {article: {sentence_id: {offset_tuple: entity_type}}}
    """
    articles=AutoVivification()

    for root, dirs, files in os.walk('resources'):
        for file_name in files:
            if file_name.startswith('cleaned') and file_name.endswith('gold'):
                file_name=os.path.join(root,file_name)
                with open(file_name, 'r') as f:
                    for line in f:
                        _, article, sentence_id, i_start, i_end, i_entity_type, _, \
                        _, j_start, j_end, j_entity_type, _ = line.rstrip('\n').split()
                        articles[article][int(sentence_id)][(int(i_start),int(i_end))]=i_entity_type
                        articles[article][int(sentence_id)][(int(j_start),int(j_end))]=j_entity_type

    return articles

def pronoun_reader():
    ls = []
    with open("resources/pronouns.txt", "r") as f_in:
        for line in f_in:
            ls.append(line.rstrip())
    return ls

def augmented_tree_reader():
    """
    Converts all nonparented trees into augmented trees. Stores augmented trees in a dict of the form
    {article:{sentence_id:tree}}
    """
    augmented_tree_dict=AutoVivification()
    for article in entity_types:
        for sent_id in entity_types[article]:
            t=NONPARENTED_SENTENCES[article][sent_id]
            t_copy=t.copy(deep=True)
            augment_tree(t_copy,sent_id,article)
            augmented_tree_dict[article][sent_id]=t_copy

    augmented_tree_dict['NYT20001229.2047.0291'][21]=Tree("")

    return augmented_tree_dict



FeatureRow = namedtuple("FeatureRow", ["relation_type", "article", "i_sentence",
                                       "i_offset_begin","i_offset_end",
                                       "i_entity_type", "i_token",
                                       "j_sentence", "j_offset_begin",
                                       "j_offset_end", "j_entity_type",
                                       "j_token", "i_cleaned", "j_cleaned"])

##########################
# final` data structures #
##########################

basedir = "stanford-full-pipeline"


all_stanford = LazyDict(basedir, stanford_general_opener)
RAW_SENTENCES = SuperLazyDict(all_stanford, stanford_raw_reader)
POS_SENTENCES = SuperLazyDict(all_stanford, stanford_pos_reader)
SYNTAX_PARSE_SENTENCES = SuperLazyDict(all_stanford, stanford_tree_reader)
NONPARENTED_SENTENCES = SuperLazyDict(all_stanford, stanford_nonparented_tree_reader)
PRONOUN_SET = set(pronoun_reader())
entity_types=gather_entities()
AUGMENTED_TREES=augmented_tree_reader()


TITLE_SET= {"chairman", "Chairman", "director", "Director", "president", "President", "manager", "Manager", "executive",
            "CEO", "Officer", "officer", "consultant", "Chief", "CFO", "COO", "CTO", "CMO", "founder", "shareholder",
            "researcher", "professor", "principal", "Principal", "minister", "Minister", "prime", "Prime", "chief",
            "Chief", "prosecutor", "Prosecutor", "queen", "Queen", "leader", "Leader", "secretary", "Secretary",
            "ex-Leader", "ex-leader", "coach", "Coach", "composer", "Composer", "head", "Head", "governor", "Governor",
            "judge", "Judge", "democrat", "Democrat", "republican", "Republican", "senator", "Senator", "congressman",
            "Congressman", "congresswoman", "Congresswoman", "analyst", "Analyst", "sen", "Sen", "Rep", "rep", "MP",
            "mp", "justice", "Justice", "co-chairwoman", "co-chair", "co-chairman", "Mr.", "mr.", "Mr", "mr", "Ms.",
            "ms.", "Mrs.", "mrs."}

if __name__ == "__main__":
    """outfile=open('test_entity_dict.txt','w')
    for k,v in entity_types.iteritems():
        outfile.write(k+'\n')
        for k2,v2 in v.iteritems():
            outfile.write('\t'+k2+'\n')
            for k3,v3 in v2.iteritems():
                outfile.write('\t\t'+str(k3)+'\t'+v3+'\n')

    outfile.close()"""


