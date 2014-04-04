# coding=utf-8
"""Necessità 'l ci 'nduce, e non diletto."""
import re
import os
from collections import defaultdict
from file_reader import FeatureRow, feature_list_reader, get_original_data
from helper import Alphabet
from feature_functions import *

__author__ = 'keelan'

import argparse

class Featurizer:

    RELATION_CLASSES = {"PHYS", "PER-SOC", "OTHER-AFF", "GPE-AFF", "DISC", "ART", "EMP-ORG", "no_rel"}


    def __init__(self, original_data, tree_functions, features, no_tag=False):
        self.tree_functions = tree_functions
        self.feature_functions = zip(range(1, len(features)+1), features)
        self.no_tag = no_tag
        self.original_data = original_data
        self.value_alphabet = Alphabet()
        self.value_alphabet.add("__NULL__") #SVMlight doesn't like 0 value for features

    def build_mallet_features(self):
        pass

    def build_features(self):
        self.new_features = []
        for feats in self.original_data:
            new_row = []
            if not self.no_tag:
                new_row.append(feats.relation_type)
            for func in self.tree_functions:
                new_row.append("|BT|")
                new_row.append(func(feats)._pprint_flat('', '()', False))
            new_row.append("|ET|")
            if self.feature_functions:
                for i,func in self.feature_functions:
                    cell = func(feats)
                    value = cell.split("=")[1]
                    if value == "False":
                        continue
                    try:
                        value_index = self.value_alphabet.get_index(value)
                    except KeyError:
                        self.value_alphabet.add(value)
                        value_index = self.value_alphabet.get_index(value)
                    finally:
                        new_row.append("{:d}:{:d}".format(i,value_index))

                new_row.append("|EV|")
            new_row = [s.encode("utf-8") for s in new_row]
            self.new_features.append(new_row)



    def build_relation_class_vectors(self):
        self.all_vectors = defaultdict(list)
        for relation_class in self.RELATION_CLASSES:
            vector_append = self.all_vectors[relation_class].append
            for row in self.new_features:
                if row[0].startswith(relation_class):
                    new_row = ["+1"] + row[1:]
                else:
                    new_row = ["-1"] + row[1:]
                vector_append(new_row)

    def write_multiple_vectors(self, basedir, file_suffix):
        for relation,feature_vectors in self.all_vectors.iteritems():
            with open(os.path.join(basedir, "{}-{}".format(relation, file_suffix)), "w") as f_out:
                for row in feature_vectors:
                    f_out.write("{}\n".format(" ".join(row)))

    def write_no_tag(self, basedir, file_suffix):
        with open(os.path.join(basedir, file_suffix), "w") as f_out:
            for row in self.new_features:
                f_out.write("{}\n".format(" ".join(row)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    parser.add_argument("output_dir")
    parser.add_argument("file_suffix")
    parser.add_argument("tree_list")
    parser.add_argument("feature_list")
    parser.add_argument("-a", "--answers", help="the input file has the answers", action="store_true")

    all_args = parser.parse_args()

    tree_funcs = feature_list_reader(all_args.tree_list, locals())
    feature_funcs = feature_list_reader(all_args.feature_list, locals())

    data = get_original_data(all_args.input_file)
    #tree_funcs=[]
    #feature_funcs=[et1_country,country_et2,poss_pronoun_per,poss_pronoun_relword,per_relword,per_org,per_nns,poss_title,per_title,nnp_title]
    f = Featurizer(data, tree_funcs, feature_funcs, not all_args.answers)
    f.build_features()
    if all_args.answers:
        f.build_relation_class_vectors()
        f.write_multiple_vectors(all_args.output_dir, all_args.file_suffix)
    else:
        f.write_no_tag(all_args.output_dir, all_args.file_suffix)
    print "built your new feature vectors at {}".format(all_args.output_dir)