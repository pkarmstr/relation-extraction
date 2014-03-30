# coding=utf-8
"""Necessità 'l ci 'nduce, e non diletto."""
import re
from feature_functions import *
from file_reader import FeatureRow, feature_list_reader, get_original_data

__author__ = 'keelan'

import argparse

class Featurizer:

    def __init__(self, original_data, tree_functions, features, no_tag=False):
        self.file_path = file_path
        self.tree_functions = tree_functions
        self.feature_functions = list(enumerate(features))
        self.no_tag = no_tag
        self.original_data = original_data

    def build_features(self):
        self.new_features = []
        for feats in self.original_data:
            new_row = []
            for func in self.tree_functions:
                new_row.append("|BT|")
                new_row.append(func(feats))
            new_row.append("|ET|")
            for i,func in self.feature_functions:
                cell = func(feats)
                if cell.split("=")[1] == "True":
                    new_row.append("{:d}:{:d}".format(i,1))

            new_row.append("|EV|")
            self.new_features.append(new_row)

    def write_new_features(self, file_path):
        with open(file_path, "w") as f_out:
            for row in self.new_features:
                f_out.write("{}\n".format(" ".join(row)))

if __name__ == "__main__":
    from feature_functions import *
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    parser.add_argument("output_file")
    parser.add_argument("feature_list")
    parser.add_argument("-a", "--answers", help="the input file has the answers", action="store_true")

    all_args = parser.parse_args()

    feature_funcs = []
    feature_funcs.extend(feature_list_reader(all_args.feature_list))
    if all_args.answers:
        feature_funcs.insert(0, relation_type)
    data = get_original_data(args.input_file)
    f = Featurizer(all_args.input_file, feature_funcs, not all_args.answers)
    f.build_features()
    f.write_new_features(all_args.output_file)
    print "built your new feature vectors at {}".format(all_args.output_file)