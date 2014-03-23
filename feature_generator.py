# coding=utf-8
"""Necessità 'l ci 'nduce, e non diletto."""
import re
from feature_functions import *
from file_reader import FeatureRow, feature_list_reader

__author__ = 'keelan'

import codecs
import argparse

class Featurizer:
    INT_INDEXES = [2, 3, 4, 7, 8, 9]

    def __init__(self, file_path, features, no_tag=False):
        self.file_path = file_path
        self.feature_functions = features
        self.no_tag = no_tag
        self.original_data = self.get_original_data()

    def prepare_line(self, line):
        line = line.rstrip().split()
        if self.no_tag:
            line.insert(0, "")
        del line[12]
        del line[6]
        i_token = line[6]
        j_token = line[11]
        for i in self.INT_INDEXES:
            line[i] = int(line[i])
        line.append(self._clean(i_token))
        line.append(self._clean(j_token))

        return line


    def get_original_data(self):
        gold_data = []
        with codecs.open(self.file_path, "r") as f_in:
            for line in f_in:
                prepped = self.prepare_line(line)
                gold_data.append(FeatureRow(*prepped))

        return gold_data

    def _clean(self, token):
        """
        1) Removes non-alpha (but not the "-") from the beginning of the token
        2) Removes possessive 's from the end
        3) Removes O', d', and ;T from anywhere (O'Brien becomes Brien, d'Alessandro becomes Alessandro, etc.)
        """
        return [re.sub(r"\W", r"", word) for word in token.split("_")]

    def build_features(self):
        self.new_features = []
        for feats in self.original_data:
            new_row = []
            for func in self.feature_functions:
                new_row.append(func(feats))

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
    f = Featurizer(all_args.input_file, feature_funcs, not all_args.answers)
    f.build_features()
    f.write_new_features(all_args.output_file)
    print "built your new feature vectors at {}".format(all_args.output_file)