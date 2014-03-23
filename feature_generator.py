"""Necessità 'l ci 'nduce, e non diletto."""
from feature_functions import *

__author__ = 'keelan'

import codecs
import argparse

def feature_list_reader(file_path):
    ls = []
    with open(file_path, "r") as f_in:
        for line in f_in:
            ls.append(eval(line.rstrip()))

    return ls

class Featurizer:
    def __init__(self, file_path, features, no_tag=False):
        self.file_path = file_path
        self.feature_functions = features
        self.no_tag = no_tag
        self.original_data = self.get_original_data()


    def get_original_data(self):
        gold_data = []
        with codecs.open(self.file_path, "r") as f_in:
            for line in f_in:
                line = line.rstrip().split()
                i_token = line[5]
                j_token = line[10]
                line[1] = int(line[1])
                line[2] = int(line[2])
                line[3] = int(line[3])
                line[6] = int(line[6])
                line[7] = int(line[7])
                line[8] = int(line[8])
                if self.no_tag:
                    line.append(self._clean(i_token))
                    line.append(self._clean(j_token))
                    line.append("")
                else:
                    line.insert(-1, self._clean(i_token))
                    line.insert(-1, self._clean(j_token))
                gold_data.append(FeatureRow(*line))

        return gold_data

    def _clean(self, token):
        """
        1) Removes non-alpha (but not the "-") from the beginning of the token
        2) Removes possessive 's from the end
        3) Removes O', d', and ;T from anywhere (O'Brien becomes Brien, d'Alessandro becomes Alessandro, etc.)
        """
        return "_".join([re.sub(r"\W", r"", word) for word in token.split("_")])

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
        feature_funcs.insert(0, is_coreferent)
    f = Featurizer(all_args.input_file, feature_funcs, not all_args.answers)
    f.build_features()
    f.write_new_features(all_args.output_file)
    print "built your new feature vectors at {}".format(all_args.output_file)