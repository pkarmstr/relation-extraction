__author__ = 'keelan'

import sys
import os
import shlex
import linecache
from operator import itemgetter
from os.path import join
from subprocess import Popen, PIPE
from feature_generator import Featurizer
from file_reader import feature_list_reader, get_original_data
from feature_functions import *

class Pipeline:

    def __init__(self, basedir, svmdir, type="dev"):
        self.basedir = basedir
        self.svmdir = svmdir
        self.type = type
        self.svm_args = "-t 5"
        self.training_data = get_original_data("resources/cleaned-train.gold")
        self.test_data = get_original_data("resources/cleaned-{:s}.notag".format(type))

    def set_up(self):
        basedir_contents = os.listdir(self.basedir)
        files = [f for f in basedir_contents if os.path.isfile(join(self.basedir, f))]
        directories = [f for f in basedir_contents if os.path.isdir(join(self.basedir, f))]

        if "feature_list.txt" not in files or\
            "tree_list.txt" not in files:
            sys.exit("You need feature_list.txt and tree_list.txt in the base directory")
        if "gold_files" not in directories:
            os.mkdir(join(self.basedir, "gold_files"))
        if "models" not in directories:
            os.mkdir(join(self.basedir, "models"))
        if "tagged_files" not in directories:
            os.mkdir(join(self.basedir, "tagged_files"))
        self.tree_funcs = feature_list_reader(
            join(self.basedir, "tree_list.txt"), globals()
        )
        self.feature_funcs = feature_list_reader(
            join(self.basedir, "feature_list.txt"), globals()
        )

    def build_features(self):
        f_training = Featurizer(self.training_data, self.tree_funcs, self.feature_funcs)
        f_training.build_features()
        f_training.build_relation_class_vectors()
        f_training.write_multiple_vectors(join(self.basedir, "gold_files"), "train.gold")

        f_test = Featurizer(self.test_data, self.tree_funcs, self.feature_funcs, no_tag=True)
        f_test.build_features()
        f_test.write_no_tag(self.basedir, "{:s}.notag".format(self.type))

    def run_svm_learn(self):
        processes = []
        svm_learn = join(self.svmdir, "svm_learn")
        for gold_file in os.listdir(join(self.basedir, "gold_files")):
            prefix = gold_file.split("-train.gold")[0]
            train = join(self.basedir, "gold_files", gold_file)
            model = join(self.basedir, "models", "{:s}.model".format(prefix))
            args = shlex.split("{:s} {:s} {:s} {:s}".format(svm_learn, self.svm_args, train, model))
            p = Popen(args, stdout=PIPE)
            processes.append(p)
            print "Building the model for {:s}".format(prefix)
        for p in processes:
            p.wait()

    def run_svm_classify(self):
        processes = []
        svm_classify = join(self.svmdir, "svm_classify")
        for model in os.listdir(join(self.basedir, "models")):
            prefix = model.split(".model")[0]
            model = join(self.basedir, "models", model)
            notag = join(self.basedir, "{:s}.notag".format(self.type))
            tagged = join(self.basedir, "tagged_files", "{:s}.tagged".format(prefix))
            args = shlex.split("{:s} {:s} {:s} {:s}".format(svm_classify, notag, model, tagged))
            p = Popen(args, stdout=PIPE)
            processes.append(p)
            print "Classifying with the {:s} model".format(prefix)
        for p in processes:
            p.wait()

    def _prepare_line(self, line):
        return float(line.rstrip())

    def translate_svm_output(self):
        relation_class = ["PHYS", "PER-SOC", "OTHER-AFF", "GPE-AFF", "DISC", "ART", "EMP-ORG"]
        file_locations = [join(self.basedir, "tagged_files", "{:s}.tagged".format(f)) for f in relation_class]
        relation_ids = zip(relation_class, file_locations)
        max_type = []
        class_indices = []
        with open(join(self.basedir, "tagged_files", "no_rel.tagged"), "r") as f_in:
            for i,line in enumerate(f_in):
                val = self._prepare_line(line)
                if val <= 0.5:
                    class_indices.append(i)
                    max_type.append(("no_rel", -10000)) #giant magic number
                else:
                    max_type.append(("no_rel", val))

        for rel_class,f in relation_ids:
            for index in class_indices:
                val = self._prepare_line(linecache.getline(f, index))
                if val > max_type[index][1]:
                    max_type[index] = (rel_class, val)
            linecache.clearcache()
        return zip(*max_type)[0]

    def evaluate(self):
        output = join(self.basedir, "final_output.tagged")
        gold = "resources/cleaned-{:s}.gold".format(self.type)
        redirect = "> {:s}/eval.txt".format(self.basedir)
        args = shlex.split("{:s} {:s} {:s} {:s} {:s}".format("python2.7",
                                                             "evaluator_finegrained.py",
                                                             gold,
                                                             output,
                                                             redirect))
        p = Popen(args, stdout=PIPE)
        p.wait()

    def run(self):
        print "setting up...",
        self.set_up()
        print "[DONE]"
        print "building features..."
        self.build_features()
        print "built all features"
        print "building the models..."
        self.run_svm_learn()
        print "built all of the models"
        print "classifying across all models..."
        self.run_svm_classify()
        print "finished classification..."
        answers = self.translate_svm_output()
        with open(join(self.basedir, "final_output.tagged"), "w") as f_out:
            f_out.write("\n".join(answers))
        print "wrote the output into final_output.tagged [DONE]"
        self.evaluate()
        print "wrote the evaluation into eval.txt"



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("basedir", help="the directory in which the experiment wil be")
    parser.add_argument("svmdir", help="the directory in which the svm binaries live")
    parser.add_argument("type", help="dev or test")
    args = parser.parse_args()

    pl = Pipeline(args.basedir, args.svmdir, args.type)
    pl.run()
