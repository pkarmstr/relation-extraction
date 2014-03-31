#!/usr/bin/python
"""
CAUTION: NOT TESTED

compute the accuracy of a relation classifier

Outputs overall precision, recall, and f-measure (like the original evaluator)
Also outputs precision, recall, and f-measure for the relation types in two ways:

1) Long labels: e.g., EMP-ORG.Employ-Executive.reverse
2) Shortened labels (aka "labelclass"): e.g., EMP-ORG
"""

import sys, re

if len(sys.argv) != 3:
    sys.exit("usage: evaluator_finegrained.py [gold_file][output_file]")

#gold standard file
goldfh = open(sys.argv[1], 'r')
#system output
testfh = open(sys.argv[2], 'r')

gold_tag_list = []
#gold_word_list = []
test_tag_list = []

emptyline_pattern = re.compile(r'^\s*$')

for gline in goldfh.readlines():
    if not emptyline_pattern.match(gline):
        parts = gline.split()
        #print parts
        gold_tag_list.append(parts[0])


for tline in testfh.readlines():
    if not emptyline_pattern.match(tline):
        parts = tline.split()
        #print parts
        test_tag_list.append(parts[0])

test_total = 0
test_labelclass_total = 0
gold_total = 0
gold_labelclass_total = 0
correct_longlabels = 0
correct_labelclass = 0

#print gold_tag_list
#print test_tag_list

label_dict={}

for i in range(len(gold_tag_list)):
    #print gold_tag_list[i]
    #print test_tag_list[i]
    for j in range(len(gold_tag_list[i])):
        gold_tag = gold_tag_list[i][j]
        test_tag = test_tag_list[i][j]
        gold_tag_labelclass=gold_tag_list[i][j].split('.')[0]
        test_tag_labelclass=test_tag_list[i][j].split('.')[0]
        if gold_tag != "no_rel":
            gold_total += 1
            gold_labelclass_total += 1
            try:
                label_dict[gold_tag][1] += 1
                label_dict[gold_tag_labelclass][1] += 1
            except KeyError:
                label_dict[gold_tag] = [0, 1, 0]
                label_dict[gold_tag_labelclass] = [0, 1, 0]
        if test_tag != "no_rel":
            test_total += 1
            test_labelclass_total += 1
            try:
                label_dict[test_tag][0] += 1
                label_dict[test_tag_labelclass][0] += 1
            except KeyError:
                label_dict[test_tag] = [1, 0, 0]
                label_dict[test_tag_labelclass] = [1, 0, 0]
        if gold_tag != "no_rel" and gold_tag == test_tag:
            correct_longlabels += 1
            try:
                label_dict[gold_tag][2] += 1
            except KeyError: #this should never happen though
                label_dict[gold_tag] = [0, 0, 1]
        if gold_tag != "no_rel" and gold_tag_labelclass == test_tag_labelclass:
            correct_labelclass += 1
            try:
                label_dict[gold_tag_labelclass][2] += 1
            except KeyError: #this should never happen though
                label_dict[gold_tag_labelclass] = [0, 0, 1]


precision_longlabels = float(correct_longlabels) / test_total
recall_longlabels = float(correct_longlabels) / gold_total
f_longlabels = precision_longlabels * recall_longlabels * 2 / (precision_longlabels + recall_longlabels)

print
print "          " + "\t" + "Precision\t" + "Recall\t" + "F-Measure"
print "LONG" + "\t\t" + str(round(precision_longlabels,2)) + "\t\t"+ \
      str(round(recall_longlabels,2)) + "\t"+ str(round(f_longlabels,2))

for label in sorted(label_dict.keys()):

    if label_dict[label][0] != 0:
        p = \
           float(label_dict[label][2]) / label_dict[label][0]
    else:
        p = 0.0

    if label_dict[label][1] != 0:
        r = \
           float(label_dict[label][2]) / label_dict[label][1]
    else:
        r = 0.0

    if (p + r) != 0:
        f = p * r * 2 / (p + r)
    else:
        f = 0.0

    print label + "\t\t" + str(round(p,2)) + "\t\t"+ \
      str(round(r,2)) + "\t"+ str(round(f,2))