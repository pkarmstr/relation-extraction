#!/bin/sh
PYTHONPATH="$PYTHONPATH:/home/g/grad/pkarmstr/python27/lib/python2.7/site-packages"
export PYTHONPATH
EXPERIMENT_DIR="$1"
TYPE="$2"
PYTHON="/home/g/grad/pkarmstr/python27/bin/python2.7"
REL="/home/g/grad/pkarmstr/infoextraction/relation-extraction"

# build all of the feature vectors

touch $EXPERIMENT_DIR/tree_list.txt

$PYTHON $REL/feature_generator.py resources/cleaned-train.gold \
	$EXPERIMENT_DIR train.gold $EXPERIMENT_DIR/tree_list.txt $EXPERIMENT_DIR/feature_list.txt -a -m

awk '{print $NF}' $REL/resources/REL-"$TYPE"set.gold > $EXPERIMENT_DIR/$TYPE.gold

$PYTHON $REL/feature_generator.py resources/cleaned-"$TYPE".notag \
	$EXPERIMENT_DIR $TYPE.notag $EXPERIMENT_DIR/feature_list.txt -m

# training
$REL/mallet-maxent-classifier.sh -train \
	-model=$EXPERIMENT_DIR/model \
	-gold=$EXPERIMENT_DIR/train.gold

# testing - doesn't work yet?
$REL/mallet-maxent-classifier.sh -classify  \
	-model=$EXPERIMENT_DIR/model \
	-input=$EXPERIMENT_DIR/$TYPE.notag > $EXPERIMENT_DIR/$TYPE.tagged

# evaluation
python $REL/evaluator_finegrained.py $EXPERIMENT_DIR/$TYPE.gold \
	$EXPERIMENT_DIR/$TYPE.tagged > $EXPERIMENT_DIR/"$TYPE"_eval.txt

echo Finished everything, results at $EXPERIMENT_DIR/"$TYPE"_eval.txt
