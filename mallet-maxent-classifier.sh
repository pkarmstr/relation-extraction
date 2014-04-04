#!/bin/sh
#

BASEDIR=/home/j/clp/chinese/tools/mallet_maxent

# Mallet maxent classifier
#new Mallet:
#MALLET_HOME=/home/j/clp/chinese/tools/mallet-2.0.5
#MALLET_LIB=$MALLET_HOME/lib

#old mallet
MALLET_HOME=/home/j/clp/chinese/tools/lib/mallet
MALLET_LIB=$MALLET_HOME/lib

export JAVA=/usr/bin/java
export JAVAC=/usr/bin/javac
export CLASSPATH=$BASEDIR/classifier/classes:$MALLET_HOME/class:$MALLET_LIB/mallet-deps.jar


if [ "$1" = make ] 
then
    shift
    $JAVAC -d $BASEDIR/classifier/classes "$@"
else 
    $JAVA -mx3000m  MaxentClassifier "$@"
    
fi
