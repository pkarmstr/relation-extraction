__author__ = 'anya'

"""
NOTE:
There is one tree that gives an error:

article: 'NYT20001229.2047.0291'
sentence_id: 21

Problem: comma after "Washington, D.C." gets removed

Can we just hardcode it when we make path-enclosed trees?
"""

import os
import re
import codecs
from nltk.tree import Tree, ParentedTree
from operator import itemgetter
from file_reader import *


def augment_tree(t, sent, article):
    """
    Given an nltk Tree, replaces named entities with a constituent that has a node <entity_type>.
    Example:

    ORIGINAL:
    (S (NP (NNP Barack) (NNP Obama)) (VP (VBD sang)))

    AUGMENTED:
    (S (NP (E-PER (NNP Barack) (NNP Obama))) (VP (VBD sang)))

    For now, handling only the easy case where all tokens of an entity are dominated by one NP
    """

    #sort the tuples in descending order first;
    #this seems to help with the index-shifting problem
    tuple_list=entity_types[article][sent]
    reverse_sorted_tuples=sorted(tuple_list, key = itemgetter(0), reverse = True)

    for tpl in reverse_sorted_tuples:
        entity_type=entity_types[article][sent][tpl]
        _add_entity(t,tpl,entity_type)

def _add_entity(t,tpl,entity_type):
    """
    Does the work of adding the entity-type node
    """

    parent_positions=[]
    parents=[]

    first_parent_position=t.leaf_treeposition(tpl[0])[:-1]
    first_grandparent_position=first_parent_position[:-1]

    for i in range(tpl[0],tpl[-1]):
        parent_position=t.leaf_treeposition(i)[:-1]
        parent=t[parent_position]
        parent_positions.append(parent_position)
        parents.append(parent)

    if 'parent_position' in locals():
        grandparent_position=parent_position[:-1]
        grandparent=t[grandparent_position]

        if grandparent_position==first_grandparent_position:
            # augment the nodes ONLY if every token in the mention has the same grandparent
            # i.e., if 'Barack Hussein Obama' is one NP, replace it with (NP (E-PER (NNP Barack)(NNP Hussein)(NNP Obama)))
            # but if we have "National Rifle" in one NP and "Association" in another NP, we don't bother adding E-ORG at all
            # (hopefully that doesn't exclude too many trees)
            aug_node='E-'+entity_type

            new_tree=Tree(aug_node,parents)

            if len(parent_positions)>1:
                if parent_positions[-1][-1]!=len(grandparent.leaves())-1: #if the last member of the tuple is NOT the rightmost child
                    #giving up on slices; collecting all of gp's children, then adding b
                    new_leaves=new_tree.leaves()
                    new_kids=[]
                    for kid in grandparent:
                        if kid[0] not in new_leaves:
                            new_kids.append(kid)
                        elif kid[0]==new_leaves[0]:
                            new_kids.append(new_tree)
                        else:
                            pass
                    new_grandparent=Tree(grandparent.node,new_kids)
                    ggparent=t[grandparent_position[:-1]]
                    ggparent[grandparent_position[-1]]=new_grandparent
                else: #it is the rightmost child
                    grandparent[parent_positions[0][-1]:len(grandparent.leaves())]=[new_tree]
            else: #one-word node
                grandparent[parent_positions[0][-1]]=new_tree


if __name__ == "__main__":

    outfile=codecs.open('test_tree_converter.txt','w','utf-8')
    #outfile=codecs.open('test_tree_converter_onesent.txt','w','utf-8')

    for article in entity_types:
    #for article in ['NYT20001229.2047.0291']:
        print article
        outfile.write('-----------------------------------\n')
        outfile.write(article + '\n')
        outfile.write('-----------------------------------\n')
        for sent_id in entity_types[article]:
        #for sent_id in [21]:
            print sent_id
            outfile.write('-----------------------------------\n')
            outfile.write("sent_id=" + str(sent_id) + '\n')
            outfile.write('-----------------------------------\n')
            test_tree=NONPARENTED_SENTENCES[article][sent_id]
            test_tree_copy=test_tree.copy(deep=True)
            len_before=len(test_tree_copy.leaves())
            print "len of tree before = ", len_before

            """print "TREE BEFORE:"
            print test_tree_copy
            print
            print"""

            augment_tree(test_tree_copy,sent_id,article)

            """print "TREE AFTER:"
            print test_tree_copy
            print
            print"""

            len_after=len(test_tree_copy.leaves())
            print "len of tree after = ", len_after
            print "Kept the length? ", len_before==len_after
            print "Length increased? ", len_before<len_after
            print
            outfile.write(test_tree_copy.pprint())
            outfile.write('\n\n')

    outfile.close()







