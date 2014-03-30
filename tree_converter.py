__author__ = 'anya'

import os
import re
import codecs
from nltk.tree import Tree, ParentedTree
from operator import itemgetter
from file_reader import *

punct=set(['-LRB-','-RRB-','``','`','_','__','-','--','"'])

"""fake_dict={}
sent_dict={}
entity_dict={}
entity_dict2={}

entity_dict[(0,1)]='GPE'
entity_dict[(1,3)]='ORG'
entity_dict[(7,10)]='PER'

entity_dict2[(0,2)]='PER'
entity_dict2[(2,5)]='FAKEV'
entity_dict2[(31,34)]='ORG'

sent_dict[3]=entity_dict
sent_dict[4]=entity_dict2

fake_dict['APW20001007.0339.0149']=sent_dict"""

test_tree=Tree("(ROOT (S (NP (NP (NP (NNP WASHINGTON)) (PRN (-LRB- -LRB-) (NP (NNP AP)) (-RRB- -RRB-))) (SBAR (S (NP (CD __) (NNPS Republicans)) (VP (VBP give) (NP (NNP George) (NNP W.) (NNP Bush)) (NP (NP (NN credit)) (PP (IN for) (S (VP (VBG promoting) (NP (DT a) (JJ Russian) (NN role)) (PP (IN in) (S (VP (VBG smoothing) (NP (DT the) (NN transition)) (PP (IN from) (NP (NN despot))) (PP (TO to) (NP (NP (NN democrat)) (PP (IN in) (NP (NNP Yugoslavia)))))))))))))))) (VP (VBD _) (NP (NP (DT an) (NN idea)) (VP (VBN dismissed) (PP (IN in) (NP (NN debate))) (PP (IN as) (ADJP (JJ risky))) (PP (IN by) (NP (NNP Al) (NNP Gore))))) (SBAR (RB even) (IN as) (S (NP (PRP$ his) (NN boss)) (VP (VBD was) (VP (VBG trying) (S (VP (TO to) (VP (VB get) (S (NP (NNP Moscow)) (VP (TO to) (VP (VB step) (PP (IN in))))))))))))) (. .)))")

test_tree2=Tree("(ROOT (S (S (S (NP (NNP Michele) (NNP Roy)) (VP (VBD was) (RB not) (VP (VBN hurt) (PP (IN during) (NP (NP (DT the) (NN dispute)) (PP (IN at) (NP (PRP$ their) (NN home))) (NP-TMP (RB early) (NNP Sunday))))))) (, ,) (CC but) (S (NP (NNP Roy)) (VP (VP (VBD admitted) (S (VP (VBG pulling) (NP (DT a) (NN bedroom) (NN door)) (PP (IN off) (NP (NP (PRP$ its) (NNS hinges)) (CC and) (JJ damaging) (NP (DT another)))) (PP (IN after) (NP (NP (PRP$ his) (NN wife)) (VP (VBN called) (NP (NNP Greenwood) (NNP Village) (NN police)))))))) (CC and) (VP (VBD hung) (PRT (RP up)) (PP (IN without) (NP (NN speaking))))))) (, ,) (NP (DT the) (NN report)) (VP (VBD said)) (. .)))")

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

    #for tpl in fake_dict[article][sent]:
    #for tpl in entity_types[article][sent]:
    for tpl in reverse_sorted_tuples:
        #entity_type=fake_dict[article][sent][tpl]
        entity_type=entity_types[article][sent][tpl]
        _add_entity(t,tpl,entity_type)

def _add_entity(t,tpl,entity_type):
    """
    Does the work of adding the entity-type node
    """

    parent_positions=[]
    parents=[]

    #print "len of tree ", len(t.leaves())
    #print tpl[0]
    #print t.leaf_treeposition(tpl[0])

    #first_parent_position=t.leaf_treeposition(tpl[0])[:-1] #need to change this; first parent should be the parent of a regular word
    first_parent_position=0
    first_grandparent_position=0

    for j in range(tpl[0],tpl[-1]):
        if _is_regular_word(t,j):
            first_parent_position=t.leaf_treeposition(j)[:-1]
            first_grandparent_position=first_parent_position[:-1]
            break

    for i in range(tpl[0],tpl[-1]):
        if _is_regular_word(t,i):
            parent_position=t.leaf_treeposition(i)[:-1]
            #print parent_position
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
                    grandparent[parent_positions[0][-1]:len(parent_positions)]=[new_tree]
                else:
                    grandparent[parent_positions[0][-1]:len(parent_positions)+1]=[new_tree]
            else:
                grandparent[parent_positions[0][-1]]=new_tree

def _is_regular_word(t,index):
    """
    We don't want to include the following in our entity nodes:
        - backticks
        - quotation marks
        - dashes
        - underscores
        - parentheses
    """
    return t.leaves()[index] not in punct

if __name__ == "__main__":

    outfile=codecs.open('test_tree_converter.txt','w','utf-8')
    #outfile=codecs.open('test_tree_converter_onesent.txt','w','utf-8')

    for article in entity_types:
    #for article in ['NYT20001111.1247.0093']:
        print article
        outfile.write('-----------------------------------\n')
        outfile.write(article + '\n')
        outfile.write('-----------------------------------\n')
        for sent_id in entity_types[article]:
        #for sent_id in [13]:
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
            print"""
           # print test_tree_copy.__repr__()
            augment_tree(test_tree_copy,sent_id,article)
            """print "TREE AFTER:"
            print test_tree_copy
            print"""
            len_after=len(test_tree_copy.leaves())
            print "len of tree after = ", len_after
            print "Kept the length? ", len_before==len_after
            print
            outfile.write(test_tree_copy.pprint())
            outfile.write('\n\n')

    outfile.close()







