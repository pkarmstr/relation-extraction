__author__ = 'keelan'

from file_reader import RAW_SENTENCES, SYNTAX_PARSE_SENTENCES, POS_SENTENCES, FeatureRow
import re
from nltk.tree import ParentedTree

###################
# basic functions #
###################

def relation_type(fr):
    return "relation_type={}".format(fr.relation_type)


####################
# Anya's functions #
####################




######################
# Keelan's functions #
######################




#####################
# Julia's functions #
#####################


def i_token(fr):
    """return the i_token"""
    return "i_token={}".format(fr.i_token)

def j_token(fr):
    """ return the j_token"""
    return "j_token={}".format(fr.j_token)

def i_entity_type(fr):
    """return i_entity type"""
    return "i_entity_type={}".format(fr.i_entity_type)

def j_entity_type(fr):
    """return j_i_entity_type"""
    return "j_entity_type={}".format(fr.i_entity_type)

def _get_mentions_in_order_(fr):
    """return a pair of tuples. The first one corresponds to mention1 and its info,
    the second one to mention2 and its info. i = mention1 and j=mention2 don't always hold"""
    if int(fr.i_offset_begin)<int(fr.j_offset_begin):
        mention1 = (fr.i_token,fr.i_offset_begin,fr.i_offset_end, fr.i_entity_type, fr.i_sentence)
        mention2 = (fr.j_token,fr.j_offset_begin,fr.j_offset_end, fr.j_entity_type,fr.j_sentence)
    else:
        mention2 = (fr.i_token,fr.i_offset_begin,fr.i_offset_end, fr.i_entity_type, fr.i_sentence)
        mention1 = (fr.j_token,fr.j_offset_begin,fr.j_offset_end, fr.j_entity_type, fr.j_sentence)
    return (mention1,mention2)

def bow_mention1(fr):
    """return the words in mention2 eg. [George,Bush]"""
    mention1 = _get_mentions_in_order_(fr)[0]
    sent=POS_SENTENCES[fr.article][int(mention1[4])]
    mention_token = mention1[0]
    return "bow_mention1={}".format(mention_token.split("_"))

def bow_mention2(fr):
    """return the words in mention2"""
    mention2 = _get_mentions_in_order_(fr)[1]
    sent=POS_SENTENCES[fr.article][int(mention2[4])]
    mention_token = mention2[0]
    return "bow_mention1={}".format(mention_token.split("_"))

def _get_words_in_between_(fr):
    """return the words between m1 and m2"""
    sent=POS_SENTENCES[fr.article][int(fr.i_sentence)]
    mention1 =  _get_mentions_in_order_(fr)[0]
    mention2 =  _get_mentions_in_order_(fr)[1]
    first_token_index = int(mention1[2])
    later_token_index = int(mention2[1])
    w_in_between = sent[first_token_index:later_token_index]
    return w_in_between

def first_word_inbetween(fr):
    """return the first word between m1 and m2"""
    return "first_word_inbetween={}".format(_get_words_in_between_(fr)[0][0])

def last_word_inbetween(fr):
    """return the last word between m1 and m2"""
    words = _get_words_in_between_(fr)
    return "last_word_inbetween={}".format(words[len(words)-1][0])

def other_words_inbetween(fr):
    """return words between m1 and m2 excluding the first and last words"""
    words = _get_words_in_between_(fr)
    words.pop(0)
    words.pop()
    return "other_words_inbetween={}".format([w for w, pos in words])

def first_word_before_m1(fr):
    """return first word before m1"""
    mention1 = _get_mentions_in_order_(fr)[0]
    sent=POS_SENTENCES[fr.article][int(mention1[4])]
    return "first_word_before_m1={}".format(sent[int(mention1[1])-1][0])


def first_word_before_m2(fr):
    """return first word before m2"""
    mention2 = _get_mentions_in_order_(fr)[1]
    sent=POS_SENTENCES[fr.article][int(mention2[4])]
    return "first_word_before_m2={}".format(sent[int(mention2[1])-1][0])

def second_word_before_m1(fr):
    """return second word before m1"""
    mention1 = _get_mentions_in_order_(fr)[0]
    sent=POS_SENTENCES[fr.article][int(mention1[4])]
    try:
        return "second_word_before_m1={}".format(sent[int(mention1[1])-2][0])
    except:
        IndexError
        return "second_word_before_m1=NONE"

def second_word_before_m2(fr):
    """return second word before m2"""
    mention2 = _get_mentions_in_order_(fr)[1]
    sent=POS_SENTENCES[fr.article][int(mention2[4])]
    try:
        return "second_word_before_m2={}".format(sent[int(mention2[1])-2][0])
    except:
        IndexError
        return "second_word_before_m2=NONE"



def path_enclosed_tree(fr):
    """****MONSTER FUNCTION!!!!****
    Return the path enclosed tree between m1 and m2 as PatentedTree
    The path enclosed tree is the smallest common
    sub-tree including the two entities, but not necessary the lowest_common_ancestor. In other
    words, the sub-tree is enclosed by the shortest
    path linking the two entities in the parse tree (this
    path is also commonly-used as the path tree feature
i   n the feature-based methods).

    That is, the path enclosed tree includes mention1, and every branch to the right of it, until
    mention2. In this function, the path enclosed tree built in the following way:
    the left branch of it includes mention1 and branches to right of it that still are on the left child
    of the lowest common ancestor. The right branch of the path-enclosed tree includes mention2, and the
    branches to the right of it that are on the right child of the lowest common ancestor.
    Both branches are merged in one tree, with the lowest_common_ancestor node, yielding the
    path eclosed tree.
    """


    if fr.i_sentence!=fr.j_sentence:
        return "Not in same sentence" #just in case
    else:
        #find lowest common ancestor
        s_tree = SYNTAX_PARSE_SENTENCES[fr.article][int(fr.i_sentence)]

        ##testing Anya's augmented trees:
        #augtree1="(ROOT (S (S (S (NP (E-PER (NNP Michele) (NNP Roy))) (VP (VBD was) (RB not) (VP (VBN hurt) (PP (IN during) (NP (NP (DT the) (NN dispute)) (PP (IN at) (NP (PRP$ their) (NN home))) (NP-TMP (RB early) (NNP Sunday))))))) (, ,) (CC but) (S (NP (NNP Roy)) (VP (VP (VBD admitted) (S (VP (VBG pulling) (NP (DT a) (NN bedroom) (NN door)) (PP (IN off) (NP (NP (PRP$ its) (NNS hinges))(CC and) (JJ damaging) (NP (DT another)))) (PP (IN after) (NP (NP (PRP$ his) (NN wife)) (VP (VBN called) (NP (E-ORG (NNP Greenwood) (NNP Village) (NN police))))))))) (CC and) (VP (VBD hung) (PRT (RP up)) (PP (IN without) (NP (NN speaking))))))) (, ,) (NP (DT the) (NN report)) (VP (VBD said)) (. .)))"
        #augtree2 ="(ROOT (S (NP (NP (NP (E-GPE (NNP WASHINGTON))) (PRN (-LRB- -LRB-) (NP (NNP AP)) (-RRB- -RRB-))) (SBAR (S (NP (CD __) (NNPS Republicans)) (VP (VBP give) (NP (E-PER (NNP George) (NNP W.) (NNP Bush))) (NP (NP (NN credit)) (PP (S (VP (VBG promoting) (NP (DT a) (JJ Russian) (NN role)) (PP (IN in) (S (VP (VBG smoothing) (NP (DT the) (NN transition)) (PP (IN from) (NP (NN despot))) (PP(TO to) (NP (NP (NN democrat)) (PP (IN in) (NP (NNP Yugoslavia)))))))))))))))) (VP (VBD _) (NP (NP (DT an) (NN idea)) (VP (VBN dismissed) (PP (IN in) (NP (NN debate))) (PP (IN as) (ADJP (JJ risky))) (PP (IN by) (NP (NNP Al) (NNP Gore))))) (SBAR (RB even) (IN as) (S (NP (PRP$ his) (NN boss)) (VP (VBD was) (VP (VBG trying) (S (VP (TO to) (VP (VB get) (S (NP (NNP Moscow)) (VP (TO to) (VP (VB step) (PP (IN in))))))))))))) (. .)))"
        #augtree3 = "(ROOT (NP (NP (NP (E-GPE (NNP KABUL)) (, ,) (E-GPE (NNP Afghanistan))) (PRN (-LRB- -LRB-) (NP (NNP AP)) (-RRB- -RRB-))) (NP (NP (CD _)) (SBAR (S (NP (NP (DT The) (NN ruling) (E-ORG (NNP Taliban)) (E-ORG (NN militia))) (PP (IN on) (NP (NNP Monday)))) (VP (VP (VBD released) (NP (CD 137) (E-PER (JJ Shiite)) (E-PER (JJ Muslim)) (E-PER (NNS prisoners))) (SBAR (S (NP (E-ORG (PRP it))) (VP (VBD had) (VP (VBN held) (PP (IN for) (NP (QP (RB nearly) (CD two)) (NNS years)))))))) (CC and) (VP (VBD urged) (NP (DT the) (E-ORG (NN opposition))) (S (VP (TO to) (VP (VB follow) (NP (NP (NP (NN suit) (CC and) (NN release)) (E-ORG (NN government)) (E-PER (NNS prisoners))) (SBAR (S (NP (E-ORG (PRP it))) (VP (VBZ is) (VP (VBG holding)))))))))))))) (. .)))"
        #augtree4= "(ROOT (S (NP (DT The) (VBN freed) (E-PER (NNS men))) (PRN (, ,) (S (NP (DT all)) (VP (VBD said) (S (VP (TO to) (VP (VB be) (NP (NP (E-PER (NNS fighters))) (VP (VBG belonging) (PP (TO to) (NP (DT the) (E-ORG (NN opposition)) (E-ORG (NN alliance))))))))))) (, ,)) (VP (VBD were) (VP (VBN released) (ADVP (RB ahead)) (PP (IN of) (NP (NP (DT the) (E-PER (JJ Islamic)) (JJ holy) (NN month)) (PP (IN of) (NP (NNP Ramadan))))) (, ,) (SBAR (WHADVP (WRB when)) (S (NP (JJ devout) (E-PER (NNPS Muslims))) (ADVP (RB fast) (PP (IN from) (NP (NN sunrise)))) (VP (TO to) (VP (VB sunset))))))) (. .)))"
        #augtree7="(ROOT (S (NP (NP (DT The) (E-ORG (NN opposition)) (E-ORG (NN alliance))) (, ,) (SBAR (WHNP (E-ORG (WDT which))) (S (VP (VP (VBZ controls) (NP (NP (QP (RB barely) (CD five)) (NN percent)) (PP (IN of) (NP (E-GPE (NNP Afghanistan)))))) (CC and) (VP (VBZ is) (VP (VBG fighting) (NP (NP (DT a) (NN war)) (PP (IN against) (NP (DT the) (JJ dominant) (E-ORG (NNP Taliban)))))))))) (, ,)) (VP (VBZ is) (VP (ADVP (RB mostly)) (VBN made) (ADVP (IN up) (PP (IN of) (NP (NP (DT the) (E-GPE (NN country)) (POS 's)) (NN minority)))) (NP (ADJP (JJ ethnic) (CC and) (JJ religious)) (E-PER (NNS groups))))) (. .)))"
        #augtree33="(ROOT (SBARQ (SBAR (NP (NN __)) (IN If) (S (NP (E-ORG (NNP CBS))) (VP (VBZ shows) (NP (NP (DT the) (JJ first) (NN episode)) (PP (IN of) (`` ``) (NP (NNP Survivor) (NNP II)) ('' ''))) (PP (IN after) (NP (NP (E-ORG (PRP$ its)) (NN broadcast)) (PP (IN of) (NP (NNP Super) (NNP Bowl) (NNP XXXV)))))))) (, ,) (SQ (MD will) (NP (DT the) (E-ORG (NN network))) (VP (ADVP (RB someday)) (VBP show) (NP (NP (DT the) (JJ first) (NN episode)) (PP (IN of) (NP (`` ``) (NP (NNP Survivor) (NNP XXXV)) ('' '') (PP (IN after) (NP (NP (E-ORG (PRP$ its)) (NN broadcast)) (PP (IN of) (NP (NNP Super) (NNP Bowl) (NNP LXVIII)))))))))) (. ?)))"
        #augtree34="(ROOT (S (NP (NN __)) (VP (MD Will) (S (ADJP (JJ cheerful)) (SBAR (S (NP (NP (NN chorus) (E-PER (NNS members))) (VP (VBN dressed) (PP (IN in) (NP (JJ V-neck) (NNS sweaters))))) (VP (VBP interrupt) (NP (E-PER (PRP$ their)) (NNS serenades)) (PP (TO to) (NP (JJ online) (NN shopping))) (ADVP (RB long) (RB enough) (S (VP (TO to) (VP (VB tell) (NP (NP (DT an) (NN advertising) (E-PER (NN columnist))) (, ,) (S (`` ``) (NP (PRP You)) (VP (VBP ask) (NP (NP (DT a) (NN lot)) (PP (IN of) (NP (NP (NNS questions)) (PP (IN for) (NP (E-PER (NN someone))))))) (PP (IN from) (NP (E-GPE (NNP Brooklyn))))) ('' '')))))))))))) (. ?)))"
        #s_tree = ParentedTree.parse(augtree4)
        #print "printing leaves corresponding to indices"
        #print s_tree.leaves()[int(fr.i_offset_begin)] #checking indices first...
        #print s_tree.leaves()[int(fr.j_offset_begin)]

        mention1 = _get_mentions_in_order_(fr)[0]
        mention2= _get_mentions_in_order_(fr)[1]
        first_entity_index = int(mention1[1])
        later_entity_index = int(mention2[2])-1
        first_token = mention1[0]
        later_token = mention2[0]
        i_tuple = s_tree.leaf_treeposition(first_entity_index)
        j_tuple = s_tree.leaf_treeposition(later_entity_index)
        #print "printing indices"
        #print "Where first entity starts: ", first_entity_index
        #print "Index of last word of later entity: ", later_entity_index
        first_tree = s_tree[i_tuple[0:-1]]
        later_tree= s_tree[j_tuple[0:-1]]
        lwca_tuple=s_tree.treeposition_spanning_leaves(first_entity_index, later_entity_index)
        lowest_common_ancestor = s_tree[lwca_tuple]
        #lowest_common_ancestor.draw()


        ###The following 3 functions generate a tree where the left branch contains the M1 path (
        ##and all right branches, etc; the right branch contains the M2 path and
        ##the left branches (left to M2). Finally, add the trees that might be in the middle (eg.in
        ##case of ternary trees): (S left_brach, (,,), right_brach).

        def from_root_to_m1(pos_token_tree):
            """#Get the path from root to the entity mention1 and everything right to it
            up to the lowest_common_ancestor node"""

            #initiate left_branch with token and pos, and its
            #right siblings.
            children_to_add = []
            found = False
            same_subtree = False

            #Building the "proto" left_branch tree: add the (POS Mention1) tree and all its right siblings.
            # Don't add anything until Mention1 is found. Not going up yet
            for child in pos_token_tree.parent():
                j_in_leaves = len(set(first_token.split("_")).intersection(set(child.leaves())))>0
                if child == pos_token_tree: #eg. (JJ Republican)
                    children_to_add.append(child.copy(deep=True))
                    found = True
                elif child == later_tree: #M2 is in that same subtree!
                    children_to_add.append(child.copy(deep=True))
                    same_subtree = True #Eg. Mention1 = Republican and M2= candidate.
                    break #don't want to keep adding stuff after M2!
                elif j_in_leaves and not same_subtree:
                    break #M2 is deep embedded in tree sibling to (POS Mention1). #from_root_to_M2 will take care of it.
                elif found:
                    children_to_add.append(child.copy(deep=True))

            #proto left-branch eg.
            left_branch = ParentedTree(pos_token_tree.parent().node, children_to_add)

            #check whether M1 and M2 in same pre-leaf phrase (eg. NP Republican candidate)
            if same_subtree:
                return left_branch #no need to keep going upwards, this is the path-enclosed tree.
            else:
                if pos_token_tree.parent() == lowest_common_ancestor:
                    # (POS Mention1) will be the left branch of the path-enclosed tree.
                    left_branch = pos_token_tree.copy(deep=True)
                    return left_branch
                else:
                    #we have to go further up
                    subtree=pos_token_tree.parent()

                ##Keep going up, looping over the children of each parent, adding branches that are
                ##right to m1 until the lowest common ancestor is hit.
                found = False
                seen = False
                while isinstance(subtree.parent(),ParentedTree) and \
                                subtree.parent()!=lowest_common_ancestor:
                    children = []
                    children.append(left_branch)
                    for child in subtree.parent():
                        if child == subtree:
                            seen = True
                            found = True
                        elif found and seen: #= if m1 was found and the current subtree is on the right side of m1
                            children.append(child.copy(deep=True))
                    left_branch = ParentedTree(subtree.parent().node,children)
                    subtree = subtree.parent()
                    seen = False
                return left_branch

        def from_root_to_m2(pos_token_tree):
            """Get the path from root to the entity mention3 and the preceding branches"""
            if s_tree[i_tuple[:-2]]== s_tree[j_tuple[:-2]]: #tokens have the same parent
                return #from_root_to_m1 has taken care of this
            else:
            #initiate right branch with token and pos, and its
            #left siblings, if any eg. NNP W. NNP Bush
                children_to_add = []
                for child in pos_token_tree.parent():
                        if child == pos_token_tree:
                            children_to_add.append(child.copy(deep=True))
                            break
                        children_to_add.append(child.copy(deep=True))
                right_branch = ParentedTree(pos_token_tree.parent().node, children_to_add)
                subtree = pos_token_tree.parent()

                ##keep going upwards adding nodes and left branches, but ignoring right branches
                while isinstance(subtree.parent(),ParentedTree) and \
                                subtree.parent()!=lowest_common_ancestor:
                    children = []
                    for child in subtree.parent():
                        if child == subtree:
                            break
                        else:
                            children.append(child.copy(deep=True))
                    children.append(right_branch)
                    right_branch = ParentedTree(subtree.parent().node,children) #start from the bottom
                    subtree = subtree.parent()
                return right_branch

        def merge_both_branches(left_branch, right_branch):
            """Merge left and right branch with the lowest_common_ancestor_node"""
            if right_branch == None:
                result_tree = left_branch
            else:
                children = [left_branch]
                m1_visited = False
                m2_visited = False
                for child in lowest_common_ancestor:
                    i_in_leaves = len(set(first_token.split("_")).intersection(set(child.leaves())))>0
                    j_in_leaves = len(set(later_token.split("_")).intersection(set(child.leaves())))>0
                    if m2_visited and m1_visited:
                        break
                    if j_in_leaves:
                        m2_visited = True
                    elif m1_visited and not m2_visited:
                        children.append(child.copy(deep=True))
                    if i_in_leaves:
                        m1_visited = True
                children.append(right_branch)
                result_tree = ParentedTree(lowest_common_ancestor.node,children)
            return result_tree


        return merge_both_branches(from_root_to_m1(first_tree),from_root_to_m2(later_tree))


