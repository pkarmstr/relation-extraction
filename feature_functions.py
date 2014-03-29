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

def path_enclosed_tree(fr):
    """****MONSTER FUNCTION!!!!****"""

    if fr.i_sentence!=fr.j_sentence:
        return "Not in same sentence" #just in case
    else:
        #find lowest common ancestor
        s_tree = SYNTAX_PARSE_SENTENCES[fr.article][int(fr.i_sentence)]
        ##testing Anya's augmented trees:
        #augtree1="(ROOT (S (S (S (NP (E-PER (NNP Michele) (NNP Roy))) (VP (VBD was) (RB not) (VP (VBN hurt) (PP (IN during) (NP (NP (DT the) (NN dispute)) (PP (IN at) (NP (PRP$ their) (NN home))) (NP-TMP (RB early) (NNP Sunday))))))) (, ,) (CC but) (S (NP (NNP Roy)) (VP (VP (VBD admitted) (S (VP (VBG pulling) (NP (DT a) (NN bedroom) (NN door)) (PP (IN off) (NP (NP (PRP$ its) (NNS hinges))(CC and) (JJ damaging) (NP (DT another)))) (PP (IN after) (NP (NP (PRP$ his) (NN wife)) (VP (VBN called) (NP (E-ORG (NNP Greenwood) (NNP Village) (NN police))))))))) (CC and) (VP (VBD hung) (PRT (RP up)) (PP (IN without) (NP (NN speaking))))))) (, ,) (NP (DT the) (NN report)) (VP (VBD said)) (. .)))"
        #augtree2 ="(ROOT (S (NP (NP (NP (E-GPE (NNP WASHINGTON))) (PRN (-LRB- -LRB-) (NP (NNP AP)) (-RRB- -RRB-))) (SBAR (S (NP (CD __) (NNPS Republicans)) (VP (VBP give) (NP (E-PER (NNP George) (NNP W.) (NNP Bush))) (NP (NP (NN credit)) (PP (S (VP (VBG promoting) (NP (DT a) (JJ Russian) (NN role)) (PP (IN in) (S (VP (VBG smoothing) (NP (DT the) (NN transition)) (PP (IN from) (NP (NN despot))) (PP(TO to) (NP (NP (NN democrat)) (PP (IN in) (NP (NNP Yugoslavia)))))))))))))))) (VP (VBD _) (NP (NP (DT an) (NN idea)) (VP (VBN dismissed) (PP (IN in) (NP (NN debate))) (PP (IN as) (ADJP (JJ risky))) (PP (IN by) (NP (NNP Al) (NNP Gore))))) (SBAR (RB even) (IN as) (S (NP (PRP$ his) (NN boss)) (VP (VBD was) (VP (VBG trying) (S (VP (TO to) (VP (VB get) (S (NP (NNP Moscow)) (VP (TO to) (VP (VB step) (PP (IN in))))))))))))) (. .)))"
        #s_tree = ParentedTree.parse(augtree1)
        #print "printing leaves corresponding to indices"
        #print s_tree.leaves()[int(fr.i_offset_begin)] #checking indices first...
        #print s_tree.leaves()[int(fr.j_offset_begin)]

        if int(fr.j_offset_begin)>int(fr.i_offset_begin):
            first_entity_index = int(fr.i_offset_begin)
            later_entity_index = int(fr.j_offset_end)-1
            first_token = fr.i_token
            later_token = fr.j_token

        else:
            first_entity_index = int(fr.j_offset_begin)
            later_entity_index = int(fr.i_offset_end)-1
            first_token = fr.j_token
            later_token = fr.i_token

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


