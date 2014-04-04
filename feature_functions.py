__author__ = 'keelan'

import re
from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import WordNetError as wn_error
from nltk.tree import Tree,ParentedTree
from file_reader import RAW_SENTENCES, SYNTAX_PARSE_SENTENCES, POS_SENTENCES, PRONOUN_SET, \
    entity_types, RELATIONSHIPS_AND_GROUPS, COUNTRIES, NATIONALITIES, OFFICIALS, PROFESSIONS, \
    TITLE_SET, POSSESSIVE_PRONOUNS, COREF, DEPENDENCIES, AUGMENTED_TREES


phrase_heads = {"PP":["IN"],
                "NP":['NN', 'NNS', 'NNP', 'NNPS', 'JJ', "PRP"], #JJ as NP for examples like "many of...".
                "VP":["VBD","VBZ","VB", "VBP","MD", "VBN", "VBP"],
                "ADJP": ["JJ"],
                "NP-TMP":['NN', 'NNS', 'NNP', 'NNPS'],
                "WHADVP":["WRB"],
                "WHNP":["WDT", "WP",],
                "ADVP":["RB"],
                }


###################
# basic functions #
###################

def relation_type(fr):
    return fr.relation_type.split(".")[0]

def _get_words_in_between_(fr):
    """return the words between m1 and m2"""
    sent=POS_SENTENCES[fr.article][int(fr.i_sentence)]
    mention1 =  _get_mentions_in_order_(fr)[0]
    mention2 =  _get_mentions_in_order_(fr)[1]
    first_token_index = int(mention1[2])
    later_token_index = int(mention2[1])
    w_in_between = sent[first_token_index:later_token_index]

    return w_in_between
def _get_mentions_in_order_(fr):
    """return a pair of tuples. The first one corresponds to mention1 and its info,
    the second one to mention2 and its info. i = mention1 and j=mention2 don't always hold"""
    if int(fr.i_offset_begin)<int(fr.j_offset_begin):
        mention1 = (fr.i_token,int(fr.i_offset_begin),
                    int(fr.i_offset_end), fr.i_entity_type, int(fr.i_sentence))
        mention2 = (fr.j_token,int(fr.j_offset_begin),
                    int(fr.j_offset_end), fr.j_entity_type, int(fr.j_sentence))
    else:
        mention2 = (fr.i_token,int(fr.i_offset_begin),
                    int(fr.i_offset_end), int(fr.i_entity_type), int(fr.i_sentence))
        mention1 = (fr.j_token,int(fr.j_offset_begin),
                    int(fr.j_offset_end), int(fr.j_entity_type), int(fr.j_sentence))
    return (mention1,mention2)

def _get_lowest_common_ancestor_(fr,s_tree):
    """return the lowest common ancestor tree """
    s_tree = SYNTAX_PARSE_SENTENCES[fr.article][int(fr.i_sentence)]
    mention1 = _get_mentions_in_order_(fr)[0]
    mention2= _get_mentions_in_order_(fr)[1]
    first_entity_index = int(mention1[1])
    later_entity_index = int(mention2[2])-1
    lwca_tuple=s_tree.treeposition_spanning_leaves(first_entity_index, later_entity_index+1)
    lowest_common_ancestor = s_tree[lwca_tuple]
    return lowest_common_ancestor

def _find_head_of_tree_(tree):
    """given a tree, return its head word"""
    result = "None"
    if tree.node not in phrase_heads.keys():
        for child in tree:
            if isinstance(child,ParentedTree):
                if child.node in ["WHNP", "MD", "VP", "S", "SQ", "SBAR"]:
                    result= _find_head_of_tree_(child)
                    break
                else:
                    result = child
    else:
        for child in tree:
            if isinstance(child,ParentedTree):
                sibling = child.right_sibling()
                next_is_not_head = isinstance(sibling,ParentedTree) and \
                                   sibling.node not in phrase_heads[tree.node]
                if child.node in phrase_heads[tree.node]:
                    if next_is_not_head:
                        result= child[0]
                    elif not isinstance(sibling,ParentedTree):
                        result= child[0]
                        break
                elif child.node == tree.node and next_is_not_head:
                    result = _find_head_of_tree_(child)
                    break
            else:
                result = child
    return result



####################
# Anya's functions #
####################
def i_pos_j_pos(fr):
    """Returns the POS of the two mentions, comma-separated: [NNP,VBD]"""
    i_pos=POS_SENTENCES[fr.article][fr.i_sentence][fr.i_offset_begin][1]
    j_pos=POS_SENTENCES[fr.article][fr.j_sentence][fr.j_offset_begin][1]

    return "i_pos_j_pos=[{},{}]".format(i_pos,j_pos)

def general_pos_ij(fr):
    """
    Returns the shortened POS of the two mentions, comma-separated.
    I've decided that the first letter of the POS is good enough.
    """
    i_pos=POS_SENTENCES[fr.article][fr.i_sentence][fr.i_offset_begin][1][0]
    j_pos=POS_SENTENCES[fr.article][fr.j_sentence][fr.j_offset_begin][1][0]

    return "general_pos_ij=[{},{}]".format(i_pos,j_pos)

def _is_pronoun(word):
    return word is not None and word.lower() in PRONOUN_SET

def same_hypernym(fr):
    """
    True if the two mentions have the same hypernym in WordNet.
    In multiword mentions, considering only the last word (I'm assuming last word=head).
    Not considering pronouns.
    Most of the logic was borrowed from Julia's WN function in the coref project - thank you.
    """

    try:

        i_final=wn.morphy(re.sub(r"\W", r"",fr.i_token.split('_')[-1]))
        j_final=wn.morphy(re.sub(r"\W", r"",fr.j_token.split('_')[-1]))

        if i_final is None or j_final is None:
            return "same_hypernym={}".format(False)

        if _is_pronoun(i_final) or _is_pronoun(j_final):
            return "same_hypernym={}".format(False)

        i_synsets=wn.synsets(i_final)
        j_synsets=wn.synsets(j_final)

        for i_synset in i_synsets:
            i_hypernym_set=set(i_synset.hypernyms())
            for j_synset in j_synsets:
                j_hypernym_set=set(j_synset.hypernyms())
                if i_hypernym_set.intersection(j_hypernym_set):
                    return "same_hypernym={}".format(True)

        return "same_hypernym={}".format(False)

    except wn_error:
        return "same_hypernym={}".format(False)

def lowest_common_hypernym(fr):
    """
    Returns the lowest common hypernym of the two mentions (based on WordNet).
    Again assuming that the last word = head word, and that it represents the phrase.
    Also considering only the first sense.
    """
    try:

        i_final=wn.morphy(re.sub(r"\W", r"",fr.i_token.split('_')[-1]))
        j_final=wn.morphy(re.sub(r"\W", r"",fr.j_token.split('_')[-1]))

        if i_final is None or j_final is None:
            return "lowest_common_hypernym={}".format(False)

        if _is_pronoun(i_final) or _is_pronoun(j_final):
            return "lowest_common_hypernym={}".format(False)

        i_synsets=wn.synsets(i_final)
        j_synsets=wn.synsets(j_final)

        lowest_common_hypernym=i_synsets[0].lowest_common_hypernyms(j_synsets[0])[0]

        return "lowest_common_hypernym={}".format(lowest_common_hypernym)

    except wn_error:
        return "lowest_common_hypernym={}".format(False)

def et12(fr):
    """Returns the entity types of the two mentions, comma-separated."""
    return "et12=[{},{}]".format(fr.i_entity_type,fr.j_entity_type)

def num_mentions_inbetween(fr):
    """Returns the number of other mentions between mention 1 and mention 2. Uses the entity_types dict."""
    i_end=fr.i_offset_end
    j_begin=fr.j_offset_begin
    all_mention_tuples=entity_types[fr.article][int(fr.i_sentence)]
    in_between_tuples=[tpl for tpl in all_mention_tuples if tpl[0]>i_end and tpl[0]<j_begin]

    return "num_mentions_inbetween={}".format(len(in_between_tuples))

def num_words_inbetween(fr):
    """Number of words between m1 and m2. Relies on Julia's functions."""
    return "num_words_inbetween={}".format(len(_get_words_in_between_(fr)))

def mention_overlap(fr):
    """
    m1 contains m2 or m2 contains m1.
    (This is true only 17 times in the training set. In all cases, the mentions are exactly the same,
    so this feature will probably not be very useful.)
    """
    result=(int(fr.i_offset_begin)<=int(fr.j_offset_begin) and int(fr.i_offset_end)>=int(fr.j_offset_end)) or \
           (int(fr.i_offset_begin)>=int(fr.j_offset_begin) and int(fr.i_offset_end)<=int(fr.j_offset_end))
    return "mention_overlap={}".format(result)

#### gazetter features start here; might be useless, but they're fun to try ####
def _is_rel_or_group(token):
    result=False
    for word in token.split('_'):
        if word.lower() in RELATIONSHIPS_AND_GROUPS:
            result=True
    return result

def _is_country(token):
    result=False
    for word in token.split('_'):
        if word.title() in COUNTRIES:
            result=True
    return result

def _is_nationality(token):
    result=False
    for word in token.split('_'):
        if word.title() in NATIONALITIES:
            result=True
    return result

def _is_official(token):
    result=False
    for word in token.split('_'):
        if word.lower() in OFFICIALS:
            result=True
    return result

def _is_profession(token):
    result=False
    for word in token.split('_'):
        if word.lower() in PROFESSIONS:
            result=True
    return result

def _is_title(token):
    result=False
    for word in token.split('_'):
        if word.lower() in TITLE_SET:
            result=True
    return result

def _is_possessive_pronoun(token):
    result=False
    for word in token.split('_'):
        if word.lower() in POSSESSIVE_PRONOUNS:
            result=True
    return result

def poss_pronoun_per(fr):
    """
    True if i is a possessive pronoun and j is PER.
    From what I can see in the data, they always occur in that order and never in reverse order.
    """
    result=_is_possessive_pronoun(fr.i_token) and fr.j_entity_type=='PER'
    #print fr.i_token, fr.j_token, "poss_pronoun_per={}".format(result)
    return "poss_pronoun_per={}".format(result)

def poss_pronoun_relword(fr):
    """
    True if i is a possessive pronoun and j is a word for a family relationship or group.
    From what I can see in the data, they always occur in that order and never in reverse order.
    """
    result=_is_possessive_pronoun(fr.i_token) and _is_rel_or_group(fr.j_token)
    #print fr.i_token, fr.j_token, "poss_pronoun_relword={}".format(result)
    return "poss_pronoun_relword={}".format(result)

def per_relword(fr):
    """
    True if i is a PER and j is a word for a family relationship or group,
    or the other way around.
    """
    result=(fr.i_entity_type=="PER" and _is_rel_or_group(fr.j_token)) or \
           (fr.j_entity_type=="PER" and _is_rel_or_group(fr.i_token))
    #print fr.i_token, fr.j_token, "per_relword={}".format(result)
    return "per_relword={}".format(result)

def per_org(fr):
    """
    True if i is PER and j is ORG or the other way around.
    """
    result=(fr.i_entity_type=='PER' and fr.j_entity_type=='ORG') or \
           (fr.i_entity_type=='ORG' and fr.j_entity_type=='PER')
    #print fr.i_token, fr.j_token, "per_org={}".format(result)
    return "per_org={}".format(result)

def per_nns(fr):
    """
    True if i is PER and j is a plural noun.
    The hope is that plural nouns will describes groups like "friends" and "neighbors."
    Of course, that doesn't cover things like "team", "party", or "Beatles", but we can try it.
    """
    j_pos=POS_SENTENCES[fr.article][fr.j_sentence][fr.j_offset_begin][1]
    result=fr.i_entity_type=='PER' and j_pos=='NNS'
    #print fr.i_token, fr.j_token, "per_nns={}".format(result)
    return "per_nns={}".format(result)

def poss_title(fr):
    """
    poss + profession OR title OR official
    """
    result=_is_possessive_pronoun(fr.i_token) and (_is_profession(fr.j_token) or _is_official(fr.j_token) or _is_title(fr.j_token))
    #print fr.i_token, fr.j_token, "poss_title={}".format(result)
    return "poss_title={}".format(result)

def per_title(fr):
    """
    per + profession OR title OR official
    """
    result=fr.i_entity_type=='PER' and (_is_profession(fr.j_token) or _is_official(fr.j_token) or _is_title(fr.j_token))
    #print fr.i_token, fr.j_token, "per_title={}".format(result)
    return "per_title={}".format(result)

def nnp_title(fr):
    """
    nnp + profession OR title OR official
    or the other way around
    """
    i_pos=POS_SENTENCES[fr.article][fr.i_sentence][fr.i_offset_begin][1]
    j_pos=POS_SENTENCES[fr.article][fr.j_sentence][fr.j_offset_begin][1]
    result=(i_pos.startswith("NNP") and (_is_profession(fr.j_token) or _is_official(fr.j_token) or _is_title(fr.j_token))) or \
           (j_pos.startswith("NNP") and (_is_profession(fr.i_token) or _is_official(fr.i_token) or _is_title(fr.i_token)))
    #print fr.i_token, fr.j_token, "nnp_title={}".format(result)
    #print
    return "nnp_title={}".format(result)

def et1_country(fr):
    """the entity type of M1 when M2 is a country name"""
    result=False
    if _is_country(fr.j_token):
        result=fr.i_entity_type
    #print fr.i_token, fr.j_token, "et1_country={}".format(result)
    return "et1_country={}".format(result)

def country_et2(fr):
    """the entity type of M2 when M1 is a country name"""
    result=False
    if _is_country(fr.i_token):
        result=fr.j_entity_type
    #print fr.i_token, fr.j_token, "country_et2={}".format(result)
    return "country_et2={}".format(result)

# dependency features
def et1_dw1(fr):
    """combination of the entity type and the dependent word(s) for M1"""
    #print fr.article
    #print DEPENDENCIES[fr.article].has_key(int(fr.i_sentence))
    et1_dependencies=DEPENDENCIES[fr.article][int(fr.i_sentence)+1]

    dep_list=[dep_word for (dep_index,dep_word),(gov_index,gov_word,dep_type) in et1_dependencies.items() if int(fr.i_offset_end)==gov_index]
    #print fr.i_token, fr.j_token, "et1_dw1={},{}".format(fr.i_entity_type,dep_list)
    #print et1_dependencies.items()
    #print
    return "et1_dw1={},{}".format(fr.i_entity_type,dep_list)

def h1_dw1(fr):
    """combination of the head word (=last word for now) and the dependent word(s) for M1"""
    #print fr.article
    #print DEPENDENCIES[fr.article].has_key(int(fr.i_sentence))
    et1_dependencies=DEPENDENCIES[fr.article][int(fr.i_sentence)+1]

    dep_list=[dep_word for (dep_index,dep_word),(gov_index,gov_word,dep_type) in et1_dependencies.items() if int(fr.i_offset_end)==gov_index]
    #print fr.i_token, fr.j_token, "h1_dw1={},{}".format(fr.i_token.split('_')[-1],dep_list)
    #print et1_dependencies.items()
    #print
    return "h1_dw1={},{}".format(fr.i_token.split('_')[-1],dep_list)

def et2_dw2(fr):
    """combination of the entity type and the dependent word(s) for M2"""
    #print fr.article
    #print DEPENDENCIES[fr.article].has_key(int(fr.i_sentence))
    et2_dependencies=DEPENDENCIES[fr.article][int(fr.i_sentence)+1]

    dep_list=[dep_word for (dep_index,dep_word),(gov_index,gov_word,dep_type) in et2_dependencies.items() if int(fr.j_offset_end)==gov_index]
    #print fr.i_token, fr.j_token, "et2_dw2={},{}".format(fr.j_entity_type,dep_list)
    #print et1_dependencies.items()
    #print
    return "et2_dw2={},{}".format(fr.j_entity_type,dep_list)

def h2_dw2(fr):
    """combination of the head word (=last word for now) and the dependent word(s) for M1"""
    #print fr.article
    #print DEPENDENCIES[fr.article].has_key(int(fr.i_sentence))
    et2_dependencies=DEPENDENCIES[fr.article][int(fr.i_sentence)+1]

    dep_list=[dep_word for (dep_index,dep_word),(gov_index,gov_word,dep_type) in et2_dependencies.items() if int(fr.j_offset_end)==gov_index]
    #print fr.i_token, fr.j_token, "h2_dw2={},{}".format(fr.i_token.split('_')[-1],dep_list)
    #print et1_dependencies.items()
    #print
    return "h2_dw2={},{}".format(fr.j_token.split('_')[-1],dep_list)

def _dep_path_to_root(offset_end):
    """Returns a list of token"""
    pass

######################
# Keelan's functions #
######################

def rule_resolve(fs):
    dcoref = COREF[fs.article]
    for group in dcoref:
        found_i = False
        found_j = False
        for referent in group:
            if _coref_helper(referent, fs.sentence, fs.offset_begin, fs.offset_end, fs.i_cleaned):
                found_i = True
            if _coref_helper(referent, fs.sentence_ref, fs.offset_begin_ref, fs.offset_end_ref, fs.j_cleaned):
                found_j = True

            if found_i and found_j:
                return "rule_resolve=True"

    return "rule_resolve=False"

def _coref_helper(i, sentence, offset_begin, offset_end, cleaned):
    """heuristically, i[2] and i[3] will have a later index, especially i[3]"""
    cleaned = cleaned.replace("_", " ")
    return i[1] == sentence and \
           i[2]-2 <= offset_begin <= i[2] and \
           i[3]-3 <= offset_end <= i[3]


#####################
# Julia's functions #
#####################


def i_token(fr):
    """return the i_token or the antecedent if i is a pronoun"""
    i_mention = (fr.i_token,int(fr.i_offset_begin),
                 int(fr.i_offset_end), fr.i_entity_type, int(fr.i_sentence))
    antecedent = _get_antecedent_(i_mention,fr.article)[0]
    token = "_".join(antecedent.split())
    return "i_token={}".format(token)

def j_token(fr):
    """ return the j_token or the antecedent if j is a pronoun"""
    j_mention = (fr.j_token,int(fr.j_offset_begin),
                 int(fr.j_offset_end), fr.j_entity_type, int(fr.j_sentence))
    antecedent = _get_antecedent_(j_mention,fr.article)[0]
    token = "_".join(antecedent.split())
    return "j_token={}".format(token)

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
        mention1 = (fr.i_token,int(fr.i_offset_begin),
                    int(fr.i_offset_end), fr.i_entity_type, int(fr.i_sentence))
        mention2 = (fr.j_token,int(fr.j_offset_begin),
                    int(fr.j_offset_end), fr.j_entity_type,int(fr.j_sentence))
    else:
        mention2 = (fr.i_token,int(fr.i_offset_begin),
                    int(fr.i_offset_end),fr.i_entity_type, int(fr.i_sentence))
        mention1 = (fr.j_token,int(fr.j_offset_begin),
                    int(fr.j_offset_end), fr.j_entity_type, int(fr.j_sentence))
    return (mention1,mention2)

def bow_mention1(fr):
    """return the words in mention2 eg. [George,Bush]"""
    mention1 = _get_mentions_in_order_(fr)[0]
    mention_token = _get_antecedent_(mention1,fr.article)[0]
    token = mention_token.split()
    return "bow_mention1={}".format(token)

def bow_mention2(fr):
    """return the words in mention2"""
    mention2 = _get_mentions_in_order_(fr)[1]
    mention_token = _get_antecedent_(mention2,fr.article)[0]
    token = mention_token.split()
    return "bow_mention2={}".format(token)

def _get_words_in_between_(fr):
    """return the words between m1 and m2"""
    sent=POS_SENTENCES[fr.article][int(fr.i_sentence)]
    mention1 =  _get_mentions_in_order_(fr)[0]
    mention2 =  _get_mentions_in_order_(fr)[1]
    first_token_index = int(mention1[2])
    later_token_index = int(mention2[1])
    w_in_between = sent[first_token_index:later_token_index]
    return w_in_between

def first_word_in_between(fr):
    """return the first word between m1 and m2"""
    words_in_between = _get_words_in_between_(fr)
    if len(words_in_between)>=1:
        first = words_in_between[0][0]
    else:
        first = "None"
    return "first_word_in_between={}".format([first])

def last_word_in_between(fr):
    """return the last word between m1 and m2"""
    words = _get_words_in_between_(fr)
    if len(words)>=1:
        last = words[len(words)-1][0]
    else:
        last = "None"
    return "last_word_in_between={}".format([last])

def bow_tree(fr):
    """ return words between m1 and m2 excluding the first and last words"""
    words = _get_words_in_between_(fr)
    if len(words)>=1:
    	words.pop()
    if len(words)>=1: 
    	words.pop()
    children = [ParentedTree(w,["*"]) for w,pos in words]
    bow_tree = ParentedTree("BOW",children)
    return bow_tree

def first_word_before_m1(fr):
    """return first word before m1"""
    mention1 = _get_mentions_in_order_(fr)[0]
    sent=POS_SENTENCES[fr.article][int(mention1[4])]
    return "first_word_before_m1={}".format([sent[int(mention1[1])-1][0]])


def first_word_before_m2(fr):
    """return first word before m2"""
    mention2 = _get_mentions_in_order_(fr)[1]
    sent=POS_SENTENCES[fr.article][int(mention2[4])]
    return "first_word_before_m2={}".format([sent[int(mention2[1])-1][0]])

def second_word_before_m1(fr):
    """return second word before m1"""
    mention1 = _get_mentions_in_order_(fr)[0]
    sent=POS_SENTENCES[fr.article][int(mention1[4])]
    try:
        return "second_word_before_m1={}".format([sent[int(mention1[1])-2][0]])
    except IndexError:
        return "second_word_before_m1=[None]"

def second_word_before_m2(fr):
    """return second word before m2"""
    mention2 = _get_mentions_in_order_(fr)[1]
    sent=POS_SENTENCES[fr.article][int(mention2[4])]
    try:
        return "second_word_before_m2={}".format([sent[int(mention2[1])-2][0]])
    except IndexError:
        return "second_word_before_m2=[None]"


def head_of_m1_coref(fr):
    """return the head of the NP in which M1 occurs"""
    mention1 = _get_mentions_in_order_(fr)[0]
    pos=POS_SENTENCES[fr.article][mention1[4]][mention1[1]][1]
    if pos == "PRP": #don't check antecedent with possessive pronouns, just personal pronouns
        antecedent = _get_antecedent_(mention1,fr.article)
        s_tree=SYNTAX_PARSE_SENTENCES[fr.article][antecedent[3]]
        m1_tuple = s_tree.leaf_treeposition(antecedent[1])
    else:
        s_tree=SYNTAX_PARSE_SENTENCES[fr.article][mention1[4]]
        m1_tuple = s_tree.leaf_treeposition(mention1[1])
    parent = s_tree[m1_tuple[0:-2]]
    return "head_of_m1_coref={}".format([_find_head_of_tree_(parent)])

def head_of_m2_coref(fr):
    """return the head of the NP in which M2 occurs"""
    mention2 = _get_mentions_in_order_(fr)[1]
    pos=POS_SENTENCES[fr.article][mention2[4]][mention2[1]][1]
    if pos == "PRP": #pronoun, do everyting for antecedent
        antecedent = _get_antecedent_(mention2,fr.article)
        s_tree=SYNTAX_PARSE_SENTENCES[fr.article][antecedent[3]]
        m2_tuple = s_tree.leaf_treeposition(antecedent[1])
    else:
        s_tree=SYNTAX_PARSE_SENTENCES[fr.article][mention2[4]]
        m2_tuple = s_tree.leaf_treeposition(mention2[1])
    parent = s_tree[m2_tuple[0:-2]]
    return "head_of_m2_coref={}".format([_find_head_of_tree_(parent)])

def _head_of_m1_(fr):
    """return the head of the NP in which M1 occurs"""
    mention1 = _get_mentions_in_order_(fr)[0]
    s_tree=SYNTAX_PARSE_SENTENCES[fr.article][mention1[4]]
    m1_tuple = s_tree.leaf_treeposition(mention1[1])
    parent = s_tree[m1_tuple[0:-2]]
    return _find_head_of_tree_(parent)

def _head_of_m2_(fr):
    """return the head of the NP in which M1 occurs"""
    mention2 = _get_mentions_in_order_(fr)[1]
    s_tree=SYNTAX_PARSE_SENTENCES[fr.article][mention2[4]]
    m1_tuple = s_tree.leaf_treeposition(mention2[1])
    parent = s_tree[m1_tuple[0:-2]]
    return _find_head_of_tree_(parent)

def same_head(fr):
    """return whether both entities have the same head"""
    mention1_head = head_of_m1_coref(fr).split("=")[1]
    mention2_head = head_of_m2_coref(fr).split("=")[1]
    return "same_head={}".format(mention1_head == mention2_head)


def first_np_head_in_between(fr):
    """
    if there are other NP between both entities,
    return the head of the first one
    """
    heads = boh_np_tree(fr)
    if len(heads)>=1:
       head = heads[0].node
    else:
        head = None
    return "first_np_head_in_between={}".format([head])


def first_head_in_between(fr):
    """
    if there are other phrases between both entities,
    return the head of the first one
    """

    heads = boh_tree(fr)
    if len(heads)>=1:
       head = heads[0].node
    else:
        head = None
    return "first_head_in_between={}".format([head])


def last_np_head_in_between(fr):
    """
    if there are other NP phrases in-between both entities,
    return the head of the last one
    """

    heads = boh_np_tree(fr)
    if len(heads)>=1:
       head = heads[-1].node
    else:
        head = None
    return "last_np_head_in_between={}".format([head])


def last_head_in_between(fr):
    """
    if there are other  phrases in_between both entities,
    return the head of the last one
    """
    heads = boh_tree(fr)
    if len(heads)>=1:
       head = heads[-1].node
    else:
        head = None
    return "last_head_in_between={}".format([head])


def boh_np_tree(fr):
    """
    return a bag of heads tree with the heads of the NPs in_between
    mention1 and mention2

    """
    mention1= _get_mentions_in_order_(fr)[0]
    mention2 = _get_mentions_in_order_(fr)[1]
    head_of_m1= _head_of_m1_(fr)
    head_of_m2= _head_of_m2_(fr)
    s_tree=SYNTAX_PARSE_SENTENCES[fr.article][mention1[4]]
    i = mention1[1]+1
    heads = []
    while i < mention2[1]:
        word_tuple = s_tree.leaf_treeposition(i)
        pos_index = word_tuple[-2]
        parent = s_tree[word_tuple[0:-2]]
        head = None
        sum = 0
        for j,child in enumerate(parent[pos_index:]):
            if child.node in ['NN', 'NNS', 'NNP', 'NNPS', 'WHNP', "PRP"] and \
                            child[0] != head_of_m1 and child[0]!= head_of_m2:
                head = child[0]
                sum = j
        if isinstance(head,unicode):
            heads.append(head)
        i+=sum + 1

    children = [ParentedTree(w,["*"]) for w in heads]
    boh_tree = ParentedTree("BOH-NPs",children)
    return boh_tree

def boh_tree(fr):
    """Return a flatten tree with all heads between m1 and m2"""

    mention1= _get_mentions_in_order_(fr)[0]
    mention2 = _get_mentions_in_order_(fr)[1]
    head_of_m1= _head_of_m1_(fr)
    head_of_m2= _head_of_m2_(fr)
    s_tree=SYNTAX_PARSE_SENTENCES[fr.article][mention1[4]]
    i = mention1[1]+1
    heads = []
    while i < mention2[1]:
        word_tuple = s_tree.leaf_treeposition(i)
        pos_index = word_tuple[-2]
        parent = s_tree[word_tuple[0:-2]]
        head = None
        sum = 0
        for j,child in enumerate(parent[pos_index:]):
            if parent.node in phrase_heads.keys():
                if parent.node in phrase_heads.keys():
                    candidate_head = child.node in phrase_heads[parent.node]
                    not_head_of_m1 = child[0] != head_of_m1
                    not_head_of_m2 = child[0] != head_of_m2
                    if not (isinstance(child.right_sibling(), ParentedTree) and
                                    child.right_sibling().node in phrase_heads[parent.node]):
                        if candidate_head and not_head_of_m1 and not_head_of_m2:
                            head = child[0]
                            sum = j
        if isinstance(head,unicode):
            heads.append(head)
        i+=sum +1

    children = [ParentedTree(w,["*"]) for w in heads]
    boh_tree = ParentedTree("BOH",children)
    return boh_tree

def first_np_head_before_m1(fr):
    """
    return the head of the first NP before mention1
    """
    mention1= _get_mentions_in_order_(fr)[0]
    head_of_m1= _head_of_m1_(fr)
    s_tree=SYNTAX_PARSE_SENTENCES[fr.article][mention1[4]]
    i = 0
    head = None
    while i < mention1[1]:
        word_tuple = s_tree.leaf_treeposition(i)
        pos_index = word_tuple[-2]
        parent = s_tree[word_tuple[0:-2]]
        sum = 0
        for j,child in enumerate(parent[pos_index:]):
            if child.node in ['NN', 'NNS', 'NNP', 'NNPS'] and \
                            child[0] != head_of_m1:
                head = child[0]
                sum = j
        i+=sum + 1

    return "first_np_head_before_m1={}".format([head])

def first_head_before_m1(fr):
    """
    return the head of the first phrase before mention1
    """
    mention1= _get_mentions_in_order_(fr)[0]
    head_of_m1= _head_of_m1_(fr)
    s_tree=SYNTAX_PARSE_SENTENCES[fr.article][mention1[4]]
    i = 0
    head = None
    while i < mention1[1]:
        word_tuple = s_tree.leaf_treeposition(i)
        pos_index = word_tuple[-2]
        parent = s_tree[word_tuple[0:-2]]
        sum = 0
        for j,child in enumerate(parent[pos_index:]):
            if parent.node in phrase_heads.keys():
                if child.node in phrase_heads[parent.node] and \
                                child[0] != head_of_m1:
                    head = child[0]
                    sum = j
        i+=sum + 1

    return "first_head_before_m1={}".format([head])

def second_np_head_before_m1(fr):
    """return the second to last NP head before m1"""

    mention1= _get_mentions_in_order_(fr)[0]
    head_of_m1= _head_of_m1_(fr)
    s_tree=SYNTAX_PARSE_SENTENCES[fr.article][mention1[4]]
    first_head_before_m1 = eval(first_np_head_before_m1(fr).split("=")[1])[0]
    i = 0
    head = None
    while i < mention1[1]:
        word_tuple = s_tree.leaf_treeposition(i)
        pos_index = word_tuple[-2]
        parent = s_tree[word_tuple[0:-2]]
        sum = 0
        for j,child in enumerate(parent[pos_index:]):
            if child.node in ['NN', 'NNS', 'NNP', 'NNPS'] and \
                            child[0] != head_of_m1 and child[0]!=first_head_before_m1:
                head = child[0]
                sum = j
        i+=sum + 1
    return "second_np_head_before_m1={}".format([head])

def second_head_before_m1(fr):
    """return the second to last head before m1"""

    mention1= _get_mentions_in_order_(fr)[0]
    head_of_m1= _head_of_m1_(fr)
    s_tree=SYNTAX_PARSE_SENTENCES[fr.article][mention1[4]]
    first_before_m1 = eval(first_head_before_m1(fr).split("=")[1])[0]
    i = 0
    head = None
    while i < mention1[1]:
        word_tuple = s_tree.leaf_treeposition(i)
        pos_index = word_tuple[-2]
        parent = s_tree[word_tuple[0:-2]]
        sum = 0
        for j,child in enumerate(parent[pos_index:]):
            if parent.node in phrase_heads.keys():
                if child.node in phrase_heads[parent.node] and \
                                child[0] != head_of_m1 and child[0]!=first_before_m1:
                    head = child[0]
                    sum = j
        i+=sum+1
    return "second_head_before_m1={}".format([head])

def second_np_head_before_m2(fr):
    """
    return the second to last NP head before m2
    """
    heads = boh_np_tree(fr)
    if len(heads)>=2:
        head = heads[-2].node
        return "second_np_head_before_m2={}".format([head])
    else:
        return "second_np_head_before_m2=None"


def second_head_before_m2(fr):
    """
    return the second to last head before m2
    """
    heads = boh_tree(fr)
    if len(heads)>=2:
        head = heads[-2].node
        return "second_head_before_m2={}".format([head])
    else:
        return "second_head_before_m2=None"

def no_words_in_between(fr):
    """return whether there are words between m1 and m2"""
    return "no_words_in_between={}".format(len(_get_words_in_between_(fr))==0)

def no_phrase_in_between(fr):
    """return whether there are phrases between both entities"""
    no_phrase = len(boh_tree(fr).leaves()) == 0
    return "no_phrase_in_between={}".format(no_phrase)


def lp_tree(fr):
    """return a flatten tree with the nodes of the phrases in the path from m1
    to m2 (duplicates removed)"""

    s_tree = SYNTAX_PARSE_SENTENCES[fr.article][int(fr.i_sentence)]
    lwca=_get_lowest_common_ancestor_(fr,s_tree)
    mention1 = _get_mentions_in_order_(fr)[0]
    mention2 = _get_mentions_in_order_(fr)[1]
    left_tree = s_tree[s_tree.leaf_treeposition(int(mention1[1]))[0:-1]]
    right_tree= s_tree[s_tree.leaf_treeposition(int(mention2[2])-1)[0:-1]]
    nodes_left_branch = []
    nodes_right_branch=[]
    curr_tree = left_tree
    if isinstance(lwca,ParentedTree):
        while curr_tree!=lwca.parent():
            if not (len(nodes_left_branch)>0 and nodes_left_branch[-1]== curr_tree.node):
                nodes_left_branch.append(curr_tree.node)
            curr_tree = curr_tree.parent()
        curr_tree = right_tree
        while curr_tree!=lwca:
            if not (len(nodes_right_branch)>0 and nodes_right_branch[-1]== curr_tree.node):
                nodes_right_branch.append(curr_tree.node)
            curr_tree = curr_tree.parent()
        nodes_right_branch.reverse()
        path = nodes_left_branch + nodes_right_branch
        children = [ParentedTree(node,["*"]) for node in path]
        label_path = ParentedTree("LP",children)
        return label_path
    else:
        label_path = ParentedTree("LP",[lwca])
        return label_path

def lp_head_tree(fr):
    """
    return a flatten tree with the nodes of the phrases in the path from m1
    to m2 (duplicates removed) augmented with the head word of the lowest
    common ancestor
    """
    s_tree = SYNTAX_PARSE_SENTENCES[fr.article][int(fr.i_sentence)]
    lwca=_get_lowest_common_ancestor_(fr,s_tree)
    mention1 = _get_mentions_in_order_(fr)[0]
    mention2 = _get_mentions_in_order_(fr)[1]
    left_tree = s_tree[s_tree.leaf_treeposition(int(mention1[1]))[0:-1]]
    right_tree= s_tree[s_tree.leaf_treeposition(int(mention2[2])-1)[0:-1]]
    nodes_left_branch = []
    nodes_right_branch=[]
    curr_tree = left_tree
    if isinstance(lwca,ParentedTree):
        while curr_tree!=lwca:
            if not (len(nodes_left_branch)>0 and nodes_left_branch[-1].node== curr_tree.node):
                nodes_left_branch.append(ParentedTree(curr_tree.node,["*"]))
            curr_tree = curr_tree.parent()
        if nodes_left_branch[-1].node == lwca.node: nodes_left_branch.pop()
        nodes_left_branch.append(ParentedTree(lwca.node,[_find_head_of_tree_(lwca)])) #add head of lwca
        curr_tree = right_tree
        while curr_tree!=lwca:
            if not (len(nodes_right_branch)>0 and nodes_right_branch[-1].node== curr_tree.node):
                nodes_right_branch.append(ParentedTree(curr_tree.node,["*"]))
            curr_tree = curr_tree.parent()
        nodes_right_branch.reverse()
        path = nodes_left_branch + nodes_right_branch
        label_path = ParentedTree("LP-head",path)
    else:
        label_path = ParentedTree("LP-head",[lwca])
    return label_path

#i'm writing this right now....
#def _get_antecedent_(mention_tuple):
def _get_antecedent_(mention_tuple, article):
    """If the token is a pronound, return its antecedent. Else,
    return the pronoun.
    Return as (token,start,end,sentence)"""
    antecedent = (mention_tuple[0], mention_tuple[1], mention_tuple[2],mention_tuple[4])
    target_group = None
    mention_referent = None
    if _is_pronoun(mention_tuple[0]):
        dcoref = COREF[article]
        for group in dcoref:
            for referent in group:
                if referent[1] == mention_tuple[4] and referent[2] == mention_tuple[1]:
                    target_group = group
                    break
            if isinstance(target_group,set):
                break
        try:
            for referent in target_group:
                if referent == mention_referent:
                    continue
                else:
                    text = referent[0]
                    sent = referent[1]
                    end = referent[3]-1
                    start = referent[2]
                    text_tag = POS_SENTENCES[article][sent][end][1]
                    if text_tag in ["NNP","NNPS"]:
                        antecedent = (text,start,end,sent)
                        break
                    elif text_tag in ["NN","NNS"]:
                        antecedent = (text,start,end,sent)
            return antecedent
        except TypeError: #sometimes sentence indices in COREF and our data don't match
            return (mention_tuple[0], mention_tuple[1], mention_tuple[2],mention_tuple[4])

    else:
        return (mention_tuple[0], mention_tuple[1], mention_tuple[2],mention_tuple[4])




def path_enclosed_tree(fr):
    if fr.i_sentence!=fr.j_sentence:
        return ParentedTree("None",["*"]) #just in case
    else:
        s_tree = SYNTAX_PARSE_SENTENCES[fr.article][int(fr.i_sentence)]
        return _generate_enclosed_tree(fr,s_tree)

def path_enclosed_tree_augmented(fr):
    if fr.i_sentence!=fr.j_sentence:
        return ParentedTree("None",["*"]) #just in case
    else:
        s_tree = ParentedTree.convert(AUGMENTED_TREES[fr.article][int(fr.i_sentence)])
        return _generate_enclosed_tree(fr,s_tree)


def _generate_enclosed_tree(fr,s_tree):
    """****MONSTER FUNCTION!!!!****
    Return the path enclosed tree between m1 and m2 as PatentedTree
    The path enclosed tree is the smallest common
    sub-tree including the two entities [JB:but not necessary the lowest_common_ancestor]. In other
    words, the sub-tree is enclosed by the shortest
    path linking the two entities in the parse tree (this
    path is also commonly-used as the path tree feature in the feature-based methods)
    [Zhang et al. 2006]
    [JB]:
    That is, the path enclosed tree includes mention1, and every branch to the right of it, until
    mention2. In this function, the path enclosed tree is built in the following way:
    the left branch of it includes mention1 and branches to the right of it that still are on the left child
    of the lowest common ancestor. The right branch of the path-enclosed tree includes mention2, and the
    branches to the left of it that are on the right child of the lowest common ancestor.
    Both branches are merged in one tree, with the lowest_common_ancestor node, yielding the
    path enclosed tree.
    """

    mention1 = _get_mentions_in_order_(fr)[0]
    mention2= _get_mentions_in_order_(fr)[1]
    first_entity_index = int(mention1[1])
    later_entity_index = int(mention2[2])-1
    first_token = mention1[0]
    later_token = mention2[0]
    i_tuple = s_tree.leaf_treeposition(first_entity_index)
    j_tuple = s_tree.leaf_treeposition(later_entity_index)
    first_tree = s_tree[i_tuple[0:-1]]
    later_tree= s_tree[j_tuple[0:-1]]
    lowest_common_ancestor = _get_lowest_common_ancestor_(fr,s_tree)


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
            j_in_leaves = len(set(later_token.split("_")).intersection(set(child.leaves())))>0
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
                #left_branch.draw()
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
                if j_in_leaves and m1_visited:
                    m2_visited = True
                elif m1_visited and not m2_visited:
                    children.append(child.copy(deep=True))
                if i_in_leaves:
                    m1_visited = True
            children.append(right_branch)
            result_tree = ParentedTree(lowest_common_ancestor.node,children)
        return result_tree
    return merge_both_branches(from_root_to_m1(first_tree),from_root_to_m2(later_tree))
