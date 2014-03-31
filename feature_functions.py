__author__ = 'keelan'

import re
from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import WordNetError as wn_error
from file_reader import RAW_SENTENCES, SYNTAX_PARSE_SENTENCES, POS_SENTENCES, PRONOUN_SET, entity_types

###################
# basic functions #
###################

def relation_type(fr):
    return "relation_type={}".format(fr.relation_type)

#####################
# Julia's functions #
#####################
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
        mention1 = (fr.i_token,int(fr.i_offset_begin),int(fr.i_offset_end), fr.i_entity_type, int(fr.i_sentence))
        mention2 = (fr.j_token,int(fr.j_offset_begin),int(fr.j_offset_end), fr.j_entity_type,int(fr.j_sentence))
    else:
        mention2 = (fr.i_token,int(fr.i_offset_begin),int(fr.i_offset_end), int(fr.i_entity_type), int(fr.i_sentence))
        mention1 = (fr.j_token,int(fr.j_offset_begin),int(fr.j_offset_end), int(fr.j_entity_type), int(fr.j_sentence))
    return (mention1,mention2)

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
    return word.lower() in PRONOUN_SET

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

    except:
        wn_error
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

        if _is_pronoun(i_final) or _is_pronoun(j_final):
            return "lowest_common_hypernym={}".format(False)

        i_synsets=wn.synsets(i_final)
        j_synsets=wn.synsets(j_final)

        lowest_common_hypernym=i_synsets[0].lowest_common_hypernyms(j_synsets[0])[0]

        return "lowest_common_hypernym={}".format(lowest_common_hypernym)

    except:
        wn_error
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



######################
# Keelan's functions #
######################

