__author__ = 'keelan'

import os
import re
import codecs
import xml.etree.ElementTree as et
from collections import defaultdict,namedtuple
from nltk.tree import Tree,ParentedTree
from corenlp import parse_parser_xml_results
from operator import itemgetter
from nltk.corpus import gazetteers as gz
from nltk.corpus import wordnet as wn


INT_INDEXES = [2, 3, 4, 7, 8, 9]

def pos_split(string):
    if not string.startswith("_"):
        return string.split("_")
    else:
        return [string[:-2], string[-1:]]

def feature_list_reader(file_path, local_names):
    ls = []
    with open(file_path, "r") as f_in:
        for line in f_in:
            ls.append(local_names[line.rstrip()])

    return ls

class LazyDict:

    def __init__(self, file_path, opener):
        self.d = {}
        self.file_path = file_path
        self.opener = opener

    def __getitem__(self, item):
        try:
            return self.d[item]
        except KeyError:
            res = self.opener(self.file_path, item)
            self.d[item] = res
            return res

class SuperLazyDict:

    def __init__(self, lazy_dict, reader):
        self.lazy_dict = lazy_dict
        self.reader = reader
        self.d = {}

    def __getitem__(self, item):
        try:
            return self.d[item]
        except KeyError:
            res = self.reader(self.lazy_dict[item])
            self.d[item] = res
            return res

class AutoVivification(dict):
    """
    Implementation of perl's autovivification feature.
    It allows one to populate all the levels of a nested dict at the same time.
    (http://stackoverflow.com/questions/651794/whats-the-best-way-to-initialize-a-dict-of-dicts-in-python)
    """
    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value

def stanford_raw_reader(nlp):
    all_sentences = []
    for s in nlp["sentences"]:
        all_sentences.append(s["text"])
    return all_sentences

def stanford_pos_reader(nlp):
    all_sentences = []
    for s in nlp["sentences"]:
        sentence = []
        for w in s["words"]:
            sentence.append([w[0], w[1]["PartOfSpeech"]])
        all_sentences.append(sentence)
    return all_sentences

def stanford_tree_reader(nlp):
    all_sentences = []
    for s in nlp["sentences"]:
        all_sentences.append(ParentedTree.parse(s["parsetree"]))
    return all_sentences

def stanford_nonparented_tree_reader(nlp):
    all_sentences = []
    for s in nlp["sentences"]:
        all_sentences.append(Tree(s["parsetree"]))
    return all_sentences

def stanford_general_opener(file_path, f_name):
    with open(os.path.join(file_path, f_name+".head.coref.raw.xml"), "r") as f_in:
        return parse_parser_xml_results(f_in.read())

def stanford_coref_reader(nlp):
    all_coref_groups = []
    for group in nlp["coref"]:
        chain = set()
        for pair in group:
            for i in pair:
                chain.add((i[0], i[1], i[3], i[4]))
        all_coref_groups.append(chain)
    return all_coref_groups

def prepare_line(line):
    line = line.rstrip().split()
    if len(line) == 11:
        line.insert(0, "")
    i_token = line[6]
    j_token = line[11]
    for i in INT_INDEXES:
        line[i] = int(line[i])
    line.append(_clean(i_token))
    line.append(_clean(j_token))

    return line

def _clean(token):
    """
    1) Removes non-alpha (but not the "-") from the beginning of the token
    2) Removes possessive 's from the end
    3) Removes O', d', and ;T from anywhere (O'Brien becomes Brien, d'Alessandro becomes Alessandro, etc.)
    """
    # this needs to be fixed
    return [re.sub(r"\W", r"", word) for word in token.split("_")]


def get_original_data(file_path):
    gold_data = []
    with codecs.open(file_path, "r") as f_in:
        for line in f_in:
            prepped = prepare_line(line)
            gold_data.append(FeatureRow(*prepped))

    return gold_data

def gather_entities():
    """
    Loops through the training, dev, and test data and creates a dictionary that contains entity types
    that correspond to offset indices.

    Format of the dictionary:
    {article: {sentence_id: {offset_tuple: entity_type}}}
    """
    articles=AutoVivification()

    for root, dirs, files in os.walk('resources'):
        for file_name in files:
            if file_name.startswith('cleaned') and file_name.endswith('gold'):
                file_name=os.path.join(root,file_name)
                with open(file_name, 'r') as f:
                    for line in f:
                        _, article, sentence_id, i_start, i_end, i_entity_type, _, \
                        _, j_start, j_end, j_entity_type, _ = line.rstrip('\n').split()
                        articles[article][int(sentence_id)][(int(i_start),int(i_end))]=i_entity_type
                        articles[article][int(sentence_id)][(int(j_start),int(j_end))]=j_entity_type

    return articles

def pronoun_reader():
    ls = []
    with open("resources/pronouns.txt", "r") as f_in:
        for line in f_in:
            ls.append(line.rstrip())
    return ls

def rels_and_groups_reader():
    ls = []
    with open("resources/relationships_and_groups.txt", "r") as f_in:
        for line in f_in:
            ls.append(line.rstrip())
    return ls

def officials_reader():
    officials=[]
    for hyponym in wn.synset('skilled_worker.n.01').hyponyms():
        officials.extend(hyponym.name.split('.')[0].split('_'))
        for h1 in hyponym.hyponyms():
            officials.extend(h1.name.split('.')[0].split('_'))
            for h2 in h1.hyponyms():
                officials.extend(h2.name.split('.')[0].split('_'))
                for h3 in h2.hyponyms():
                    officials.extend(h3.name.split('.')[0].split('_'))
    return officials

def augmented_tree_reader():
    """
    Converts all nonparented trees into augmented trees. Stores augmented trees in a dict of the form
    {article:{sentence_id:tree}}
    """
    augmented_tree_dict=AutoVivification()
    for article in entity_types:
        for sent_id in entity_types[article]:
            t=NONPARENTED_SENTENCES[article][sent_id]
            t_copy=t.copy(deep=True)
            augment_tree(t_copy,sent_id,article)
            augmented_tree_dict[article][sent_id]=t_copy

    augmented_tree_dict['NYT20001229.2047.0291'][21]=Tree("(ROOT (S (NP (E-PER (NNP Bush))) (VP (VBD got) (NP (NP (CD 271) (JJ electoral) (NNS votes)) (, ,) (ADJP (ADJP (NP (CD one)) (JJR more)) (SBAR (IN than) (S (NP (E-PER (PRP he))) (VP (VBD needed) (PP (IN for) (NP (NP (DT a) (NN majority)) (CC and) (NP (CD five) (JJR more)))) (PP (IN than) (NP (E-PER (NNP Gore)))))))) (, ,) (SBAR (WHNP (E-PER (WP who))) (S (VP (VBD lost) (NP (NP (CD one) (NN vote)) (PP (IN in) (NP (DT the) (E-PER (NNP Electoral) (NNP College))))) (SBAR (WHADVP (WRB when)) (S (NP (DT a) (NAC (E-GPE (NNP Washington) (, ,) (NNP D.C.))(, ,)) (E-PER (NN elector))) (VP (VBD left) (NP (E-PER (PRP$ her)) (NN ballot) (NN blank)) (S (VP (TO to) (VP (VB protest) (NP (NP (DT the) (NNP District)) (PP (IN of) (NP (NP (NP (NNP Columbia) (POS 's)) (NN lack)) (PP (IN of) (NP (NP (VBG voting) (NN power)) (PP (IN in) (NP (E-ORG (NNP Congress)))))))))))))))))))) (. .)))")

    return augmented_tree_dict


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

def stanford_dependency_reader():
    """
    Reads in the dependency parses into a dict:
    dependencies[article][sentence_id][(dependent_index,dependent_token)]=(governor_index,governor_token,dep_type)
    offset_end is equivalent to the index. For multiword mentions, this will be equivalent to the last word.
    Note that key=dependent, and value=governor, so we're basically following the arrows backwards.
    Example:
    APW20001001.2021.0521
    	10
		    ('20', 'said') : ('0', 'ROOT', 'root')
		    ('2', 'Short') : ('3', 'term', 'amod')
		    ('3', 'term') : ('8', 'say', 'tmod')
		    ('5', 'anyone') : ('6', 'objective', 'nn')
    """
    dependencies=AutoVivification()
    for root, dirs, files in os.walk('stanford-full-pipeline'):
        for file_name in files:
            article=file_name[:21]
            #print article
            f=os.path.join(root,file_name)
            with open(f, 'r') as article_file:
                xml_tree=et.parse(article_file)
                rt=xml_tree.getroot()
                for sentence in rt.find('document').find('sentences').findall('sentence'):
                    sentence_id=sentence.attrib['id']
                    #print '\t',sentence_id
                    for dep in sentence.findall('dependencies')[0]:
                        dep_type=dep.attrib['type']
                        governor=dep.findall('governor')[0]
                        dependent=dep.findall('dependent')[0]
                        governor_index=governor.attrib['idx']
                        dependent_index=dependent.attrib['idx']
                        governor_token=governor.text
                        dependent_token=dependent.text
                        dependencies[article][sentence_id][(dependent_index,dependent_token)]=\
                            (governor_index,governor_token,dep_type)
                        #print '\t\t',(dependent_index,dependent_token),":",(governor_index,governor_token,dep_type)
    return dependencies





FeatureRow = namedtuple("FeatureRow", ["relation_type", "article", "i_sentence",
                                       "i_offset_begin","i_offset_end",
                                       "i_entity_type", "i_token",
                                       "j_sentence", "j_offset_begin",
                                       "j_offset_end", "j_entity_type",
                                       "j_token", "i_cleaned", "j_cleaned"])

##########################
# final` data structures #
##########################

basedir = "stanford-full-pipeline"


all_stanford = LazyDict(basedir, stanford_general_opener)
RAW_SENTENCES = SuperLazyDict(all_stanford, stanford_raw_reader)
POS_SENTENCES = SuperLazyDict(all_stanford, stanford_pos_reader)
SYNTAX_PARSE_SENTENCES = SuperLazyDict(all_stanford, stanford_tree_reader)
NONPARENTED_SENTENCES = SuperLazyDict(all_stanford, stanford_nonparented_tree_reader)
COREF = SuperLazyDict(all_stanford, stanford_coref_reader)
PRONOUN_SET = set(pronoun_reader())
entity_types=gather_entities()
AUGMENTED_TREES=augmented_tree_reader()
RELATIONSHIPS_AND_GROUPS=set(rels_and_groups_reader())
COUNTRIES=set(gz.words('countries.txt'))
NATIONALITIES=set(gz.words('nationalities.txt'))
OFFICIALS=officials_reader() #these are bit silly; will probably discard"""
DEPENDENCIES=stanford_dependency_reader()
POSSESSIVE_PRONOUNS=['my','mine','your','yours','her','hers','his','our','ours','their','theirs']
TITLE_SET= {"chairman", "Chairman", "director", "Director", "president", "President", "manager", "managers","Manager", "executive",
            "CEO", "Officer", "officer", "consultant", "CFO", "COO", "CTO", "CMO", "founder", "shareholder",
            "researcher", "professor", "principal", "Principal", "minister", "Minister", "prime", "Prime", "chief",
            "Chief", "prosecutor", "Prosecutor", "queen", "Queen", "leader", "Leader", "secretary", "Secretary",
            "ex-Leader", "ex-leader", "coach", "Coach", "composer", "Composer", "head", "Head", "governor", "Governor",
            "judge", "Judge", "democrat", "Democrat", "republican", "Republican", "senator", "Senator", "congressman",
            "Congressman", "congresswoman", "Congresswoman", "analyst", "Analyst", "sen", "Sen", "Rep", "rep", "MP",
            "mp", "justice", "Justice", "co-chairwoman", "co-chair", "co-chairman", "Mr.", "mr.", "Mr", "mr", "Ms.",
            "ms.", "Mrs.", "mrs.","secretary-general","Secretary-General","doctor","Doctor"}

#obtained from WordNet by getting hypernyms of hypernyms of hypernyms of 'professional.n.01'
#lightly edited
PROFESSIONS=set(['practitioner', 'homeopath', 'gongorist', 'clinician', 'careerist', 'career', 'man',
                 'career', 'girl', 'publisher', 'lawyer', 'conveyancer', 'barrister', 'serjeant-at-law', 'counsel',
                 'counsel', 'defense', 'attorney',
                 'advocate', 'public', 'defender', 'solicitor', 'law', 'agent', 'referee',
                 'lawyer', 'prosecutor', 'attorney', 'professional', 'nurse', 'head', 'nurse', 'scrub', 'foster-nurse', 'midwife',
                 'practical', 'nurse', 'graduate', 'nurse', 'matron', 'visiting', 'nurse', 'registered',
                 'nurse', 'practitioner', 'nurse-midwife', 'probationer', 'pharmacist', 'pharmacologist',
                 'practitioner', 'inoculator', 'doctor', 'physician', 'abortionist', 'general',
                 'practitioner', 'intern', 'physician', 'specialist', 'allergist',
                 'angiologist', 'gastroenterologist', 'extern', 'veterinarian', 'surgeon', 'dentist', 'periodontist',
                 'orthodontist', 'pedodontist', 'dental', 'surgeon', 'exodontist', 'prosthodontist', 'endodontist',
                 'medical', 'officer', 'surgeon', 'surgeon', 'general', 'bonesetter', 'medical', 'assistant',
                 'electrologist', 'librarian', 'cataloger', 'craftsman', 'critic', 'literary', 'critic', 'educator', 'schoolmaster',
                 'lector', 'academician', 'professor', 'assistant', 'principal', 'headmistress',
                 'chancellor', 'headmaster', 'housemaster', 'teacher', 'fellow', 'missionary', 'cyril',
                 'dancing-master', 'instructress', 'english', 'teacher', 'catechist', 'schoolteacher', 'games-master',
                 'schoolmarm', 'music', 'teacher', 'piano', 'teacher', 'art', 'teacher', 'governess', 'docent', 'riding',
                 'master', 'demonstrator', 'preceptor', 'coach', 'envoy'])

if __name__ == "__main__":
    pass


