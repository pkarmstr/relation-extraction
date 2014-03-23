"""cleaning the training, dev and test data using everything that coreNLP
provides"""

__author__ = 'keelan'

from file_reader import FeatureRow, all_stanford
import sys


class Cleaner:
    """better than a Polish maid"""

    def __init__(self, input_file, tokenized):
        self.data_dict = all_stanford
        self.bad_rows = 0
        self.total_rows = 0
        self.original_data = self.open_gold_data(input_file)
        self.tokenized = self.open_tokenization(tokenized)
        self.log = []

    def write_log(self):
        with open("cleaning_log.txt", "w") as f_out:
            f_out.write("\n".join(self.log))

    def preprocess_tokenize(self, output_file):
        out = []
        for fr in self.original_data:
            out.append("{:s} | {:s}".format(" ".join(fr.token.split("_")),
                                            " ".join(fr.token_ref.split("_"))))
        with open(output_file, "w") as f_out:
            f_out.write("\n".join(out))


    def open_gold_data(self, gold_file):
        original_data = []
        with open(gold_file, "r") as f_in:
            for line in f_in:
                line = line.rstrip().split()
                if line == []:
                    continue
                if len(line) == 11:
                    line.extend(["", "", ""])
                else:
                    line.insert(-1, "")
                    line.insert(-1, "")
                original_data.append(FeatureRow(*line))
                self.total_rows += 1
        return original_data

    def open_tokenization(self, f):
        tokenized = []
        with open(f, "r") as f_in:
            for line in f_in:
                line = line.rstrip().split("|")
                tokenized.append((line[0].strip().split(), line[1].strip().split()))
        return tokenized

    def get_correct_offset(self, tokenized, sentence, offset_begin, offset_end):
        if len(tokenized) > offset_end - offset_begin:
            offset_end = len(tokenized) + offset_begin

        if tokenized == sentence[offset_begin:offset_end]:
            return (offset_begin, offset_end)
        while tokenized != sentence[offset_begin:offset_end]:
            offset_begin += 1
            offset_end += 1
            if offset_end > len(sentence):
                #raise IndexError("{:d} invalid index, token={:s}".format(offset_end, tokenized))
                self.bad_rows += 1
                return (-1, -1)
        return (offset_begin, offset_end)

    def build_new_data(self):
        referent_cache = []
        j_offset = []
        all_new = []
        for i, fr in enumerate(self.original_data):
            try:
                nlp_data = self.data_dict[fr.article]
            except KeyError:
                self.update_cache(fr.article)
                nlp_data = self.data_dict[fr.article]

            referent_tmp = [fr.j_token, fr.j_sentence]
            if referent_tmp != referent_cache:
                j_sentence = nlp_data["sentences"][int(fr.j_sentence)]["text"]
                tokenized_referent = self.tokenized[i][1]
                begin_ref, end_ref = self.get_correct_offset(tokenized_referent,
                                                      j_sentence,
                                                      int(fr.offset_begin_ref)-1,
                                                      int(fr.offset_end_ref)-1
                )
                j_offset = [tokenized_referent, begin_ref, end_ref]
                referent_cache = referent_tmp

            tokenized = self.tokenized[i][0]
            i_sentence = nlp_data["sentences"][fr.i_sentence]["text"]
            offset_begin, offset_end = self.get_correct_offset(tokenized,
                                                               i_sentence,
                                                               int(fr.offset_begin)-1,
                                                               int(fr.offset_end)-1
            )
            if offset_begin == -1 or j_offset[1] == -1:
                if offset_begin == -1:
                    self.log.append("word: {:s} | offset: ({}, {}) | sentence: {:s}".format(
                        tokenized, fr.i_offset_begin, fr.i_offset_end, i_sentence))
                else:
                    self.log.append("word: {:s} | offset: ({}, {}) | sentence: {:s}".format(
                        j_offset[0], fr.j_offset_begin, fr.j_offset_end, j_sentence))

                continue
            new_row = " ".join([fr.article, fr.sentence, str(offset_begin), str(offset_end),
                                fr.entity_type, "_".join(tokenized), fr.sentence_ref,
                                str(j_offset[1]), str(j_offset[2]), fr.entity_type_ref,
                                "_".join(j_offset[0]), fr.is_referent])
            all_new.append(new_row)
        print "{:d} out of {:d} rows didn't have a match".format(self.bad_rows, self.total_rows)
        self.write_log()
        return all_new

    def write_new(self, file_name, data):
        with open(file_name, "w") as f_out:
            f_out.write("\n".join(data))

if __name__ == "__main__":
    c = Cleaner(sys.argv[1], sys.argv[2], sys.argv[3])
    data = c.build_new_data()
    c.write_new(sys.argv[4], data)
    print "DONE"