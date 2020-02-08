import re
# # import numpy as np
# # import nltk
# # from collections import defaultdict
# # nltk.download('averaged_perceptron_tagger')
# # Regex_1 = r"'([A-Za-z0-9_\./\\-]*)'"
# # Regex_2 = r" ([A-Za-z0-9_\./\\-]*)' "
# # Regex_3 = r" '([A-Za-z0-9_\./\\-]*)"
# # Regex_5 = r"((^.*\b(not|cannot|never)\b.)|(.n't))"
# #
# # featureDict = {}
# # specialApostropheWords = {"’em", "’tis", "’70s"}
# # negation_ending_tokens = {"but", "nevertheless", "however", ".", "?", "!"}
# #
# #
# # def load_corpus(corpus_path):
# #     values = []
# #     with open(corpus_path, 'r') as f:
# #         lines = f.readlines()
# #
# #     for line in lines:
# #         values.append(line.strip().split(': ')[1])
# #
# #     return values
# #
# #
# # def add_space(match):  #### function to add space in the substring
# #     input_string = match.group(0)
# #     output_string = ""
# #
# #     for word in input_string.split():
# #         if word in specialApostropheWords:
# #             output_string += word + " "
# #         else:
# #             for index, letter in enumerate(word):
# #                 if index == 0 and letter == "’":
# #                     output_string += letter + " "
# #                 elif index == len(word) - 1 and letter == "’":
# #                     output_string += " " + letter
# #                 else:
# #                     output_string += letter
# #
# #             output_string += " "
# #     return output_string.strip()
# #
# #
# # def tokenize(snippet):
# #     token = []
# #     ans_regex_1 = re.sub(Regex_1, add_space, snippet)
# #     ans_regex_2 = re.sub(Regex_2, add_space, ans_regex_1)
# #     ans_regex_3 = re.sub(Regex_3, add_space, ans_regex_2)
# #     return ans_regex_3.split()
# #
# #
# # def tag_edits(tokenized_snippet):
# #     meta_tag = "EDIT_"  # not a part of the vocab, as per slides
# #     isOpenBracket = 0
# #     for i in range(len(tokenized_snippet)):
# #
# #         if tokenized_snippet[i][0] == "[":
# #             isOpenBracket = 1
# #             tokenized_snippet[i] = meta_tag + tokenized_snippet[i][1:]
# #             continue
# #
# #         elif tokenized_snippet[i][-1] == "]":
# #             isOpenBracket = 0
# #             tokenized_snippet[i] = meta_tag + tokenized_snippet[i][0:-1]
# #
# #         if isOpenBracket == 1:
# #             tokenized_snippet[i] = meta_tag + tokenized_snippet[i]
# #
# #     return tokenized_snippet
# #
# #
# # # def tag_negation(tokenized_snippet):
# # #     # Initializations
# # #     orignal = tokenized_snippet
# # #
# # #     changed_index = list()
# # #     new_tokens = list()
# # #     negation_tagged_pos = list()
# # #     meta_tagged_pos = list()
# # #
# # #     meta_tag = "EDIT_"
# # #     negation_tag = "NOT_"
# # #
# # #     counter = 0
# # #     neg_tagger = 0  # Flag for neg_tagger, OFF = 0, ON = 1
# # #
# # #     comparatives = {"JJR", "RBR"}
# # #
# # #     for i in range(len(orignal)):
# # #         if meta_tag in orignal[i]:
# # #             new_tokens.append(orignal[i].replace(meta_tag, ""))
# # #             changed_index.append(i)
# # #         else:
# # #             new_tokens.append(orignal[i])
# # #
# # #     pos_tag_token_input = list()
# # #     for token in new_tokens:
# # #         if token == '':
# # #             continue
# # #         else:
# # #             pos_tag_token_input.append(token)
# # #
# # #     parts_of_speech = nltk.pos_tag(pos_tag_token_input)
# # #
# # #     for word, pos in parts_of_speech:
# # #
# # #         if counter in changed_index:
# # #             meta_tagged_pos.append((meta_tag + word, pos))
# # #         else:
# # #             meta_tagged_pos.append((word, pos))
# # #         counter += 1
# # #
# # #     # TODO: Make it in a single loop
# # #
# # #     for i in range(0, len(meta_tagged_pos)):
# # #
# # #         word, pos = meta_tagged_pos[i]
# # #         match = re.search(Regex_5, word)
# # #
# # #         # Found a negative word and neg_tagger flag is OFF, need to code for corner case
# # #         if match is not None and neg_tagger == 0:
# # #
# # #             if match.group(0) == "not":
# # #                 if i + 1 < len(meta_tagged_pos) and meta_tagged_pos[i + 1][
# # #                     0] == "only":  # next word is only, then don't tag anything
# # #                     neg_tagger = 0
# # #             else:
# # #                 neg_tagger = 1
# # #                 word = word
# # #
# # #         # Found a negative word and the neg_tagger flag is ON, need to code for corner case
# # #         elif match is not None and neg_tagger == 1:
# # #             neg_tagger = 0
# # #
# # #         # Did not find a negative word, but the pos is in the comparative and the neg_tagger flag is ON
# # #         elif match is None and ((pos in comparatives) or (word in negation_ending_tokens)) and neg_tagger == 1:
# # #             neg_tagger = 0
# # #
# # #         # No match found but the neg_tagger flag is ON
# # #         elif match is None and neg_tagger == 1:
# # #             word = negation_tag + word
# # #
# # #         negation_tagged_pos.append((word, pos))
# # #
# # #     return negation_tagged_pos
# # #
# #
# # def get_features(preprocessed_snippet):
# #     meta_tag = "EDIT_"
# #     array = np.zeros((len(featureDict)), dtype=int)
# #     for word, tag in preprocessed_snippet:
# #         if word.find(meta_tag) == -1:
# #             if word in featureDict:
# #                 array[featureDict[word]] += 1
# #     return array
# #
# #
# #
# #
# # training_snippets  = load_corpus("text.txt")
# # tokenlist = list()
# # meta_tag = "EDIT_"
# # negation_tag = "NOT_"
# # counter = 0
# # tag_negate_tokens = list()
# # training_samples = len(training_snippets)
# # Y_train = np.zeros(training_samples, dtype=int)
# # loopCounter = 0
# #
# # for sentence in training_snippets:
# #     token = tokenize(sentence)
# #     # Y_train[loopCounter] = label
# #     tag_edit_token = tag_edits(token)
# #     # tag_negate_token = tag_negation(tag_edit_token)
# #     tag_negate_tokens.append(tag_edit_token)
# #     for word , pos in tag_edit_token:
# #         if word.find(meta_tag) == -1:
# #             if word not in featureDict:
# #                 featureDict[word] = counter
# #                 counter += 1
# #     loopCounter += 1
# #
# # X_train = np.empty([training_samples,len(featureDict)])
# # print(X_train)

# TODO complete the code for pos tag

import numpy as np
from keras.preprocessing import text, sequence


def Punctuation(string):
    # punctuation marks
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

    # traverse the given string and if any punctuation
    # marks occur replace it with null
    for x in string.lower():
        if x in punctuations:
            string = string.replace(x, "")

            # Print string without punctuation
    return string


def datasetText():
    transcript_path = 'text.txt'
    labels_path = 'labels.txt'

    emotion_dict = {'ang': 0,
                    'hap': 1,
                    'exc': 2,
                    'sad': 3,
                    'fru': 4,
                    'fea': 5,
                    'sur': 6,
                    'neu': 7,
                    'xxx': 8,
                    'oth': 8}
    with open(labels_path, 'r') as f:
        content = f.read()
    useful_regex = re.compile(r'\[.+\]\n', re.IGNORECASE)

    # lines=content.splitlines()
    labels = []
    info = {}
    # lines=content.readl
    lines = re.findall(useful_regex, content)
    for line in lines[1:]:
        # print(line)
        wavfile = line.strip().split('\t')[1]
        label = line.strip().split('\t')[2]
        info[wavfile]= label

    with open('text.txt', 'r') as f:
        lines1 = f.readlines()
    texts = {}
    for line in lines1:
        # print(line)
        texts[line.strip().split(': ')[0].split(' ')[0]] = Punctuation(line.strip().split(': ')[1])
    return texts,info
    # train = []
    # for val in info:
    #     for t in texts:
    #         if val[0] == t[0]:
    #             print(val[0])
    #             train.append(t[1])
    #             labels.append(emotion_dict[val[1]])
    # max_features = 12000
    # tokenizer = text.Tokenizer(lower=False, num_words=max_features)
    # tokenizer.fit_on_texts(train)
    # sorted_by_word_count = sorted(tokenizer.word_counts.items(), key=lambda x: x[1], reverse=True)
    # tokenizer.word_index = {}
    # i = 0
    # for word, count in sorted_by_word_count:
    #     if i == max_features:
    #         break
    #     tokenizer.word_index[word] = i + 1  # <= because tokenizer is 1 indexed
    #     i += 1
    # X_train = tokenizer.texts_to_sequences(train)
    # X_train = sequence.pad_sequences(X_train, padding='post')
    # X_train = np.array(X_train)
    # # todo add the count prt here
    # return X_train, labels


datasetText()
