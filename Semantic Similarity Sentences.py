import nltk
from nltk import word_tokenize
from nltk import sent_tokenize
from nltk.corpus import wordnet as wn
import numpy as np
from nltk.corpus import brown
import math
#CC coordinating conjunction
#CD cardinal digit
#DT determiner
#EX existential there (like: “there is” … think of it like “there exists”)
#FW foreign word
#IN preposition/subordinating conjunction
#JJ adjective ‘big’
#JJR adjective, comparative ‘bigger’
#JJS adjective, superlative ‘biggest’
#LS list marker 1)
#MD modal could, will
#NN noun, singular ‘desk’
#NNS noun plural ‘desks’
#NNP proper noun, singular ‘Harrison’
#NNPS proper noun, plural ‘Americans’
#PDT predeterminer ‘all the kids’
#POS possessive ending parent’s
#PRP personal pronoun I, he, she
#PRP$ possessive pronoun my, his, hers
#RB adverb very, silently,
#RBR adverb, comparative better
#RBS adverb, superlative best
#RP particle give up
#TO, to go ‘to’ the store.
#UH interjection, errrrrrrrm
#VB verb, base form take
#VBD verb, past tense took
#VBG verb, gerund/present participle taking
#VBN verb, past participle taken
#VBP verb, sing. present, non-3d take
#VBZ verb, 3rd person sing. present takes
#WDT wh-determiner which
#WP wh-pronoun who, what
#WP$ possessive wh-pronoun whose
#WRB wh-abverb where, when
PHI = 0.2
BETA = 0.45
ALPHA = 0.2
DELTA = 0.9
ETA = 0.4
total_words = 0
word_freq_brown = {}
def proper_synset(word_one , word_two):
    pair = (None,None)
    maximum_similarity = -1
    synsets_one = wn.synsets(word_one)
    synsets_two = wn.synsets(word_two)
    if(len(synsets_one)!=0 and len(synsets_two)!=0):
        for synset_one in synsets_one:
            for synset_two in synsets_two:
                similarity = wn.path_similarity(synset_one,synset_two)
                if(similarity == None):
                    sim = -2
                elif(similarity > maximum_similarity):
                    maximum_similarity = similarity
                    pair = synset_one,synset_two
    else:
        pair = (None , None)
    return pair
def length_between_words(synset_one , synset_two):
    length = 100000000
    if synset_one is None or synset_two is None:
        return 0
    elif(synset_one == synset_two):
        length = 0
    else:
        words_synet1 = set([word.name() for word in synset_one.lemmas()])
        words_synet2 = set([word.name() for word in synset_two.lemmas()])
        if(len(words_synet1) + len(words_synet2) > len(words_synet1.union(words_synet2))):
            length = 0
        else:
            length = synset_one.shortest_path_distance(synset_two)
            if(length is None):
                return 0
    return math.exp( -1 * ALPHA * length)
def depth_common_subsumer(synset_one,synset_two):
    height = 100000000
    if synset_one is None or synset_two is None:
        return 0
    elif synset_one == synset_two:
        height = max([hypernym[1] for hypernym in synset_one.hypernym_distances()])
    else:
        hypernym_one = {hypernym_word[0]:hypernym_word[1] for hypernym_word in synset_one.hypernym_distances()}
        hypernym_two = {hypernym_word[0]:hypernym_word[1] for hypernym_word in synset_two.hypernym_distances()}
        common_subsumer = set(hypernym_one.keys()).intersection(set(hypernym_two.keys()))
        if(len(common_subsumer) == 0):
            height = 0
        else:
            height = 0
            for cs in common_subsumer:
                val = [hypernym_word[1] for hypernym_word in cs.hypernym_distances()]
                val = max(val)
                if val > height : height = val

    return (math.exp(BETA * height) - math.exp(-BETA * height))/(math.exp(BETA * height) + math.exp(-BETA * height))
def word_similarity(word1,word2):
    synset_wordone,synset_wordtwo = proper_synset(word1,word2) 
    return length_between_words(synset_wordone,synset_wordtwo) * depth_common_subsumer(synset_wordone,synset_wordtwo)

def I(search_word):
    global total_words
    if(total_words == 0):
        for sent in brown.sents():
            for word in sent:
                word = word.lower()
                if word not in word_freq_brown:
                    word_freq_brown[word] = 0
                word_freq_brown[word] +=1
                total_words+=1
    count = 0 if search_word not in word_freq_brown else word_freq_brown[search_word]
    ret = 1.0 - (math.log(count+1)/math.log(total_words+1))
    return ret
def most_similar_word(word,sentence):
    most_similarity = 0
    most_similar_word = ''
    for w in sentence:
        sim  =  word_similarity(w,word)
        if sim > most_similarity:
            most_similarity = sim
            most_similar_word = w
    if most_similarity <= PHI:
        most_similarity = 0
    return most_similar_word,most_similarity 

def gen_sem_vec(sentence , joint_word_set):
    semantic_vector = np.zeros(len(joint_word_set))
    i = 0
    for joint_word in joint_word_set:
        sim_word = joint_word 
        beta_sim_measure = 1
        if (joint_word in sentence):
            pass
        else:
            sim_word,beta_sim_measure = most_similar_word(joint_word,sentence)
            beta_sim_measure = 0 if beta_sim_measure <= PHI else beta_sim_measure
        sim_measure = beta_sim_measure * I(joint_word) * I(sim_word)
        semantic_vector[i] = sim_measure
        i+=1
    return semantic_vector
def sent_sim(sent_set_one, sent_set_two , joint_word_set):
    sem_vec_one = gen_sem_vec(sent_set_one,joint_word_set)
    sem_vec_two = gen_sem_vec(sent_set_two,joint_word_set)
    return np.dot(sem_vec_one,sem_vec_two.T) / (np.linalg.norm(sem_vec_one) * np.linalg.norm(sem_vec_two))
def word_order_similarity(sentence_one , sentence_two):
    token_one  = word_tokenize(sentence_one)
    token_two = word_tokenize(sentence_two)
    joint_word_set = list(set(token_one).union(set(token_two)))
    r1 = np.zeros(len(joint_word_set))
    r2 = np.zeros(len(joint_word_set))

    en_joint_one = {x[1]:x[0] for x in enumerate(token_one)}
    en_joint_two = {x[1]:x[0] for x in enumerate(token_two)}
    set_token_one = set(token_one)
    set_token_two = set(token_two)
    i = 0

    for word in joint_word_set:
        if word in set_token_one:
            r1[i] = en_joint_one[word]
        else:
            sim_word , sim = most_similar_word(word , list(set_token_one))
            if sim > ETA : 
                r1[i] = en_joint_one[sim_word]
            else:
                r1[i] = 0
        i+=1
    j = 0
    for word in joint_word_set:
        if word in set_token_two:
            r2[j] = en_joint_two[word]
        else:
            sim_word , sim = most_similar_word(word , list(set_token_two))
            if sim > ETA : 
                r2[j] = en_joint_two[sim_word]
            else:
                r2[j] = 0
        j+=1
    return 1.0 - (np.linalg.norm(r1 - r2) / np.linalg.norm(r1 + r2))
def main(sentence_one,sentence_two):
    sent_set_one = set(filter(lambda x : not (x == '.' or x == '?') , word_tokenize(sentence_one)))
    sent_set_two = set(filter(lambda x : not (x == '.' or x == '?') , word_tokenize(sentence_two)))
    joint_word_set = list(sent_set_one.union(sent_set_two)) 
    sentence_similarity = (DELTA * sent_sim(sent_set_one,sent_set_two,list(joint_word_set))) + ((1.0 - DELTA) * word_order_similarity(sentence_one,sentence_two))
    return sentence_similarity
def file_sem(f):
    contents = open(f).read().strip()
    ind_sentences = sent_tokenize(contents)
    print(ind_sentences)
    no_of_sentences = len(ind_sentences)
    sent_sim_matr = np.zeros((no_of_sentences,no_of_sentences))
    i = 0
    print(ind_sentences)
    while(i < no_of_sentences):
        j = i
        while(j < no_of_sentences):
            sent_sim_matr[i][j] = main(ind_sentences[i],ind_sentences[j])
            sent_sim_matr[j][i] = sent_sim_matr[i][j]
            j+=1
        i+=1
    return sent_sim_matr
def intro():
    print("\nEnter the option you want:\n")
    print("1.Sentence Similarity between a sentence which is typed in and files containing different sentences.")
    print("2.Sentence similarity between two sentences.")
    print("3.Sentence similarity between a sentence and a list sentences you type in.\n")
    option = int(input("Your choice : "))
    if option == 1:
        sent_one = input("Enter the sentence you want to compare senmantically :")
        #tokens=nltk.word_tokenize(sent_one)
        #print("Parts of Speech: ", nltk.pos_tag(tokens))
        filepath = 'listsentences.txt'
        with open(filepath) as fp:
            line = fp.readline()
            cnt = 1
            while line:
                print("Sentence {}: {}".format(cnt, line.strip()))
                line = fp.readline()
                cnt += 1
        with open(filepath) as fp1:
            sentences = fp1.readlines()
            for sentence in sentences:
                #token1=nltk.word_tokenize(sentence)
                #print("Parts of Speech: ", nltk.pos_tag(token1))
                prob_sim_sent = main(sent_one , sentence)
                print(prob_sim_sent)
    elif option == 2:
        sent_one = input("Enter the first sentence : ")
        #tokens=nltk.word_tokenize(sent_one)
        #print("Parts of Speech: ", nltk.pos_tag(tokens))
        sent_two = input("Enter the second sentence : ")
        #token1=nltk.word_tokenize(sent_two)
        #print("Parts of Speech: ", nltk.pos_tag(token1))
        prob_sim_sent = main(sent_one , sent_two)
        print(prob_sim_sent)
        #print("Similarity between\n"+sent_one+"\n"+sent_two+"\n\n is : ",prob_sim_sent)
    elif option ==3:
        sent_one = input("Enter the first sentence :")
        #tokens=nltk.word_tokenize(sent_one)
        #print("Parts of Speech: ", nltk.pos_tag(tokens))
        list_sentences =[]
        n = int(input("Enter number sentences: "))
        for i in range(0,n):
            ele= input()
            list_sentences.append(ele)
        for sentence in list_sentences:
            #token1=nltk.word_tokenize(sentence)
            #print("Parts of Speech: ", nltk.pos_tag(token1))
            prob_sim_sent = main(sent_one,sentence)
            print(prob_sim_sent)
    else:
        global max_count
        if max_count < 4 : print("Wrong Choice Try again"); max_count+=1 
        else: print("Wrong choice time exceeded!");exit()
        intro()
if __name__ == "__main__":  
    print("-------------------Sentence Similarity--------------------------")
    intro()
    print("Want to try once again? if yes press 1 or else 0")
    excited = int(input())
    while(excited == 1):
        intro()
        print("Want to try once again?")
        excited = int(input())