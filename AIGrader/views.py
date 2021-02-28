"""
Routes and views for the flask application.
"""

from datetime import datetime
from flask import render_template
from AIGrader import app
from flask import request
from nltk import word_tokenize, pos_tag, sent_tokenize, Text
from collections import Counter
from enchant.checker import SpellChecker
from gingerit.gingerit import GingerIt
from keras.models import load_model
import tensorflow as tf
import language_tool_python
import spacy
from spacy.matcher import Matcher
from AIGrader import semantic_similarity
from langdetect import detect
import math



#@app.route('/home')
@app.route('/')
def hello():
    global question_input
    global text_input
    global lan_question
    global lan_text
    global word_count

    question_input = 'Enter your question here... '
    text_input = 'Enter your essay here...'
    lan_question = 'en'
    lan_text = 'en'
    word_count = 101

    return render_template(
        'WebPage1.html', question = question_input, text = text_input,lan_question=lan_question, lan_text = lan_text, word_count=word_count
        )

@app.route('/submit', methods=['POST'])
def submit_textarea():
    global score
    load_keras_model()

    topic_essay = request.form.get("question")

    # store the given text in a variable
    text = request.form.get("text")

    # split the text to get each line in a list
    text2 = text.split()
    # Total number of word counts

    lan_question = detect(topic_essay)
    lan_text = detect(text)

    word_count = wordCount(text)

    if lan_question != 'en':

        score = None
        question = topic_essay
        text = text
        lan_question = lan_question

        return render_template('WebPage1.html',question = topic_essay, text = text, lan_question = lan_question, word_count = word_count)
    
    elif word_count <= 100 or lan_text != 'en':

        score = None
        question = topic_essay
        text = text
        lan_text = lan_text

        return render_template('WebPage1.html',question = topic_essay, text = text, lan_text = lan_text,lan_question=lan_question,word_count = word_count)


    else:

        word_limit = 600
        total_sentence, long_sentences = longSentence(text)
        tense = determine_tense_input(text)
        verb_counter = verbCounter(text)
        spell_checker = spellingChecker(text)
        grammar_checker = grammarChecker(text)
        unique_counter = uniqueVocabulary(text)
        passive_sentences = passiveSentences(text)
        semantic_similarity = semanticSimilarity(text)
        topic_similarity = essayTopic(topic_essay,text)

        limit_ratio = 1 - (word_count / word_limit)
        long_sentences = long_sentences / total_sentence
        max_tense = max(tense.values())
        tense_density = max_tense / (tense["past"] + tense["present"] + tense["future"])
        spell_error = spell_checker / word_count
        grammar_error = grammar_checker / word_count
        unique_vocabulary = unique_counter / word_count
        word_counter_encoded = (0.27 / 230) * word_count
        sentence_encoded = (0.22 / 13) * total_sentence
        semantic_score = (0.413 / 0.949) * semantic_similarity
        topic_score = (0.56 / 0.96) * topic_similarity
        passive_score = (total_sentence - passive_sentences) / total_sentence

        # change the text (add 'Hi' to each new line)
        text_changed = ''.join(['<br>Hi ' + line for line in text2])
        # (i used <br> to materialize a newline in the returned value)
        print(long_sentences, tense_density, spell_error, grammar_error,unique_vocabulary, word_counter_encoded, sentence_encoded,passive_sentences,semantic_similarity,topic_similarity)
        print(grammar_checker,spell_checker,unique_counter)
        results = model.predict([[limit_ratio, long_sentences,passive_score, tense_density, spell_error, grammar_error,semantic_score,topic_score,unique_vocabulary, word_counter_encoded, sentence_encoded]]) > 0.3

        get_index = []
        for index, result in enumerate(results[0]):
            if result == True:
                get_index.append(index)

        score = int((max(get_index) / 12) * 30)

        if word_count <= 420:
            score -= 1
        else:
            score += 1
        
    
        if grammar_checker >= 10:
            score -= 1
        else:
            score += 1

        if unique_counter >= 200:
            score += 1
        else:
            score -= 1

        if spell_checker >= 3:
            score -= 1
        else:
            score += 1


        #return {"You entered: {}".format(grammar_checker),"Total Word Count : {}".format(word_count)}
        #return 'Total sentences: {} Total long sentences{} tense{} result{} tense_density{} max_tense {} grammar_error{}'.format(total_sentence, long_sentences, max_tense,score,tense_density,max_tense, grammar_checker)
        return render_template('WebPage1.html',value = score,question = topic_essay, text = text, lan_question = lan_question, lan_text = lan_text, word_count = word_count)

def wordCount(text):
    
    text = text.split()
    return len(text)

def longSentence(text):
    
    long_sentence_counter = 0
    total_sentence = text.count('.')
    list_of_sentences = text.split(".")
    
    for sentence in list_of_sentences:
        if len(sentence) > 15:
            long_sentence_counter =+ 1
            
    return total_sentence, long_sentence_counter


def determine_tense_input(sentence):
    text = word_tokenize(sentence)
    tagged = pos_tag(text)

    tense = {}
    tense["future"] = len([word for word in tagged if word[1] == "MD"])
    tense["present"] = len([word for word in tagged if word[1] in ["VBP", "VBZ","VBG"]])
    tense["past"] = len([word for word in tagged if word[1] in ["VBD", "VBN"]]) 
    return(tense)

def verbCounter(text):
    
    tokens = word_tokenize(text.lower())
    text = Text(tokens)
    tags = pos_tag(text)

    counts = Counter(tag for word, tag in tags)
    return counts

def spellingChecker(text):
    chkr = SpellChecker("en_US", text)
    chkr.set_text(text)
    errors_count = 0
    for err in chkr:
        errors_count += 1 

    return errors_count


def grammarChecker(text):

    list_of_sentences = text.split(".")
    total_mistakes = 0
    
    for sentence in list_of_sentences:
    
        mistakes = len(ginger(sentence)['corrections'])
        total_mistakes = total_mistakes + mistakes


    #parser = GingerIt()
    #grammar_error_counter = len(parser.parse(text)['corrections'])
    #tool = language_tool_python.LanguageTool('en-US')
    #grammar_error_counter = tool.check(text)

    #print(grammar_error_counter)
    return total_mistakes



def uniqueVocabulary(text):

    text = text.lower()
    words = text.split()
    words = [word.strip('.,!;()[]') for word in words]
    words = [word.replace("'s", '') for word in words]

    unique_counter = 0
    unique = []

    for word in words:
        if word not in unique:
            unique.append(word)
            unique_counter += 1

    return unique_counter

def passiveSentences(text):

    nlp = spacy.load('en_core_web_sm')
    matcher = Matcher(nlp.vocab)

    passive_rule = [[{'DEP':'nsubjpass'},{'DEP':'aux','OP':'*'},{'DEP':'auxpass'},{'TAG':'VBN'}]]
    matcher.add('passive_rule',passive_rule)

    doc = nlp(text)
    matches = matcher(doc)
         
    return len(matches)

def semanticSimilarity(text):
    sentences = sent_tokenize(text)
    essay_semantic_similarity = semantic_similarity.intra_para_semantic_similarity(text)
    num_of_sentences = len(sentences)
    essay_semantic_similarity = essay_semantic_similarity * (math.log(num_of_sentences, 2))
    
    return essay_semantic_similarity

def essayTopic(topic_essay,text):
    sentences = sent_tokenize(text)
    num_of_sentences = len(sentences)
    topic_essay_semantic_similarity = semantic_similarity.inter_para_semantic_similarity(topic_essay, text)
    topic_essay_semantic_similarity = topic_essay_semantic_similarity * (math.log(num_of_sentences, 2))
    
    return topic_essay_semantic_similarity

def ginger(text):
    parser = GingerIt()
    grammar_error_counter = parser.parse(text)
    return grammar_error_counter
    

def load_keras_model():

    global model

    model = load_model('C:/Users/User/source/repos/AIGrader/AIGrader/aigrader2.h5')
    



def home():
    """Renders the home page."""
    return render_template(
        'index.html',
        title='Home Page',
        year=datetime.now().year,
    )

@app.route('/contact')
def contact():
    """Renders the contact page."""
    return render_template(
        'contact.html',
        title='Contact',
        year=datetime.now().year,
        message='Your contact page.'
    )

@app.route('/about')
def about():
    """Renders the about page."""
    return render_template(
        'about.html',
        title='About',
        year=datetime.now().year,
        message='Your application description page.'
    )
