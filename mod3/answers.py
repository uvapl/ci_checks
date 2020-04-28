import numpy as np
import pandas as pd
import spacy as spacy

from numpy import nan
from pandas import DataFrame, Series


def test_1(mse_random):
    print("Testing mse value: ", end='')
    assert mse_random > 3.0, "expected mse value > 3.0"
    assert mse_random < 4.0, "expected mse value < 4.0"
    print("success!")


def test_2(mse_mean, mse_cf_item_based):
    print("Testing mse value: ", end='')
    assert mse_mean < mse_cf_item_based, f"expected mse > {mse_cf_item_based:.4f}"
    print('success!')


def test_4(mse_genres):
    print("Testing mse value: ", end='')
    assert mse_genres > 0.8 and (
        mse_genres < 0.9), "expected mse value around 0.84"
    print("success!")


def test_7(bag_of_words):
    def _test_series(student_answer, solution):
        assert type(
            student_answer) == Series, "Expected your answer to be of type Series"
        assert set(student_answer.index) == set(
            solution.index), f"solution index: {solution.index}, your index: {student_answer.index}"
        for word in solution.index:
            assert solution[word] == student_answer[
                word], f"Expected {solution[word]} at '{word}' but found {student_answer[word]}"

    nlp = spacy.load('en_core_web_sm')

    print(
        "Testing bag of words with a text that does NOT contain punctuation: ",
        end="")
    solution = Series([1, 1, 1], index=['duck', 'cross', 'road']).astype(
        "int64")
    doc = nlp("the duck crossed the road again and again")
    student_answer = bag_of_words(doc)
    _test_series(student_answer, solution)
    print("Succes!")

    print("Testing bag of words with a text that does contain punctuation: ",
          end="")
    solution = Series([1, 1, 1], index=['duck', 'cross', 'road']).astype(
        "int64")
    doc = nlp("the duck crossed, the road again and again.")
    student_answer = bag_of_words(doc)
    _test_series(student_answer, solution)
    print("Succes!")


def test_8(doc, bag_of_words, term_frequency, nlp):
    print(
        "Testing term_frequency on the text 'english english english word word random': ",
        end="")
    solution = Series([0.5, 0.3333333333333333, 0.16666666666666666],
                      index=['english', 'word', 'random']).astype("float64")

    def _test_series(student_answer, solution):
        assert type(
            student_answer) == Series, "Expected your answer to be of type Series"
        assert set(student_answer.index) == set(
            solution.index), f"solution index: {solution.index}, your index: {student_answer.index}"
        for word in solution.index:
            assert solution[word] == student_answer[
                word], f"Expected {solution[word]} at '{word}' but found {student_answer[word]}"

    doc = nlp("english english english word word random")
    bag = bag_of_words(doc)
    student_answer = term_frequency(bag)
    _test_series((student_answer * 100).map(int), (solution * 100).map(int))
    print("Succes!")


def test_9(nlp, bag_of_words, inverse_document_frequency):
    print("Testing inverse document frequency: ", end="")

    def _test_series(student_answer, solution):
        assert type(
            student_answer) == Series, "Expected your answer to be of type Series"
        assert set(student_answer.index) == set(
            solution.index), f"solution index: {solution.index}, your index: {student_answer.index}"
        for word in solution.index:
            assert solution[word] == student_answer[
                word], f"Expected {solution[word]} at '{word}' but found {student_answer[word]}"

    solution = Series(
        [1.0986122886681098, 0.0, 0.4054651081081644, 1.0986122886681098],
        index=['english', 'random', 'roof', 'word']).astype("float64")

    bag1 = bag_of_words(nlp("english english english word word random"))
    bag2 = bag_of_words(nlp("roof roof random"))
    bag3 = bag_of_words(nlp("roof random"))
    student_answer = inverse_document_frequency([bag1, bag2, bag3])
    _test_series(student_answer, solution)
    print("Succes!")


def test_11(nlp, bag_of_words, tf_idf):
    idf = pd.read_pickle('./data/idf.pkl')

    def _test_series(student_answer, solution):
        assert type(
            student_answer) == Series, "Expected your answer to be of type Series"
        assert set(student_answer.index) == set(
            solution.index), f"solution index: {solution.index}, your index: {student_answer.index}"
        for word in solution.index:
            assert solution[word] == student_answer[
                word], f"Expected {solution[word]} at '{word}' but found {student_answer[word]}"

    print("Testing tf_idf: ", end="")
    solution = Series(
        [0.9700282369019263, 0.0649750369302339, 0.4454819004385067],
        index=['english', 'word', 'random']).astype("float64")

    bag = bag_of_words(nlp("english english english word word random"))
    student_answer = tf_idf(bag, idf)
    _test_series(student_answer, solution)
    print("Succes!")


def test_16(frame):
    print("Testing: ", end="")

    frame = frame.astype("float64")
    solution = DataFrame([[1.0, 0.33510586690428007, 0.44173321095505513,
                           0.7517859342501927, 0.5248965400138786],
                          [0.33510586690428007, 1.0, 0.7453350795732844,
                           0.3677176990934737, 0.6501385254601899],
                          [0.44173321095505513, 0.7453350795732844, 1.0,
                           0.48254533689518175, 0.7573737418959072],
                          [0.7517859342501927, 0.3677176990934737,
                           0.48254533689518175, 1.0, 0.580948308759836],
                          [0.5248965400138786, 0.6501385254601899,
                           0.7573737418959072, 0.580948308759836, 1.0]],
                         columns=['recommender', 'warren', 'biden',
                                  'machinelearning', 'brexit'],
                         index=['recommender', 'warren', 'biden',
                                'machinelearning', 'brexit'])
    print(type(frame))
    pd.testing.assert_frame_equal(frame, solution)
    print("Success")


def test_18(compute_vector, nlp_headlines, similarity):
    print("Testing compute_vector: ", end="")
    print("test deleted, will be checked manually.")
