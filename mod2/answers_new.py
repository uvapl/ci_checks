import numpy as np
import pandas as pd

from numpy import nan
from pandas import DataFrame, Series


def test_1(get_rating, ratings):
    print('Check output type: ', end='')
    _solution = get_rating(ratings, 182, 2710)
    _expected_type = float
    assert isinstance(_solution,
                      _expected_type), f'expected output to be of type {_expected_type}, not {type(_solution)}'
    print('success!')

    print('Check get_rating(): ', end='')
    for _user, _movie, _solution in [(182, 2571, 5.0), (182, 2710, 4.5),
                                     (182, 4306, 4.0), (195, 2571, 3.0),
                                     (195, 2710, 1.0), (195, 4306, 3.0),
                                     (204, 2571, 4.5), (204, 2710, 5.0),
                                     (204, 4306, 4.0), (376, 2571, 3.5),
                                     (376, 2710, 1.5), (376, 4306, 4.0),
                                     (542, 2571, 5.0), (542, 2710, 0.5),
                                     (542, 4306, 5.0), (182, 1234, np.nan),
                                     (195, 5678, np.nan)]:
        if np.isnan(_solution):
            assert np.isnan(get_rating(ratings, _user,
                                       _movie)), f'Return NaN if there is no rating for the user/movie combination'
        else:
            assert get_rating(ratings, _user,
                              _movie) == _solution, f'The rating for user {_user}, {_movie} should be {_solution}'
    print('Success!')


def test_2(pivot_ratings, ratings):
    print('Check output type: ', end='')
    _student_answer = pivot_ratings(ratings)
    _expected_type = pd.DataFrame
    assert isinstance(_student_answer,
                      _expected_type), f'expected output to be of type {_expected_type}, not {type(_solution)}'
    print('success!')

    print('Testing pivot on complete data set: ', end='')
    _solution = DataFrame([[5.0, 3.0, 4.5, 3.5, 5.0], [4.5, 1.0, 5.0, 1.5,
                                                       0.5],
                           [4.0, 3.0, 4.0, 4.0, 5.0]],
                          columns=[182, 195, 204, 376, 542],
                          index=[2571, 2710, 4306])
    _student_answer = pivot_ratings(ratings)

    pd.testing.assert_frame_equal(_student_answer, _solution,
                                  check_names=False, check_like=True)
    print('success!')

    print('Testing pivot on incomplete data set: ', end='')

    _incomplete_ratings = DataFrame(
        [[182, 2571, 5.0, 1054779786], [182, 2710, 4.5, 1063284735],
         [195, 4306, 3.0, 994032742], [204, 2571, 4.5, 1327183462],
         [204, 2710, 5.0, 1327185697], [376, 2571, 3.5, 1364994024],
         [376, 2710, 1.5, 1364994544], [542, 2571, 5.0, 1163386800],
         [542, 2710, 0.5, 1163387159]],
        columns=['userId', 'movieId', 'rating', 'timestamp'],
        index=[0, 1, 5, 6, 7, 9, 10, 12, 13])

    _solution = DataFrame(
        [[5.0, nan, 4.5, 3.5, 5.0], [4.5, nan, 5.0, 1.5, 0.5],
         [nan, 3.0, nan, nan, nan]],
        columns=[182, 195, 204, 376, 542],
        index=[2571, 2710, 4306])

    _student_answer = pivot_ratings(_incomplete_ratings)

    pd.testing.assert_frame_equal(_student_answer, _solution,
                                  check_names=False, check_like=True)
    print('success!')


def test_8(create_similarity_matrix_manhattan, utility_matrix):
    print('Check output type: ', end='')
    _student_answer = create_similarity_matrix_manhattan(utility_matrix)
    _expected_type = pd.DataFrame
    assert isinstance(_student_answer,
                      _expected_type), f'expected output to be of type {_expected_type}, not {type(_student_answer)}'
    print('success!')

    print('Testing create_similarity_matrix_manhattan on complete data set: ',
          end='')
    _student_answer = create_similarity_matrix_manhattan(utility_matrix)
    _solution = DataFrame([[1.0, 0.09523809523809523, 0.3333333333333333],
                           [0.09523809523809523, 1.0, 0.08695652173913043],
                           [0.3333333333333333, 0.08695652173913043, 1.0]],
                          columns=[2571, 2710, 4306], index=[2571, 2710, 4306])
    np.testing.assert_allclose(_student_answer, _solution, rtol=1e-1)
    print('success!')

    print(
        'Testing create_similarity_matrix_manhattan on incomplete data set: ',
        end='')
    _utility_matrix = DataFrame(
        [[5.0, nan, 4.5, nan, 5.0], [4.5, nan, 5.0, 1.5, 0.5],
         [nan, 3.0, nan, nan, nan]], columns=[182, 195, 204, 376, 542],
        index=[2571, 2710, 4306])
    _student_answer = create_similarity_matrix_manhattan(_utility_matrix)
    _solution = DataFrame(
        [[1.0, 0.15384615384615385, nan], [0.15384615384615385, 1.0, nan],
         [nan, nan, 1.0]], columns=[2571, 2710, 4306],
        index=[2571, 2710, 4306])
    
    np.testing.assert_allclose(_student_answer, _solution, rtol=1e-1)
    print('success!')
    
    print('\n\nOutput second test:')
    print(_student_answer)

def test_10(create_similarity_matrix_euclid, utility_matrix):
    print('Check output type: ', end='')
    _student_answer = create_similarity_matrix_euclid(utility_matrix)
    _expected_type = pd.DataFrame
    assert isinstance(_student_answer, _expected_type), f'expected output to be of type {_expected_type}, not {type(_expected_type)}'
    print('success!')

    print('Testing create_similarity_matrix_euclid on complete data set: ',
          end='')
    _student_answer = create_similarity_matrix_euclid(utility_matrix)
    _solution = DataFrame([[1.0, 0.1571856809867317, 0.4494897427831781],
                           [0.1571856809867317, 1.0, 0.15072240113145766],
                           [0.4494897427831781, 0.15072240113145766, 1.0]],
                          columns=[2571, 2710, 4306], index=[2571, 2710, 4306])
    np.testing.assert_allclose(_student_answer, _solution, rtol=1e-1)
    print('success!')

    print('Testing create_similarity_matrix_euclid on incomplete data set: ',
          end='')
    _utility_matrix = DataFrame(
        [[5.0, nan, 4.5, nan, 5.0], [4.5, nan, 5.0, 1.5, 0.5],
         [nan, 3.0, nan, nan, nan]], columns=[182, 195, 204, 376, 542],
        index=[2571, 2710, 4306])
    _student_answer = create_similarity_matrix_euclid(_utility_matrix)
    _solution = DataFrame(
        [[1.0, 0.18001097668719743, nan], [0.18001097668719743, 1.0, nan],
         [nan, nan, 1.0]], columns=[2571, 2710, 4306],
        index=[2571, 2710, 4306])
    np.testing.assert_allclose(_student_answer, _solution, rtol=1e-1)
    print('success!')


def test_11(create_similarity_matrix_cosine, utility_matrix):
    print('Check output type: ', end='')
    _student_answer = create_similarity_matrix_cosine(utility_matrix)
    _expected_type = pd.DataFrame
    assert isinstance(_student_answer, _expected_type), f'expected output to be of type {_expected_type}, not {type(_student_answer)}'
    print('success!')

    print('Testing create_similarity_matrix_cosine on complete data set: ',
          end='')
    _solution = DataFrame([[0.9999999999999999, 0.8347319075206693, 0.992843914076249],
                             [0.8347319075206693, 1.0, 0.7829084180122037],
                             [0.992843914076249, 0.7829084180122037, 0.9999999999999998]],
                            columns=[2571, 2710, 4306], index=[2571, 2710, 4306])
    _student_answer = create_similarity_matrix_cosine(utility_matrix)

    np.testing.assert_allclose(_student_answer, _solution, rtol=1e-1)
    print('success!')

    print('Testing create_similarity_matrix_cosine on incomplete data set: ',
          end='')
    
    _utility_matrix = DataFrame(
        [[5.0, nan, 4.5, nan, 5.0], [4.5, nan, 5.0, 1.5, 0.5],
         [nan, 3.0, nan, nan, nan]], columns=[182, 195, 204, 376, 542],
        index=[2571, 2710, 4306])
    
    _solution = DataFrame([[1.0000000000, 0.8347319075, 0.9928439141],
                           [0.8347319075, 1.0000000000, 0.7829084180],
                           [0.9928439141, 0.7829084180, 1.0000000000]],
                          columns=[2571, 2710, 4306], index=[2571, 2710, 4306])
    
    _student_answer = create_similarity_matrix_cosine(_utility_matrix)

    np.testing.assert_allclose(_student_answer, _solution, rtol=1e-1)
    print('success!')

    print(
        "Testing create_similarity_matrix_cosine on a dataset with a movie that has only been rated with 0: ",
        end='')
    
    _utility_matrix = DataFrame(
        [[5.0, nan, 4.5, nan, 5.0], [4.5, nan, 5.0, 1.5, 0.5],
         [0.0, 0.0, 0.0, 0.0, 0.0]], columns=[182, 195, 204, 376, 542],
        index=[2571, 2710, 4306])
    
    _solution = DataFrame([[1.0000000000, 0.8480436727, 0.000000000],
                           [0.8480436727, 1.0000000000, 0.0000000000], 
                           [0.0000000000, 0.0000000000, 0.0000000000]],
                          columns=[2571, 2710, 4306], index=[2571, 2710, 4306])
        
    _student_answer = create_similarity_matrix_cosine(_utility_matrix)

    np.testing.assert_allclose(_student_answer, _solution, rtol=1e-1)
    print('success!')


def test_12(mean_center_columns, utility_matrix,
            create_similarity_matrix_cosine):
    print('Check output type: ', end='')
    _student_answer = mean_center_columns(utility_matrix)
    _expected_type = pd.DataFrame
    assert isinstance(_student_answer, _expected_type), f'expected output to be of type {_expected_type}, not {type(_solution)}'
    print('success!')

    print('Testing mean_center_columns on complete data set: ', end='')
    _solution1 = DataFrame([[0.5, 0.6666666666666665, 0.0, 0.5, 1.5],
                            [0.0, -1.3333333333333335, 0.5, -1.5, -3.0],
                            [-0.5, 0.6666666666666665, -0.5, 1.0, 1.5]],
                           columns=[182, 195, 204, 376, 542],
                           index=[2571, 2710, 4306])
    _student_answer = mean_center_columns(utility_matrix)

    np.testing.assert_allclose(_student_answer, _solution1, rtol=1e-1)
    print('success!')

    print('Testing create_similarity_matrix_cosine on complete data set: ',
          end='')
    _solution2 = DataFrame([[1.0, -0.9426042752501429, 0.8043933497362553],
                            [-0.9426042752501429, 1.0, -0.956600715515994],
                            [0.8043933497362553, -0.956600715515994, 1.0]],
                           columns=[2571, 2710, 4306],
                           index=[2571, 2710, 4306])
    _student_answer = create_similarity_matrix_cosine(_solution1)

    np.testing.assert_allclose(_student_answer, _solution2, rtol=1e-1)
    print('success!')

    print('Testing mean_center_columns on incomplete data set: ', end='')
    
    _utility_matrix = DataFrame(
        [[5.0, nan, 4.5, nan, 5.0], [4.5, nan, 5.0, 1.5, 0.5],
         [nan, 3.0, nan, nan, nan]], columns=[182, 195, 204, 376, 542],
        index=[2571, 2710, 4306])
    
    _solution1 = DataFrame(
        [[0.25, nan, -0.25, nan, 2.25], [-0.25, nan, 0.25, 0.0, -2.25],
         [nan, 0.0, nan, nan, nan]], columns=[182, 195, 204, 376, 542],
        index=[2571, 2710, 4306])
    _student_answer = mean_center_columns(_utility_matrix)

    np.testing.assert_allclose(_student_answer, _solution1, rtol=1e-1)
    print('success!')

    
    print('Testing create_similarity_matrix_cosine on incomplete data set: ',
          end='')
    
    _student_answer = create_similarity_matrix_cosine(_solution1)
    
    _solution2 = DataFrame([[1.0000000000, -0.9516522202, 0.0000000000], 
                            [-0.9516522202, 1.0000000000, 0.0000000000], 
                            [0.0000000000,  0.0000000000, 0.0000000000]],
        columns=[2571, 2710, 4306], index=[2571, 2710, 4306])

    np.testing.assert_allclose(_student_answer, _solution2, rtol=1e-1)
    print('success!')


def test_14(select_neighborhood, similarity, utility_matrix2):
    print('Check output type: ', end='')
    _solution = select_neighborhood(similarity[4306], utility_matrix2[123], 10)
    _expected_type = pd.Series
    assert isinstance(_solution,
                      _expected_type), f'expected output to be of type {_expected_type}, not {type(_solution)}'
    print('success!')

    print('Check select_neighborhood for user 123: ', end='')
    _solution = Series([0.4764040460, 0.2687109722, 0.569404],
                       index=[2571, 4444, 5555]).astype("float64")
    _student_answer = select_neighborhood(similarity[4306],
                                          utility_matrix2[123], 10)

    np.testing.assert_allclose(np.sort(_student_answer), np.sort(_solution),
                               rtol=1e-1)
    print('success!')

    print('Check select_neighborhood for user 456: ', end='')
    _solution = Series([0.4764040460,  0.2687109722, 0.5694040620],
                       index=[2571, 4444, 5555]).astype("float64")
    _student_answer = select_neighborhood(similarity[4306],
                                          utility_matrix2[456], 10)

    np.testing.assert_allclose(np.sort(_student_answer), np.sort(_solution),
                               rtol=1e-1)
    print('success!')

    print('Check select_neighborhood for movie 2710 and user 195: ', end='')
    _solution = Series([1.0], index=[2710]).astype("float64")
    _student_answer = select_neighborhood(similarity[2710],
                                          utility_matrix2[195], 10)

    np.testing.assert_allclose(np.sort(_student_answer), np.sort(_solution),
                               rtol=1e-1)
    print('success!')

    print('Check empty neighborhood: ', end='')
    _dummy1 = pd.Series([np.nan, np.nan, 1.0, 1.0], index=[1, 2, 3, 4])
    _dummy2 = pd.Series([1.0, 1.0, np.nan, np.nan], index=[1, 2, 3, 4])
    _student_answer = select_neighborhood(_dummy1, _dummy2, 10)
    _solution = pd.Series([], index=[])
    np.testing.assert_allclose(np.sort(_student_answer), np.sort(_solution),
                               rtol=1e-1)
    print('success!')

    print('Check only zero similarity: ', end='')
    _dummy1 = pd.Series([0.0, 0.0, 1.0, 1.0], index=[1, 2, 3, 4])
    _dummy2 = pd.Series([1.0, 1.0, np.nan, np.nan], index=[1, 2, 3, 4])
    _student_answer = select_neighborhood(_dummy1, _dummy2, 10)
    _solution = pd.Series([], index=[])
    np.testing.assert_allclose(np.sort(_student_answer), np.sort(_solution),
                               rtol=1e-1)
    print('success!')


def test_15(select_neighborhood, similarity, utility_matrix2, weighted_mean):
    print('Check output type: ', end='')
    _neighborhood1 = select_neighborhood(similarity[4306],
                                         utility_matrix2[123], 10)
    _solution = weighted_mean(_neighborhood1, utility_matrix2[123])
    _expected_type = float
    assert isinstance(_solution,
                      _expected_type), f'expected output to be of type {_expected_type}, not {type(_solution)}'
    print('success!')

    print('Testing predictions: ', end='')
    
    # test prediction 1
    _neighborhood1 = select_neighborhood(similarity[4306],
                                         utility_matrix2[123], 10)
    _prediction1 = weighted_mean(_neighborhood1, utility_matrix2[123])
    assert _prediction1 > 4.1, "expected a predicted rating between 4.1 and 4.2"
    assert _prediction1 < 4.2, "expected a predicted rating between 4.1 and 4.2"
    
    # test prediction 2
    _neighborhood2 = select_neighborhood(similarity[4306],
                                         utility_matrix2[456], 10)
    _prediction2 = weighted_mean(_neighborhood2, utility_matrix2[456])
    assert _prediction2 > 2.2, "expected a predicted rating between 2.2 and 2.3"
    assert _prediction2 < 2.3, "expected a predicted rating between 2.2 and 2.3"
    print('success!')

    print('Testing prediction for empty neighborhood: ', end='')
    _empty = Series(0, index=[])
    _weighted_mean = weighted_mean(_empty, utility_matrix2[123])
    assert _weighted_mean is np.nan, "Expected: NaN"
    print('success!')


def test_17(utility_matrix3):
    print('Check output type: ', end='')
    _expected_type = pd.DataFrame
    assert isinstance(utility_matrix3,
                      _expected_type), f'expected output to be of type {_expected_type}, not {type(utility_matrix3)}'
    print('success!')

    print('Check select_neighborhood for user 123: ', end='')
    _solution = DataFrame(
        [[5.0, 1.5, nan, 4.0, 3.5], [5.0, 4.5, 4.0, 3.0, nan],
         [3.0, 1.0, 3.0, 2.0, nan], [4.5, 5.0, 4.0, 4.0, nan],
         [3.5, 1.5, 4.0, 4.5, nan], [1.5, 5.0, nan, 1.0, 3.5],
         [5.0, 0.5, 5.0, 4.5, nan]], columns=[2571, 2710, 4306, 4444, 5555],
        index=[123, 182, 195, 204, 376, 456, 542])

    np.testing.assert_allclose(np.sort(utility_matrix3), np.sort(_solution),
                               rtol=1e-1)
    print('success!')


def test_19(centered_utility_matrix):
    print('Check output type: ', end='')
    _expected_type = pd.DataFrame
    assert isinstance(centered_utility_matrix,
                      _expected_type), f'expected output to be of type {_expected_type}, not {type(centered_utility_matrix)}'
    print('success!')

    print('Testing mean centered matrix on complete data set: ', end='')
    _solution1 = DataFrame(
        [[1.5, -2.0, nan, 0.5, 0.0], [0.875, 0.375, -0.125, -1.125, nan],
         [0.75, -1.25, 0.75, -0.25, nan], [0.125, 0.625, -0.375, -0.375, nan],
         [0.125, -1.875, 0.625, 1.125, nan], [-1.25, 2.25, nan, -1.75, 0.75],
         [1.25, -3.25, 1.25, 0.75, nan]],
        columns=[2571, 2710, 4306, 4444, 5555],
        index=[123, 182, 195, 204, 376, 456, 542])

    np.testing.assert_allclose(centered_utility_matrix, _solution1, rtol=1e-1)
    print('success!')


def test_20(similarity):
    print('Check output type: ', end='')
    _expected_type = pd.DataFrame
    assert isinstance(similarity,
                      _expected_type), f'expected output to be of type {_expected_type}, not {type(_solution)}'
    print('success!')

    print('Testing similarity of complete data set: ', end='')
    _solution1 = pd.DataFrame(
[[-0.8882176431559489, -0.5913123959890826, -2.7755575615628914e-17, 0.7749555280373117, 0.8278373543847156, 0.9091675835657295, 1.0],
[-0.575159934353966, -0.20149814827784024, -2.7755575615628914e-17, 0.1528941574312877, 0.3629750281770626, 0.6625413488689132, 1.0],     
[-0.6363636363636362, -0.6239181323861147, 0.1528941574312877, 0.6949985619375698, 0.8278373543847156, 0.9185315234055025, 1.0],
[-0.9597599188661678, -0.8386592170224152, -0.6363636363636362, -0.5913123959890826, 0.6625413488689132, 0.7180944542557169, 1.0],
[-0.9597599188661678, -0.8699715932280792, -0.575159934353966, 0.6949985619375698, 0.7749555280373117, 0.9159337782355624, 1.0],
[-0.8882176431559489, -0.8699715932280792, -0.8429430938550735, -0.6239181323861147, 0.3629750281770626, 0.7180944542557169, 1.0],
[-0.8429430938550735, -0.8386592170224152, -0.20149814827784024, 0.9091675835657295, 0.9159337782355624, 0.9185315234055025, 1.0]],
            columns=[123, 182, 195, 204, 376, 456, 542],
            index=[123, 182, 195, 204, 376, 456, 542])
    
    np.testing.assert_allclose(np.sort(similarity), np.sort(_solution1), rtol=0.25)
    print('success!')


def test_21(neighborhood1, neighborhood2):
    print('Check output type: ', end='')
    _expected_type = pd.Series
    assert isinstance(neighborhood1,
                      _expected_type), f'expected output to be of type {_expected_type}, not {type(neighborhood1)}'
    print('success!')

    print('Check select_neighborhood for user 123: ', end='')
    _solution = Series(
        [0.774956, 0.827837, 0.909168],
        index=[195, 376, 542]).astype("float64")
    np.testing.assert_allclose(np.sort(neighborhood1), np.sort(_solution),
                               rtol=1e-1)
    print('success!')

    print('Check select_neighborhood for user 456: ', end='')
    _solution = Series([0.3629750282, 0.7180944543],
                       index=[182, 204]).astype("float64")
    np.testing.assert_allclose(np.sort(neighborhood2), np.sort(_solution),
                               rtol=1e-1)
    print('success!')


def test_22(prediction1, prediction2):
    print('Check output type: ', end='')
    _expected_type = float
    assert isinstance(prediction1,
                      _expected_type), f'expected output to be of type {_expected_type}, not {type(_solution)}'
    print('success!')

    print("Testing predictions: ", end='')
    assert prediction1 > 3.9, "expected a predicted rating for user 123 above 3.9"
    assert prediction1 < 4.1, "expected a predicted rating for user 123 under 4.1"
    assert prediction2 > 3.9, "expected a predicted rating for user 456 above 3.9"
    assert prediction2 < 4.1, "expected a predicted rating for user 456 under 4.1"
    print('success!')


def test_24(utility, similarity):
    print('Check utility: ', end='')
    _partial_utility_solution = DataFrame(
        [[2.5, nan, nan, 4.0, nan, 2.5, 4.0, 2.5, nan, nan],
         [nan, nan, 3.5, nan, 4.0, 4.0, nan, nan, 2.0, nan],
         [3.5, 4.5, nan, 5.0, nan, nan, 4.0, nan, 4.5, 4.5],
         [nan, 3.5, nan, 4.0, nan, nan, 3.0, 4.0, nan, 4.5],
         [nan, 5.0, 4.5, nan, 4.5, 4.0, nan, 3.0, 4.5, 5.0]],
        columns=[68, 105, 177, 182, 232, 600, 603, 606, 608, 610],
        index=[1, 2, 16, 32, 47])
    _partial_student_solution = utility.loc[
        [1, 2, 16, 32, 47], [68, 105, 177, 182, 232, 600, 603, 606, 608, 610]]

    np.testing.assert_allclose(_partial_student_solution,
                               _partial_utility_solution, rtol=6e-1)
    print('success!')

    print('Check similarity: ', end='')
    _partial_similarity_solution = DataFrame(
        [[ 1.0000000000,  0.0057399843,  0.0173653612, -0.0260048241,  0.1013538293],
         [ 0.0057399843,  1.0000000000, -0.2764836030, -0.2554602920, -0.4963292866],
         [ 0.0173653612, -0.2764836030,  1.0000000000, -0.0558476293,  0.2085305218],
         [-0.0260048241, -0.2554602920, -0.0558476293,  1.0000000000,  0.4225094954],
         [ 0.1013538293, -0.4963292866,  0.2085305218,  0.4225094954,  1.0000000000]],
                                     columns=[1, 2, 16, 32, 47],
                                     index=[1, 2, 16, 32, 47])
    
    _partial_student_solution = similarity.loc[
        [1, 2, 16, 32, 47], [1, 2, 16, 32, 47]]

    np.testing.assert_allclose(_partial_student_solution,
                               _partial_similarity_solution, rtol=6e-1)
    print('success!')


def test_25(predict_ratings_item_based, similarity_items, utility_items, test_data):
    print('Computing solution: ', end='')
    _partial_solution = DataFrame(
        [[68, 2, 2.5, 3.4879827138689024], [68, 47, 4.0, 3.457850808877052],
         [68, 110, 2.5, 3.625071363263896], [68, 344, 2.5, 3.4579973887255604],
         [68, 593, 3.5, 3.4942987600576014]],
        columns=['userId', 'movieId', 'rating', 'predicted rating'],
        index=[1, 3, 6, 14, 27])
    _student_solution = predict_ratings_item_based(similarity_items,
                                                   utility_items, test_data[
                                                       ['userId', 'movieId',
                                                        'rating']]).head()

    print('success!')

    print('Testing layout of prediction: ', end='')
    for column in ['userId', 'movieId', 'rating', 'predicted rating']:
        assert column in _student_solution.columns, f'expected column {column} in output'

    print('success!')

    print('Testing values of prediction: ', end='')
    np.testing.assert_allclose(_student_solution, _partial_solution, rtol=2e-1)
    print('success!')


def test_26(mse, predicted_item_based):
    print('Testing mse item based: ', end='')
    np.testing.assert_allclose(mse(predicted_item_based), 0.6, rtol=0.15)
    print('success!')


def test_27(mse, predicted_user_based, predicted_item_based):
    print('Testing user based versus item based prediction: ', end='')
    assert mse(predicted_user_based) > mse(
        predicted_item_based), "expected a slightly lower error for item based prediction"
    print('success!')


def test_28(mse_random):
    print('Testing: ', end='')
    _student_solution = mse_random
    _solution = 3.2
    assert np.allclose(_student_solution, _solution,
                       atol=0.5), f'expected value around {_solution}'
    print('success!')


def test_29(mse_item_mean):
    print('Testing: ', end='')
    _student_solution = mse_item_mean
    _solution = 0.68
    assert np.allclose(_student_solution, _solution,
                       atol=0.2), f'expected value around {_solution}'
    print('success!')


def test_31(recommended, hidden, predicted_item_based, treshold_recommended):
    print('Testing: ', end='')
    tol = 5

    recommended_items_solution = 448
    recommended_items = recommended(predicted_item_based,
                                    treshold_recommended).shape[0]

    assert abs(recommended_items - recommended_items_solution) < tol, f'expected value around {recommended_items_solution}'

    hidden_items_solution = 545
    hidden_items = hidden(predicted_item_based, treshold_recommended).shape[0]

    assert abs(hidden_items - hidden_items_solution) < tol, f'expected value around {hidden_items_solution}'

    print('success!')


def test_32(used, unused, predicted_item_based, treshold_used):
    print('Testing: ', end='')
    tol = 5

    used_items_solution = 503
    used_items = used(predicted_item_based, treshold_used).shape[0]

    assert abs(used_items_solution - used_items) < tol, f'expected value around {used_items_solution}'

    unused_items_solution = 490
    unused_items = unused(predicted_item_based, treshold_used).shape[0]

    assert abs(unused_items_solution - unused_items) < tol, f'expected value around {unused_items_solution}'

    print('success!')


def test_33(confusion_matrix):
    print('Testing: ', end='')
    solution = DataFrame([[314, 134], [189, 356]], columns=['used', 'unused'],
                         index=['recommended', 'hidden'])

    pd.testing.assert_frame_equal(confusion_matrix, solution)
    print('success!')


def test_34(precision_item_based):
    print('Testing: ', end='')
    _student_solution = precision_item_based
    _solution = 0.7
    assert np.allclose(_student_solution, _solution,
                       atol=0.2), f'expected value around {_solution}'
    print('success!')


def test_35(recall_item_based):
    print('Testing: ', end='')
    _student_solution = recall_item_based
    _solution = 0.62
    assert np.allclose(_student_solution, _solution,
                       atol=0.2), f'expected value around {_solution}'
    print('success!')
