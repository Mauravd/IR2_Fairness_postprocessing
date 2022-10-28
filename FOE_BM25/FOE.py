import numpy as np
from cvxopt import spmatrix, matrix, sparse, solvers

# import from other files
from csvPrinting import createPCSV
import Birkhoff
from bm25 import generate_pre_ranking
solvers.options['show_progress'] = False

def runFOEIR(ranking, dataSetName, algoName, k=40, query_rep=1):
    """
    Start the calculation of the ranking for FOEIR under a given fairness constraint
    either Disparate Impact (DI), Disparate Treatment (DT), or Demographic Parity (DP)

    @param ranking_backup: List of candidate objects ordered color-blindly
    @param dataSetName: Name of the data set the candidates are from
    @param algoName: Name of inputed algorithm
    @param k: Length of the ranking to return, if longer than default,
    set to default because otherwise computation will run out of memory

    return the new ranking and the path where to print the ranking to
    """
    # I used this one to be safe and not change ranking, actually now ranking itself is a backup and I modified
    # ranking_backup in the code (only in baseline mode)
    ranking_backup = ranking.copy()
    ranking_scores = ranking_backup['y_pred']
    # initialize as empty string
    rankingResultsPath = ''

    # 
    # ranking_backup.sort(key=lambda candidate: candidate.learnedScores, reverse=True)

    # set k to maximum default value
    if k > len(ranking_scores):
        k = len(ranking_scores)
    
    # initialize rankings as repetitions of the original ranking
    # in case LP cannot find any solutions, the original ranking will be outputted as the solution
    # TODO?
    # rankings = query_rep * [ranking[0:k]]


    # check for which constraint to comput the ranking
    if algoName == 'FOEIR-DTC':

        x, isRanked = solveLPWithDTC(ranking_backup, k, dataSetName, algoName)

        # return x
    if isRanked == True:

        x = np.reshape(x, (k, k))
        x = np.asarray(x, dtype='float64')

        helperPath = algoName + 'Ranking'

        # crate csv file with doubly stochastic matrix inside
        createPCSV(x, dataSetName, helperPath, k)

        # creat the new ranking, if not possible, isRanked will be false and newRanking
        # will be equal to ranking
        # rankings, isRanked = createRanking(x, ranking_backup, k, scores=scores, query_rep=query_rep)
        rankings, isRanked = createRanking(x, ranking_backup, k, query_rep=query_rep)
    else:
        createPCSV(np.array([]), algoName, 'NO_SOL_FOUND', ranking[0].query)
    rankingResultsPath = algoName + '/' + dataSetName + "ranking.csv"

    return rankings, rankingResultsPath, True  # True -> isRanked 


def createRanking(x, nRanking, k, query_rep=1):
    """
    Calculates the birkhoff-von-Neumann decomopsition using package available at
    https://github.com/jfinkels/birkhoff

    @param x: doubly stochastic matrix
    @param nRanking: nRanking: List with candidate objects from the data set ordered color-blindly
    @param k: length of the ranking

    return the a list with candidate objects ordered according to the new ranking
    """

    result = Birkhoff.birkhoff_von_neumann_decomposition(x)
    
    for r in result:
        if not np.array_equal(np.sum(r[1], axis=1), np.ones(r[1].shape[0])) or \
                not np.array_equal(np.sum(r[1], axis=0), np.ones(r[1].shape[0])):
            result.remove(r)
            createPCSV(r[1], str(nRanking['query_id']), 'NOT_VALID_PERMUTATION', r[0])

    # instead of choosing the matrix with highest probability we sample based on the coefficients (probabilities)
    matrices = [i[1] for i in result]
    coefficients = [i[0] for i in result]  # [i[0] if i[0] > 0 else 0 for i in result]

    # normalize coefficients between 0 and 1
    if len(coefficients) > 1:
        coefficients_max = max(coefficients)
        coefficients_min = min(coefficients)
        coefficients = [(i - coefficients_min) / (coefficients_max - coefficients_min) for i in coefficients]

    # normalize coefficients to sum to 1
    coefficients = np.array(coefficients)
    coefficients /= sum(coefficients)

    try:
        matrix_indices = np.random.choice(list(range(len(matrices))), query_rep, p=coefficients)
        matrix_indices = np.random.choice(list(range(len(matrices))), query_rep, p=coefficients)
        rankings = [matrices[i] for i in matrix_indices]
    except:
        print('query number: ', nRanking['query'])

    # back up the original ranking
    nRanking_orig = nRanking.copy()
    for ranking in rankings:
        nRanking = nRanking_orig.copy()
        # get positions of each document
        positions = np.nonzero(ranking)[1]
        # candidate_indices = np.nonzero(ranking)[0]
        # convert numpy array to iterable list
        positions = positions.tolist()
        # candidate_indices = candidate_indices.tolist()
        

        # sort candidates according to new index
        print(nRanking)
        for i in nRanking:
            if i == 'query_id':
                continue
            Z = [x for _, x in sorted(zip(positions, nRanking[i]))]
            nRanking[i] = Z

    return nRanking, True


def solveLPWithDTC(ranking, k, dataSetName, algoName):
    """
    Solve the linear program with DTC

    @param ranking: list of candidate objects in the ranking
    @param k: length of the ranking
    @param dataSetName: Name of the data set the candidates are from
    @param algoName: Name of inputed algorithm

    return doubly stochastic matrix as numpy array
    """

    print('Start building LP with DTC.')
    # to return to the square matrix replace all 'm's with 'k'
    # calculate the attention vector v using 1/log(1+indexOfRanking)
    u = []
    unproU = 0
    proU = 0
    proCount = 0
    unproCount = 0
    proListX = []
    unproListX = []

    # for candidate in ranking[:k]:
    #     u.append(candidate.learnedScores)
    for score in ranking['y_pred']:
        u.append(score)

    # initialize v with DCG
    v = np.arange(1, (k + 1), 1)
    v = 1 / np.log2(1 + v + 1)
    v = np.reshape(v, (1, k))

    arrayU = np.asarray(u)
    
    # normalize input
    arrayU = (arrayU - np.min(arrayU)) / (np.max(arrayU) - np.min(arrayU))
    
    I = []
    J = []
    I2 = []

    # set up indices for column and row constraints
    for j in range(k * k):
        J.append(j)

    for i in range(k):
        for j in range(k):
            I.append(i)

    for i in range(k):
        for j in range(k):
            I2.append(j)

    for i in range(k):
        # if ranking[i].isProtected:
        if ranking['groups'][i] == 'L':
            proCount += 1
            proListX.append(i)
            proU += arrayU[i]

        else:
            unproCount += 1
            unproListX.append(i)
            unproU += arrayU[i]

    arrayU = np.reshape(arrayU, (k, 1))

    if len(ranking) < 10:
        topk = len(ranking)
    else:
        topk = 10

    h = np.array([1] * topk + [0] * (k - topk))
    h = np.reshape(h, (1, k))
  
    uv = arrayU.dot(v)
    uv = uv.flatten()

    # negate objective function to convert maximization problem to minimization problem
    uv = np.negative(uv)

    # check if there are protected items
    if proCount == 0:
        print(
            'Cannot create a P for ' + algoName + ' on data set ' + dataSetName + ' because no protected items in data set.')
        return 0, False
    # check if there are unprotected items
    if unproCount == 0:
        print(
            'Cannot create a P for ' + algoName + ' on data set ' + dataSetName + ' because no unprotected items in data set.')
        return 0, False

    proU = proU / proCount
    unproU = unproU / unproCount

    initf = np.zeros((k, 1))

    initf[proListX] = 1 / (proCount * proU)
    initf[unproListX] = -(1 / (unproCount * unproU))

    f1 = initf.dot(v)

    f1 = f1.flatten()
    f1 = np.reshape(f1, (1, k * k))

    f = matrix(f1)

    # set up constraints x <= 1
    A = spmatrix(1.0, range(k * k), range(k * k))
    # set up constraints x >= 0
    A1 = spmatrix(-1.0, range(k * k), range(k * k))
    # set up constraints that sum(rows)=1
    M = spmatrix(1.0, I, J)
    # set up constraints sum(columns)=1
    M1 = spmatrix(1.0, I2, J)

    # values for sums columns == 1
    h1 = matrix(1.0, (k, 1))
    # values for sums rows == 1
    b = matrix(1.0, (k * k, 1))
    # values for x >= 0
    d = matrix(0.0, (k * k, 1))

    c = matrix(uv)
    # assemble constraint matrix as sparse matrix
    G = sparse([M, M1, A, A1, f])
    # assemble constraint values
    h = matrix([h1, h1, b, d, 0.0])

    print('Start solving LP with DTC.')
    try:
        sol = solvers.lp(c, G, h)
    except Exception:
        print('Cannot create a P for ' + algoName + ' on data set ' + dataSetName + '.')
        return 0, False
    print('Finished solving LP with DTC.')
    if sol['x'] is None:
        print('Cannot create a P for ' + algoName + ' on data set ' + dataSetName + '.')
        return 0, False
    return np.array(sol['x']), True


ranking = generate_pre_ranking()
dataSetName = 'trec'
algoName = 'FOEIR-DTC'
x = runFOEIR(ranking, dataSetName, algoName, k=40, query_rep=1)
print(x)