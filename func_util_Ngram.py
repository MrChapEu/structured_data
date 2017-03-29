from nltk.util import ngrams
import numpy as np
def getNgram(word,N):
    output = []
    for n in range(1,N+1):
        for ngram in ngrams(word,n):
            output.append(' '.join(str(lettre) for lettre in ngram))
    return output

def getNgramDict(words,N):
    output = []
    for word in words:
        output.append(getNgram(list(word),N))
    return np.unique([i for oneList in output for i in oneList])

def getGT_oneWord(word,ngramDict,lengthOfDict,N):
    output = np.zeros(lengthOfDict)
    ngram = getNgram(word,N)
    for i in range(lengthOfDict):
        if ngramDict[i] in ngram:
            output[i]=1
    return output

def getGround_Truth(words,ngramDict,N):
    output = []
    lengthOfDict = len(ngramDict)
    for word in words:
        output.append(list(getGT_oneWord(word,ngramDict,lengthOfDict,N)))
    return output

def ground_truth(words,N):
    ngramDict = getNgramDict(words,N)
    gt = getGround_Truth(words,ngramDict,N)
    return ngramDict,gt
