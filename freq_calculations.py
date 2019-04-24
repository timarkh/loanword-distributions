import re
import os
import random
import math
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from json_doc_reader import JSONDocReader


rxBadToken = re.compile('^[^_]*[a-z0-9à-ʸ΄-ϿԱ-῾=.,?!:;()«»_/ћöü]')


def load_freq_wordlist(fname):
    """
    Load frequency list of tokens, return a dictionary.
    """
    # rxBadToken = re.compile('[a-z0-9à-ʸ΄-ϿԱ-῾_=.,?!:;()-]')
    freqDict = {}
    cntBadTokens = 0
    cntGoodTokens = 0
    fIn = open(fname, 'r', encoding='utf-8-sig')
    for line in fIn:
        if '\t' not in line:
            continue
        token, freq = line.lower().strip().split('\t')
        freq = int(freq)
        if rxBadToken.search(token) is not None:
            cntBadTokens += freq
        else:
            cntGoodTokens += freq
            if token in freqDict:
                freqDict[token] += freq
            else:
                freqDict[token] = freq
    fIn.close()
    print(cntGoodTokens, 'good tokens,', cntBadTokens, 'bad tokens.')
    return freqDict


def load_year_list(fname):
    """
    Load list of lemmata with data about their first mention.
    """
    # rxBadToken = re.compile('[a-z0-9à-ʸ΄-ϿԱ-῾_=.,?!:;()-]')
    rxBadLemma = re.compile('^[^_]*[a-z0-9à-ʸ΄-ϿԱ-῾=.?!:;()]')
    lemmaMentionDict = {}
    cntBadLemmata = 0
    cntGoodLemmata = 0
    fIn = open(fname, 'r', encoding='utf-8-sig')
    for line in fIn:
        if '\t' not in line:
            continue
        lemma, grdic, freq, year1, year2 = line.lower().strip().split('\t')
        freq = int(freq)
        year1 = int(year1)
        year2 = int(year2)
        if rxBadLemma.search(lemma) is not None:
            cntBadLemmata += freq
        else:
            cntGoodLemmata += freq
            if lemma not in lemmaMentionDict or lemmaMentionDict[lemma] > year1:
                lemmaMentionDict[lemma] = year1
    fIn.close()
    print(cntGoodLemmata, 'good lemmata,', cntBadLemmata, 'bad lemmata.')
    return lemmaMentionDict


def load_lemma_replacements(fname):
    """
    Load tsv list of replacements for ambiguous lemmata, return a dictionary.
    """
    replacements = {}
    fIn = open(fname, 'r', encoding='utf-8-sig')
    for line in fIn:
        if '\t' not in line:
            continue
        lemma, replacement = line.strip().split('\t')
        replacements[lemma] = replacement
    fIn.close()
    return replacements


def load_parsed_wordlist(fname, replacements=None, tag=',rus', include='', exclude='',
                         fname_tags=''):
    """
    Load analyzed word list in XML, return dictionary {token -> lemma}
    and dictionary {lemma -> is_tagged}.
    """
    if replacements is None:
        replacements = {}
    rxAna = re.compile('<ana +lex="([^<>"]+)"[^<>]*gr="([^<>"]+)"')
    rxToken = re.compile('>([^<>]+)</w>')
    lemmaDict = {}
    taggedLemmaDict = {}
    if len(exclude) > 0:
        exclude = re.compile(exclude)
    else:
        exclude = None
    if len(include) > 0:
        include = re.compile(include)
    else:
        include = None
    fIn = open(fname, 'r', encoding='utf-8-sig')
    for line in fIn:
        mToken = rxToken.search(line)
        if mToken is None:
            continue
        token = mToken.group(1).lower()
        anas = rxAna.findall(line)
        if include is not None:
            anas = [ana for ana in anas if include.search(ana[1]) is not None]
        if exclude is not None:
            anas = [ana for ana in anas if exclude.search(ana[1]) is None]
        anaSet = set(l[0] for l in anas)
        lemmata = '/'.join(l for l in sorted(anaSet))
        if len(lemmata) <= 0:
            continue
        if lemmata in replacements:
            lemmata = replacements[lemmata]
        lemmaDict[token] = lemmata
        taggedLemmaDict[lemmata] = any(tag in l[1] for l in anas)
    fIn.close()
    if fname_tags != '':
        annotatedLemmata = set()
        fIn = open(fname_tags, 'r', encoding='utf-8-sig')
        for line in fIn:
            if line.startswith('P'):
                taggedLemmaDict[line.strip('\r\n LP')] = True
            annotatedLemmata.add(line.strip('\r\n LP'))
        lemmaDict = {t: lemmaDict[t] for t in lemmaDict if lemmaDict[t] in annotatedLemmata}
        taggedLemmaDict = {l: taggedLemmaDict[l] for l in taggedLemmaDict if l in annotatedLemmata}
    print(len(lemmaDict), 'analyses loaded.')
    return lemmaDict, taggedLemmaDict


def lemma_count_dictribution(freqDict, lemmaDict, fname):
    """
    Calculate number of different lemmata and distribution
    of types by lemmata for each token frequency.
    Write the output to fname.
    """
    fOut = open(fname, 'w', encoding='utf-8')
    prevFreq = 0
    lemmataFreq = {'_unattributed': 0}
    typeCounter = 0
    for token, freq in sorted(freqDict.items(), key=lambda x: (x[1], x[0])):
        if prevFreq != 0 and freq != prevFreq:
            fOut.write('==  Frequency: ' + str(prevFreq) + '  ==\n')
            fOut.write('* Different attributed types: ' + str(typeCounter) + '\n')
            fOut.write('* Total attributed tokens: ' + str(typeCounter * prevFreq) + '\n')
            fOut.write('* Different lemmata: ' + str(len(lemmataFreq) - 1) + '\n')
            fOut.write('* Distribution of types by lemma:\n')
            fOut.write('{' + ', '.join(str(v) + ': ' + str(sum(1 for l in lemmataFreq
                                                           if lemmataFreq[l] == v))
                                       for v in sorted(set(lemmataFreq.values()))) + '}\n')
            fOut.write('\n'.join(lemma + '\t' + str(nTypes)
                                 for lemma, nTypes in sorted(lemmataFreq.items(), key=lambda x: (-x[1], x[0]))) + '\n\n\n')
            typeCounter = 0
            lemmataFreq = {'_unattributed': 0}
        prevFreq = freq
        if token in lemmaDict:
            lemma = lemmaDict[token]
            if lemma in lemmataFreq:
                lemmataFreq[lemma] += 1
            else:
                lemmataFreq[lemma] = 1
            typeCounter += 1
        else:
            lemmataFreq['_unattributed'] += 1
    fOut.close()


def lemma_forms_dictribution(freqDict, lemmaDict):
    """
    For each lemma frequency, calculate the distribution of lemmata
    over number of forms they have in the corpus.
    Return the dictionary {lfreq -> {ntypes -> nlemmata}}
    """
    lemmaFreqs = lemma_freqs(freqDict, lemmaDict)
    lfreq2ntypes = {}
    lemma2ntypes = {}
    for token in lemmaDict:
        if token not in freqDict:
            continue
        lemma = lemmaDict[token]
        try:
            lemma2ntypes[lemma] += 1
        except KeyError:
            lemma2ntypes[lemma] = 1
    for lemma in lemma2ntypes:
        freq = lemmaFreqs[lemma]
        if freq not in lfreq2ntypes:
            lfreq2ntypes[freq] = {}
        try:
            lfreq2ntypes[freq][lemma2ntypes[lemma]] += 1
        except KeyError:
            lfreq2ntypes[freq][lemma2ntypes[lemma]] = 1
    return lfreq2ntypes


def write_lemma_forms_dictribution(freqDict, lemmaDict, fname):
    """
    For each lemma frequency, calculate the distribution of lemmata
    over number of forms they have in the corpus.
    Write the output to fname.
    """
    lfreq2ntypes = lemma_forms_dictribution(freqDict, lemmaDict)
    fOut = open(fname, 'w', encoding='utf-8')
    for freq in sorted(lfreq2ntypes):
        fOut.write('Frequency: ' + str(freq) + '\n')
        fOut.write(json.dumps(lfreq2ntypes[freq], sort_keys=True, indent=2))
        fOut.write('\n\n')
    fOut.close()


def print_ambiguous(lemmaDict):
    """
    Print all ambiguous lemmata.
    """
    ambiguousLemmata = set()
    for lemma in sorted(lemmaDict.values()):
        if '/' in lemma:
            ambiguousLemmata.add(lemma)
    for lemma in sorted(ambiguousLemmata):
        print(lemma)


def print_random(freqDictWf, unanalyzed, freq, n):
    """
    Print n random unanalyzed tokens whose frequency == freq.
    """
    unanalyzed2choose = []
    for token in freqDictWf:
        if freqDictWf[token] == freq and token in unanalyzed:
            unanalyzed2choose.append(token)
    if n > len(unanalyzed2choose):
        n = len(unanalyzed2choose)
    if n <= 0:
        return
    randomSample = random.sample(unanalyzed2choose, n)
    print('\n'.join(sorted(randomSample)))


def write_unparsed(fname, freqDict, lemmaDict, min_len=5, min_freq=5, min_different=5):
    """
    Write a list of unparsed tokens that could belong to a single
    lexeme with total frequency >= freq and at least min_different
    potential tokens in the corpus. Several tokens are considered
    to be possible instances of the same lexeme if they have common
    prefix of at least min_len characters.
    """
    fOut = open(fname, 'w', encoding='utf-8')
    outTokens = set()
    prefixes = {}
    for token in freqDict:
        if token in lemmaDict or len(token) < min_len:
            continue
        try:
            prefixes[token[:min_len]].append(token)
        except KeyError:
            prefixes[token[:min_len]] = [token]
    for token in freqDict:
        if token in lemmaDict or len(token) < min_len:
            continue
        freq = freqDict[token]
        possibleTokens = {token}
        pfx = token[:min_len]
        for t in prefixes[pfx]:
            if t != token:
                possibleTokens.add(t)
                freq += freqDict[t]
        if freq >= min_freq and len(possibleTokens) >= min_different:
            outTokens |= possibleTokens
    fOut.write('\n'.join(t for t in sorted(outTokens)))
    fOut.close()


def lemma_freq(lemma, freqDict, lemmaDict):
    """
    Return total frequency of a lemma.
    """
    cnt = 0
    for token in lemmaDict:
        if token in freqDict and lemmaDict[token] == lemma:
            cnt += freqDict[token]
    return cnt


def lemma_freqs(freqDict, lemmaDict):
    """
    Return a dictionary with lemma frequencies.
    """
    lemmaFreqs = {}
    for token in freqDict:
        if token not in lemmaDict:
            continue
        try:
            lemmaFreqs[lemmaDict[token]] += freqDict[token]
        except KeyError:
            lemmaFreqs[lemmaDict[token]] = freqDict[token]
    return lemmaFreqs


def tagged_share(freqDictLex, fname, corrections_fname='',
                 mentionDict=None):
    """
    Write share of tagged lemmata for each frequency rank.
    """
    freq2total = {}
    freq2tagged = {}
    freq2year = {}
    corrections = {}
    if corrections_fname != '':
        fIn = open(corrections_fname, 'r', encoding='utf-8-sig')
        for line in fIn:
            if '\t' not in line:
                continue
            m = re.search('^([0-9]+)\t([0-9]+)\t([0-9]+)$', line.strip())
            if m is None:
                continue
            corrections[int(m.group(1))] = [int(m.group(2)), int(m.group(3))]
        print('Corrections loaded: ' + json.dumps(corrections, sort_keys=True))
        fIn.close()
    for lemma, pos, isTagged in freqDictLex:
        freq = freqDictLex[(lemma, pos, isTagged)]
        if freq <= 0:
            continue
        year = 0
        if mentionDict is not None:
            if lemma in mentionDict:
                year = mentionDict[lemma]
            else:
                year = 1970
        try:
            freq2total[freq] += 1
            freq2year[freq] += year
        except KeyError:
            freq2total[freq] = 1
            freq2tagged[freq] = 0
            freq2year[freq] = year
        if isTagged:
            freq2tagged[freq] += 1
    for freq in corrections:
        try:
            freq2total[freq] += corrections[freq][0]
            freq2tagged[freq] += corrections[freq][1]
        except KeyError:
            freq2total[freq] = corrections[freq][0]
            freq2tagged[freq] = corrections[freq][1]
    fOut = open(fname, 'w', encoding='utf-8')
    totalCnt = 0
    taggedCnt = 0
    yearCnt = 0
    curRank = 1
    x = []
    yProb = []
    yCumulative = []
    # for i in sorted(freq2total):
    #     print(i, freq2total[i])
    for freq in sorted(freq2total, reverse=True):
        totalCnt += freq2total[freq]
        taggedCnt += freq2tagged[freq]
        yearCnt += freq2year[freq]
        x.append(curRank + math.floor(freq2total[freq] / 2))
        yProb.append(freq2tagged[freq] / freq2total[freq])
        yCumulative.append(taggedCnt / totalCnt)
        fOut.write(str(curRank + math.floor(freq2total[freq] / 2)) + '\t'
                   + str(taggedCnt / totalCnt).replace('.', ',') + '\t'
                   + str(freq2tagged[freq] / freq2total[freq]).replace('.', ',') + '\t'
                   + str(yearCnt / totalCnt).replace('.', ',') + '\t'
                   + str(freq2year[freq] / freq2total[freq]).replace('.', ',') + '\n')
        curRank += freq2total[freq]
    fOut.close()
    yProbSmooth = []
    for i in range(len(yProb)):
        # moving average
        nPoints = 0
        curSum = 0
        for shift in range(-25, 26):
            if 0 <= i + shift < len(x):
                if -25 <= x[i] - x[i + shift] <= 25:
                    nPoints += 1
                    curSum += yProb[i + shift]
        yProbSmooth.append(curSum / nPoints)
    plt.subplot(2, 2, 1)
    plt.title('Probability', fontsize=8)
    plt.plot(x, yProbSmooth, 'g-', lw=0.5)
    plt.xlabel('rank')
    plt.grid(True, which='both', color=(0.6, 0.6, 0.6), lw=0.25)
    plt.ylim(0, 0.56)
    plt.subplot(2, 2, 2)
    plt.title('Probability (log x)', fontsize=8)
    plt.plot(x, yProbSmooth, 'g-', lw=0.5)
    plt.xlabel('rank')
    plt.grid(True, which='both', color=(0.6, 0.6, 0.6), lw=0.25)
    plt.xscale('log')
    plt.xlim(100, 40000)
    plt.ylim(0, 0.56)
    plt.subplot(2, 2, 3)
    plt.title('Cumul. prob.', fontsize=8)
    plt.plot(x, yCumulative, 'r-', lw=0.5)
    plt.xlabel('rank')
    plt.grid(True, which='both', color=(0.6, 0.6, 0.6), lw=0.25)
    plt.ylim(0, 0.56)
    plt.subplot(2, 2, 4)
    plt.title('Cumul. prob. (log x)', fontsize=8)
    plt.plot(x, yCumulative, 'r-', lw=0.5)
    plt.xlabel('rank')
    plt.grid(True, which='both', color=(0.6, 0.6, 0.6), lw=0.25)
    plt.xscale('log')
    plt.xlim(100, 40000)
    plt.ylim(0, 0.56)
    plt.savefig(fname[:-4] + '.png', dpi=300)
    plt.clf()


def write_adjustments_file(fname, freqDictWf, unanalyzed, word2lemma,
                           unparsed_freq_counts, tagged_shares_by_freq):
    """
    Calculate and write adjustments to the lemma frequency list and
    the number of tagged lemmata for each frequencies, based on numbers
    obtained manually from a sample of unanalyzed tokens.
    """
    # lfreq2ntypes = lemma_forms_dictribution(freqDict, lemmaDict)
    adjustments = {}
    for freq in unparsed_freq_counts:
        if freq not in tagged_shares_by_freq:
            continue
        curUnparsedTokens = unparsed_freq_counts[freq]
        # curTotalParsedLemmata = sum(lfreq2ntypes[freq].values())
        # denominator = sum(ntypes * lfreq2ntypes[freq][ntypes]
        #                   for ntypes in lfreq2ntypes[freq]) / curTotalParsedLemmata
        curLemmata = set()
        curParsedTokens = 0
        for word in freqDictWf:
            if freqDictWf[word] <= freq and word in word2lemma:
                curParsedTokens += 1
                curLemmata |= set(word2lemma[word])
        # Assume that token/lemma rate for unparsed tokens with given frequency
        # is the same as for parsed tokens with the same frequency:
        print('Frequency: ' + str(freq) + ', parsed types: ' + str(curParsedTokens)
              + ', lemma/type ratio: ' + str(len(curLemmata) / curParsedTokens) + '.')
        curUnparsedLemmata = curUnparsedTokens * (len(curLemmata) / curParsedTokens)
        adjustments[freq] = [math.floor(curUnparsedLemmata), math.floor(curUnparsedLemmata * tagged_shares_by_freq[freq])]
    fOut = open(fname, 'w', encoding='utf-8')
    fOut.write('freq\ttotal\ttagged\n')
    for freq in sorted(adjustments):
        fOut.write(str(freq) + '\t' + str(adjustments[freq][0]) + '\t' + str(adjustments[freq][1]) + '\n')
    fOut.close()


def write_zipf(fname, freqDict, lemmaDict, corrections_fname=''):
    """
    Write the rank-frequency table.
    """
    lemmaFreqs = lemma_freqs(freqDict, lemmaDict)
    freq2total = {}
    corrections = {}
    if corrections_fname != '':
        fIn = open(corrections_fname, 'r', encoding='utf-8-sig')
        for line in fIn:
            if '\t' not in line:
                continue
            m = re.search('^([0-9]+)\t([0-9]+)\t([0-9]+)$', line.strip())
            if m is None:
                continue
            corrections[int(m.group(1))] = int(m.group(2))
        print('Corrections loaded: ' + json.dumps(corrections, sort_keys=True))
        fIn.close()
    for lemma in set(lemmaDict.values()):
        if lemma not in lemmaFreqs:
            continue
        freq = lemmaFreqs[lemma]
        if freq <= 0:
            continue
        try:
            freq2total[freq] += 1
        except KeyError:
            freq2total[freq] = 1
    for freq in corrections:
        try:
            freq2total[freq] += corrections[freq]
        except KeyError:
            freq2total[freq] = corrections[freq]
    fOut = open(fname, 'w', encoding='utf-8')
    totalCnt = 0
    curRank = 1
    for freq in sorted(freq2total, reverse=True):
        totalCnt += freq2total[freq]
        fOut.write(str(curRank + math.floor(freq2total[freq] / 2)) + '\t'
                   + str(freq) + '\n')
        curRank += freq2total[freq]
    fOut.close()


def simulate_lemma_distibution(N=25000, NR=7000, corpus_size=800000, share_tagged=0.09,
                               s=-1.2, sR=-0.95, lower_boundR=1):
    """
    Simulate a corpus where the native vocabulary has N lemmata, the
    dominant language has NR lemmata, and overall share of borrowed
    tokens is share_tagged. Assume Zipf's distribution of lemmata
    in both languages with parameters s and sR.
    """
    A = (1 - share_tagged) / sum(r ** s for r in range(1, N+1))
    AR = share_tagged / sum(r ** sR for r in range(lower_boundR, NR+1))
    print(AR * sum(r ** sR for r in range(lower_boundR, NR+1)))
    freqDict = {'w' + str(i): 0 for i in range(1, N+1)}
    freqDict.update({'wR' + str(i): 0 for i in range(lower_boundR, NR+1)})
    population = ['w' + str(i) for i in range(1, N+1)] + ['wR' + str(i) for i in range(lower_boundR, NR+1)]
    weights = [A * r ** s for r in range(1, N+1)] + [AR * r ** sR for r in range(lower_boundR, NR+1)]
    print(sum(weights))
    corpus = random.choices(population, weights, k=corpus_size)
    for w in corpus:
        freqDict[w] += 1
    freqDict = {w: freqDict[w] for w in freqDict if freqDict[w] > 0}
    lemmaDict = {w: w for w in freqDict}
    taggedLemmaDict = {'wR' + str(i): True for i in range(lower_boundR, NR + 1) if 'wR' + str(i) in freqDict}
    taggedLemmaDict.update({'w' + str(i): False for i in range(1, N + 1) if 'w' + str(i) in freqDict})
    print('Total native lemmata:', sum(1 for w in taggedLemmaDict if not taggedLemmaDict[w]))
    print('Total borrowed lemmata:', sum(1 for w in taggedLemmaDict if taggedLemmaDict[w]))
    return freqDict, lemmaDict, taggedLemmaDict


def choose_random_unparsed(freqDictWf, unanalyzed, n=100):
    """
    Choose and print n random unanalyzed tokens.
    """
    unparsed = list(unanalyzed)
    sample = random.choices(unparsed, k=n)
    for t in sample:
        print(t + '\t' + str(freqDictWf[t]))


def clean_dicts(lemmaDict, freqDict):
    # Remove duplicate parts of hyphen-separated words
    rxHyphen = re.compile('[^-]+-')
    allLemmata = [l for l in freqDict]
    for lemma in allLemmata:
        if rxHyphen.search(lemma) is not None:
            if lemma not in lemmaDict:
                del freqDict[lemma]
            else:
                parts = lemma.split('-')
                if all (p in lemmaDict for p in parts):
                    del freqDict[lemma]
                    del lemmaDict[lemma]


def add_word_to_dicts(word, lemmaReplacements, tag, excludeWords,
                      freqDictWf, freqDictLex, word2lemma, unanalyzedTokens):
    """
    Add one word from a JSON corpus file to the frequency dictionaries.
    """
    wf = word['wf'].lower()
    if rxBadToken.search(wf) is not None:
        return
    try:
        freqDictWf[wf] += 1
    except KeyError:
        freqDictWf[wf] = 1
    if 'ana' not in word or len(word['ana']) <= 0 or all('lex' not in ana or 'gr.pos' not in ana
                                                         for ana in word['ana']):
        unanalyzedTokens.add(wf)
        return
    lemmata = '/'.join(sorted(set(ana['lex'] for ana in word['ana'] if 'lex' in ana)))
    if lemmata in lemmaReplacements:
        word['ana'] = [ana for ana in word['ana'] if 'lex' in ana and ana['lex'] == lemmaReplacements[lemmata]]
        lemmata = lemmaReplacements[lemmata]
    if excludeWords is not None and len(excludeWords) > 0:
        if any(all(condition in ana and excludeWords[condition] in ana[condition] for condition in excludeWords)
               for ana in word['ana']):
            return
    pos = '/'.join(sorted(set(ana['gr.pos'] for ana in word['ana'] if 'gr.pos' in ana)))
    isTagged = any(all(condition in ana and tag[condition] in ana[condition] for condition in tag)
                   for ana in word['ana'])
    lexeme = (lemmata, pos, isTagged)
    try:
        freqDictLex[lexeme] += 1
    except KeyError:
        freqDictLex[lexeme] = 1
    try:
        word2lemma[wf].add(lemmata)
    except KeyError:
        word2lemma[wf] = {lemmata}


def process_corpus(dirname, format='json', lemmaReplacements=None, tag=None,
                   excludeWords=None, yearRange=None):
    """
    Read all corpus files in tsakorpus JSON or gzipped JSON,
    return a frequency dictionary of wordforms, a frequency
    dictionary of lexemes where each lexeme is a tuple
    (lemma, pos, isTagged), and a set of unanalyzed tokens.
    """
    if tag is None:
        tag = {'gr.add': 'rus'}
    if lemmaReplacements is None:
        lemmaReplacements = {}
    freqDictWf = {}
    freqDictLex = {}
    word2lemma = {}
    unanalyzedTokens = set()
    iterSent = JSONDocReader(format=format)
    for path, dirs, files in os.walk(dirname):
        print('Entering', path)
        for fname in files:
            if (not ((format == 'json'
                      and fname.lower().endswith('.json'))
                     or (format == 'json-gzip'
                         and fname.lower().endswith('.json.gz')))):
                continue
            fnameFull = os.path.join(path, fname)
            if yearRange is not None:
                docMeta = iterSent.get_metadata(fnameFull)
                if 'year_from' in docMeta and int(docMeta['year_from']) < yearRange[0]:
                    continue
                if 'year_to' in docMeta and int(docMeta['year_to']) > yearRange[1]:
                    continue
            for s, bLast in iterSent.get_sentences(fnameFull):
                if 'words' not in s or 'lang' not in s or s['lang'] != 0:
                    continue
                for word in s['words']:
                    if 'wtype' not in word or word['wtype'] != 'word' or 'wf' not in word or len(word['wf']) <= 0:
                        continue
                    add_word_to_dicts(word, lemmaReplacements, tag, excludeWords, freqDictWf,
                                      freqDictLex, word2lemma, unanalyzedTokens)
    totalAnalyzed = sum(freqDictLex.values())
    totalUnanalyzed = sum(freqDictWf.values()) - totalAnalyzed
    print('Analyzed words: ' + str(totalAnalyzed) + ', unanalyzed words: ' + str(totalUnanalyzed) + '.')
    word2lemma = {w: [l for l in sorted(word2lemma[w])] for w in word2lemma}
    return freqDictWf, freqDictLex, word2lemma, unanalyzedTokens


def write_freq_dict(freqDict, fnameOut):
    fOut = open(fnameOut, 'w', encoding='utf-8')
    for k in sorted(freqDict, key=lambda x: (-freqDict[x], random.random())):
        kStr = k
        if type(kStr) == tuple:
            kStr = '\t'.join(str(p) for p in k)
        fOut.write(kStr + '\t' + str(freqDict[k]) + '\n')
    fOut.close()


def write_unanalyzed(unanalyzedTokens, fnameOut):
    fOut = open(fnameOut, 'w', encoding='utf-8')
    for token in sorted(unanalyzedTokens):
        fOut.write(token + '\n')
    fOut.close()


def load_data(fnameWf, fnameLex, fnameWord2lemma, fnameUnanalyzed):
    """
    Load the data previously collected by process_corpus.
    """
    freqDictWf = {}
    freqDictLex = {}
    word2lemma = {}
    unanalyzed = set()
    fWf = open(fnameWf, 'r', encoding='utf-8')
    for line in fWf:
        line = line.strip()
        if len(line) <= 2:
            continue
        wf, freq = line.split('\t')
        freqDictWf[wf] = int(freq)
    fWf.close()
    fLex = open(fnameLex, 'r', encoding='utf-8')
    for line in fLex:
        line = line.strip()
        if len(line) <= 2:
            continue
        lemma, pos, isTagged, freq = line.split('\t')
        lex = (lemma, pos, isTagged == 'True')
        freqDictLex[lex] = int(freq)
    fLex.close()
    fWord2lemma = open(fnameWord2lemma, 'r', encoding='utf-8')
    word2lemma = json.loads(fWord2lemma.read())
    fWord2lemma.close()
    fUnanalyzed = open(fnameUnanalyzed, 'r', encoding='utf-8')
    for line in fUnanalyzed:
        line = line.strip()
        if len(line) <= 0:
            continue
        unanalyzed.add(line)
    fUnanalyzed.close()
    return freqDictWf, freqDictLex, word2lemma, unanalyzed


def write_unanalyzed_by_freq(unanalyzed, freqDictWf, fnameOut):
    unanalyzedByFreq = {}
    for word in unanalyzed:
        freq = freqDictWf[word]
        try:
            unanalyzedByFreq[freq] += 1
        except KeyError:
            unanalyzedByFreq[freq] = 1
    fOut = open(fnameOut, 'w', encoding='utf-8')
    fOut.write(json.dumps(unanalyzedByFreq, ensure_ascii=False, indent=2, sort_keys=True))
    fOut.close()


def write_word2lemma(word2lemma, fnameOut):
    fOut = open(fnameOut, 'w', encoding='utf-8')
    fOut.write(json.dumps(word2lemma, ensure_ascii=False, indent=1, sort_keys=True))
    fOut.close()


if __name__ == '__main__':
    lang = 'udm_vk'
    # freqDict, lemmaDict, taggedLemmaDict = simulate_lemma_distibution()

    # corpusDir = 'J:/Corpus/NewCorpusPlatform/tsakorpus/tsakorpus_repo/corpus/udmurt'
    corpusDir = 'J:/Corpus/NewCorpusPlatform/tsakorpus/tsakorpus_repo/corpus/udmurt_social_networks'
    # corpusDir = 'J:/Corpus/NewCorpusPlatform/tsakorpus/tsakorpus_repo/corpus/erzya'
    # corpusDir = 'J:/Corpus/NewCorpusPlatform/tsakorpus/tsakorpus_repo/corpus/moksha'
    # corpusDir = 'J:/Corpus/NewCorpusPlatform/tsakorpus/tsakorpus_repo/corpus/komi-zyrian'
    lemmaReplacements = load_lemma_replacements(os.path.join(lang, 'lemma_replacements.csv'))
    freqDictWf, freqDictLex, word2lemma, unanalyzed = process_corpus(corpusDir,
                                                                     format='json-gzip', lemmaReplacements=lemmaReplacements,
                                                                     tag={'gr.add': 'rus'},
                                                                     yearRange=[1975, 2020])
    write_freq_dict(freqDictWf, os.path.join(lang, 'wordlist.csv'))
    write_freq_dict(freqDictLex, os.path.join(lang, 'lexlist.csv'))
    write_word2lemma(word2lemma, os.path.join(lang, 'word2lemma.csv'))
    write_unanalyzed(unanalyzed, os.path.join(lang, 'unanalyzed.csv'))
    # freqDictWf, freqDictLex, word2lemma, unanalyzed = load_data(os.path.join(lang, 'wordlist.csv'),
    #                                                             os.path.join(lang, 'lexlist.csv'),
    #                                                             os.path.join(lang, 'word2lemma.csv'),
    #                                                             os.path.join(lang, 'unanalyzed.csv'))
    write_unanalyzed_by_freq(unanalyzed, freqDictWf, os.path.join(lang, 'unanalyzed_by_freq.csv'))
    # freqDict = load_freq_wordlist(os.path.join(lang, 'wordlist.csv'))
    # mentionDict = load_year_list(os.path.join(lang, 'lemmata_first_mentions.csv'))
    # lemmaDict, taggedLemmaDict = load_parsed_wordlist(os.path.join(lang, 'wordlist.csv-parsed.txt'),
    #                                                   lemmaReplacements, tag=',rus', include='', exclude='',
    #                                                   fname_tags='')
    # clean_dicts(lemmaDict, freqDict)
    # write_zipf(os.path.join(lang, 'zipf_lemmata.csv'),
    #            freqDict, lemmaDict,
    #            os.path.join(lang, 'plot_adjustments.csv'))
    # choose_random_unparsed(freqDictWf, unanalyzed, n=100)
    # For rus in myv_vk with PNs:
    # write_adjustments_file(os.path.join(lang, 'plot_adjustments.csv'),
    #                        freqDict, lemmaDict, n_unparsed_tokens=33673,
    #                        unparsed_freq_shares={1: 0.17, 2: 0.08, 3: 0.07, 4: 0.02, 5: 0.02},
    #                        tagged_shares_by_freq={1: 2 / 17, 2: 4 / 17, 3: 4 / 17, 4: 4 / 17, 5: 4 / 17})

    # For rus in myv_vk without PNs:
    # write_adjustments_file(os.path.join(lang, 'plot_adjustments_wo_PN.csv'),
    #                        freqDict, lemmaDict, n_unparsed_tokens=33673,
    #                        unparsed_freq_shares={1: 0.11, 2: 0.05, 3: 0.06, 4: 0.02, 5: 0.02},
    #                        tagged_shares_by_freq={1: 2 / 11, 2: 4 / 13, 3: 4 / 13, 4: 4 / 13, 5: 4 / 13})

    # For PN in myv_vk with rus:
    # write_adjustments_file(os.path.join(lang, 'plot_adjustments_PN.csv'),
    #                        freqDict, lemmaDict, n_unparsed_tokens=33673,
    #                        unparsed_freq_shares={1: 0.17, 2: 0.08, 3: 0.07, 4: 0.02, 5: 0.02, 6: 0.005, 7: 0.005},
    #                        tagged_shares_by_freq={1: 5 / 17, 2: 5 / 17, 3: 5 / 17, 4: 5 / 17, 5: 5 / 17,
    #                                               6: 5 / 17, 7: 5 / 17})

    # For rus un myv_main with PNs:
    # write_adjustments_file(os.path.join(lang, 'plot_adjustments.csv'),
    #                        freqDict, lemmaDict, n_unparsed_tokens=58575,
    #                        unparsed_freq_shares={1: 0.21, 2: 0.14, 3: 0.11 / 4, 4: 0.11 / 4,
    #                                              5: 0.11 / 4, 6: 0.11 / 4, 7: 0.01, 8: 0.01, 9: 0.01},
    #                        tagged_shares_by_freq={1: 6 / 21, 2: 4 / 14, 3: 5 / 14, 4: 5 / 14,
    #                                               5: 5 / 14, 6: 5 / 14, 7: 5 / 14, 8: 5 / 14, 9: 5 / 14})

    # For udm_main:
    # write_adjustments_file(os.path.join(lang, 'plot_adjustments.csv'),
    #                        freqDictWf, unanalyzed, word2lemma,
    #                        unparsed_freq_counts={1: math.floor(64950 * 0.69),
    #                                              2: math.floor(12781 * 0.76),
    #                                              3: math.floor(5062 * 0.73),
    #                                              4: math.floor(2769 * 0.76),
    #                                              5: math.floor(1681 * 0.76),
    #                                              6: math.floor(1110 * 0.76)
    #                                              },
    #                        tagged_shares_by_freq={1: 34 / 69,
    #                                               2: 30 / 76,
    #                                               3: 26 / 73,
    #                                               4: 33 / 76,
    #                                               5: 33 / 76,
    #                                               6: 33 / 76
    #                                               })
    # For myv_main:
    # write_adjustments_file(os.path.join(lang, 'plot_adjustments.csv'),
    #                        freqDictWf, unanalyzed, word2lemma,
    #                        unparsed_freq_counts={1: math.floor(38634 * 0.76),
    #                                              2: math.floor(7872 * 0.72),
    #                                              3: math.floor(2938 * 0.72),
    #                                              4: math.floor(1523 * 0.72),
    #                                              5: math.floor(851 * 0.72),
    #                                              6: math.floor(635 * 0.72)
    #                                              },
    #                        tagged_shares_by_freq={1: 22 / 76,
    #                                               2: 25 / 72,
    #                                               3: 26 / 72,
    #                                               4: 26 / 72,
    #                                               5: 26 / 72,
    #                                               6: 26 / 72
    #                                               })
    # For kpv_main:
    # write_adjustments_file(os.path.join(lang, 'plot_adjustments.csv'),
    #                        freqDictWf, unanalyzed, word2lemma,
    #                        unparsed_freq_counts={1: math.floor(10414 * 0.83),
    #                                              2: math.floor(1977 * 0.81),
    #                                              3: math.floor(745 * 0.9)
    #                                              },
    #                        tagged_shares_by_freq={1: 33 / 83,
    #                                               2: 32 / 81,
    #                                               3: 40 / 90
    #                                               })

    # tagged_share(freqDict, lemmaDict, taggedLemmaDict,
    #              os.path.join(lang, 'loan_share.csv'),
    #              os.path.join(lang, 'plot_adjustments.csv'),
    #              mentionDict=mentionDict)
    # tagged_share(freqDict, lemmaDict, taggedLemmaDict,
    #              os.path.join(lang, 'rus_share.csv'),
    #              os.path.join(lang, 'plot_adjustments.csv'),
    #              mentionDict=mentionDict)
    tagged_share(freqDictLex,
                 os.path.join(lang, 'rus_share.csv'),
                 corrections_fname=os.path.join(lang, 'plot_adjustments.csv'))
    # tagged_share(freqDictLex,
    #              os.path.join(lang, 'rus_share.csv'))
    # write_lemma_forms_dictribution(freqDict, lemmaDict, os.path.join(lang, 'nforms_for_each_lemma_freq.csv'))
    # lemma_count_dictribution(freqDictWf, freqDictLex, os.path.join(lang, 'lemma_counts_by_token_freq.csv'))
    # print_ambiguous(lemmaDict)
    # for i in range(11):
    #     print(i)
    #     print_random(freqDictWf, unanalyzed, i, 100)
    # write_unparsed(os.path.join(lang, 'unparsed2look.txt'), freqDict, lemmaDict, min_len=6, min_freq=5)
