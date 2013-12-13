# Natural Language Toolkit: Chunk format conversions
#
# Copyright (C) 2001-2013 NLTK Project
# Author: Edward Loper <edloper@gmail.com>
#         Steven Bird <stevenbird1@gmail.com> (minor additions)
# URL: <http://nltk.org/>
# For license information, see LICENSE.TXT
from __future__ import print_function, unicode_literals, division

import re

from nltk.tree import Tree, SVTree
from nltk.tag.util import str2tuple
from nltk.compat import python_2_unicode_compatible

##//////////////////////////////////////////////////////
## EVALUATION
##//////////////////////////////////////////////////////

from nltk.metrics import accuracy as _accuracy
def accuracy(chunker, gold):
    """
    Score the accuracy of the chunker against the gold standard.
    Strip the chunk information from the gold standard and rechunk it using
    the chunker, then compute the accuracy score.

    :type chunker: ChunkParserI
    :param chunker: The chunker being evaluated.
    :type gold: tree
    :param gold: The chunk structures to score the chunker on.
    :rtype: float
    """

    gold_tags = []
    test_tags = []
    for gold_tree in gold:
        test_tree = chunker.parse(gold_tree.flatten())
        gold_tags += tree2conlltags(gold_tree)
        test_tags += tree2conlltags(test_tree)

#    print 'GOLD:', gold_tags[:50]
#    print 'TEST:', test_tags[:50]
    return _accuracy(gold_tags, test_tags)


# Patched for increased performance by Yoav Goldberg <yoavg@cs.bgu.ac.il>, 2006-01-13
#  -- statistics are evaluated only on demand, instead of at every sentence evaluation
#
# SB: use nltk.metrics for precision/recall scoring?
#
class ChunkScore(object):
    """
    A utility class for scoring chunk parsers.  ``ChunkScore`` can
    evaluate a chunk parser's output, based on a number of statistics
    (precision, recall, f-measure, misssed chunks, incorrect chunks).
    It can also combine the scores from the parsing of multiple texts;
    this makes it significantly easier to evaluate a chunk parser that
    operates one sentence at a time.

    Texts are evaluated with the ``score`` method.  The results of
    evaluation can be accessed via a number of accessor methods, such
    as ``precision`` and ``f_measure``.  A typical use of the
    ``ChunkScore`` class is::

        >>> chunkscore = ChunkScore()           # doctest: +SKIP
        >>> for correct in correct_sentences:   # doctest: +SKIP
        ...     guess = chunkparser.parse(correct.leaves())   # doctest: +SKIP
        ...     chunkscore.score(correct, guess)              # doctest: +SKIP
        >>> print('F Measure:', chunkscore.f_measure())       # doctest: +SKIP
        F Measure: 0.823

    :ivar kwargs: Keyword arguments:

        - max_tp_examples: The maximum number actual examples of true
          positives to record.  This affects the ``correct`` member
          function: ``correct`` will not return more than this number
          of true positive examples.  This does *not* affect any of
          the numerical metrics (precision, recall, or f-measure)

        - max_fp_examples: The maximum number actual examples of false
          positives to record.  This affects the ``incorrect`` member
          function and the ``guessed`` member function: ``incorrect``
          will not return more than this number of examples, and
          ``guessed`` will not return more than this number of true
          positive examples.  This does *not* affect any of the
          numerical metrics (precision, recall, or f-measure)

        - max_fn_examples: The maximum number actual examples of false
          negatives to record.  This affects the ``missed`` member
          function and the ``correct`` member function: ``missed``
          will not return more than this number of examples, and
          ``correct`` will not return more than this number of true
          negative examples.  This does *not* affect any of the
          numerical metrics (precision, recall, or f-measure)

        - chunk_label: A regular expression indicating which chunks
          should be compared.  Defaults to ``'.*'`` (i.e., all chunks).

    :type _tp: list(Token)
    :ivar _tp: List of true positives
    :type _fp: list(Token)
    :ivar _fp: List of false positives
    :type _fn: list(Token)
    :ivar _fn: List of false negatives

    :type _tp_num: int
    :ivar _tp_num: Number of true positives
    :type _fp_num: int
    :ivar _fp_num: Number of false positives
    :type _fn_num: int
    :ivar _fn_num: Number of false negatives.
    """
    def __init__(self, **kwargs):
        self._correct = set()
        self._guessed = set()
        self._tp = set()
        self._fp = set()
        self._fn = set()
        self._max_tp = kwargs.get('max_tp_examples', 100)
        self._max_fp = kwargs.get('max_fp_examples', 100)
        self._max_fn = kwargs.get('max_fn_examples', 100)
        self._chunk_label = kwargs.get('chunk_label', '.*')
        self._tp_num = 0
        self._fp_num = 0
        self._fn_num = 0
        self._count = 0
        self._tags_correct = 0.0
        self._tags_total = 0.0

        self._measuresNeedUpdate = False

    def _updateMeasures(self):
        if (self._measuresNeedUpdate):
           self._tp = self._guessed & self._correct
           self._fn = self._correct - self._guessed
           self._fp = self._guessed - self._correct
           self._tp_num = len(self._tp)
           self._fp_num = len(self._fp)
           self._fn_num = len(self._fn)
           self._measuresNeedUpdate = False

    def score(self, correct, guessed):
        """
        Given a correctly chunked sentence, score another chunked
        version of the same sentence.

        :type correct: chunk structure
        :param correct: The known-correct ("gold standard") chunked
            sentence.
        :type guessed: chunk structure
        :param guessed: The chunked sentence to be scored.
        """
        self._correct |= _chunksets(correct, self._count, self._chunk_label)
        self._guessed |= _chunksets(guessed, self._count, self._chunk_label)
        self._count += 1
        self._measuresNeedUpdate = True
        # Keep track of per-tag accuracy (if possible)
        try:
            correct_tags = tree2conlltags(correct)
            guessed_tags = tree2conlltags(guessed)
        except ValueError:
            # This exception case is for nested chunk structures,
            # where tree2conlltags will fail with a ValueError: "Tree
            # is too deeply nested to be printed in CoNLL format."
            correct_tags = guessed_tags = ()
        self._tags_total += len(correct_tags)
        self._tags_correct += sum(1 for (t,g) in zip(guessed_tags,
                                                     correct_tags)
                                  if t==g)

    def accuracy(self):
        """
        Return the overall tag-based accuracy for all text that have
        been scored by this ``ChunkScore``, using the IOB (conll2000)
        tag encoding.

        :rtype: float
        """
        if self._tags_total == 0: return 1
        return self._tags_correct/self._tags_total

    def precision(self):
        """
        Return the overall precision for all texts that have been
        scored by this ``ChunkScore``.

        :rtype: float
        """
        self._updateMeasures()
        div = self._tp_num + self._fp_num
        if div == 0: return 0
        else: return float(self._tp_num) / div

    def recall(self):
        """
        Return the overall recall for all texts that have been
        scored by this ``ChunkScore``.

        :rtype: float
        """
        self._updateMeasures()
        div = self._tp_num + self._fn_num
        if div == 0: return 0
        else: return float(self._tp_num) / div

    def f_measure(self, alpha=0.5):
        """
        Return the overall F measure for all texts that have been
        scored by this ``ChunkScore``.

        :param alpha: the relative weighting of precision and recall.
            Larger alpha biases the score towards the precision value,
            while smaller alpha biases the score towards the recall
            value.  ``alpha`` should have a value in the range [0,1].
        :type alpha: float
        :rtype: float
        """
        self._updateMeasures()
        p = self.precision()
        r = self.recall()
        if p == 0 or r == 0:    # what if alpha is 0 or 1?
            return 0
        return 1/(alpha/p + (1-alpha)/r)

    def missed(self):
        """
        Return the chunks which were included in the
        correct chunk structures, but not in the guessed chunk
        structures, listed in input order.

        :rtype: list of chunks
        """
        self._updateMeasures()
        chunks = list(self._fn)
        return [c[1] for c in chunks]  # discard position information

    def incorrect(self):
        """
        Return the chunks which were included in the guessed chunk structures,
        but not in the correct chunk structures, listed in input order.

        :rtype: list of chunks
        """
        self._updateMeasures()
        chunks = list(self._fp)
        return [c[1] for c in chunks]  # discard position information

    def correct(self):
        """
        Return the chunks which were included in the correct
        chunk structures, listed in input order.

        :rtype: list of chunks
        """
        chunks = list(self._correct)
        return [c[1] for c in chunks]  # discard position information

    def guessed(self):
        """
        Return the chunks which were included in the guessed
        chunk structures, listed in input order.

        :rtype: list of chunks
        """
        chunks = list(self._guessed)
        return [c[1] for c in chunks]  # discard position information

    def __len__(self):
        self._updateMeasures()
        return self._tp_num + self._fn_num

    def __repr__(self):
        """
        Return a concise representation of this ``ChunkScoring``.

        :rtype: str
        """
        return '<ChunkScoring of '+repr(len(self))+' chunks>'

    def __str__(self):
        """
        Return a verbose representation of this ``ChunkScoring``.
        This representation includes the precision, recall, and
        f-measure scores.  For other information about the score,
        use the accessor methods (e.g., ``missed()`` and ``incorrect()``).

        :rtype: str
        """
        return ("ChunkParse score:\n" +
                ("    IOB Accuracy: %5.1f%%\n" % (self.accuracy()*100)) +
                ("    Precision:    %5.1f%%\n" % (self.precision()*100)) +
                ("    Recall:       %5.1f%%\n" % (self.recall()*100))+
                ("    F-Measure:    %5.1f%%" % (self.f_measure()*100)))

        
import numpy
from collections import defaultdict
import itertools
from scipy import stats as stats
import pandas
from collections import defaultdict
class SV_Stats():
    """Holder class for routines used to collect stats about SVTrees."""
    def __init__(self, trees):
        # Ratios over training set
        self._sb_ratios = [(self.sb_index(t) + 1) / (self.chunk_count(t)) for t in trees]
        self._vb_ratios = [(self.vb_index(t) + 1) / (self.chunk_count(t)) for t in trees]
        # Beta functions from ratios
        self._sb_alpha, self._sb_beta = stats.beta.fit(self._sb_ratios)[:2]
        self._vb_alpha, self._vb_beta = stats.beta.fit(self._vb_ratios)[:2]
        # rudish prob dist by separations
        self._sepprobs = self.rude_sepprob(trees)
        # transition probability matrix
        trans = defaultdict(int)
        tot_trans = 0.0
        labels = set()
        for tree in trees:
            si = self.sb_index(tree)
            vi = self.vb_index(tree)
            segment = tree[si:vi+1] if si < vi else tree[vi:si+1]
            fromch = segment[0]
            labels.add(fromch.label() if isinstance(fromch, Tree) else '<other>')
            for chunk in segment[1:]:
                toch = chunk
                fromlabel = fromch.label() if isinstance(fromch, Tree) else '<other>'
                tolabel   = toch.label() if isinstance(toch, Tree) else '<other>'
                labels.add(tolabel)
                trans[(fromlabel, tolabel)] += 1
                tot_trans += 1
                fromch = toch
        # smooth out 0 values
        for c in itertools.product(labels, repeat=2):
            if c not in trans.keys():
                trans[c] = 0.5
                tot_trans += .5
        self.trans_prob = pandas.DataFrame(index=sorted(labels), columns=sorted(labels))
        for fromk, tok in itertools.product(labels, repeat=2):
            self.trans_prob[fromk][tok] = trans[(fromk, tok)] / tot_trans


    def chunk_count(self, tree):
        return len(tree)

    def sb_index(self, tree):
        """Return index of subject chunk in tree with one sb."""
        index = -1
        if not isinstance(tree, SVTree):
            raise ValueError("Tree must be instance of SVTree")
        if not tree.label() == 'S':
            raise ValueError("Tree must be root of sentence tree")
        for chunkno, chunk in enumerate(tree):
            if not isinstance(chunk, SVTree):
                continue
            if chunk.gram_role() == 'sb':
                index = chunkno
        if index < 0:
            raise RuntimeError("Unable to find subject in \n%s" % tree.pprint())
        return index

    def vb_index(self, tree):
        """Return index of verb chunk in tree with one vb."""
        index = -1
        if not isinstance(tree, SVTree):
            raise ValueError("Tree must be instance of SVTree")
        if not tree.label() == 'S':
            raise ValueError("Tree must be root of sentence tree")
        for chunkno, chunk in enumerate(tree):
            if not isinstance(chunk, SVTree):
                continue
            if chunk.gram_role() == 'vb':
                index = chunkno
        if index < 0:
            raise RuntimeError("Unable to find verb in \n%s" % tree.pprint())
        return index

    def sv_sep(self, tree):
        """Return degree of separation between subject and verb (vb - sb)."""
        return self.vb_index(tree) - self.sb_index(tree)

    def sep_range(self, trees):
        """Return range of sv_sep in trees."""
        low = min([self.sv_sep(t) for t in trees])
        high = max([self.sv_sep(t) for t in trees])
        return (low, high + 1)

    def rude_sepprob(self, trees):
        seps = self.sv_seps(trees)
        for k in range(-30,30):
            if k not in seps.keys():
                seps[k] = .25
            if k in seps.keys() and seps[k] == 0 and k != 0:
                seps[k] += 1
        tot = sum(seps.values())
        return pandas.Series( { i : seps[i] / tot for i in seps } )

    def max_chunk_count(self, trees):
        """Return max number chunks in a sentence from list of sentences."""
        maxcount = 0
        for treeno, tree in enumerate(trees):
            if not isinstance(tree, SVTree):
                raise ValueError("Tree %d is not SVTree." % treeno)
            maxcount = max(maxcount, self.chunk_count(tree))
        return maxcount

    def chunk_counts(self, trees):
        """Return numpy array: element i = no. trees containing i chunks."""
        v = numpy.array([0 for c in range(self.max_chunk_count(trees) + 1)])
        for treeno, tree in enumerate(trees):
            if not isinstance(tree, SVTree):
                raise ValueError("Tree %d is not SVTree." % treeno)
            v[self.chunk_count(tree)] += 1
        return v

    def sb_indices(self, trees):
        """Return numpy array: element i = no. trees with sb at ith chunk."""
        v = numpy.array([0 for c in range(self.max_chunk_count(trees))])
        for treeno, tree in enumerate(trees):
            if not isinstance(tree, SVTree):
                raise ValueError("Tree %d is not SVTree." % treeno)
            v[self.sb_index(tree)] += 1
        return v

    def vb_indices(self, trees):
        """Return numpy array: element i = no. trees with vb at ith chunk."""
        v = numpy.array([0 for c in range(self.max_chunk_count(trees))])
        for treeno, tree in enumerate(trees):
            if not isinstance(tree, SVTree):
                raise ValueError("Tree %d is not SVTree." % treeno)
            v[self.vb_index(tree)] += 1
        return v

    def sv_seps(self, trees):
        """Return array indexed sep_range representing no. sentences with sv_sep i."""
        v = defaultdict(float)
        for i in range(*self.sep_range(trees)): v[i]
        for treeno, tree in enumerate(trees):
            if not isinstance(tree, SVTree):
                raise ValueError("Tree %d is not SVTrees." % treeno)
            v[self.sv_sep(tree)] += 1
        return dict(v)

    def NP_indices(self, tree):
        """Return list of all indices (start=1) of NPs in tree."""
        inds = list()
        for chunkno, chunk in enumerate(tree, start=1):
            if not isinstance(chunk, SVTree):
                continue
            if chunk.label() == u'NP':
                inds.append(chunkno)
        return inds

    def VP_indices(self, tree):
        """Return list of all indices (start=1) of VPs in tree."""
        inds = list()
        for chunkno, chunk in enumerate(tree, start=1):
            if not isinstance(chunk, SVTree):
                continue
            if chunk.label() == u'VP':
                inds.append(chunkno)
        return inds

    def priorprob(self, ind, chs, flag):
        """Helper function for priors, to make things more readable."""
        a = self._sb_alpha if flag == 'sb' else self._vb_alpha
        b = self._sb_beta  if flag == 'sb' else self._vb_beta
        return stats.beta.cdf(ind / chs, a, b) - stats.beta.cdf((ind-1) / chs, a, b)

    def priors(self, tree):
        """Return dataframe of prior probabilities that NP:VP comnination == main sb:vb."""
        nps = self.NP_indices(tree)
        vps = self.VP_indices(tree)
        chs = self.chunk_count(tree)
        v   = [ self.priorprob(vp, chs, 'vb') for vp in vps ]
        sd  = { i : v for i in nps }
        sbp = pandas.DataFrame(sd, index=vps)
        s   = [ self.priorprob(np, chs, 'sb') for np in nps ]
        vbp = pandas.Series(s, index=nps)
        priors = pandas.DataFrame(sbp.mul(vbp), index=vps)
        norm = priors.sum().sum()
        return priors.applymap(lambda x : x / norm)

    def likelihood(self, segment):
        """Return likelihood of segment of chunks."""
        likelihood = 1.0
        fromch = segment[0]
        for chunk in segment[1:]:
            toch = chunk
            fromlabel = fromch.label() if isinstance(fromch, Tree) else '<other>'
            tolabel   = toch.label() if isinstance(toch, Tree) else '<other>'
            likelihood *= self.trans_prob[fromlabel][tolabel]
            fromch = toch
        return likelihood

    def update(self, priors, tree):
        """Return updated matrix == prior * likelihood(NP:VP segment of tree)."""
        post = priors.copy(deep=True)
        for si in post:
            for vi in post[si].index:
                segment = tree[si-1:vi] if si<vi else tree[vi-1:si]
                post[si][vi] *= self.likelihood(segment)
        return post

    def update_by_sep(self, priors, trees):
        """Return updated matrix == prior * likelihood(NP:VP segment of tree)."""
        post = priors.copy(deep=True)
        for si in post:
            for vi in post[si].index:
                post[si][vi] *= self._sepprobs[vi - si]
        return post

    def max_ind(self, mat):
        """Return max index, as (col, row) for max value in mat."""
        s = mat.unstack().copy()
        s.sort()
        return (s.index[-1][0] - 1, s.index[-1][1] - 1)


# extract chunks, and assign unique id, the absolute position of
# the first word of the chunk
def _chunksets(t, count, chunk_label):
    pos = 0
    chunks = []
    for child in t:
        if isinstance(child, Tree):
            if re.match(chunk_label, child.label()):
                chunks.append(((count, pos), child.freeze()))
            pos += len(child.leaves())
        else:
            pos += 1
    return set(chunks)


def tagstr2tree(s, chunk_label="NP", root_label="S", sep='/'):
    """
    Divide a string of bracketted tagged text into
    chunks and unchunked tokens, and produce a Tree.
    Chunks are marked by square brackets (``[...]``).  Words are
    delimited by whitespace, and each word should have the form
    ``text/tag``.  Words that do not contain a slash are
    assigned a ``tag`` of None.

    :param s: The string to be converted
    :type s: str
    :param chunk_label: The label to use for chunk nodes
    :type chunk_label: str
    :param root_label: The label to use for the root of the tree
    :type root_label: str
    :rtype: Tree
    """

    WORD_OR_BRACKET = re.compile(r'\[|\]|[^\[\]\s]+')

    stack = [Tree(root_label, [])]
    for match in WORD_OR_BRACKET.finditer(s):
        text = match.group()
        if text[0] == '[':
            if len(stack) != 1:
                raise ValueError('Unexpected [ at char %d' % match.start())
            chunk = Tree(chunk_label, [])
            stack[-1].append(chunk)
            stack.append(chunk)
        elif text[0] == ']':
            if len(stack) != 2:
                raise ValueError('Unexpected ] at char %d' % match.start())
            stack.pop()
        else:
            if sep is None:
                stack[-1].append(text)
            else:
                stack[-1].append(str2tuple(text, sep))

    if len(stack) != 1:
        raise ValueError('Expected ] at char %d' % len(s))
    return stack[0]

### CONLL

_LINE_RE = re.compile('(\S+)\s+(\S+)\s+([IOB])-?(\S+)?')
def conllstr2tree(s, chunk_types=('NP', 'PP', 'VP'), root_label="S"):
    """
    Return a chunk structure for a single sentence
    encoded in the given CONLL 2000 style string.
    This function converts a CoNLL IOB string into a tree.
    It uses the specified chunk types
    (defaults to NP, PP and VP), and creates a tree rooted at a node
    labeled S (by default).

    :param s: The CoNLL string to be converted.
    :type s: str
    :param chunk_types: The chunk types to be converted.
    :type chunk_types: tuple
    :param root_label: The node label to use for the root.
    :type root_label: str
    :rtype: Tree
    """

    stack = [Tree(root_label, [])]

    for lineno, line in enumerate(s.split('\n')):
        if not line.strip(): continue

        # Decode the line.
        match = _LINE_RE.match(line)
        if match is None:
            raise ValueError('Error on line %d' % lineno)
        (word, tag, state, chunk_type) = match.groups()

        # If it's a chunk type we don't care about, treat it as O.
        if (chunk_types is not None and
            chunk_type not in chunk_types):
            state = 'O'

        # For "Begin"/"Outside", finish any completed chunks -
        # also do so for "Inside" which don't match the previous token.
        mismatch_I = state == 'I' and chunk_type != stack[-1].label()
        if state in 'BO' or mismatch_I:
            if len(stack) == 2: stack.pop()

        # For "Begin", start a new chunk.
        if state == 'B' or mismatch_I:
            chunk = Tree(chunk_type, [])
            stack[-1].append(chunk)
            stack.append(chunk)

        # Add the new word token.
        stack[-1].append((word, tag))

    return stack[0]


_LINE_RE_PLUS = re.compile('(\S+)\s+(\S+)\s+([IOB])-?(\S+)?\s+(\S+)')
def conllstrplus2tree(s, chunk_types=('NP', 'PP', 'VP'), top_node="S"):
    """
    Return a chunk structure for a single sentence
    encoded in the given CONLL 2000 style string.
    This function converts a CoNLL IOB string into a tree.
    It uses the specified chunk types
    (defaults to NP, PP and VP), and creates a tree rooted at a node
    labeled S (by default).

    :param s: The CoNLL string to be converted.
    :type s: str
    :param chunk_types: The chunk types to be converted.
    :type chunk_types: tuple
    :param top_node: The node label to use for the root.
    :type top_node: str
    :rtype: Tree
    """

    stack = [SVTree(top_node, None, [])]

    for lineno, line in enumerate(s.split('\n')):
        if not line.strip(): continue

        # Decode the line.
        match = None
        gram_role = None
        if len(line.split()) == 4:
            match = _LINE_RE_PLUS.match(line)
            if match is None:
                raise ValueError('Error on line %d' % lineno)
            (word, tag, state, chunk_type, gram_role) = match.groups()
        else:
            match = _LINE_RE.match(line)
            if match is None:
                raise ValueError('Error on line %d' % lineno)
            (word, tag, state, chunk_type) = match.groups()

        #print("Line %d:" % (lineno))
        #print(match.groups())

        # If it's a chunk type we don't care about, treat it as O.
        if (chunk_types is not None and
            chunk_type not in chunk_types):
            state = 'O'

        # For "Begin"/"Outside", finish any completed chunks -
        # also do so for "Inside" which don't match the previous token.
        mismatch_I = state == 'I' and chunk_type != stack[-1].label()
        if state in 'BO' or mismatch_I:
            if len(stack) == 2: stack.pop()

        # For "Begin", start a new chunk.
        if state == 'B' or mismatch_I:
            chunk = SVTree(chunk_type, None, [])
            stack[-1].append(chunk)
            stack.append(chunk)

        # Add the new word token.
        stack[-1].append((word, tag))

        # If a grammar role is indicated, label chunk
        if gram_role is not None:
            if state == 'O':
                raise ValueError('Error on line %d: non-chunk words cannot take grammar role\n%s' % (lineno, s))
            stack[-1].set_gram_role(gram_role)

    return stack[0]


def tree2conlltags(t):
    """
    Return a list of 3-tuples containing ``(word, tag, IOB-tag)``.
    Convert a tree to the CoNLL IOB tag format.

    :param t: The tree to be converted.
    :type t: Tree
    :rtype: list(tuple)
    """

    tags = []
    for child in t:
        try:
            category = child.label()
            prefix = "B-"
            for contents in child:
                if isinstance(contents, Tree):
                    raise ValueError("Tree is too deeply nested to be printed in CoNLL format")
                tags.append((contents[0], contents[1], prefix+category))
                prefix = "I-"
        except AttributeError:
            tags.append((child[0], child[1], "O"))
    return tags

def conlltags2tree(sentence, chunk_types=('NP','PP','VP'),
                   root_label='S', strict=False):
    """
    Convert the CoNLL IOB format to a tree.
    """
    tree = Tree(root_label, [])
    for (word, postag, chunktag) in sentence:
        if chunktag is None:
            if strict:
                raise ValueError("Bad conll tag sequence")
            else:
                # Treat as O
                tree.append((word,postag))
        elif chunktag.startswith('B-'):
            tree.append(Tree(chunktag[2:], [(word,postag)]))
        elif chunktag.startswith('I-'):
            if (len(tree)==0 or not isinstance(tree[-1], Tree) or
                tree[-1].label() != chunktag[2:]):
                if strict:
                    raise ValueError("Bad conll tag sequence")
                else:
                    # Treat as B-*
                    tree.append(Tree(chunktag[2:], [(word,postag)]))
            else:
                tree[-1].append((word,postag))
        elif chunktag == 'O':
            tree.append((word,postag))
        else:
            raise ValueError("Bad conll tag %r" % chunktag)
    return tree

def tree2conllstr(t):
    """
    Return a multiline string where each line contains a word, tag and IOB tag.
    Convert a tree to the CoNLL IOB string format

    :param t: The tree to be converted.
    :type t: Tree
    :rtype: str
    """
    lines = [" ".join(token) for token in tree2conlltags(t)]
    return '\n'.join(lines)

### IEER

_IEER_DOC_RE = re.compile(r'<DOC>\s*'
                          r'(<DOCNO>\s*(?P<docno>.+?)\s*</DOCNO>\s*)?'
                          r'(<DOCTYPE>\s*(?P<doctype>.+?)\s*</DOCTYPE>\s*)?'
                          r'(<DATE_TIME>\s*(?P<date_time>.+?)\s*</DATE_TIME>\s*)?'
                          r'<BODY>\s*'
                          r'(<HEADLINE>\s*(?P<headline>.+?)\s*</HEADLINE>\s*)?'
                          r'<TEXT>(?P<text>.*?)</TEXT>\s*'
                          r'</BODY>\s*</DOC>\s*', re.DOTALL)

_IEER_TYPE_RE = re.compile('<b_\w+\s+[^>]*?type="(?P<type>\w+)"')

def _ieer_read_text(s, root_label):
    stack = [Tree(root_label, [])]
    # s will be None if there is no headline in the text
    # return the empty list in place of a Tree
    if s is None:
        return []
    for piece_m in re.finditer('<[^>]+>|[^\s<]+', s):
        piece = piece_m.group()
        try:
            if piece.startswith('<b_'):
                m = _IEER_TYPE_RE.match(piece)
                if m is None: print('XXXX', piece)
                chunk = Tree(m.group('type'), [])
                stack[-1].append(chunk)
                stack.append(chunk)
            elif piece.startswith('<e_'):
                stack.pop()
#           elif piece.startswith('<'):
#               print "ERROR:", piece
#               raise ValueError # Unexpected HTML
            else:
                stack[-1].append(piece)
        except (IndexError, ValueError):
            raise ValueError('Bad IEER string (error at character %d)' %
                             piece_m.start())
    if len(stack) != 1:
        raise ValueError('Bad IEER string')
    return stack[0]

def ieerstr2tree(s, chunk_types = ['LOCATION', 'ORGANIZATION', 'PERSON', 'DURATION',
               'DATE', 'CARDINAL', 'PERCENT', 'MONEY', 'MEASURE'], root_label="S"):
    """
    Return a chunk structure containing the chunked tagged text that is
    encoded in the given IEER style string.
    Convert a string of chunked tagged text in the IEER named
    entity format into a chunk structure.  Chunks are of several
    types, LOCATION, ORGANIZATION, PERSON, DURATION, DATE, CARDINAL,
    PERCENT, MONEY, and MEASURE.

    :rtype: Tree
    """

    # Try looking for a single document.  If that doesn't work, then just
    # treat everything as if it was within the <TEXT>...</TEXT>.
    m = _IEER_DOC_RE.match(s)
    if m:
        return {
            'text': _ieer_read_text(m.group('text'), root_label),
            'docno': m.group('docno'),
            'doctype': m.group('doctype'),
            'date_time': m.group('date_time'),
            #'headline': m.group('headline')
            # we want to capture NEs in the headline too!
            'headline': _ieer_read_text(m.group('headline'), root_label),
            }
    else:
        return _ieer_read_text(s, root_label)


def demo():

    s = "[ Pierre/NNP Vinken/NNP ] ,/, [ 61/CD years/NNS ] old/JJ ,/, will/MD join/VB [ the/DT board/NN ] ./."
    import nltk
    t = nltk.chunk.tagstr2tree(s, chunk_label='NP')
    print(t.pprint())
    print()

    s = """
These DT B-NP
research NN I-NP
protocols NNS I-NP
offer VBP B-VP
to TO B-PP
the DT B-NP
patient NN I-NP
not RB O
only RB O
the DT B-NP
very RB I-NP
best JJS I-NP
therapy NN I-NP
which WDT B-NP
we PRP B-NP
have VBP B-VP
established VBN I-VP
today NN B-NP
but CC B-NP
also RB I-NP
the DT B-NP
hope NN I-NP
of IN B-PP
something NN B-NP
still RB B-ADJP
better JJR I-ADJP
. . O
"""

    conll_tree = conllstr2tree(s, chunk_types=('NP', 'PP'))
    print(conll_tree.pprint())

    # Demonstrate CoNLL output
    print("CoNLL output:")
    print(nltk.chunk.tree2conllstr(conll_tree))
    print()


if __name__ == '__main__':
    demo()

