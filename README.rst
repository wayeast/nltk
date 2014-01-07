Repo: nltk
-------------
Features
----------
- *Python*
- *NLTK*
- ``pandas`` *Python statistical package*

Description
---------------
Sentence parsing is one of the common functions of natural language processing
packages.  Though there exist NLP packages that perform full-sentence parsing,
this is often a low-performance task.  Motivated by an interest in extracting
just 1) a main subject and 2) a main verb phrase from any given sentence, my
approach in this project was to perform a partial parse to identify these two
items of interest without building an entire parse tree from the input.  The
method I chose to use was a purely probabilistic one that takes as input a
sentence that has been tokenized, part-of-speech tagged, and chunked, and
produces as output a subject index, verb index tuple representing the most
probable locations of the main subject and main verb in a chunk tree.

The model here is an extension of the code base of the Natural Language Toolkit
(NLTK).  The NLTK *chunk* module stores a sentence as an nltk.Tree data structure
-- a class that inherits from the Python list class and represents a sentence
as a list of chunks.  To allow for the tagging and prediction of subject and
verb relationships, I created a class SVTree (**tree.py, lines 780-856**) that
inherited from the nltk.Tree class and added a data member *gram_role* that
could be either *sb* or *vb*.

The actual model (**chunk/util.py, lines 304-38**) consists of three pieces: 1) a
beta distribution *s* representing the prior probability that the main subject of
a sentence will occur at a given point in a sentence, 2) a beta distribution
*v*
representing the probability that the main verb of a sentence will occur at a
given point in a sentence, and 3) a likelihood function that I tested as either
*i*) a table T[c\ :subscript:`i` -> c\ :subscript:`i+1`\ ] of chunk transition
probabilities occurring over the
segment of chunks between a subject and verb chunk, or *ii*) a probability
distribution *sep\_prob*\(d) giving the probability that the difference *index*\
(verb) - *index*\(subject) == d.

The algorithm takes advantage of the fact that a subject must be a noun phrase
chunk and a verb must be a verb phrase chunk, so when a new sentence is input
to the model (as a sequence of chunks with chunk labels), the indices of all
*n*
noun phrases and all *m* verb phrases are first extracted.  An *n* x *m* prior
probability matrix *prior*\[n][m] is constructed, from the model's beta
distributions as a function of their relative position in the sentence, showing
the probability that noun phrase n\ :subscript:`i` and verb phrase m\ :subscript:`j`
are the main subject and
verb of the sentence (**chunk/util.py, lines 468-80**).  This probability is the
normalized product of *s*\(n\ :subscript:`i` == main subject) and *v*\(m\
:subscript:`j` == main verb).

Next, the prior probability is updated (**chunk/util.py, lines 494-509**).  When
testing using the chunk transition table T, this update accounted for 
L([n\ :sub:`i`\ ,
... , m\ :sub:`j`\ ]) -- the likelihood of the occurrence of the segment of chunks between
n\ :sub:`i` and m\ :sub:`j` (**chunk/util.py, lines 482-92**).  This is represented as the joint
probability of all of the chunk transitions from c\ :subscript:`i` to c\
:subscript:`i+1` that occur along
the segment, or the product of all probabilities T[c\ :subscript:`i` -> c\
:subscript:`i+1`\
].  An *n* x *m*
posterior probability matrix *post*\[n][m] is constructed where *post*\[\
n\ :subscript:`i`\ ][m\ :subscript:`j`\ ] =
*prior*\[n\ :subscript:`i`\ ][m\ :subscript:`j`\ ] \* L([n\ :subscript:`i`\ ,
... , m\ :subscript:`j`\
]).  When testing using the index difference
probability distribution *sep_prob*\(d), the update accounted for the probability
of the degree of subject-verb separation.  Similarly to the previous method,
this is calculated as *post*\[n\ :subscript:`i`\ ][m\ :subscript:`j`\ ] =
*prior*\[n\ :subscript:`i`\ ]\
[m\ :subscript:`j`\ ] \* *sep_prob*\[j - i].

From the resulting posterior matrix, the combination of subject index, verb
index with the highest probability is returned as the model.s prediction.

**NOTE** that this fork of the NLTK source code was used as a base for a graduate class
project whose aim was to predict the main subject and verb of a sentence using
probabilistic modelling methods.  This is not the NLTK project home; the NLTK code
here is not up to date; see the text of the original NLTK README file in README.orig.


