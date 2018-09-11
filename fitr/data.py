# -*- coding: utf-8 -*-
# Fitr. A package for fitting reinforcement learning models to behavioural data
# Copyright (C) 2018 Abraham Nunes
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# CONTACT INFO:
#   Abraham Nunes
#    Email: nunes@dal.ca
#
# ============================================================================
import re
import autograd.numpy as np
from nltk.util import skipgrams

class BehaviouralData(object):
    """ A flexible and generic object to store and process behavioural data across tasks

    Arguments:

        ngroups: Integer number of groups represented in the dataset. Only > 1 if data are merged
        nsubjects: Integer number of subjects in dataset
        ntrials: Integer number of trials done by each subject
        dict: Dictionary storage indexed by subject.
        params: `ndarray((nsubjects, nparams + 1))` parameters for each (simulated) subject
        meta: Array of covariates of type `ndarray((nsubjects, nmetadata_features+1))`
        tensor: Tensor representation of the behavioural data of type `ndarray((nsubjects, ntrials, nfeatures))`
    """
    def __init__(self, nsubjects):
        """
        Arguments:
            nsubjects: Integer number of subjects to be included in dataset.
        """
        self.ngroups   = 1
        self.nsubjects = nsubjects
        self.ntrials = None
        self.dict = None
        self.params = None
        self.meta = None
        self.tensor = None
        self.initialize_data_dictionary()

    def initialize_data_dictionary(self):
        self.dict = {}
        self.params = []
        self.meta = []
        for i in range(self.nsubjects):
            self.dict[i] = []

    def add_subject(self, subject_index, parameters, subject_meta):
        """ Appends a new subject to the dataset

        Arguments:

            subject_index: Integer identification for subject
            parameters: `list` of parameters for the subject
            subject_meta: Some covariates for the subject (`list`)
        """
        self.meta.append([subject_index] + subject_meta)
        self.params.append([subject_index] + parameters)

    def update(self, subject_index, behav_data):
        """ Adds behavioural data to the dataset

        Arguments:

            subject_index: Integer index for the subject
            behav_data: 1-dimensional `ndarray` of flattened data
        """
        self.dict[subject_index].append(behav_data)

    def numpy_tensor_to_bdf(self, X):
        """ Creates `BehaviouralData` formatted set from a dataset stored in a numpy `ndarray`.

        Arguments:

            X: `ndarray((nsubjects, ntrials, m))` with `m` being the size of flattened single-trial data
        """
        self.nsubjects, self.ntrials, _ = X.shape
        self.tensor = X

    def make_tensor_representations(self):
        """ Creates a tensor with all subjects' data

        #### Notes

        Assumes that all subjects did same number of trials.
        """
        self.ntrials = len(self.dict[0])
        m = len(self.dict[0][0])
        self.tensor  = np.empty((self.nsubjects, self.ntrials, m))
        for i in range(self.nsubjects):
            self.tensor[i,:,:] = self.dict[i]

        # Do the same for parameters (convert to np array)
        self.params = np.array(self.params)
        self.meta   = np.array(self.meta)

    def make_behavioural_ngrams(self, n):
        """ Creates N-grams of behavioural data """
        #TODO: explain difference between kmer and ngram here
        nsubjects = self.nsubjects
        ntrials   = self.ntrials
        nfeatures = self.tensor[0].shape[1]
        self.kmer_tensor = np.empty((nsubjects,ntrials-n+1,nfeatures*n))
        self.behavioural_ngrams = []
        for i in range(nsubjects):
            kmers = make_behavioural_kmers(self.tensor[i],n)
            ngram = [re.sub('[\.\s\[\]]','',str(x)) for x in kmers]
            self.behavioural_ngrams.append(ngram)
            self.kmer_tensor[i,:,:] = kmers
            ngtunique = np.unique(self.kmer_tensor[i], axis=0)

        self.behavioural_ngrams = np.array(self.behavioural_ngrams)
        self.vocabulary = np.unique(self.behavioural_ngrams.flatten())
        self.ntokens = self.vocabulary.size

    def make_cooccurrence_matrix(self, k, dtype=np.float32):
        X = np.zeros((self.ntokens, self.ntokens), dtype=np.float32)
        word_int, int_word = hash_vocabulary(self.vocabulary)

        for i, S in enumerate(self.behavioural_ngrams):
            skipgram_generator = skipgrams(S, 2, k)
            for s in skipgram_generator:
                idx1 = word_int[s[0]]
                idx2 = word_int[s[1]]
                X[idx1, idx2] += 1

        self.cooccurrence_matrix = X

    def unpack_tensor(self, x_dim, u_dim, r_dim=1, terminal_dim=1, get='sarsat'):
        """ Unpacks data stored in tensor format into separate arrays for states, actions, rewards, next states, and next actions.

        Arguments:

            x_dim : Task state space dimensionality (`int`)
            u_dim : Task action space dimensionality (`int`)
            r_dim : Reward dimensionality (`int`, default=1)
            terminal_dim : Dimensionality of the terminal state indicator (`int` , default=1)
            get : String indicating the order that data are stored in the array. Can also be shortened such that fewer elements are returned. For example, the default is `sarsat`.

        Returns:

            List with data, where each element is in the order of the argument `get`
        """
        out = []
        i_start = 0
        for l in get:
            if l == 's':
                i_end = i_start + x_dim
            elif l == 'a':
                i_end = i_start + u_dim
            elif l == 'r':
                i_end = i_start + r_dim
            elif l == 't':
                i_end = i_start + terminal_dim

            out.append(self.tensor[:,:,i_start:i_end])
            i_start += i_end-i_start
        return out

def hash_vocabulary(V):
    """ Creates a hash table for each unique behavioural N-gram """
    word_int = {} # Mappings from vocabulary to integers
    int_word = {} # Mapping from integer to vocabulary
    for i, w in enumerate(V):
        word_int[w] = i
        int_word[i] = w
    return word_int, int_word

def make_behavioural_kmers(x, n):
    """ Creates K-mers of behavioural data """
    ntrials = x.shape[0]
    y = np.empty((ntrials-n+1, x.shape[1]*n))
    for i in range((ntrials-n)+1):
        y[i,:] = x[i:i+n,:].flatten()
    return y

def merge_behavioural_data(datalist):
    """ Combines BehaviouralData objects.

    Arguments:

        datalist: List of BehaviouralData objects

    Returns:

        `BehaviouralData` with data from multiple groups merged.
    """
    ngroups = len(datalist)

    data = datalist[0] # The first group will be the reference
    if data.tensor is None: data.make_tensor_representations()
    data.ngroups = ngroups

    for i,D in enumerate(datalist[1:]):
        if D.ntrials == data.ntrials:
            data.nsubjects += D.nsubjects
            if D.tensor is None: D.make_tensor_representations()
            data.tensor = np.concatenate((data.tensor, D.tensor), axis=0)
            data.meta = np.concatenate((data.meta, D.meta), axis=0)
            data.params = np.concatenate((data.params, D.params), axis=0)
        else:
            print("Could not merge group %s due to unequal trial number" %D.ntrials)

    newidx = np.arange(data.nsubjects)
    data.meta[:,0] = newidx
    data.params[:,0] = newidx
    return data


def fast_unpack(x, ranges):
    """ Unpacks data stored in BDF tensor format (`ndarray(dim=1)`) quickly.

    This function is useful for trial-by trial extraction of data in likelihood calculations.

    Arguments:

        x: `ndarray(nfeatures)`. Vector to be unpacked.
        ranges: `list` of `ndarray` objects or more `list` objects. Each element is a range indexing the columns of x to be extracted.

    Returns:

        `list` with `ndarray` objects, where each element corresponds to the columns specified by respective elements of `ranges`.

    Examples:

    ```python
    ranges = [np.arange(2), np.arange(2)+2, 4, 4+np.arange(2)]
    D = np.outer(np.ones(5), np.arange(7))
    x, u, r, x_ = fast_unpack(D[0], ranges)
    ```
    """
    return [x[range_i] for i, range_i in enumerate(ranges)]
