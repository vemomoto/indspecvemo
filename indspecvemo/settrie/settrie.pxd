# distutils: language=c
#cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False
'''
@author Samuel Fischer
'''

from datrie cimport BaseTrie, BaseState, BaseIterator

cdef bint has_subset_c(BaseTrie trie, BaseState trieState, str setarr, int index, int size)

"""
cdef void delete_subsets_c(BaseTrie trie, BaseState trieState, str setarr, 
                          int index, int size, str trace)
"""

cdef class SetTrie():
    cdef BaseTrie trie
    cdef bint touched
    cdef int maxSizeBound