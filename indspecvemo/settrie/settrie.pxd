# distutils: language=c
#cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False
'''
@author Samuel Fischer
'''

from datrie cimport BaseTrie, BaseState, BaseIterator

ctypedef unsigned char char_type

cdef bint has_subset(BaseTrie trie, BaseState trieState, str setarr, int index, int size) except +
cdef bint has_subset_c(BaseTrie trie, BaseState trieState, char* setarr, int index, int size) except +

cdef class SetTrie():
    cdef BaseTrie trie
    cdef bint touched
    cdef int maxSizeBound