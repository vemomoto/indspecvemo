# cython: profile=False
"""
Cython wrapper for libdatrie.
"""

from cpython.version cimport PY_MAJOR_VERSION
from cython.operator import dereference as deref
from libc.stdlib cimport malloc, free
from libc cimport stdio
from libc cimport string
cimport stdio_ext
cimport cdatrie

import itertools
import warnings
import sys
import tempfile

try:
    from collections.abc import MutableMapping
except ImportError:
    from collections import MutableMapping

try:
    import cPickle as pickle
except ImportError:
    import pickle

cdef class BaseTrie:
    cdef AlphaMap alpha_map
    cdef cdatrie.Trie *_c_trie

    cpdef bint is_dirty(self)

    cdef void _setitem(self, unicode key, cdatrie.TrieData value)

    
    cdef cdatrie.TrieData _getitem(self, unicode key) except -1

    
    cpdef bint _delitem(self, unicode key) except -1
    
    @staticmethod
    cdef int len_enumerator(cdatrie.AlphaChar *key, cdatrie.TrieData key_data,
                            void *counter_ptr)
    
    cdef cdatrie.TrieData _setdefault(self, unicode key, cdatrie.TrieData value)

    
    cpdef suffixes(self, unicode prefix)

    cdef list _prefix_items(self, unicode key)

    cdef list _prefix_values(self, unicode key)

    cdef _longest_prefix_item(self, unicode key, default)

    cdef _longest_prefix_value(self, unicode key, default)

    cpdef items(self, unicode prefix)

    
    cpdef keys(self, unicode prefix)

    cpdef values(self, unicode prefix)

    cdef _index_to_value(self, cdatrie.TrieData index)


cdef class Trie(BaseTrie):
    """
    Wrapper for libdatrie's trie.
    Keys are unicode strings, values are Python objects.
    """

    cdef list _values

    cpdef items(self, unicode prefix)

    cpdef values(self, unicode prefix)

    cdef _index_to_value(self, cdatrie.TrieData index)


cdef class _TrieState:
    cdef cdatrie.TrieState* _state
    cdef BaseTrie _trie

    cpdef walk(self, unicode to)

    cdef bint walk_char(self, cdatrie.AlphaChar char)

    cpdef copy_to(self, _TrieState state)

    cpdef rewind(self)

    cpdef bint is_terminal(self)

    cpdef bint is_single(self)

    cpdef bint is_leaf(self)



cdef class BaseState(_TrieState):
    """
    cdatrie.TrieState wrapper. It can be used for custom trie traversal.
    """
    cpdef int data(self)


cdef class State(_TrieState):
    cpdef data(self)


cdef class _TrieIterator:
    cdef cdatrie.TrieIterator* _iter
    cdef _TrieState _root

    
    cpdef bint next(self)

    cpdef unicode key(self)


cdef class BaseIterator(_TrieIterator):
    """
    cdatrie.TrieIterator wrapper. It can be used for custom datrie.BaseTrie
    traversal.
    """
    cpdef cdatrie.TrieData data(self)


cdef class Iterator(_TrieIterator):
    """
    cdatrie.TrieIterator wrapper. It can be used for custom datrie.Trie
    traversal.
    """
    cpdef data(self)


cdef (cdatrie.Trie* ) _load_from_file(f) except NULL


cdef class AlphaMap:
    """
    Alphabet map.

    For sparse data compactness, the trie alphabet set should
    be continuous, but that is usually not the case in general
    character sets. Therefore, a map between the input character
    and the low-level alphabet set for the trie is created in the
    middle. You will have to define your input character set by
    listing their continuous ranges of character codes creating a
    trie. Then, each character will be automatically assigned
    internal codes of continuous values.
    """

    cdef cdatrie.AlphaMap *_c_alpha_map


    cdef AlphaMap copy(self)

    cpdef _add_range(self, cdatrie.AlphaChar begin, cdatrie.AlphaChar end)


cdef cdatrie.AlphaChar* new_alpha_char_from_unicode(unicode txt)
cdef unicode unicode_from_alpha_char(cdatrie.AlphaChar* key, int len)

