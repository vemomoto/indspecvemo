# distutils: language=c
#cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False
'''
Created on 05.07.2016

@author: Samuel
'''


from datrie cimport BaseTrie, BaseState, BaseIterator
from math import inf

cdef bint has_subset_c(BaseTrie trie, BaseState trieState, str setarr, 
                        int index, int size):
    cdef BaseState trieState2 = BaseState(trie)
    cdef int i
    trieState.copy_to(trieState2)
    for i in range(index, size):
        if trieState2.walk(setarr[i]):
            if trieState2.is_terminal() or has_subset_c(trie, trieState2, setarr, 
                                                        i, size): 
                return True
            trieState.copy_to(trieState2)
    return False

"""
# issue: deletes also the tested item
cdef void delete_subsets_c(BaseTrie trie, BaseState trieState, str setarr, 
                           int index, int size, str trace):
    cdef bint leaf = trieState.is_leaf()
    if trieState.is_terminal():
        trie._delitem(trace)
        if leaf:
            return
        
    cdef BaseState trieState2 = BaseState(trie)
    cdef int i
    trieState.copy_to(trieState2)
    for i in range(index, size):
        if trieState2.walk(setarr[i]):
            delete_subsets_c(trie, trieState2, setarr, i, size, trace+setarr[i])
            trieState.copy_to(trieState2)
    
# issue: childres are all descendents not only the direct descendents
cdef bint has_superset_c(BaseTrie trie, BaseState trieState, str setarr, 
                        int index, int size):
    
    if index >= size:
        return False
    
    cdef BaseIterator trieIter = BaseIterator(trieState)
    cdef str currentElem = setarr[index]
    cdef str elem
    cdef bint found 
    cdef BaseState trieState2 = BaseState(trie)
    
    while trieIter.next():
        elem = iter.key()[0]
        if elem > currentElem:
            break
        
        trieState.copy_to(trieState2)
        trieState2.walk(elem)
        
        if elem == currentElem:
            found = has_superset_c(trie, trieState2, setarr, 
                                   index+1, size)
        else:
            found = has_superset_c(trie, trieState2, setarr, 
                                   index, size)
        
        if found: 
            return True
    
    return False
    
    cdef int i
    
    
    trieState.copy_to(trieState2)
    for i in range(index, size):
        if trieState2.walk(setarr[i]):
            if trieState2.is_leaf() or has_subset_c(trie, trieState2, setarr, 
                                                    i, size): 
                return True
            trieState.copy_to(trieState2)
    return True
"""

cdef class SetTrie():
    
    def __init__(self, alphabet, initSet=[]):
        if not hasattr(alphabet, "__iter__"):
            alphabet = range(alphabet)
        self.trie = BaseTrie("".join(chr(i+1) for i in alphabet))
        self.touched = False
        for i in initSet:
            self.trie[chr(i+1)] = 0
            if not self.touched:
                self.touched = True

    def has_subset(self, superset):
        cdef BaseState trieState = BaseState(self.trie)
        setarr = "".join(chr(i+1) for i in superset)
        return bool(has_subset_c(self.trie, trieState, setarr, 0, len(setarr)))
    
    def extend(self, sets):
        for s in sets:
            self.trie["".join(chr(i+1) for i in sorted(s))] = 0
            if not self.touched:
                self.touched = True
    
    """
    def delete_subsets(self, superset=None):
        cdef str elem 
        cdef BaseState trieState = BaseState(self.trie)
        cdef BaseIterator trieIter = BaseIterator(BaseState(self.trie))
        if superset is None:
            while trieIter.next():
                elem = trieIter.key()
                delete_subsets_c(self.trie, trieState, elem, 0, len(elem), '')
        else:
            elem = "".join(chr(i+1) for i in superset)
            delete_subsets_c(self.trie, trieState, elem, 0, len(elem), '')
    """
    
    def delete_supersets(self):
        cdef str elem 
        cdef BaseState trieState = BaseState(self.trie)
        cdef BaseIterator trieIter = BaseIterator(BaseState(self.trie))
        if trieIter.next():
            elem = trieIter.key()
            while trieIter.next():
                self.trie._delitem(elem)
                if not has_subset_c(self.trie, trieState, elem, 0, len(elem)):
                    self.trie._setitem(elem, 0)
                elem = trieIter.key()
            if has_subset_c(self.trie, trieState, elem, 0, len(elem)):
                val = self.trie.pop(elem)
                if not has_subset_c(self.trie, trieState, elem, 0, len(elem)):
                    self.trie._setitem(elem, val)
            
    
    def update_by_settrie(self, SetTrie setTrie, maxSize=inf, initialize=True):
        cdef BaseIterator trieIter = BaseIterator(BaseState(setTrie.trie))
        cdef str s
        if initialize and not self.touched and trieIter.next():
            for s in trieIter.key():
                self.trie._setitem(s, 0)
            self.touched = True
        
        while trieIter.next():
            self.update(trieIter.key(), maxSize, True)
    
    def update(self, otherSet, maxSize=inf, isString=False):
        cdef str otherSetStr
        if isString:
            otherSet = set(otherSet)
        else:
            otherSet = set(chr(i+1) for i in otherSet)
        cdef str subset, newSubset, elem
        cdef list disjointList = []
        cdef BaseTrie trie = self.trie
        cdef int l
        cdef BaseIterator trieIter = BaseIterator(BaseState(self.trie))
        if trieIter.next():
            subset = trieIter.key()
            while trieIter.next():
                if otherSet.isdisjoint(subset):
                    disjointList.append(subset)
                    trie._delitem(subset)
                subset = trieIter.key()
            if otherSet.isdisjoint(subset):
                disjointList.append(subset)
                trie._delitem(subset)
        
        cdef BaseState trieState = BaseState(self.trie)
        
        for subset in disjointList:
            l = len(subset)
            if l < maxSize:
                if l+1 > self.maxSizeBound:
                    self.maxSizeBound = l+1
                for elem in otherSet:
                    newSubset = subset + elem
                    trieState.rewind()
                    if not has_subset_c(self.trie, trieState, newSubset, 0, 
                                        len(newSubset)):
                        trie[newSubset] = 0
    
    def get_frozensets(self):
        return (frozenset(ord(t)-1 for t in subset) for subset in self.trie)
        
    def clear(self):
        self.touched = False
        self.trie.clear()
    
    def prune(self, maxSize):
        cdef bint changed = False
        cdef BaseIterator trieIter 
        cdef str k
        if self.maxSizeBound > maxSize:
            self.maxSizeBound = maxSize
            trieIter = BaseIterator(BaseState(self.trie))
            k = ''
            while trieIter.next():
                if len(k) > maxSize:
                    self.trie._delitem(k)
                    changed = True
                k = trieIter.key()
            if len(k) > maxSize:
                self.trie._delitem(k)
                changed = True
        return changed
    
    def __nonzero__(self):
        return self.touched
    
    def __repr__(self):
        return str([set(ord(t)-1 for t in subset) for subset in self.trie])
    
    def __iter__(self):
        return self.get_frozensets()
    
    def get_trie(self):
        return self.trie
        