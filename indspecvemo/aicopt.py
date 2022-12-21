'''
Created on 22.05.2020

@author: Samuel
'''
import os
from itertools import combinations as itercombinations, product as iterproduct
from concurrent.futures import as_completed
from collections import defaultdict
from settrie import SetTrie
import warnings

import numpy as np
from pebble import ProcessPool

DEBUGMODE = False

def DefaultResultHandler(result):
    if hasattr(result, "fun"):
        if hasattr(result, "success"):
            error = not result.success
        else:
            error = None
        return result.fun, result, error
    else:
        if hasattr(result, "__iter__"):
            return result
        return result, None
        

def update_dependencies(covariates, fixed, dependencies):
    
    for i, (covariateConsidered, isFixed, dependency) in \
                            enumerate(zip(covariates, fixed, dependencies)):
        if dependency == i:
            continue
        
        if not covariateConsidered:
            if isFixed:
                # remove dependency
                dependencies[i] = i
            else:
                if fixed[dependency]:
                    if covariates[dependency]:
                        # remove dependency
                        dependencies[i] = i
                    else:
                        # fix to False
                        fixed[i] = True
                        covariates[i] = False
            continue
        
        if isFixed:
            if fixed[dependency]:
                if covariates[dependency]: 
                    # remove dependency
                    dependencies[i] = i
                else:
                    # note inconsistency
                    raise ValueError("The given fixed values do not satsify "
                                     "the given dependencies.") 
            else:
                # fix to True
                covariates[dependency] = True
                fixed[dependency] = True
                return update_dependencies(covariates, fixed, dependencies)
        elif fixed[dependency]: 
            if covariates[dependency]: 
                # remove dependency
                dependencies[i] = i
            else:
                # fix to False
                fixed[i] = True
                covariates[i] = False
                return update_dependencies(covariates, fixed, dependencies)
        
    return covariates, fixed, dependencies


class AICOptimizer():
    
    def __init__(self, fun, baseCovariates, args=[], dependencies=None,
                 checkIntegrity=True, resultExtractor=None, maxWorkers=None):
        """
        Initialize the AICOptimizer
        
        Parameters
        ----------
        fun : callable
            A method returning the negative likelihood of the model for a 
            given set of covariates. First argument must be a boolean array
            indicating for each parameter whether it is to be considered in the
            model (see `baseCovariates`). Other positional arguments of fun will 
            be passed via the `args` argument (below).
            
        """
        self._fun = fun
        self._pool = None
        
        if not isinstance(baseCovariates, np.ndarray):
            if hasattr(baseCovariates, "__iter__"):
                baseCovariates = np.array(baseCovariates, bool)
            else:
                baseCovariates = np.ones(baseCovariates, bool)
        
        if np.ma.is_masked(baseCovariates):
            fixed = baseCovariates.mask | (~baseCovariates.data)
            baseCovariates = baseCovariates.data
        else:
            fixed = ~baseCovariates
            
        if dependencies is not None:
            baseCovariates, fixed, dependencies = update_dependencies(
                baseCovariates.copy(), fixed, dependencies.copy())
            
            if (dependencies == np.arange(dependencies.size)).all():
                dependencies = None
        
        self._checkIntegrity = checkIntegrity
        self._baseCovariates = baseCovariates
        self._dependencies = dependencies
        self._basePermutation = frozenset(np.nonzero(~fixed)[0])
        self._bestAIC = np.inf
        self._jobs = {}
        self._finishedJobs = set()
        self._activeLevel = len(self._basePermutation)
        self._results = np.zeros(100, dtype=[
            ("permutation", object),
            ("covariates", str(baseCovariates.size)+"bool"),
            ("level", int),
            ("nLL", float),
            ("AIC", float),
            ("childLevel", int),
            ("processed", bool),
            ("error", bool),
            ("info", object),
            ])
        if maxWorkers is None:
            maxWorkers = os.cpu_count()
        self._maxWorkers = maxWorkers
        self._resultCount = 0
        self._submissionComplete = False
        self._limitReached = False
        self._args = args
        if resultExtractor:
            self._resultHandler = resultExtractor
        else:
            self._resultHandler = DefaultResultHandler
        self._noParentTrie = SetTrie(self._basePermutation)
        self._permutationGenerator = []
        self._maxNLLs = defaultdict(lambda: -np.inf)
        self._mininmalRejectionLevel = self._activeLevel
        
    @property
    def results(self):
        return self._results[:self._resultCount]
    
    def set_active_level(self, level, forceReset=False):
        
        increase = self._activeLevel < level
        if self._activeLevel == level and not forceReset:
            return
        
        if DEBUGMODE:
            print("@@@LEVEL set to", level)
        
        if level > 0 and not self._limitReached:
            self._submissionComplete = False
        
        self._activeLevel = level
        results = self.results
        processed = (results["childLevel"] < level) & (results["level"] > level) & (~results["error"])
        self._permutationGenerator = []
        
        if increase or not (processed == results["processed"]).all():
            self._results["processed"] = False
            self._noParentTrie.clear()
        else:
            self._noParentTrie.prune(level)
            
    def get_permutations(self, level, fixedIndexSet=None):
        
        permutation = self._basePermutation
        if fixedIndexSet:
            fixedIndexSet = frozenset(fixedIndexSet)
            level -= len(fixedIndexSet)
            permutation = permutation.difference(fixedIndexSet) 
        
        permutations = itercombinations(permutation, level)
        if fixedIndexSet is None:
            permutations = map(frozenset, permutations)
        else:
            permutations = map(lambda x: fixedIndexSet.union(x), permutations)
            
        if self._dependencies is None:
            yield from permutations
            return
        
        dependencies = self._dependencies 
        
        for perm in permutations:
            if perm.issuperset(dependencies[list(perm)]):
                yield perm
        
    def get_restricted_permutations(self):
        
        level = self._activeLevel
        
        if self._mininmalRejectionLevel <= self._activeLevel:
            noParents = []
        else:
            results = self.results
            processing = np.nonzero(~results["processed"])[0]
            results = results[processing]
            processing = processing[(results["childLevel"] < level)
                                                    & (results["level"] > level)
                                                    & (~results["error"])]
            
            noParents = self._results["permutation"][processing]
        
        if not len(noParents):
            if self._permutationGenerator:
                return self._permutationGenerator
            elif not self._noParentTrie:
                self._permutationGenerator = self.get_permutations(level)
                return self._permutationGenerator
        
        if level < 1:
            return []
        
        
        tmpTrie = SetTrie(self._basePermutation)
        tmpTrie.extend(self._basePermutation.difference(p) for p in noParents)
        tmpTrie.delete_supersets()
        self._noParentTrie.update_by_settrie(tmpTrie, level)
        
        self._results["processed"][processing] = True
        fixedIndexSets = self._noParentTrie.get_frozensets()
        
        dependencies = self._dependencies 
        
        def permutationGenerator():
            for fixedIndexSet in fixedIndexSets:
                
                if len(fixedIndexSet) == level:
                    if fixedIndexSet.issuperset(dependencies[list(fixedIndexSet)]):
                        yield frozenset(fixedIndexSet)
                    continue
                
                yield from self.get_permutations(level, fixedIndexSet)
        
        self._permutationGenerator = permutationGenerator()
        return self._permutationGenerator
        
    
    def cancel_job(self, permutation):
        job = self._jobs.get(permutation, None)
        if job and job.cancel(): 
            del self._jobs[permutation]
        
    def cancel_all_jobs(self):
        for job in self._jobs.values():
            job.cancel()
            
        self._jobs.clear()
    
    def sanity_check(self, permutation, nLL):
        level = len(permutation)
        results = self.results
        permutations = results["permutation"]
        resetLevel = None
        isError = False
        errorCount = 0
        
        if level > self._activeLevel:
            considered = np.nonzero((results["level"] < level) & 
                                    (results["nLL"] < nLL))[0]
            for perm, c in zip(permutations[considered], considered):
                if perm.issubset(permutation):
                    isError = True
                    if DEBUGMODE:
                        print("@@@EE1", self.permutation_to_covariates(permutation).astype(int), 
                              self.permutation_to_covariates(perm).astype(int),
                              results["nLL"][c], nLL)
                    errorCount += 1
                    break
        
        for l in range(level+1, len(self._basePermutation)+1):
            maxLevelNLL = self._maxNLLs[l]
            if maxLevelNLL > nLL:
                updateMaxLevelNLL = False
                considered = np.nonzero((results["level"]==l) & 
                                        (results["nLL"] > nLL) & 
                                        (~results["error"]))[0]
                for i in considered:
                    if permutations[i].issuperset(permutation):
                        results["error"][i] = True
                        if DEBUGMODE:
                            print("@@@EE2", self.permutation_to_covariates(permutations[i]).astype(int), 
                                  self.permutation_to_covariates(permutation).astype(int),
                                  results["nLL"][i], nLL
                                  )
                        errorCount += 1
                        if results["nLL"][i] == maxLevelNLL:
                            updateMaxLevelNLL = True
                            
                        if results["childLevel"][i] < l-1:
                            resetLevel = l
                if updateMaxLevelNLL:
                    remainingNLLs = results["nLL"][
                                    (results["level"]==l) & (~results["error"])]
                    if not remainingNLLs.size:
                        self._maxNLLs[l] = -np.inf
                    else:
                        self._maxNLLs[l] = np.max(remainingNLLs)
        
        if resetLevel is not None:
            self.set_active_level(resetLevel, True)      
        
        if errorCount:
            if errorCount > 1:
                warnings.warn(("At least {} results are erroneous. I ignore "
                               "these results").format(errorCount))
                if DEBUGMODE:
                    print(("@@@E At least {} results are erroneous. I ignore "
                                   "these results").format(errorCount))
            else:
                warnings.warn("A result is erroneous. I ignore it.")
                if DEBUGMODE:
                    print("@@@E A result is erroneous. I ignore it.")
        
        return isError
            
    def add_result(self, permutation, nLL, isError, info=None): 
        
        covariates = self.permutation_to_covariates(permutation)
        
        level = len(permutation)
        if self._resultCount == self._results.size:
            self._results = np.append(self._results, np.zeros_like(self._results))
        
        self._jobs.pop(permutation, None)
        self._finishedJobs.add(permutation)
        
        AIC = 2*(nLL+level)
        
        updateBestAIC = self._bestAIC > AIC
        
        if updateBestAIC: 
            self._bestAIC = AIC
        
        
        try:
            childLevel = int(self._bestAIC / 2 - nLL)
        except:
            # if nLL is NaN or Inf assign an arbitrary level that does no harm
            childLevel = level
        
        if not isError and self._checkIntegrity:
            isError = self.sanity_check(permutation, nLL)
            
        self._results[self._resultCount] = (permutation, covariates, level, nLL, 
                                            AIC, childLevel, False, isError, 
                                            info)
        self._resultCount += 1
        
        if isError:
            return
        
        self._maxNLLs[level] = max(self._maxNLLs[level], nLL)
        
        if not self._checkIntegrity > 1:
            cancel = []
            for perm, job in self._jobs.items():
                l = len(perm) 
                if l < level and perm.issubset(permutation):
                    if l > childLevel:
                        if DEBUGMODE:
                            print("@@@C cancel job with level", l, perm, 
                                  "parent:", level, permutation, nLL, AIC, childLevel)
                        cancel.append(perm)
                    else:
                        job.__nLL = max(job.__nLL, nLL)
            
            for perm in cancel:
                self.cancel_job(perm)
                
        if updateBestAIC: 
            self.update_best_AIC(AIC)
            
    
    def update_best_AIC(self, AIC):
        self._bestAIC = AIC
        cancel = []
        
        self.results["childLevel"][~self.results["error"]
                                   ] = AIC/2 - self.results["nLL"][~self.results["error"]]
            
        for permutation, job in self._jobs.items():
            if len(permutation) > AIC/2 - job.__nLL:
                cancel.append(permutation)
        
        for permutation in cancel:
            self.cancel_job(permutation)
    
    def permutation_to_covariates(self, permutation):
        result = self._baseCovariates.copy()
        result[list(self._basePermutation)] = False
        result[list(permutation)] = True
        return result
    
    def start_job(self, permutation, nLL=-np.inf):
        if DEBUGMODE:
            print("@@@S start job with level", len(permutation), permutation)
        job = self._pool.schedule(self._fun, [self.permutation_to_covariates(permutation),
                                              *self._args])
        job.__permutation = permutation
        job.__nLL = nLL
        self._jobs[permutation] = job
    
    @property
    def jobCount(self):
        return len(self._jobs)
        
    def add_jobs(self, limit=None):
        
        openSpots = self._maxWorkers - self.jobCount
        
        if openSpots <= 0:
            return
        
        for level in range(self._activeLevel, -1, -1):
            
            self.set_active_level(level)
            
            for permutation in self.get_restricted_permutations():
                
                if permutation in self._jobs or permutation in self._finishedJobs:
                    continue 
                
                self.start_job(permutation)
                openSpots -= 1
                if limit and self._resultCount + self.jobCount >= limit:
                    self._submissionComplete = True
                    self._limitReached = True
                    if DEBUGMODE:
                        print("@@@LIMIT reached", self._resultCount, self.jobCount, limit)
                    return
                if not openSpots:
                    return
        
        self._submissionComplete = True
    
    def all_running(self):
        return all(job.running() for job in self._jobs.values()) or not self._jobs
        
    def run(self, limit=None, mininmalRejectionLevel=0, sortResults=True):
        """Optimizes the AIC of the given model
        
        Returns a structured array containing the covariates and the AIC values
        of all considered models.
        
        Parameters
        ----------
        limit : int
            Maximal number of likelihood maximizations
        mininmalRejectionLevel : int
            Number of parameters that need to be switched off before likelihood
            optimizations can be omitted. This argument is not necessary if
            the likelihood maximizations always yield correct results. Otherwise,
            the argument can prevent that bad optimmizations with many,
            potentially unidentifiable, parameters block the consideration of
            promising candidates with less parameters. Note that setting this
            parameter to a large value will dramatically increase computation
            time, since at least `(n choose mininmalRejectionLevel)` will be 
            considered.
        sortResults : bool
            If True, the results will be sorted in ascending order of the AIC.
            
        """
        i = 0
        self._limitReached = False
        self._mininmalRejectionLevel = self._activeLevel - mininmalRejectionLevel
        
        with ProcessPool(max_workers=self._maxWorkers) as pool:
            self._pool = pool
            self.add_jobs(limit)
            try:
                while self._jobs:
                    i += 1
                    completedJob = next(as_completed(self._jobs.values()))
                    
                    permutation = completedJob.__permutation
                    result = self._resultHandler(completedJob.result())
                    nLL, info = result[:2]
                    if len(result) == 3:
                        isError = result[2]
                    else:
                        isError = False
                        
                    self.add_result(permutation, nLL, isError, info)
                    
                    # if there is another result available already, 
                    # postpone the addition of new jobs
                    if self.all_running() and not self._submissionComplete:
                        self.add_jobs(limit)
            finally:
                self.cancel_all_jobs()
        
        return self.extract_results(sortResults)
        
    def extract_results(self, sortResults=True):
        results = self.results
        
        cleanResult = np.zeros_like(self.results, dtype=[
            ("deltaAIC", float),
            ("AIC", float),
            ("nLL", float),
            ("covariates", str(self._baseCovariates.size)+"bool"),
            ("error", bool),
            ("info", object)
            ])
        
        cleanResult[["covariates", "nLL", "AIC", "error", "info"]] = results[
                                ["covariates", "nLL", "AIC", "error", "info"]]
        cleanResult["deltaAIC"] = cleanResult["AIC"] - self._bestAIC
        
        if sortResults:
            return cleanResult[np.argsort(cleanResult["deltaAIC"])]
        else:
            return cleanResult

def testFun(x):
    from time import sleep
    print(x.sum(), "Start")
    timeRequirement = x.sum()/100
    sleep(timeRequirement)
    
    print(x.sum(), x)
    if not x[:3].any():
        return -1
    if x.sum() > x.size-5:
        return 0
    return 15
    
    vals = x * np.arange(x.size)
    result = -np.sum(vals*vals[::-1])/10
    return result

def testFun2(x):
    from time import sleep
    print(x.sum(), "Start")
    timeRequirement = x.sum()/100
    sleep(timeRequirement)
    
    print(x.sum(), x)
    if x.sum() > x.size-5:
        return 0
    return 15
    
    vals = x * np.arange(x.size)
    result = -np.sum(vals*vals[::-1])/10
    return result

"""

def futuresTest():
    
    #with ProcessPoolExecutor() as pool:
    with ProcessPool() as pool:
        def submitWithID(arg, ID):
            #fut = pool.submit(fun, arg)
            fut = pool.schedule(fun, [arg])
            fut.__ID = ID
            #setattr(fut, "ID", ID)
            return fut
        jobs = {}
        for i in [4,3,2,5,7,1]:
            ID = i
            jobs[ID] = submitWithID(i, ID)
        for i in range(10):
            fut = next(as_completed(jobs.values()))
            try:
                print(fut.__ID, fut.result())
            except CancelledError:
                print("Retrieved cancelled result", fut.__ID)
                continue
            finally:
                jobs.pop(fut.__ID)
            if i > 3:
                if 5 in jobs: 
                    print("Cancelling", 5, jobs[5].cancel())
                    
            ID = fut.__ID+i*1.1
            jobs[ID] = submitWithID(i*1.1, ID)
"""

def stepAICTest():
    dim = 20
    x0 = np.ma.ones(dim, bool)
    x0[1] = False
    x0[[1,4]] = np.ma.masked
    dependencies = np.arange(dim)
    dependencies[3] = 1
    dependencies[5] = 4
    dependencies[6] = 5
    manager = AICOptimizer(testFun, x0, [], dependencies)
    results = manager.run()
    print(results)
    print(results.size)
if __name__ == '__main__':
    from vemomoto_core.tools.simprofile import profile
    profile("stepAICTest()", globals(), locals())
    #print("done")
    pass