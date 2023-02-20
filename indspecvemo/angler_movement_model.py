'''
Created on 09.05.2020

@author: Samuel M. Fischer
'''
from functools import partial
from itertools import repeat, count, chain as iterchain,  product as iterproduct
from collections import defaultdict, OrderedDict
import os
from copy import copy
import sys

import numpy as np
from scipy import signal
from numpy.lib import recfunctions
import scipy.optimize as op
from scipy.stats import nbinom, vonmises
from scipy.sparse import csr_matrix
from scipy.special import binom, hyp2f1
import pandas as pd
import numdifftools as nd

from matplotlib import pyplot as plt
    
import autograd.numpy as ag
from autograd import grad, hessian

from hybrid_vector_model.hybrid_vector_model import BaseTrafficFactorModel, \
    _non_join, create_observed_predicted_mean_error_plot, safe_delattr
from vemomoto_core.tools.hrprint import HierarchichalPrinter
from vemomoto_core.tools.saveobject import SeparatelySaveable
from vemomoto_core.concurrent.concurrent_futures_ext import ProcessPoolExecutor
from vemomoto_core.npcollections.npext import convert_R_pos,\
    convert_R_0_1, convert_R_pos_reverse
from autograd.scipy.special import gammaln

from vemomoto_core.tools.doc_utils import inherit_doc, staticmethod_inherit_doc
from ci_rvm import find_CI_bound

from .aicopt import AICOptimizer

MAX_WORKERS = os.cpu_count()

# If working on a server, decrease the utilized cpu cores
if MAX_WORKERS > 40:
    MAX_WORKERS //= 3
    import matplotlib
    matplotlib.use("Agg")

if __name__ == '__main__':
    # The first command line argument specifies the output file to which all output
    # will be written.     
    from vemomoto_core.tools.tee import Tee                
    if len(sys.argv) > 1:
        teeObject = Tee(sys.argv[1])
        
IDTYPE = int
IDTYPE_A = "|S9"
class ndarray_flex(np.ndarray):
    
    def pick(self, other):
        result = ndarray_flex(self.size, dtype=object)
        resultarr = result.array
        resultarr[:] = [other[a] for a in self]
        return result
    
    @staticmethod
    def from_csr_matrix(arr):
        result = ndarray_flex(arr.shape[0], dtype = object)
        resultarr = result.array
        for i in range(result.size):
            resultarr[i] = arr[i].data
        result.alldata = arr.data
        return result
    
    @staticmethod
    def from_arr_list(arrs):
        data = np.concatenate([arr.ravel() for arr in arrs])
        result = ndarray_flex(len(arrs), dtype = object)
        resultarr = result.array
        counter = 0
        for i, arr in enumerate(arrs):
            resultarr[i] = np.reshape(data[counter:counter+arr.size], arr.shape)
            counter += arr.size
        result.alldata = data
        return result
    
    def __getitem__(self, index):
        if not isinstance(index, tuple) or len(index) <= 1 or not isinstance(index[0], slice):
            return np.ndarray.__getitem__(self, index)
        rIndex = index[1:]
        result = ndarray_flex(self[index[0]].size, dtype=object)
        result.array[:] = [a[rIndex] for a in self[index[0]]]
        return result
    
    def __setitem__(self, index, val):
        if not hasattr(index, "__len__") or len(index) <= 1:
            if isinstance(index, slice) or hasattr(index, "__iter__"):
                any(np.ndarray.__setitem__(a, None, val) for a in 
                    np.ndarray.__getitem__(self, index))
            else:
                np.ndarray.__getitem__(self, index)[:] = val
        else:
            rIndex = index[1:]
            if isinstance(index[0], slice):
                if hasattr(val, "__iter__"):
                    any(np.ndarray.__setitem__(a, rIndex, v) for a, v in 
                        zip(np.ndarray.__getitem__(self, index[0]), val))
                else:
                    any(np.ndarray.__setitem__(a, rIndex, val) for a in 
                        np.ndarray.__getitem__(self, index[0]))
            else: 
                np.ndarray.__getitem__(self, index[0])[rIndex] = val
            
    @property
    def array(self):
        return np.asarray(self)
    
    @property
    def fullsize(self):
        return sum(a.size for a in self)
    
    @property
    def fullshape(self):
        return tuple(a.shape for a in self)
    
    def sum(self, axis=None):
        if axis is None:
            return sum(a.sum() for a in self)
        if not axis:
            return np.sum(self)
        if self[0].ndim == 1:
            result = np.ndarray(self.size)
            resultarr = result
        else:
            result = ndarray_flex(self.size, dtype=object)
            resultarr = result.array
        axis -= 1
        resultarr[:] = [a.sum(axis) for a in self]
        return result
    
    def prod(self, axis=None):
        if axis is None:
            return np.prod(a.prod() for a in self)
        if not axis:
            return np.prod(self)
        if self[0].ndim == 1:
            result = np.ndarray(self.size)
            resultarr = result
        else:
            result = ndarray_flex(self.size, dtype=object)
            resultarr = result.array
        axis -= 1
        resultarr[:] = [a.prod(axis) for a in self]
        return result
    
    def max(self, axis=None):
        if axis is None:
            return max(a.max() for a in self)
        if not axis:
            return np.max(self)
        if self[0].ndim == 1:
            result = np.ndarray(self.size)
            resultarr = result
        else:
            result = ndarray_flex(self.size, dtype=object)
            resultarr = result.array
        axis -= 1
        resultarr[:] = [a.max(axis) for a in self]
        return result

def convolve_positive(a, b, minval=1e-50, axes=1):
    result = signal.oaconvolve(a, b, axes=axes)
    result = np.maximum(result, minval, result)
    return result

def date_to_int(date_str):
    return int((np.datetime64(date_str, 'D')-np.datetime64('1899-12-31', 'D')).astype(int))+1

def create_permuations(arr, fixedTrue=[], fixedFalse=[]):
    return list(sum(j, []) for j in 
                iterproduct(*(
                    [[True]*i] if j in fixedTrue else 
                    ([[False]*i] if j in fixedFalse else
                    [[True]*i, [False]*i])  
                              for j, i in enumerate(arr))))

def create_constrained_permuations(arr, maxDisabled=None, fixedTrue=[], fixedFalse=[]):
    permutations = create_permuations(arr, fixedTrue, fixedFalse)
    minEnabled = np.sum(arr) - maxDisabled
    return [l for l in permutations if np.sum(l)>=minEnabled]

def _count_occurrence(arr):
    counter = defaultdict(lambda: count())
    return [next(counter[val]) for val in arr]

def toAbsoluteDependencies(relativeDependencies):
    return relativeDependencies + np.arange(len(relativeDependencies))

class AICPrintManager(HierarchichalPrinter):
    def __init__(self, **printerArgs):
        HierarchichalPrinter.__init__(self, **printerArgs)
        self._models = []
        self.bestAIC = np.inf
        self.bestModel = None
    
    def add_model(self, nLL, parametersConsidered, *parameters, silent=True):
        AIC = (np.sum(parametersConsidered) + nLL) * 2
        modelInfo = (AIC, nLL, parametersConsidered, *parameters)
        self._models.append(modelInfo)
        if AIC < self.bestAIC:
            self.bestAIC = AIC
            self.bestModel = modelInfo
        if not silent:
            self.prst("AIC: {:8.2f} | -l: {:8.2f} | ".format(AIC, nLL), 
                      parametersConsidered, " | ".join(str(i) for i in parameters))
            
    def print_AIC_overview(self):
        
        models = sorted(self._models, key=lambda x: x[0])
        bestAIC = models[0][0]
        
        self.increase_print_level()
        
        for AIC, nLL, parametersConsidered, *parameters in models:
            self.prst("DeltaAIC: {:8.2f} | AIC: {:8.2f} | -l: {:8.2f} | ".format(
                AIC-bestAIC, AIC, nLL, parametersConsidered), 
                " | ".join(str(i) for i in parameters))
        
        self.decrease_print_level()
        

def resultHandler(result):
    return result.fun, (result.x, result.xOriginal), not result.success

class TimeFactorModel():
    
    SIZE = 6
    "(`int`) -- Maximal number of parameters in the implemented model."
    
    PERMUTATIONS = create_permuations([1]*6)
    
    LABELS = np.array([
        "Year addition constant",
        "Week addition constant",
        "Year shape parameter",
        "Week shape parameter",
        "Year location parameter",
        "Week location parameter"
        ])
    
    def __init__(self, dates, parametersConsidered=None):
        self.set_parameters_considered(parametersConsidered)
        self.yearDaysTransformed, self.weekDaysTransformed = self.transformDates(dates)
    
    def transformDates(self, dates):
        # the -1 is due to the MS Excel 1900 leap year error
        dateShift = int(np.datetime64('1899-12-31', 'D').astype(int))-1
        dates = np.array(dates+dateShift, dtype='datetime64[D]')
        
        
        yearDays = (dates-np.array(dates, dtype='datetime64[Y]')).astype(int)
        weekDays = (dates.astype(int) + (np.datetime64('2020-01-06', 'D')
                                         -np.datetime64('2020-01-06', 'W')).astype(int)
                     ) % 7
        return yearDays / 365 * 2 * np.pi, weekDays / 7 * 2 * np.pi
    
    def set_parameters_considered(self, parametersConsidered):
        self.parametersConsidered = parametersConsidered
    
    def get_time_factor(self, parameters, dates=None, normalized=True):
        cY, cW, kappaY, kappaW, locY, wLocW = self.convert_parameters(parameters)
        if dates is not None:
            yearDaysTransformed, weekDaysTransformed = self.transformDates(dates)
        else:
            yearDaysTransformed, weekDaysTransformed = self.yearDaysTransformed, self.weekDaysTransformed
        result = ((cY+vonmises.pdf(yearDaysTransformed, kappaY, locY))
                  *(cW+vonmises.pdf(weekDaysTransformed, kappaW, wLocW)))
        if normalized:
            result /= result.mean()
        return result
    
    def convert_parameters(self, parameters):   
        parametersConsidered = self.parametersConsidered
        if parametersConsidered is None: # or np.all(parametersConsidered):
            parametersConsidered = np.ones(6, dtype=bool)
            #return np.concatenate((np.exp(parameters[:4]), parameters[4:]))
        
        result = np.zeros(6)
        result[2:4] = 1e-15 # to prevent numerical issues only
        i = 0
        if parametersConsidered[0]: 
            #result[0] = convert_R_pos(parameters[0])
            result[0] = np.exp(parameters[0])
            i += 1
        if parametersConsidered[1]: 
            #result[1] = convert_R_pos(parameters[i])
            result[1] = np.exp(parameters[i])
            i += 1
        if parametersConsidered[2]: 
            #result[2] = convert_R_pos(parameters[i])
            result[2] = np.exp(parameters[i])
            i += 1
        if parametersConsidered[3]: 
            #result[3] = convert_R_pos(parameters[i]) 
            result[3] = np.exp(parameters[i])  
            i += 1
        if parametersConsidered[4]: 
            result[4] = parameters[i]
            i += 1
        if parametersConsidered[5]: 
            result[5] = parameters[i]
            i += 1
        return result
    

class TimeModel(SeparatelySaveable, HierarchichalPrinter):
    
    STATIC_PARAMETERS_LABELS = [
        "Dispersion constant",
        "Mean/variance constant",
        ]
    
    @property
    def isFitted(self):
        return hasattr(self, "AIC") and hasattr(self, "parameters")
    
    def __init__(self, timeFactorModel, countData, variableDispersion=True,
                 **printerArgs):
        SeparatelySaveable.__init__(self)
        HierarchichalPrinter.__init__(self, **printerArgs)
        self.timeFactorModel = timeFactorModel
        self.countData = countData
        self.set_variable_dispersion(variableDispersion)
    
    def set_variable_dispersion(self, variableDispersion):
        self.variableDispersion = variableDispersion
    
    def convert_parameters(self, parameters):
        return np.concatenate((np.exp(parameters[:2]), 
                               self.timeFactorModel.convert_parameters(parameters[2:])))
    
    def get_r_q(self, parameters):
        c0, c1 = np.exp(parameters[:2])
        if self.variableDispersion:
            return c0*self.timeFactorModel.get_time_factor(parameters[2:]), 1/(1+c1)
        else:
            return c0, 1/(1+c1*self.timeFactorModel.get_time_factor(parameters[2:]))
    
    def negative_log_likelihood(self, parameters):
        return -nbinom.logpmf(self.countData, *self.get_r_q(parameters)).sum()
    
    def maximize_likelihood(self, parametersConsidered=None, 
                            variableDispersion=None, parameters=None,
                            continueOptimization=True):
        
        fun, jac, hess = self.get_nLL_functions(parametersConsidered,
                                                variableDispersion)
        
        if not continueOptimization and parameters is not None:
            result = op.OptimizeResult(x=parameters, success=True, status=0,
                                       fun=self.negative_log_likelihood(parameters, parametersConsidered), 
                                       nfev=1, njev=0,
                                       nhev=0, nit=0,
                                       message="parameters checked")
        else: 
            if parameters is None:
                parameters = [0]*(parametersConsidered.sum()+2) # + [0, -1, 0, 0]
                bounds = [(-8, 5)] * len(parameters)
                parameters = op.differential_evolution(fun, bounds, maxiter=20).x
            result = op.minimize(fun, parameters, jac=jac, hess=hess, method="trust-exact")
        
        result.xOriginal = self.convert_parameters(result.x)
        return result
            
    def get_nLL_functions(self, parametersConsidered, variableDispersion):
        if parametersConsidered is not None:
            self.timeFactorModel.set_parameters_considered(parametersConsidered)
        if variableDispersion is not None:
            self.set_variable_dispersion(variableDispersion)
        fun = self.negative_log_likelihood
        jac = nd.Gradient(fun) #, method='complex')
        hess = nd.Hessian(fun) #, method='complex')
        return fun, jac, hess
    
    def fit(self, permutations=None, variableDispersion=[False, True], 
            parameters=None, continueOptimization=False, get_CI=True, 
            plotFileName=None):
        
        self.prst("Fitting time models.")
        self.increase_print_level()
        
        fittedModel = False
        
        if permutations is None:
            permutations = self.timeFactorModel.PERMUTATIONS
        
        if permutations is None:
            permutations = np.ones(self.timeFactorModel.SIZE, dtype=bool)[None,:]
        
        if not hasattr(variableDispersion, "__iter__"):
            variableDispersion = [variableDispersion]
        
        permutationNo = len(permutations)
        permutations *= len(variableDispersion)
        permutations = np.array(permutations)
        variableDispersion = np.repeat(variableDispersion, permutationNo)
        
        myAICOptimizer = AICPrintManager(parentPrinter=self)
        
        if parameters is None:
            
            with ProcessPoolExecutor() as pool:
                mapObj = pool.map(self.maximize_likelihood, permutations,
                                  variableDispersion, chunksize=1)
                
                for permutation, varDisp, result in zip(permutations, variableDispersion, mapObj):
                    myAICOptimizer.add_model(result.fun, permutation, varDisp, result.x,
                                           result.xOriginal, silent=False)
            fittedModel = True
        else:
            if continueOptimization:
                result = self.maximize_likelihood(
                                  parameters["parametersConsidered"], 
                                  variableDispersion[0],
                                  parameters["paramters"])
                fittedModel = True
            else:
                result = self.maximize_likelihood(
                                  parameters["parametersConsidered"], 
                                  variableDispersion[0],
                                  parameters["parameters"],
                                  False)
            nLL = result.fun
            myAICOptimizer.add_model(nLL, parameters["parametersConsidered"], 
                                   variableDispersion[0], result.x, result.xOriginal)
        
        self.decrease_print_level()
        
        
        AIC, LL, covariates, variableDispersion, bestParameters, bestParametersOriginal = myAICOptimizer.bestModel
        
        self.prst("Choose the following covariates:")
        self.prst(covariates)
        self.prst("Choose variable dispersion:", variableDispersion)
        self.prst("Parameters (transformed):")
        self.prst(bestParameters) 
        self.prst("Parameters (original):")
        self.prst(bestParametersOriginal)
        self.prst("Negative log-likelihood:", LL, "AIC:", AIC)
        
        if myAICOptimizer.bestModel and fittedModel and get_CI: 
            self.investigate_profile_likelihood(bestParameters, covariates, 
                                                variableDispersion,
                                                disp=True, apprxtol=1.1, minstep=1e-10, nmax=400)
                 
        myAICOptimizer.print_AIC_overview()
        
        if (not self.isFitted or self.AIC >= AIC
            or (parameters is not None and not continueOptimization)):
            self.AIC = AIC
            self.parameters = bestParameters
            self.timeFactorModel.set_parameters_considered(covariates)
            self.plot_time_distribution(plotFileName)
            self.set_variable_dispersion(variableDispersion)
            
        self.decrease_print_level()
        return fittedModel
    
    def _find_profile_CI(self, x0, parametersConsidered, variableDispersion,
                         index, direction, profile_LL_args={}):
        """Searches the profile likelihood confidence interval for a given
        parameter.
        
        Parameters
        ----------
        profile_LL_args : dict
            Keyword arguments to be passed to :py:meth:`find_CI_bound`.
        
        """
        
        fun, jac, hess = self.get_nLL_functions(parametersConsidered, variableDispersion)
        
        fun_ = lambda x: -fun(x)   
        jac_ = lambda x: -jac(x)   
        hess_ = lambda x: -hess(x) 
        
        return find_CI_bound(x0, fun_, index, direction, jac_, hess_, 
                                     **profile_LL_args)
    
    def investigate_profile_likelihood(self, x0, parametersConsidered,
                                       variableDispersion,
                                       **profile_LL_args):
        """# Searches the profile likelihood confidence interval for a given
        parameter."""
        
        self.prst("Investigating the profile likelihood")
        
        self.increase_print_level()
        
        dim = len(x0)
        
        result = np.zeros((dim, 2))
        
        parametersConsidered = np.array([True]*2 + list(parametersConsidered[:2])
                                        + [True]*(self.timeFactorModel.LABELS.size-2))
        labels = self.STATIC_PARAMETERS_LABELS + list(self.timeFactorModel.LABELS[parametersConsidered[2:]])
        
        indices, directions = zip(*iterproduct(range(dim), (-1, 1)))
        
        self.prst("Creating confidence intervals")
        with ProcessPoolExecutor() as pool:
            mapObj = pool.map(self._find_profile_CI, 
                              repeat(x0), repeat(parametersConsidered),
                              repeat(variableDispersion),
                              indices, directions, repeat(profile_LL_args))
            
            for index, direction, r in zip(indices, directions, mapObj):
                result[index][(0 if direction==-1 else 1)
                              ] = np.array(self.convert_parameters(r.x))[ 
                                  parametersConsidered][index]
        
        self.prst("Printing confidence intervals and creating profile plots")
        self.increase_print_level()
        
        x0Orig = np.array(self.convert_parameters(x0))[parametersConsidered]
        
        for index, intv in enumerate(result):
            start, end = intv
            self.prst("CI for {:<40}: [{:10.4g} --- {:10.4g} --- {:10.4g}]".format(
                labels[index], start, x0Orig[index], end))
            
        self.decrease_print_level()
        self.decrease_print_level()
    
    def plot_time_distribution(self, plotFileName):
        allDays = np.arange(1, self.timeFactorModel.yearDaysTransformed.size+1)
        """
        plt.plot(allDays, (counts-nbinom.mean(*getRQ(x)))/nbinom.std(*getRQ(x)))
        """
        plt.rcParams.update({'font.size': 12})
        plt.figure(figsize=(12,4))
        plt.fill_between(allDays, nbinom.ppf(0.025, *self.get_r_q(self.parameters)),
                         nbinom.ppf(0.975, *self.get_r_q(self.parameters)), 
                         facecolor='#ff7f0e', alpha=0.4) #030764
        plt.plot(allDays, nbinom.mean(*self.get_r_q(self.parameters)), color='#d62728') #, color='k' 
        plt.plot(allDays, self.countData) #, color='#030764'
        
        plt.xlabel("Day $t$ of the study")
        xTickLabels = ["May 2018", "", "May 2019", "", "May 2020"]
        xTicks = np.linspace(0, 365*2+1, 5, dtype=int)
        plt.xticks(xTicks, xTickLabels)
        plt.xlim([-20, 751])
        
        plt.ylim([0, None])
        
        plt.yticks([0, 10, 20])
        
        plt.ylabel(r"Trip rate $\bar{\mu} \epsilon_t$")
        plt.tight_layout()
        #"""
        if plotFileName is not None:
            plt.savefig(plotFileName + ".pdf")
            plt.savefig(plotFileName + ".png", dpi=1000)
        
        plt.show()
    
    def get_time_factor(self, dates=None):
        return self.timeFactorModel.get_time_factor(self.parameters[2:], dates)
        
    
class LocalitySubbasinAnglerTrafficFactorModel(BaseTrafficFactorModel):
    
    SIZE = 24
    "(`int`) -- Maximal number of parameters in the implemented model."
    
    # If only the name of the covariate is given, the data type will be assumed
    # to be double
    ORIGIN_COVARIATES = [
        ("localityPopulation", float),
        ("localityAnglers", float),
        ("meanIncome", float),
        ("medianIncome", float),
        ]
    """(`(str, type)[]`) -- The names and types of the covariates for the sources.
    If the type is not spcified, it will default to float.
    """
    
    DESTINATION_COVARIATES = [
        ("waterArea", float),
        ("waterPerimeter", float),
        ("waterAreaConfirmed", float),
        ("waterPerimeterConfirmed", float),
        ("subbasinPopulation", float),
        ("campgrounds", float),
        ("speciesVotes", float),
        ("pageViews", float),
        ]
    "(`(str, type=double)[]`) -- The names and types of the covariates for the sinks."
    
    DEPENDENCIES = toAbsoluteDependencies(np.array([
        #0                5                10               15                20
        0, -1, 0, -1, 0, -1, 0, -1, 1, -1, 1, -1, 1, -1, 1, -1, 0, -1, 0, -1, 0, -1, 0, -1
        ]))
    
    BASE_PARAMETERS = None
    
    PARAMETERS_FIXED = [
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        ]
    
    
    """(`bool[][]`) -- Parameter combinations to be considered when selecting the optimal model.
    
    The number of columns must match the maximal number of parameters of the 
    model (see :py:attr:`SIZE`)."""
            
    LABELS = np.array([
        "Distance exponent",               #0
        "Base distance",
        #"Angler license exponent", 
        "Locality population factor", 
        "Locality population exponent", 
        "Locality mean income factor",         #5
        "Locality mean income exponent", 
        "Locality median income factor", 
        "Locality median income exponent", 
        "Water area factor",
        "Water area exponent",             #10
        "Water perimeter factor",
        "Water perimeter exponent",
        "Water area confirmed factor",
        "Water area confirmed exponent", 
        "Water perimeter confirmed factor",#15
        "Water perimeter confirmed exponent",
        "Subbasin population factor",
        "Subbasin population exponent",
        "Campground factor",             
        "Campground exponent",             #12
        "Species vote factor",
        "Species vote exponent",
        "Page visit factor",
        "Page visit exponent",
        ], dtype="object")
    """(`str[]`) -- The names of the parameters in the implemented model.
    
    The length must match the maximal number of parameters of the 
    model (see :py:attr:`SIZE`)."""
    
    BOUNDS = np.array([
        (-10, 10),
        (-10, 100),
        (-10, 100),
        (-5, 5),
        (-10, 100),
        (-5, 5),
        (-10, 100),
        (-5, 5),
        (-10, 100),
        (-10, 20),
        (-10, 100),
        (-10, 20),
        (-10, 100),
        (-10, 20),
        (-10, 100),
        (-10, 20),
        (-10, 100),
        (-10, 20),
        (-10, 100),
        (-10, 20),
        (-10, 100),
        (-10, 20),
        (-10, 100),
        (-10, 20),
        ])
    """(`(float, float)[]`) -- Reasonable bounds for the parameters (before conversion).
    
    The length must match the maximal number of parameters of the 
    model (see :py:attr:`SIZE`)."""
    
    def __init__(self, localityData, subbasinData, distanceData):
        """Constructor"""
        self.localityData = localityData
        self.subbasinData = subbasinData
        self.distanceData = distanceData
    
    def convert_parameters(self, dynamicParameters, parametersConsidered):
        """#
        Converts an array of given parameters to an array of standard (maximal)
        length and in the parameter domain of the model
        
        See `BaseTrafficFactorModel.convert_parameters`
        """
        
        result = [np.nan]*LocalitySubbasinAnglerTrafficFactorModel.SIZE
        j = 0
        for i in range(len(result)):
            if parametersConsidered[i]:
                if i == 3 or i == 5 or i == 7:
                    result[i] = dynamicParameters[j]
                else:
                    result[i] = convert_R_pos(dynamicParameters[j])
                j += 1
        
        return result
    
    def get_mean_factor(self, parameters, parametersConsidered, 
                        getProbabilities=True, getAnglerNumbers=True,
                        convertParameters=False, correctDimensionMode=False): #,
                        #DEBUG=False):
        """# 
        Gravity model for a factor proportional to the traffic flow
        between a jurisdiction and a subbasin.
        
        See `BaseTrafficFactorModel.get_mean_factor` for further details.
        """
        localityData = self.localityData
        subbasinData = self.subbasinData
        distanceData = self.distanceData
        
        if convertParameters:
            parameters = self.convert_parameters(parameters, parametersConsidered)
        
        # distance parameter
        i1 = 2
        d, d0 = parameters[:i1]
        cons_d, cons_d0 = parametersConsidered[:i1]
        
        # locality parameters
        i2 = i1 + 6
        cons_c1, cons_c2, cons_c3, cons_c4, cons_c5, cons_c6 = parametersConsidered[i1:i2]
        c1, c2, c3, c4, c5, c6 = parameters[i1:i2]
        
        # subbasin waterbody size paramters
        i1 = i2
        i2 = i1 + 8
        HS0, HS1, HS2, HS3, HS4, HS5, HS6, HS7 = parameters[i1:i2]
        cons_HS0, cons_HS1, cons_HS2, cons_HS3, cons_HS4, cons_HS5, cons_HS6, cons_HS7 = parametersConsidered[i1:i2]
        
        # subbasin tourism paramters
        i1 = i2
        i2 = i1 + 8
        HB0, HB1, HB2, HB3, HB4, HB5, HB6, HB7 = parameters[i1:i2]
        cons_HB0, cons_HB1, cons_HB2, cons_HB3, cons_HB4, cons_HB5, cons_HB6, cons_HB7 = parametersConsidered[i1:i2]
        
        power = ag.power
        npsum = ag.sum
        ones = ag.ones
        exp = ag.exp
        log = ag.log
            
        parametersConsidered = parametersConsidered+0
        
        if getProbabilities:
            
            if cons_d:
                if cons_d0:
                    distanceFactor = 1/(power(distanceData, d) + d0**d) 
                else:
                    distanceFactor = power(distanceData, -d)
            else:
                distanceFactor = np.ones_like(distanceData)
            
#             if DEBUG:
#                 print("A distanceFactor.sum()", distanceFactor.sum())
            
            sizeFactor = 1
            if cons_HS0:
                area = subbasinData["waterArea"]
                if cons_HS1:
                    HS1 = HS1 + 1e-100
                    if np.abs(HS1) > 5:
                        if not correctDimensionMode:
                            HS0 = exp(log(HS0)/HS1)
                        area = power(area*HS0, HS1)
                    else:
                        if correctDimensionMode:
                            HS0 = exp(log(HS0)*HS1)
                        area = power(area, HS1) * HS0
                else:
                    area = area * HS0
                if area.max() > 1e-30:
                    sizeFactor = sizeFactor + area
#             if DEBUG:
#                 print("B sizeFactor.sum()", sizeFactor.sum())
            
            if cons_HS2:
                perimeter = subbasinData["waterPerimeter"]
                if cons_HS3:
                    HS3 = HS3 + 1e-100
                    if np.abs(HS3) > 5:
                        if not correctDimensionMode:
                            HS2 = exp(log(HS2)/HS3)
                        perimeter = power(perimeter*HS2, HS3)
                    else:
                        if correctDimensionMode:
                            HS2 = exp(log(HS2)*HS3)
                        perimeter = power(perimeter, HS3)*HS2
                else:
                    perimeter = perimeter * HS2
                if perimeter.max() > 1e-30:
                    sizeFactor = sizeFactor + perimeter
#             if DEBUG:
#                 print("C sizeFactor.sum()", sizeFactor.sum())
            
            if cons_HS4:
                area = subbasinData["waterAreaConfirmed"]
                if cons_HS5:
                    HS5 = HS5 + 1e-100
                    if np.abs(HS5) > 5:
                        if not correctDimensionMode:
                            HS4 = exp(log(HS4)/HS5)
                        area = power(area*HS4, HS5)
                    else:
                        if correctDimensionMode:
                            HS4 = exp(log(HS4)*HS5)
                        area = power(area, HS5)*HS4
                else:
                    area = area * HS4
                if area.max() > 1e-30:
                    sizeFactor = sizeFactor + area
#             if DEBUG:
#                 print("D sizeFactor.sum()", sizeFactor.sum())
            
            if cons_HS6:
                perimeter = subbasinData["waterPerimeterConfirmed"]
                if cons_HS7:
                    HS7 = HS7 + 1e-100
                    if np.abs(HS7) > 5:
                        if not correctDimensionMode:
                            HS6 = exp(log(HS6)/HS7)
                            #HS6 = HS6**(1/HS7)
                        perimeter = power(perimeter*HS6, HS7)
                    else:
                        if correctDimensionMode:
                            HS6 = exp(log(HS6)*HS7)
                        perimeter = power(perimeter, HS7)*HS6
                else:
                    perimeter = perimeter * HS6
                if perimeter.max() > 1e-30:
                    sizeFactor = sizeFactor + perimeter
#             if DEBUG:
#                 print("E sizeFactor.sum()", sizeFactor.sum())
            
            tourismFactor = 1
            if cons_HB0:
                if cons_HB1:
                    HB1 = HB1 + 1e-100
                    if np.abs(HB1) > 5:
                        if not correctDimensionMode:
                            HB0 = exp(log(HB0)/HB1)
                        pop = power(subbasinData["subbasinPopulation"] * HB0, HB1)
                    else:
                        if correctDimensionMode:
                            HB0 = exp(log(HB0)*HB1)
                        pop = power(subbasinData["subbasinPopulation"], HB1) * HB0
                else:
                    pop = (subbasinData["subbasinPopulation"] > 0) * HB0
                if pop.max() > 1e-30:
                    tourismFactor = tourismFactor + pop 
#             if DEBUG:
#                 print("F tourismFactor.sum()", tourismFactor.sum())
            
            if cons_HB2:
                if cons_HB3:
                    HB3 = HB3 + 1e-100
                    if np.abs(HB3) > 5:
                        if not correctDimensionMode:
                            HB2 = exp(log(HB2)/HB3)
                        camp = power(subbasinData["campgrounds"]*HB2, HB3)
                    else:
                        if correctDimensionMode:
                            HB2 = exp(log(HB2)*HB3)
                        camp = power(subbasinData["campgrounds"], HB3)*HB2
                else:
                    camp = (subbasinData["campgrounds"] > 0) * HB2
                if camp.max() > 1e-30:
                    tourismFactor = tourismFactor + camp 
#             if DEBUG:
#                 print("G tourismFactor.sum()", tourismFactor.sum())
            
            if cons_HB4:
                if cons_HB5:
                    HB5 = HB5 + 1e-100
                    if np.abs(HB5) > 5:
                        if not correctDimensionMode:
                            HB4 = exp(log(HB4)/HB5)
                        spec = power(subbasinData["speciesVotes"]*HB4, HB5)
                    else:
                        if correctDimensionMode:
                            HB4 = exp(log(HB4)*HB5)
                        spec = power(subbasinData["speciesVotes"], HB5)*HB4
                else:
                    spec = (subbasinData["speciesVotes"] > 0) * HB4
                if spec.max() > 1e-30:
                    tourismFactor = tourismFactor + spec
#             if DEBUG:
#                 print("H tourismFactor.sum()", tourismFactor.sum())
            
            if cons_HB6:
                if cons_HB7:
                    HB7 = HB7 + 1e-200
                    if np.abs(HB7) > 5:
                        if not correctDimensionMode:
                            HB6 = exp(log(HB6)/HB7)
                        www = power(subbasinData["pageViews"]*HB6, HB7)
                    else:
                        if correctDimensionMode:
                            HB6 = exp(log(HB6)*HB7)
                        www = power(subbasinData["pageViews"], HB7)*HB6
                else:
                    www = (subbasinData["pageViews"] > 0) * HB6
                if www.max() > 1e-30:
                    tourismFactor = tourismFactor + www 
#             if DEBUG:
#                 print("I tourismFactor.sum()", tourismFactor.sum())
            
            result = (tourismFactor*sizeFactor)*distanceFactor
            
#             if DEBUG:
#                 print("J result.sum()", result.sum())
            
            if hasattr(result, "shape"):
                result /= npsum(result, 1)[:,None]
            else:
                result = ones(distanceData.shape)
        
        if not getAnglerNumbers:
            return result
        
        localityFactor = localityData["localityAnglers"]
        
        anglerActiveness = 1
        if cons_c1:
            population = localityData["localityPopulation"]
            if cons_c2:
                if np.abs(c2) > 5:
                    if not correctDimensionMode:
                        c1 = exp(log(c1)/c2)
                    population = power(population*c1, c2)
                else:
                    if correctDimensionMode:
                        c1 = exp(log(c1)*c2)
                    population = power(population, c2)*c1 #np.array([1]) #
            else:
                population = population * c1
            if population.max() > 1e-30:
                anglerActiveness = anglerActiveness + population 
#         if DEBUG:
#             print("K anglerActiveness.sum()", anglerActiveness.sum())
        
        if cons_c3:
            meanIncome = localityData["meanIncome"]
            if cons_c4:
                if np.abs(c4) > 5:
                    if not correctDimensionMode:
                        c3 = exp(log(c3)/c4)
                    meanIncome = power(meanIncome*c3, c4)
                else:
                    if correctDimensionMode:
                        c3 = exp(log(c3)*c4)
                    meanIncome = power(meanIncome, c4)*c3
            else:
                meanIncome = meanIncome * c3
            if meanIncome.max() > 1e-30:
                anglerActiveness = anglerActiveness + meanIncome
#         if DEBUG:
#             print("L anglerActiveness.sum()", anglerActiveness.sum())
            
        if cons_c5:
            medianIncome = localityData["medianIncome"]
            if cons_c6:
                if np.abs(c6) > 5:
                    if not correctDimensionMode:
                        c5 = exp(log(c5)/c6)
                    medianIncome = power(medianIncome*c5, c6)
                else:
                    if correctDimensionMode:
                        c5 = exp(log(c5)*c6)
                    medianIncome = power(medianIncome, c6)*c5
            else:
                medianIncome = medianIncome * c5
            if medianIncome.max() > 1e-30:
                anglerActiveness = anglerActiveness + medianIncome
#         if DEBUG:
#             print("M anglerActiveness.sum()", anglerActiveness.sum())
        
        localityFactor = localityFactor * anglerActiveness
        
#         if DEBUG:
#             print("N localityFactor.sum()", localityFactor.sum())
        if not getProbabilities:
            return ones(distanceData.shape[0]) * localityFactor
        else:
            return result * localityFactor[:,None]
    
    def convert_parameters_correct_dimension(self, parameters, parametersConsidered,
                                             convertParameters=True):
        
        if convertParameters:
            parameters = self.convert_parameters(parameters, parametersConsidered)
        
        # distance parameter
        i0 = parametersConsidered[:2].sum()
        
        result = [convert_R_pos_reverse(x) for x in parameters[:i0]]
        for i in range(i0, LocalitySubbasinAnglerTrafficFactorModel.SIZE):
            if parametersConsidered[i]:
                x = parameters[i]
                if not i % 2 and parametersConsidered[i+1]:
                    x = x**(1/parameters[i+1])
                if not i == 3 or i == 5 or i == 7:
                    x = convert_R_pos_reverse(x)
                result.append(x)
        
        return result
        
class BaseLocalitySubbasinAnglerTrafficModel(SeparatelySaveable, HierarchichalPrinter):
    
    STATIC_PARAMETERS_BOUNDS = []
    STATIC_PARAMETERS_LABELS = []
    ANGLER_DATA_DTYPE = [("anglerId", IDTYPE_A), ("localityId", IDTYPE), 
                         ("subbasinId", IDTYPE), ("date", int)]
    STATIC_PARAMETERS_SIZE = 0
    
    def __init__(self, trafficFactorModel_class, **printerArgs):
        """Constructor"""
        SeparatelySaveable.__init__(self)
        HierarchichalPrinter.__init__(self, **printerArgs)
        self._trafficFactorModel_class = trafficFactorModel_class
    
    @property 
    def hasCountData(self):
        return True
    
    def negative_log_likelihood(self, parameters, parametersConsidered, convertParameters=True):
        raise NotImplementedError()
    
    def get_nLL_functions(self, parametersConsidered, **kwargs):
        fun_ = self.negative_log_likelihood
        fun = partial(fun_, parametersConsidered=parametersConsidered, **kwargs)
        jac_ = grad(fun)
        jac_nd = nd.Gradient(fun, num_steps=2)
        def jac(x, convertParameters=True):
            y = jac_(x, convertParameters=convertParameters)
            if convertParameters and not np.isfinite(y).all():
                y = jac_nd(x, convertParameters=convertParameters)
                if convertParameters and not np.isfinite(y).all():
                    print("!JAC_NAN", parametersConsidered+0, list(x))
                    y[~np.isfinite(y)] = -1
            return y
        hess_ = hessian(fun)
        hess_nd = nd.Hessian(fun, num_steps=2)
        def hess(x, convertParameters=True):
            y = hess_(x, convertParameters=convertParameters)
            if convertParameters and not np.isfinite(y).all():
                y = hess_nd(x, convertParameters=convertParameters)
                if convertParameters and not np.isfinite(y).all():
                    print("!HESS_NAN", parametersConsidered+0, list(x))
                    y = np.eye(*y.shape)
            return y
        return fun, jac, hess
    
    def _convert_static_parameters(self, parameters):
        raise NotImplementedError()
        
        
    def convert_parameters(self, parameters, parametersConsidered):
        return (self._convert_static_parameters(parameters[:self.STATIC_PARAMETERS_SIZE]) 
                 + self._trafficFactorModel_class.convert_parameters(
                     self._trafficFactorModel_class,
                     parameters[self.STATIC_PARAMETERS_SIZE:], 
                     parametersConsidered[self.STATIC_PARAMETERS_SIZE:]))
    
    def _erase_model_fit(self):
        """Resets the gravity model to an unfitted state."""
        safe_delattr(self, "AIC")
        safe_delattr(self, "parameters")
        safe_delattr(self, "covariates")
    
    def _erase_survey_data(self):
        safe_delattr(self, "pairCountData")
        self._erase_model_fit()
    
    def _erase_distance_data(self):
        safe_delattr(self, "distanceData")
        self._erase_model_fit()
    
    def _erase_traffic_factor_model(self):
        """Erases the gravity model."""
        safe_delattr(self, "trafficFactorModel")
        self._erase_model_fit() 
    
    @property
    def isFitted(self):
        return hasattr(self, "AIC") and hasattr(self, "parameters") and hasattr(self, "covariates")
    
    def set_response_rate(self, responseRate):
        """Sets the boaters' compliance rate (for stopping at inspection/survey 
        locations) 
        
        The rate is used for both fitting the model and optimizing inspection 
        station operation
        
        Parameters
        ----------
        complianceRate : float
            Proportion of agents stopping at survey/inspection stations.
            
        """
        self.responseRate = responseRate
    
    def maximize_likelihood(self, parametersConsidered, parameters=None,
                            continueOptimization=True, disp=False): #disp=True): #
        
        fun, jac, hess = self.get_nLL_functions(parametersConsidered)
        
        if not continueOptimization and parameters is not None:
            result = op.OptimizeResult(x=parameters, success=True, status=0,
                                       fun=fun(parameters), 
                                       nfev=1, njev=0,
                                       nhev=0, nit=0,
                                       message="parameters checked")
            result.xOriginal = self.convert_parameters(
                                       result.x, parametersConsidered)
            result.jacOriginal = jac(result.xOriginal, convertParameters=False)
            return result
        
        #parameters = np.zeros(parametersConsidered.sum())
        
        if parameters is None:
            bounds = copy(self.STATIC_PARAMETERS_BOUNDS)
            
            staticParameterNumber = len(bounds)
            
            parametersConsidered[:staticParameterNumber] = True
            
            for bound in self._trafficFactorModel_class.BOUNDS[parametersConsidered[staticParameterNumber:]]:
                bounds.append(tuple(bound))
                
            np.random.seed()
            
            result = op.differential_evolution(fun, bounds, 
                                               popsize=30, maxiter=100, #300, 
                                               mutation=(0.5, 1.5),
                                               recombination=0.5,
                                               disp=disp)
            self.prst("Parameters:", parametersConsidered.astype(int))          
            self.prst("Differential evolution result:", result)
            parameters = result.x.copy()
            result.xOriginal = self.convert_parameters(result.x, parametersConsidered)
            result.jacOriginal = jac(result.xOriginal, convertParameters=False)
            
        result2 = op.minimize(fun, parameters, method="L-BFGS-B",
                              jac=jac, hess=hess,
                              bounds=None, options={"maxiter":2000,
                                                    "iprint":disp*2})
        
        result2.xOriginal = self.convert_parameters(result2.x, parametersConsidered)
        result2.jacOriginal = jac(result2.xOriginal, convertParameters=False)
        self.prst("Parameters:", parametersConsidered.astype(int))          
        self.prst("L-BFGS-B result:", result2)
        
        x0 = result2.x.copy()
        result = result2
        
        result2 = op.minimize(fun, x0, jac=jac, 
                              hess=hess, bounds=None, 
                              options={"maxiter":800, 
                                       "iprint":disp*2},
                              method="SLSQP")
        result2.xOriginal = self.convert_parameters(result2.x, parametersConsidered)
        result2.jacOriginal = jac(result2.xOriginal, convertParameters=False)
        self.prst("Parameters:", parametersConsidered.astype(int))          
        self.prst("SLSQP result:", result)
        if result2.fun < result.fun:
            parameters = result2.x.copy()
            result = result2
        try:
            maxiter = 2000
            maxiter = 1000
            result2 = op.minimize(fun, result.x, jac=jac, 
                                  hess=hess, bounds=None, 
                                  method="trust-exact",
                                  options={"maxiter":maxiter, 
                                           "disp":disp,
                                           #"gtol":1e-6,
                                           "max_trust_radius":50,
                                           })
            result2.xOriginal = self.convert_parameters(result2.x, parametersConsidered)
            result2.jacOriginal = jac(result2.xOriginal, convertParameters=False)
            self.prst("Parameters:", parametersConsidered.astype(int))          
            self.prst("Trust-exact result:", result2)
            if result2.fun < result.fun:
                result = result2
                if not result.success and result.nit == maxiter and np.mean(np.abs(result.jac)) < 0.00001:
                    print("Result success set to True")
                    result.success = True
        except ValueError:
            pass
            print("!VALUE_ERROR", parametersConsidered+0)
            
        
        if np.isnan(result.fun):
            result.fun = np.inf
            result.success = False
        
        return result
        
        
    def fit(self, basePermutation=None, dependencies=None, refit=False, parameters=None, 
            continueOptimization=False, get_CI=True, plotFileName=None, limit=np.inf):
        """Fits the traffic flow (gravity) model.
        
        Fits one or multiple candidates for the traffic flow model and selects
        the model with minimal AIC value. 
        
        Parameters
        ----------
        permutations : bool[][]
            Each row corresponds to a parameter combination of a models that 
            is to be considered. For each parameter that could be potentially 
            included, the row must contain a boolean value. Do only include 
            parameters included in the traffic factor model. If ``None``,
            the :py:attr:`PERMUTATIONS <BaseTrafficModel.PERMUTATIONS>` given
            in the traffic factor model class will be considered. If this 
            attribute is not implemented, only the full model will be considered.
        refit : bool
            Whether to repeat the fitting procedure if the model has been 
            fitted earlier.
        parameters : dict
            Dictionary with the keys ``"parametersConsidered"`` and ``"parameters"`` 
            that provides an initial guess for the optimization or the 
            corresponding solution. ``"parametersConsidered"`` contains a `bool[]` with 
            the considered parameter combination (see :py:obj:`permutations`);
            ``"parameters"`` contains a `float[]` with the values for the 
            parameters where ``flowParameters["parametersConsidered"]`` is ``True``.
        continueOptimization : bool
            If ``True``, the :py:obj:`flowParameters` will be used as initial 
            guess. Otherwise, they will be considered as the optimal 
            parameters.
        get_CI : bool
            Whether confidence intervals shall be computed after the model
            has been fitted. Note that no confidence intervals will be computed,
            if ``continueFlowOptimization is False``.
        
        """
        self.prst("Fitting flow models.")
        self.increase_print_level()
        
        fittedModel = False
        if not refit and self.isFitted:
            self.prst("A model does already exist. I skip",
                      "this step. Enforce fitting with the argument",
                      "refit=True")
            return False
        if not self.hasCountData:
            self.prst("The model has no prepared traveller data. I stop.",
                      "Call `set_count_data` if you want to",
                      "use the model.")
            return False
        if not hasattr(self, "trafficFactorModel"):
            self.prst("No traffic factor model has been specified. Call "
                      "model.set_traffic_factor_model(...)!")
            return False
        
        
        if basePermutation is None:
            basePermutation = self._trafficFactorModel_class.BASE_PARAMETERS
            if basePermutation is None:
                basePermutation = np.ones(self._trafficFactorModel_class.SIZE, 
                                          dtype=bool)
        
        if dependencies is None:
            dependencies = self._trafficFactorModel_class.DEPENDENCIES
        
        
        if len(basePermutation) == self._trafficFactorModel_class.SIZE:
            basePermutation = np.ma.concatenate(([True]*self.STATIC_PARAMETERS_SIZE,
                                                 basePermutation))
        basePermutation[:2] = np.ma.masked
        basePermutation[2:][self._trafficFactorModel_class.PARAMETERS_FIXED] = np.ma.masked
        
        basePermutation[8:10] = False
        basePermutation[8:10] = np.ma.masked
        
        basePermutation[-8] = False
        basePermutation[-8] = np.ma.masked
        
        #basePermutation = np.ma.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0]) > 0
        #basePermutation[:] = np.ma.masked
        
        if dependencies is not None and len(dependencies) == self._trafficFactorModel_class.SIZE:
            dependencies = np.concatenate((
                list(range(self.STATIC_PARAMETERS_SIZE)),
                dependencies+self.STATIC_PARAMETERS_SIZE)
            )
        
        
        if parameters is None:
            
            if self.isFitted and not refit:
                x0 = self.parameters
                basePermutation = np.ma.array(self.covariates)
                basePermutation[:] = np.ma.masked
            else: 
                x0 = None
            
            myAICOptimizer = AICOptimizer(self.maximize_likelihood, basePermutation, 
                                      [x0], dependencies, 2, resultHandler, MAX_WORKERS)
            fittedModel = True
        else:
            parametersConsidered = np.ma.array(parameters["parametersConsidered"])
            parametersConsidered[:] = np.ma.masked
            myAICOptimizer = AICOptimizer(self.maximize_likelihood, parametersConsidered, 
                                          [parameters["parameters"], continueOptimization], 
                                          dependencies, 2, resultHandler, 1)
            limit = 0
            if continueOptimization:
                fittedModel = True
        
        results = myAICOptimizer.run(limit, 0)
        results = results[np.argsort(results["deltaAIC"])]
        self.results = results
        
        self.decrease_print_level()
        
        _, AIC, nLL, covariates, error, (bestParameters, bestParametersOriginal) = results[0]
        
        self.prst("Choose the following covariates:")
        self.prst(covariates)
        self.prst("Parameters (transformed):")
        self.prst(bestParameters) 
        self.prst("Parameters (original):")
        self.prst(bestParametersOriginal)
        self.prst("Negative log-likelihood:", nLL, "AIC:", AIC)
        
        if results.size and fittedModel and get_CI: 
            self.investigate_profile_likelihood(bestParameters, covariates, 
                                                correctDimensionMode=True,
                                                disp=True)
                 
        self.increase_print_level()
        
        for deltaAIC, AIC, nLL, parametersConsidered, error, (x, xOriginal) in results:
            self.prst("DeltaAIC: {:8.2f}{} | AIC: {:8.2f} | -l: {:8.2f} | ".format(
                deltaAIC, "*" if error else " ", AIC, nLL), 
                " | ".join((str(i) for i in (parametersConsidered.astype(int), xOriginal, list(x)))))
        
        self.decrease_print_level()
            
        if (not self.isFitted or self.AIC >= AIC
            or (parameters is not None and not continueOptimization)):
            self.AIC = AIC
            self.parameters = bestParameters
            self.covariates = covariates
            self.plot_observed_predicted(plotFileName)
        
        self.decrease_print_level()
        return fittedModel
    
    def _find_profile_CI(self, x0, parametersConsidered, index,
                         direction, likelihood_args={}, CI_args={}):
        """Searches the profile likelihood confidence interval for a given
        parameter.
        
        Parameters
        ----------
        profile_LL_args : dict
            Keyword arguments to be passed to :py:meth:`find_CI_bound`.
        
        """
        
        fun, jac, hess = self.get_nLL_functions(parametersConsidered, 
                                                **likelihood_args)
        
        fun_ = lambda x: -fun(x)   
        jac_ = lambda x: -jac(x)   
        hess_ = lambda x: -hess(x) 
        
        return find_CI_bound(x0, fun_, index, direction, jac_, hess_, 
                                     **CI_args)
    
    def investigate_profile_likelihood(self, x0, parametersConsidered,
                                       correctDimensionMode=False, **CI_args):
        """# Searches the profile likelihood confidence interval for a given
        parameter."""
        
        self.prst("Investigating the profile likelihood")
        
        self.increase_print_level()
        
        dim = len(x0)
        
        result = np.zeros((dim, 2))
        
        labels = self.STATIC_PARAMETERS_LABELS + list(self.trafficFactorModel.LABELS[parametersConsidered[2:]])
        
        indices, directions = zip(*iterproduct(range(dim), (-1, 1)))
        
        if correctDimensionMode:
            x0 = np.array(x0)
            x0[2:] = self.trafficFactorModel.convert_parameters_correct_dimension(x0[2:], parametersConsidered[2:])
            print(self.negative_log_likelihood(x0, parametersConsidered, True, True))
            
        self.prst("Creating confidence intervals")
        with ProcessPoolExecutor() as pool:
            mapObj = pool.map(self._find_profile_CI, 
                              repeat(x0), repeat(parametersConsidered),
                              indices, directions, 
                              repeat({"correctDimensionMode":correctDimensionMode}),
                              repeat(CI_args))
            
            
            for index, direction, r in zip(indices, directions, mapObj):
                result[index][(0 if direction==-1 else 1)
                              ] = np.array(self.convert_parameters(r.x, 
                                        parametersConsidered))[parametersConsidered][index]
        
        self.prst("Printing confidence intervals and creating profile plots")
        self.increase_print_level()
        
        x0Orig = np.array(self.convert_parameters(x0, parametersConsidered))[parametersConsidered]
        
        for index, intv in enumerate(result):
            start, end = intv
            self.prst("CI for {:<40}: [{:10.4g} --- {:10.4g} --- {:10.4g}]".format(
                labels[index], start, x0Orig[index], end))
            
        self.decrease_print_level()
        self.decrease_print_level()
    
    def get_traffic_mean(self, parameters=None, parametersConsidered=None,
                         responseRate=None):
        raise NotImplementedError()
    
    def get_traffic_quantiles(self, p, parameters=None, parametersConsidered=None,
                             responseRate=None):
        raise NotImplementedError()
    
    def get_subbasin_probabilities(self):
        return self.trafficFactorModel.get_mean_factor(self.parameters[self.STATIC_PARAMETERS_SIZE:],
                                                       self.covariates[self.STATIC_PARAMETERS_SIZE:],
                                                       getAnglerNumbers=False, convertParameters=True)
    
    
    def read_locality_data(self, cityDataFileName):
        """Reads and saves data that can be used to determine the repulsiveness of cities in the angler traffic model.
        
        Parameters
        ----------
        cityDataFileName : str
            Name of a csv file with (ignored) header and columns separated by 
            ``,``.             
        """
        self.prst("Reading origin data file", cityDataFileName)
        dtype = [("localityId", IDTYPE)]
        for nameType in self._trafficFactorModel_class.ORIGIN_COVARIATES:
            if type(nameType) is not str and hasattr(nameType, "__iter__"): 
                if len(nameType) >= 2:
                    dtype.append(nameType[:2])
                else:
                    dtype.append((nameType[0], "double"))
            else:
                dtype.append((nameType, "double"))
        
        localityData = np.genfromtxt(cityDataFileName, delimiter=",", skip_header=True, 
                                 dtype=dtype) 
        localityData = self._trafficFactorModel_class.process_source_covariates(localityData)
        localityData.sort(order="localityId")
        self.localityData = localityData
        self.localityIdToLocalityIndex = {ID:i for i, ID in enumerate(localityData["localityId"])}
        self._erase_survey_data()
        self._erase_distance_data()
    
    
    def read_subbasin_data(self, subbasinDataFileName):
        """Reads and saves data that can be used to determine the attractiveness of subbasins.
        
        Parameters
        ----------
        subbasinDataFileName : str
            Name of a csv file with (ignored) header and columns separated by 
            ``,``. 
                    
        """
        self.prst("Reading destination data file", subbasinDataFileName)
        
        dtype = [("subbasinId", IDTYPE), ("infested", bool)]
        for nameType in self._trafficFactorModel_class.DESTINATION_COVARIATES:
            if type(nameType) is not str and hasattr(nameType, "__iter__"): 
                if len(nameType) >= 2:
                    dtype.append(nameType[:2])
                else:
                    dtype.append((nameType[0], "double"))
            else:
                dtype.append((nameType, "double"))
                
        
        subbasinData = np.genfromtxt(subbasinDataFileName, delimiter=",", skip_header = True, 
                                 dtype = dtype)
        
        subbasinData = self._trafficFactorModel_class.process_sink_covariates(subbasinData)
        subbasinData.sort(order="subbasinId")
        
        self.subbasinData = subbasinData
        self.subbasinIdToSubbasinIndex = {ID:i for i, ID in enumerate(subbasinData["subbasinId"])}
        
        #self._erase_survey_data()
        #self._erase_distance_data()
    
    def read_distance_data(self, distanceDataFileName):
        """Reads and saves data that can be used to determine the repulsiveness of cities in the angler traffic model.
        
        Parameters
        ----------
        distanceDataFileName : str
            Name of a csv file with (ignored) header and columns separated by 
            ``,``.             
        """
        self.prst("Reading distance data file", distanceDataFileName)
        dtype = [("localityId", IDTYPE), ("subbasinId", IDTYPE), ("distance", float)]
        
        distanceData = np.genfromtxt(distanceDataFileName, delimiter=",", skip_header=True, 
                                     dtype=dtype) 
        
        if (not hasattr(self, "localityIdToLocalityIndex") or 
                not hasattr(self, "subbasinIdToSubbasinIndex")):
            raise ValueError("Data for cities and subbasins must be available before "
                             "the distances between locality-subbasin pairs can be determined.")
        
        if not distanceData.size==self.subbasinData.size*self.localityData.size:
            raise ValueError("The number of locality-subbasin pairs must match the "
                             "product of the locality count and the subbasin "
                             "count.")
            
        distances = np.zeros((self.localityData.size, self.subbasinData.size))
        
        try:
            for i, j, dist in distanceData:
                distances[self.localityIdToLocalityIndex[i], self.subbasinIdToSubbasinIndex[j]] = dist
        except KeyError:
            raise ValueError(("The locality ID {} or the subbasin {} were not contained "
                              "in the locality or subbasin data set.".format(i, j)))
        
        self.distanceData = distances
        
        self._erase_model_fit()
        
        try:
            self.create_traffic_factor_model()
        except (ValueError, AttributeError):
            pass
        
    
    def read_angler_data(self, anglerDataFileName, dataFractionForFitting=1, 
                         dataFractionForValidation=None, seed=1):
        """Reads the survey observation data.
        
        Parameters
        ----------
        anglerDataFileName : str 
            Name of a csv file containing the road network. The file must be a 
            have a header (will be ignored) and columns separated by ``,``.
        dataFractionForFitting : float
            Fraction of the data that shall be used for model fitting
        dataFractionForValidation : float
            Fraction of the data that shall be used for model validation.
            If `None`, it is computed as `1-dataFractionForFitting`
        seed : int
            Seed of the random generator used when splitting the data
            into fitting and validation data
        
        """
        
        self.prst("Reading angler data", anglerDataFileName)
        self._erase_model_fit()
        
        dtype = self.ANGLER_DATA_DTYPE
        
        surveyData = np.genfromtxt(anglerDataFileName, delimiter=",",  
                                   skip_header = True, 
                                   dtype = dtype)
        
        if dataFractionForValidation is None:
            dataFractionForValidation = 1-dataFractionForFitting
        elif dataFractionForValidation+dataFractionForFitting > 1:
            raise ValueError("The sum of dataFractionForValidation and dataFractionForFitting must not exceed 1.")
        
        
        if dataFractionForFitting < 1:
            anglers, anglerIndices = np.unique(surveyData["anglerId"], 
                                               return_inverse=True)
            fitSize = int(round(dataFractionForFitting*anglers.size))
            validationSize = int(round(dataFractionForValidation*anglers.size))
            if validationSize + fitSize > anglers.size:
                validationSize -= 1
            
            anglersConsidered = np.zeros_like(anglers, dtype=int)
            anglersConsidered[:fitSize] = 1
            anglersConsidered[-validationSize:] = -1
                
            randomGenerator = np.random.default_rng(seed)
            randomGenerator.shuffle(anglersConsidered)
            
            self.validationData = surveyData[(anglersConsidered==-1)[anglerIndices]]
            surveyData = surveyData[(anglersConsidered==1)[anglerIndices]]
        else:
            self.validationData = None    
        
        
        self.dataFractionForFitting = dataFractionForFitting
        self.dataFractionForValidation = dataFractionForValidation
            
        self.surveyData = surveyData
    
    def add_place_indices(self):
        if "localityIndex" in self.surveyData.dtype.names:
            return
        
        result = []
        for data in self.surveyData, self.validationData:
            if data is None:
                result.append(None)
                continue
            localityIndices = [self.localityIdToLocalityIndex[i] for i in data["localityId"]]
            subbasinIndices = [self.subbasinIdToSubbasinIndex[i] for i in data["subbasinId"]]
            
            result.append(recfunctions.append_fields(data, 
                                                     ["localityIndex", "subbasinIndex"], 
                                                     [localityIndices, subbasinIndices], 
                                                     [int, int], usemask=False))
                                
        self.surveyData, self.validationData = result
    
    def set_traffic_factor_model_class(self, trafficFactorModel_class=None):
        """Sets the class representing the traffic factor (gravity) model.
        
        Parameters
        ----------
        trafficFactorModel_class : class
            Class of the traffic factor model. Must inherit from 
            :py:class:`BaseTrafficFactorModel`.
        
        """
        if trafficFactorModel_class is not None:
            try:
                if self._trafficFactorModel_class == trafficFactorModel_class:
                    return
            except AttributeError:        
                pass
            trafficFactorModel_class._check_integrity()
            self._trafficFactorModel_class = trafficFactorModel_class
        safe_delattr(self, "localityData")
        safe_delattr(self, "subbasinData")
        self._erase_traffic_factor_model()
    
    def create_traffic_factor_model(self):
        
        if (not hasattr(self, "_trafficFactorModel_class") or 
                not self.hasCountData):
            raise ValueError("The class of the traffic factor model "
                             "must be specified "
                             "and the count data must be computed.")
        
        self.trafficFactorModel = self._trafficFactorModel_class(self.localityData,
                                                                 self.subbasinData,
                                                                 self.distanceData)
        
        self._erase_model_fit()
    
    def plot_observed_predicted(self, saveFileName=None,
                                comparisonFileName=None):
        
        if saveFileName:
            if not os.path.isdir(saveFileName): 
                os.makedirs(saveFileName)
            saveFileName = os.path.join(saveFileName, saveFileName)
        
        if comparisonFileName:
            comparisonFileName = os.path.join(comparisonFileName, comparisonFileName)
        
        self.prst("Creating quality plots.")
        
        predicted = self.get_traffic_mean()
        err = self.get_traffic_quantiles(np.array([0.025, 0.975]), 
                                         responseRate=1)
        err[0] = predicted.ravel()-err[0]
        err[1] = err[1]-predicted.ravel()
        
        observed = self.get_traffic_observations()
        
        if not comparisonFileName and self.validationData is not None:
            comparisonObserved = self.get_traffic_observations(True)
            comparisonPredicted = predicted * (self.dataFractionForValidation/self.dataFractionForFitting)
            trivialPrediction = observed * (self.dataFractionForValidation/self.dataFractionForFitting)
            """
            mask = comparisonPredicted < comparisonPredicted.mean()
            comparisonObserved = np.ma.array(comparisonObserved, mask=mask)
            comparisonPredicted = np.ma.array(comparisonPredicted, mask=mask)
            trivialPrediction = np.ma.array(trivialPrediction, mask=mask)
            """
            
            modelMeanError = np.mean(np.abs(comparisonObserved-comparisonPredicted))
            comparisonMeanError = np.mean(np.abs(comparisonObserved-trivialPrediction))
            self.prst("Model mean error:", modelMeanError)
            self.prst("Trivial prediction mean error:", comparisonMeanError)
            self.prst("MASE error ratio:", modelMeanError/comparisonMeanError)
            
            modelMeanError = np.mean(np.abs(comparisonObserved.sum(1)-comparisonPredicted.sum(1)))
            comparisonMeanError = np.mean(np.abs(comparisonObserved.sum(1)-trivialPrediction.sum(1)))
            self.prst("Model mean error (origins):", modelMeanError)
            self.prst("Trivial prediction mean error (origins):", comparisonMeanError)
            self.prst("MASE error ratio (origins):", modelMeanError/comparisonMeanError)
            
            modelMeanError = np.mean(np.abs(comparisonObserved.sum(0)-comparisonPredicted.sum(0)))
            comparisonMeanError = np.mean(np.abs(comparisonObserved.sum(0)-trivialPrediction.sum(0)))
            self.prst("Model mean error (destinations):", modelMeanError)
            self.prst("Trivial prediction mean error (destinations):", comparisonMeanError)
            self.prst("MASE error ratio (destinations):", modelMeanError/comparisonMeanError)
            comparisonPredictedS1 = comparisonPredicted.sum(1)
            comparisonObservedS1 = comparisonObserved.sum(1)
            comparisonPredictedS0 = comparisonPredicted.sum(0)
            comparisonObservedS0 = comparisonObserved.sum(0)
            comparisonPredicted = comparisonPredicted.ravel()
            comparisonObserved = comparisonObserved.ravel()
            
        else:
            comparisonPredictedS1 = None
            comparisonObservedS1 = None
            comparisonPredictedS0 = None
            comparisonObservedS0 = None
            comparisonObserved = None
            comparisonPredicted = None
        
        create_observed_predicted_mean_error_plot(predicted.sum(1), 
                                                  observed.sum(1), 
                                                  saveFileName=_non_join(saveFileName, "_Cities"),
                                                  comparisonFileName=_non_join(comparisonFileName, "_Cities"),
                                                  comparisonPredicted = comparisonPredictedS1,
                                                  comparisonObserved = comparisonObservedS1,
                                                  )
        create_observed_predicted_mean_error_plot(predicted.sum(0), 
                                                  observed.sum(0), 
                                                  saveFileName=_non_join(saveFileName, "_subbasins"),
                                                  comparisonFileName=_non_join(comparisonFileName, "_subbasins"),
                                                  comparisonPredicted = comparisonPredictedS0,
                                                  comparisonObserved = comparisonObservedS0,
                                                  )
        
        create_observed_predicted_mean_error_plot(predicted.ravel(), 
                                                  observed.ravel(), 
                                                  np.abs(err),
                                                  saveFileName=_non_join(saveFileName, "_pairs"),
                                                  comparisonFileName=_non_join(comparisonFileName, "_pairs"),
                                                  comparisonPredicted = comparisonPredicted,
                                                  comparisonObserved = comparisonObserved,
                                                  )
        create_observed_predicted_mean_error_plot(predicted.ravel(), 
                                                  observed.ravel(), 
                                                  saveFileName=_non_join(saveFileName, "_pairs_raw"),
                                                  comparisonFileName=_non_join(comparisonFileName, "_pairs_raw"),
                                                  comparisonPredicted = comparisonPredicted,
                                                  comparisonObserved = comparisonObserved,
                                                  )
    
    def save_subbasin_predictions(self, fileName):
        
        dtype = [("subbasinId", IDTYPE), ("predicted", float), ("observed", float),
                 ("residual", float)]
        
        result = np.zeros_like(self.subbasinData, dtype=dtype)
        
        result["subbasinId"] = self.subbasinData["subbasinId"]
        
        result["predicted"] = self.get_traffic_mean().sum(0)
        result["observed"] = self.get_traffic_observations().sum(0)
        result["residual"] = result["predicted"]-result["observed"]
        
        df = pd.DataFrame(result)
        df.to_csv(fileName + "_subbasin.csv", index=False)
        
        
        
    
    
class YearlyLocalitySubbasinAnglerTrafficModel(BaseLocalitySubbasinAnglerTrafficModel):
    
    STATIC_PARAMETERS_BOUNDS = [
        (-10, 10),
        (-10, 10),
        ]
    STATIC_PARAMETERS_LABELS = [
        "Base factor",
        "Dispersion parameter",
        ]
    STATIC_PARAMETERS_SIZE = 2
    
    @property 
    def hasCountData(self):
        return hasattr(self, "pairCountData")
    
    def _convert_static_parameters(self, parameters):
        return [convert_R_pos(parameters[0]), convert_R_pos(parameters[1])]
    
    def negative_log_likelihood(self, parameters, parametersConsidered, convertParameters=True):
        
        if convertParameters:
            parameters = self.convert_parameters(parameters, parametersConsidered)
        
        alpha, c = parameters[:2]
        counts = self.pairCountData["count"]
        cities = self.pairCountData["localityIndex"]
        subbasins = self.pairCountData["subbasinIndex"]
        dataN = counts.size
        r = 1/alpha
        
        log = ag.log
        npsum = ag.sum
        
        meanFactor = self.trafficFactorModel.get_mean_factor(
            parameters[2:], parametersConsidered[2:])
        
        log_q_denominator = log(1+c*meanFactor)
        
        return (-npsum(gammaln(counts+r)) + dataN*gammaln(r) #+npsum(gammaln(counts+1)) 
                + r*npsum(log_q_denominator) - counts.sum()*log(c)
                - npsum(counts*log(meanFactor[cities, subbasins])) 
                + npsum(counts*log_q_denominator[cities, subbasins])
                ) 
        
    def get_r_q(self, parameters=None, parametersConsidered=None, 
                responseRate=None):
        if parameters is None or parametersConsidered is None:
            parameters = self.parameters
            parametersConsidered = self.covariates
        
        if responseRate is None:
            responseRate = self.responseRate
        
        parameters = self.convert_parameters(parameters, parametersConsidered)
        q = 1/(1+(parameters[1]/responseRate)
               * self.trafficFactorModel.get_mean_factor(parameters[self.STATIC_PARAMETERS_SIZE:], 
                                                         parametersConsidered[self.STATIC_PARAMETERS_SIZE:]))
        r = 1/parameters[0]
        
        return r, q
        
    def get_traffic_mean(self, parameters=None, parametersConsidered=None,
                         responseRate=None):
        r, q = self.get_r_q(parameters, parametersConsidered, responseRate)
        
        return r*(1-q)/q
    
    def get_traffic_observations(self):
        observed = np.zeros_like(self.distanceData)
        
        cities = self.pairCountData["localityIndex"]
        subbasins = self.pairCountData["subbasinIndex"]
        counts = self.pairCountData["count"]
        
        observed[cities, subbasins] = counts
        
        return observed
    
    def get_traffic_quantiles(self, p, parameters=None, parametersConsidered=None,
                             responseRate=None):
        r, q = self.get_r_q(parameters, parametersConsidered, responseRate)
        
        if type(p) == np.ndarray:
            q = q.ravel()
            return nbinom.ppf(p[:, None], r, q)
        return nbinom.ppf(p, r, q)
    
    def read_angler_data(self, anglerDataFileName):
        BaseLocalitySubbasinAnglerTrafficModel.read_angler_data(self, anglerDataFileName)
        self.set_pair_count_data()
    
    def set_pair_count_data(self):
        """Converts raw survey data in count data by locality-subbasin pair.
        
        """
        
        if (not hasattr(self, "localityIdToLocalityIndex") or 
                not hasattr(self, "subbasinIdToSubbasinIndex")):
            raise ValueError("Data for cities and subbasins must be available before "
                             "the count data per locality-subbasin pair can be determined")
        
        self.add_place_indices()
        
        pairs, count = np.unique(self.surveyData[["localityIndex", "subbasinIndex"]], 
                                 return_counts=True)
        
        pairCountData = recfunctions.append_fields(pairs, ["count"], [count], 
                                                   [int], usemask=False)
        
        if not count.all():
            pairCountData = pairCountData[pairCountData["count"]>0]
        
        self.pairCountData = pairCountData
        try:
            self.create_traffic_factor_model()
        except (ValueError, AttributeError):
            pass

class DailyLocalitySubbasinAnglerTrafficModel(BaseLocalitySubbasinAnglerTrafficModel):
    
    STATIC_PARAMETERS_BOUNDS = [
        (-5, 30),
        (-25, 10),
        ]
    STATIC_PARAMETERS_LABELS = [
        "Dispersion parameter",
        "Base factor",
        ]
    STATIC_PARAMETERS_SIZE = 2
    
    @property 
    def hasCountData(self):
        return hasattr(self, "pairDayCountData")
    
    def __init__(self, trafficFactorModel_class, timeFactorModel_class, timeInterval=(None, None), 
                 **printerArgs):
        BaseLocalitySubbasinAnglerTrafficModel.__init__(self, trafficFactorModel_class, **printerArgs)
        self.startDate, self.endDate = timeInterval
        self._timeFactorModel_class = timeFactorModel_class
    
       
    def _convert_static_parameters(self, parameters):
        return [convert_R_pos(parameters[0]), convert_R_pos(parameters[1])]
    
    def read_angler_data(self, anglerDataFileName, dataFractionForFitting=1, 
                         dataFractionForValidation=None):
        BaseLocalitySubbasinAnglerTrafficModel.read_angler_data(self, anglerDataFileName,
                                                       dataFractionForFitting,
                                                       dataFractionForValidation)
        self.set_pair_day_count_data()
        self.dayCount = self.endDate - self.startDate
    
    def set_pair_day_count_data(self):
        """Converts raw survey data in count data by locality-subbasin pair.
        
        """
        
        if (not hasattr(self, "localityIdToLocalityIndex") or 
                not hasattr(self, "subbasinIdToSubbasinIndex")):
            raise ValueError("Data for cities and subbasins must be available before "
                             "the count data per locality-subbasin pair can be determined")
        
        self.add_place_indices()
        tmp = []
        for data in self.surveyData, self.validationData:
            if data is None:
                tmp.append(None)
                continue
            
            pairDays, count = np.unique(data[["localityIndex", "subbasinIndex", "date"]], 
                                                         return_counts=True)
            
            pairDayCountData = np.zeros(pairDays.shape[0], dtype=[("localityIndex", int), 
                                                                  ("subbasinIndex", int),
                                                                  ("date", int),
                                                                  ("timeFactor", float),
                                                                  ("count", int)])
            pairDayCountData["localityIndex"] = pairDays["localityIndex"]
            pairDayCountData["subbasinIndex"] = pairDays["subbasinIndex"]
            pairDayCountData["date"] = pairDays["date"]
            pairDayCountData["count"] = count
            
            if not count.all():
                pairDayCountData = pairDayCountData[pairDayCountData["count"]>0]
            
            tmp.append(pairDayCountData)
        
        self.pairDayCountData, self.pairDayValidationData = tmp
        try:
            self.create_time_factor_model()
            self.create_traffic_factor_model()
        except (ValueError, AttributeError):
            pass
    
    def create_time_factor_model(self, startDate=None, endDate=None):
        
        timeFactorModel_class = self._timeFactorModel_class
        if startDate is None:
            startDate = self.startDate
        if endDate is None:
            endDate = self.endDate
        
        data = self.surveyData
        if startDate is None:
            startDate = np.min(data["date"])
        if endDate is None:
            endDate = np.max(data["date"])+1
        
        self.startDate = startDate
        self.endDate = endDate
        
        
        countData = []
        dates = np.arange(startDate, endDate)
        
        for date in dates:
            countData.append(np.sum(data["date"]==date))
                
        countData = np.asarray(countData)
        
        timeFactorModel = timeFactorModel_class(dates)
        self.timeModel = TimeModel(timeFactorModel, countData)
        
    def fit_time_model(self, permutations=None, variableDispersion=[False, True], **fittingArgs):
        self.timeModel.fit(permutations, variableDispersion, **fittingArgs)
        
        self.pairDayCountData["timeFactor"] = self.timeModel.get_time_factor(
                                                self.pairDayCountData["date"])
            
    
    def negative_log_likelihood(self, parameters, parametersConsidered, 
                                convertParameters=True, correctDimensionMode=False):
        
        if type(parameters) == np.ndarray and (parameters==0).all():
            DEBUG = True
            print("0 parametersConsidered", parametersConsidered)
            print("1 parameters", parameters)
        else:
            DEBUG = False
            
        if convertParameters:
            parameters = self.convert_parameters(parameters, parametersConsidered)
        
        if DEBUG:
            print("2 parameters", parameters)
        
        alpha, c = parameters[:2]
        counts = self.pairDayCountData["count"]
        cities = self.pairDayCountData["localityIndex"]
        subbasins = self.pairDayCountData["subbasinIndex"]
        timeFactors = self.pairDayCountData["timeFactor"]
        r = 1/alpha
        if r > 1e10:
            r = 1e10
        
        rt = r*timeFactors
        
        log = ag.log
        npsum = ag.sum
        
        meanFactor = self.trafficFactorModel.get_mean_factor(
            parameters[2:], parametersConsidered[2:], 
            correctDimensionMode=correctDimensionMode) #, DEBUG=DEBUG)
        
        log_q_denominator = log(1+c*meanFactor)
        if DEBUG:
            print("3 meanFactor.sum()", meanFactor.sum())
            print("4 -npsum(gammaln(counts+rt)) + npsum(gammaln(rt))", -npsum(gammaln(counts+rt)) + npsum(gammaln(rt)))
            print("5 r*self.dayCount*npsum(log_q_denominator) - counts.sum()*log(c)", r*self.dayCount*npsum(log_q_denominator) - counts.sum()*log(c))
            print("6 - npsum(counts*log(meanFactor[cities, subbasins]))", - npsum(counts*log(meanFactor[cities, subbasins])))
            print("7 npsum(counts*log_q_denominator[cities, subbasins])", npsum(counts*log_q_denominator[cities, subbasins]))
#         if type(meanFactor) == np.ndarray:
#             print(meanFactor.sum())
#             print(npsum(rt))
#             print(-npsum(gammaln(counts+rt)))
#             print(npsum(gammaln(rt)))
        
        result = (-npsum(gammaln(counts+rt)) + npsum(gammaln(rt)) # +npsum(gammaln(counts+1)) 
                + r*self.dayCount*npsum(log_q_denominator) - counts.sum()*log(c)
                - npsum(counts*log(meanFactor[cities, subbasins])) 
                + npsum(counts*log_q_denominator[cities, subbasins])
                ) 
        if type(result) == np.float64 and np.isnan(result):
            #print("!NAN_all", parametersConsidered+0, list(parameters))
            return np.inf
        """
        try:
            if not np.isfinite(result):
                return 1e100
        except:
            pass
        if type(result) == np.float64:
            print(result, list(parameters))
        """
        return result
    
    def negative_log_likelihood_independent(self, parameters, parametersConsidered, convertParameters=True):
        
        if convertParameters:
            parameters = self.convert_parameters(parameters, parametersConsidered)
        
        alpha, c = parameters[:2]
        counts = self.pairDayCountData["count"]
        cities = self.pairDayCountData["localityIndex"]
        subbasins = self.pairDayCountData["subbasinIndex"]
        timeFactors = self.pairDayCountData["timeFactor"]
        anglerNumbers = self.localityData["localityAnglers"]
        r = 1/alpha
        if r > 1e10:
            r = 1e10
        
        log = ag.log
        npsum = ag.sum
        
        meanFactor = self.trafficFactorModel.get_mean_factor(
            parameters[2:], parametersConsidered[2:], getProbabilities=False)
        meanFactor = meanFactor/(meanFactor/anglerNumbers).mean()
        
        
        probabilities = self.trafficFactorModel.get_mean_factor(
            parameters[2:], parametersConsidered[2:], getAnglerNumbers=False)
        
        rti = r*timeFactors*meanFactor[cities]
        
        log_q_denominator = log(1+c*probabilities)
        
        return (-npsum(gammaln(counts+rti)) + npsum(gammaln(rti)) #*dataN #+npsum(gammaln(counts+1)) 
                + r*self.dayCount*npsum(meanFactor*npsum(log_q_denominator, 1)) - counts.sum()*log(c)
                - npsum(counts*log(probabilities[cities, subbasins])) 
                + npsum(counts*log_q_denominator[cities, subbasins])
                ) 
        
    def get_r_q(self, parameters=None, parametersConsidered=None, dates=None):
        if parameters is None or parametersConsidered is None:
            parameters = self.parameters
            parametersConsidered = self.covariates
        
        parameters = self.convert_parameters(parameters, parametersConsidered)
        qFactor = self.trafficFactorModel.get_mean_factor(parameters[self.STATIC_PARAMETERS_SIZE:], 
                                                     parametersConsidered[self.STATIC_PARAMETERS_SIZE:])
            
            
        q = 1/(1+(parameters[1]*qFactor))
        
        if dates is not None:
            timeFactor = np.sum(self.timeModel.get_time_factor(dates))
        else:
            timeFactor = self.dayCount
        r = timeFactor/parameters[0]
        
        return r, q
        
    def get_traffic_mean(self, parameters=None, parametersConsidered=None):
        r, q = self.get_r_q(parameters, parametersConsidered)
        
        return r*(1-q)/q
    
    def get_traffic_observations(self, validation=False):
        observed = np.zeros_like(self.distanceData)
        
        if validation:
            data = self.pairDayValidationData
        else:
            data = self.pairDayCountData
        cities = data["localityIndex"]
        subbasins = data["subbasinIndex"]
        counts = data["count"]
        
        np.add.at(observed,(cities, subbasins),counts)
        
        return observed
    
    def get_traffic_quantiles(self, p, parameters=None, parametersConsidered=None,
                             responseRate=None):
        r, q = self.get_r_q(parameters, parametersConsidered)
        
        if type(p) == np.ndarray:
            q = q.ravel()
            return nbinom.ppf(p[:, None], r, q)
        return nbinom.ppf(p, r, q)
    
            
    def get_trip_rate_factor(self):
        """
        Returns mu_a * nu_a
        """
        parameters = self.convert_parameters(self.parameters, self.covariates)
        tripRateFactor = self.trafficFactorModel.get_mean_factor(self.parameters[self.STATIC_PARAMETERS_SIZE:],
                                                       self.covariates[self.STATIC_PARAMETERS_SIZE:],
                                                       getProbabilities=False, convertParameters=True
                                                       ) 
        
        
        anglerNumbers = self.localityData["localityAnglers"]
        
        tripRateFactor /= anglerNumbers
        
        return tripRateFactor * (parameters[1]/parameters[0])
       
    
        
        
class SubbasinSubbasinAnglerTrafficModel(HierarchichalPrinter, SeparatelySaveable):
    
    REGION_DATA_DTYPE = [("regionId", IDTYPE), ("subbasinId", IDTYPE)]
    subbasin_DISTANCE_DATA_DTYPE = [("from_subbasinId", IDTYPE), ("to_subbasinId", IDTYPE), ("distance", float)]
    
    PARAMETERS_BOUNDS = [
        (-20, 20),
        (-20, 20),
        (-20, 20),
        (-20, 20),
        (-20, 20),
        ]
    
    PARAMETERS_LABELS = [
        "P(same destination)",
        "P(choose in preferred area)",
        "P(report trip)",
        "alpha",
        "angler coverage",
        ]
    
    def __init__(self, fileName, **printerArgs):
        """Constructor"""
        HierarchichalPrinter.__init__(self, **printerArgs)
        SeparatelySaveable.__init__(self, extension=".amm")
        self.truncationNo = 200
        self.fileName = fileName
    
    def set_locality_subbasin_traffic_model(self, localitySubbasinTrafficModel):
        self.localitySubbasinTrafficModel = localitySubbasinTrafficModel
        self._erase_model_fit()
    
    def save(self, fileName=None):
        """Saves the model to the file ``fileName``.amm
        
        Parameters
        ----------
        fileName : str
            File name (without extension). If ``None``, the model's default 
            file name will be used.
        
        """
        if fileName is None:
            fileName = self.fileName
        if fileName is not None:
            self.prst("Saving the model as file", fileName+".amm")
            self.save_object(fileName)
    
        
    def _erase_model_fit(self):
        """Resets the gravity model to an unfitted state."""
        safe_delattr(self, "AIC")
        safe_delattr(self, "parameters")
    
    @property
    def isFitted(self):
        return hasattr(self, "parameters")
    
    def read_region_data(self, regionDataFileName):
        
        self.prst("Reading region data", regionDataFileName)
        self._erase_model_fit()
        
        dtype = self.REGION_DATA_DTYPE
        
        regionData_raw = np.genfromtxt(regionDataFileName, delimiter=",",  
                                       skip_header = True, 
                                       dtype = dtype)
        
        regions, regionIndices = np.unique(regionData_raw["regionId"], 
                                           return_inverse=True) #return_counts=True)
        regionIdToRegionIndex = {ID:i for i, ID in enumerate(regions)}
        subbasinIdToSubbasinIndex = self.localitySubbasinTrafficModel.subbasinIdToSubbasinIndex
        
        
        subbasinIndices = [subbasinIdToSubbasinIndex[i] for i in regionData_raw["subbasinId"]]
        regionToSubbasincsr = csr_matrix((subbasinIndices, (regionIndices, 
                                               _count_occurrence(regionIndices))))
        subbasinToRegioncsr = csr_matrix((regionIndices, (subbasinIndices, 
                                                  _count_occurrence(subbasinIndices))))
        
        self.regionIdToRegionIndex = regionIdToRegionIndex
        self.regionToSubbasin = ndarray_flex.from_csr_matrix(regionToSubbasincsr)
        self.subbasinToRegion = ndarray_flex.from_csr_matrix(subbasinToRegioncsr)
    
    def set_regions_by_radius(self, radius):
        
        self.prst("Setting preferred regions with radius ", radius)
        regionIdToRegionIndex = self.localitySubbasinTrafficModel.subbasinIdToSubbasinIndex
        
        regionToSubbasin = [np.nonzero(row <= radius)[0] for row 
                       in self.subbasin_subbasin_distances]
        subbasinToRegion = [np.nonzero(col <= radius)[0] for col
                       in self.subbasin_subbasin_distances.T]
        
        self.regionIdToRegionIndex = regionIdToRegionIndex
        self.regionToSubbasin = ndarray_flex.from_arr_list(regionToSubbasin)
        self.subbasinToRegion = ndarray_flex.from_arr_list(subbasinToRegion)
        self.regionRadius = radius
        
    def read_subbasin_distances(self, subbasinDistancesFileName):
        
        self.prst("Reading subbasin distance data", subbasinDistancesFileName)
        
        dtype = self.subbasin_DISTANCE_DATA_DTYPE
        
        distanceData_raw = np.genfromtxt(subbasinDistancesFileName, delimiter=",",  
                                         skip_header = True, 
                                         dtype = dtype)
        size = self.localitySubbasinTrafficModel.subbasinData.size
        distances = np.full((size, size), np.inf)
        np.fill_diagonal(distances, 0)
        subbasinIdToSubbasinIndex = self.localitySubbasinTrafficModel.subbasinIdToSubbasinIndex
        for fromId, toId, dist in distanceData_raw:
            distances[subbasinIdToSubbasinIndex[fromId], subbasinIdToSubbasinIndex[toId]] = dist
        
        self.subbasin_subbasin_distances = distances
        
    
    def _set_local_traffic_scale_and_fit(self, distance):
        self.set_regions_by_radius(distance)
        self.prepare_survey_data()
        return self.maximize_likelihood()
    
    def fit_region_radius(self, minDist, maxDist, steps=21, get_CI=True,
                                plotFileName=None):
        
        
        self.prst("Fitting the radius of the preferred regions.")
        self.increase_print_level()
        
        results = np.zeros(steps, dtype=[("radius", float),
                                         ("nLL", float),
                                         ("parameters", "5double"),
                                         ("parametersOriginal", "5double"),
                                         ("error", bool),
                                         ])
        results["radius"] = np.linspace(minDist, maxDist, steps)
        
        with ProcessPoolExecutor() as pool:
            mapObj = pool.map(self._set_local_traffic_scale_and_fit, 
                              results["radius"])
            
            for i, (radius, result) in enumerate(zip(results["radius"], mapObj)):
                self.prst("Radius {:10.4g} => -l = {:10.4f}".format(radius, result.fun))
                results[i] = (radius, result.fun, result.x, result.xOriginal, not result.success)
        
        self.local_scale_results = results
        
        bestResult = results[np.argmin(results["nLL"])]
        
        self.prst("Choose local traffic radius of {:10.4g}.".format(bestResult["radius"]))
        
        self.set_regions_by_radius(bestResult["radius"])
        self.prepare_survey_data()
        
        if get_CI: 
            self.investigate_profile_likelihood(bestResult["parameters"], disp=True)
        
        self.prst("Fitting complete. The results:")
        
        for radius, nLL, x, xOriginal, error in results:
            self.prst("Radius: {:8.2f} | -l: {:8.2f}{} | ".format(
                radius, nLL, "*" if error else " "), 
                " | ".join((str(xOriginal), str(list(x)))))
        
        if plotFileName is None:
            plotFileName = self.fileName
        
        self.parameters = bestResult["parameters"]
        self.plot_observed_predicted(plotFileName)
            
        return True
        
 
    def prepare_survey_data(self):
        
        self.localitySubbasinTrafficModel.surveyData.sort(order=["date"])
        self._erase_model_fit()
        self.localitySubbasinTrafficModel.add_place_indices()
        
        surveyData = self.localitySubbasinTrafficModel.surveyData
        
        _, uniqueAnglerIndices, anglerIndices, counts = np.unique(
                                                        surveyData["anglerId"], 
                                                        return_index=True,
                                                        return_counts=True,
                                                        return_inverse=True)
        
        self.anglerNumber = uniqueAnglerIndices.size
        
        indptr = _count_occurrence(anglerIndices)
        
        angler_tripIndices_csr = csr_matrix((np.arange(surveyData.size), 
                                             (anglerIndices, indptr)))
        
        self.angler_origins = surveyData["localityIndex"][uniqueAnglerIndices]
        self.angler_destinations = ndarray_flex.from_csr_matrix(csr_matrix((surveyData["subbasinIndex"], 
                                                   (anglerIndices, indptr))))
        
        timeSurveyData = surveyData["date"]
        timeModel = self.localitySubbasinTrafficModel.timeModel
        startDate = self.localitySubbasinTrafficModel.startDate
        shiftedDates = surveyData["date"] - startDate
        date_timeFactors = timeModel.get_time_factor(np.arange(startDate, 
                                                               self.localitySubbasinTrafficModel.endDate))
        
        Y_mult1 = defaultdict(list)
        Y_mult2 = defaultdict(list)
        
        L_convolve1 = defaultdict(list)
        L_convolve2 = defaultdict(list)
        L_convolve3= defaultdict(list)
        
        L_factors = {}
        Y_factors = {}
        
        indptr = [0]
        indices = []
        tripIndices = []
        dayTripCounts = []
        
        anglerRateFactor = self.localitySubbasinTrafficModel.get_trip_rate_factor()
        self.anglerRateFactor = anglerRateFactor[self.angler_origins]
        self.locality_anglerRateFactor = anglerRateFactor
        
        self.locality_unobservedAnglers = self.localitySubbasinTrafficModel.localityData["localityAnglers"]
        
        np.add.at(self.locality_unobservedAnglers, self.angler_origins, -1)
        
        for i, row in enumerate(angler_tripIndices_csr):
            tmpIndices, inv, counts = np.unique(timeSurveyData[row.data], return_index=True,
                                           return_inverse=True, return_counts=True)[1:]
            tripIndices.extend(row.data[tmpIndices])
            indices.extend(np.arange(tmpIndices.size))
            dayTripCounts.extend(counts)
            indptr.append(len(indices))
            
            
            if row.size <= 1:
                continue
            
            dayCounts = counts[inv]
            shiftedDatesRow = shiftedDates[row.data]
            
            dayFactors = date_timeFactors[shiftedDatesRow]
            localityFactor = anglerRateFactor[self.angler_origins[i]]
            
            dayLocalityKey = (dayFactors[0], localityFactor, dayCounts[0])
            Y_factors[dayLocalityKey] = 0
            L_factors[dayLocalityKey] = 0
            
                
            previousDay = shiftedDatesRow[0]
            for j, (day, dayCount, dayFactor) in enumerate(zip(shiftedDatesRow[1:], dayCounts[1:], dayFactors[1:]), 1):
                previousDayLocalityKey = dayLocalityKey
                previousDayFactor, _, previousDayCount = dayLocalityKey
                dayLocalityKey = (dayFactor, localityFactor, dayCount)
                if day == previousDay:
                    Y_mult1[dayFactor, dayCount, dayLocalityKey].append((i, j))
                    L_convolve1[dayLocalityKey].append((i, j))
                else:
                    betweenDateFactor = np.sum(date_timeFactors[previousDay+1:day])
                    Y_factors[dayLocalityKey] = 0
                    Y_mult2[dayFactor+previousDayFactor+betweenDateFactor, 
                            dayCount+previousDayCount, previousDayLocalityKey, dayLocalityKey].append((i, j))
                    L_factors[dayLocalityKey] = 0
                    if day - previousDay == 1:
                        L_convolve2[previousDayLocalityKey, dayLocalityKey].append((i, j))
                    else:
                        L_factors[betweenDateFactor, localityFactor] = 0
                        L_convolve3[previousDayLocalityKey, dayLocalityKey, betweenDateFactor].append((i, j))
                previousDay = day
        
        Y_factorsFrac = list(k for k in Y_factors.keys() if k[2]==1)
        Y_factorsBinom = list(k for k in Y_factors.keys() if k[2]>1)
        L_factorsBinom = list(k for k in L_factors.keys() if len(k)==2)
        L_factorsHypergeom = list(k for k in L_factors.keys() if len(k)>2)
        
        for i, key in enumerate(iterchain(Y_factorsFrac, Y_factorsBinom)):
            Y_factors[key] = i
        for i, key in enumerate(iterchain(L_factorsBinom, L_factorsHypergeom)):
            L_factors[key] = i
            
        self.angler_dayTripCounts = csr_matrix((dayTripCounts, indices, indptr))
        angler_trip_Lambda = ndarray_flex.from_csr_matrix(angler_tripIndices_csr) * 0 - 1
        angler_trip_Ypsilon = angler_trip_Lambda + 0
         
        for Y_index, tripIndexRow in enumerate(iterchain(Y_mult1.values(), Y_mult2.values())):
            for i, j in tripIndexRow:
                angler_trip_Ypsilon[i][j] = Y_index 
        
        self.Y_mult1 = np.array([(dayFactor, dayKey[1], dayCount, Y_factors[dayKey]) for 
                                 dayFactor, dayCount, dayKey in Y_mult1],
                                 dtype=[("timeFactor", float),
                                        ("localityFactor", float),
                                        ("tripCount", int),
                                        ("YFactorIndex", int)])
        self.Y_mult2 = np.array([(dayFactor, dayKey[1], dayCount, Y_factors[previousDayKey], Y_factors[dayKey])
                                 for dayFactor, dayCount, previousDayKey, dayKey 
                                 in Y_mult2],
                                 dtype=[("timeFactor", float),
                                        ("localityFactor", float),
                                        ("tripCount", int),
                                        ("YFactorIndex1", int),
                                        ("YFactorIndex2", int)])
        
        
        for L_index, tripIndexRow in enumerate(iterchain(L_convolve1.values(),
                                                        L_convolve2.values(),
                                                        L_convolve3.values())):
            for i, j in tripIndexRow:
                angler_trip_Lambda[i][j] = L_index 
            
        
        self.angler_trip_Lambda = angler_trip_Lambda
        self.angler_trip_Ypsilon = angler_trip_Ypsilon
        
        self.L_convolve1 = np.array([L_factors[dayKey] for dayKey in L_convolve1])
        self.L_convolve2 = np.array([(L_factors[previousDayKey], L_factors[dayKey])
                                     for previousDayKey, dayKey in L_convolve2])
        self.L_convolve3 = np.array([(L_factors[previousDayKey], L_factors[dayKey],
                                      L_factors[betweenDateFactor, dayKey[1]]) for 
                                      previousDayKey, dayKey, betweenDateFactor 
                                      in L_convolve3])
        
        self.Y_factorsFrac = np.array(Y_factorsFrac)[:, 0]
        self.Y_factorsBinom = np.array(Y_factorsBinom, 
                                       dtype=[("timeFactor", float),
                                              ("localityFactor", float),
                                              ("tripCount", int)])
        self.L_factorsBinom = np.array(L_factorsBinom,
                                       dtype=[("timeFactor", float),
                                              ("localityFactor", float)])
        self.L_factorsHypergeom = np.array(L_factorsHypergeom, 
                                           dtype=[("timeFactor", float),
                                                  ("localityFactor", float),
                                                  ("tripCount", int)])
        
        self.angler_dayFactor = csr_matrix((
            date_timeFactors[shiftedDates[tripIndices]], 
            indices, indptr)
        )
        
            
        subbasinProbabilities = self.localitySubbasinTrafficModel.get_subbasin_probabilities()
        rawRegionProbabilities = np.array([[probs[r].sum() for r in self.regionToSubbasin]
                                         for probs in subbasinProbabilities])
        regionProbabilities = rawRegionProbabilities/rawRegionProbabilities.sum(1)[:,None]
        
        self.angler_trip_kappa_0 = ndarray_flex.from_arr_list(
            [subbasinProbabilities[o, a] for o, a in 
             zip(self.angler_origins, self.angler_destinations)])
        
        self.angler_trip_iota_0 = ndarray_flex.from_arr_list(
            [np.insert(a[1:]==a[:-1], 0, 0) for a in self.angler_destinations])
        
        self.angler_regionCandidates = [list(set(iterchain(*self.subbasinToRegion[subbasins]))) 
                                        for subbasins in self.angler_destinations]
        
        anglerNumber = self.angler_origins.size
        angler_noRegionCandidateProbabilities = np.zeros(anglerNumber)
        angler_regionCandidate_Probabilities = []
        angler_kappa1 = []
        
        for i, (origin, subbasins, regions) in enumerate(zip(self.angler_origins,
                        self.angler_destinations, self.angler_regionCandidates)):
            angler_regionCandidate_Probabilities.append(regionProbabilities[origin, regions])
            angler_noRegionCandidateProbabilities[i] = 1-regionProbabilities[origin, regions].sum()
            kappa1 = np.array([np.in1d(subbasins, regionSubbasins) for 
                               regionSubbasins in self.regionToSubbasin[regions]], float)
            kappa1 /= rawRegionProbabilities[origin, regions][:, None]
            angler_kappa1.append(kappa1)
        
        self.angler_trip_kappa1 = ndarray_flex.from_arr_list(angler_kappa1)
        self.angler_regionCandidate_Probabilities = ndarray_flex.from_arr_list(angler_regionCandidate_Probabilities)
        self.angler_noRegionCandidateProbabilities = angler_noRegionCandidateProbabilities
        
        self.daysWithObservationCount = self.angler_dayTripCounts.data.size
        self.tripCounts = self.angler_dayTripCounts.sum(1)
        
    def negative_log_likelihood(self, parameters, convertParameters=True):
        if convertParameters:
            parameters = self.convert_parameters(parameters)
        xi_0, xi_R, nu_a, alpha, coverage = parameters
        xi_A = 1 - xi_R
        """
        power = ag.power
        log = ag.log
        npsum = ag.sum
        agbinom = binomial_coefficient
        #convolve = partial(ags.signal.convolve, axes=1)
        convolve = lambda a, b: [ag.convolve(an, bn) for an, bn in zip(a, b)]
        concatenate = ag.concatenate
        """
        power = np.power
        log = np.log
        npsum = np.sum
        convolve = convolve_positive
        concatenate = np.concatenate
        agbinom = binom
        #"""
        #nu_a /= 10
        mu_a_nu_a = self.anglerRateFactor / coverage
        locality_mu_a_nu_a = self.locality_anglerRateFactor / coverage
        qFact = alpha/(nu_a * coverage)
        
        nDay = self.localitySubbasinTrafficModel.dayCount
        truncationNo = self.truncationNo
        angler_dayFactor = self.angler_dayFactor.data / alpha
        tripCounts = self.tripCounts
        anglerNumber = mu_a_nu_a.size
        
        """
        Y_factorsFrac = alpha / self.Y_factorsFrac
        Y_factorsBinom_r = self.Y_factorsBinom["timeFactor"] / alpha - 1
        Y_factorsBinom = 1/agbinom(self.Y_factorsBinom["tripCount"]
                                 + Y_factorsBinom_r, Y_factorsBinom_r)
        Y_factors = concatenate((Y_factorsFrac, Y_factorsBinom))
        
        q1 = 1 / (1 + qFact*self.Y_mult1["localityFactor"])
        eta1 = (1-q1) * (1-nu_a)
        Y_mult1 = (power(1-eta1, self.Y_mult1["timeFactor"] / alpha 
                        + self.Y_mult1["tripCount"]) 
                   * Y_factors[self.Y_mult1["YFactorIndex"]])
        
        q2 = 1 / (1 + qFact*self.Y_mult2["localityFactor"])
        eta2 = (1-q2) * (1-nu_a)
        Y_mult2 = (power(1-eta2, self.Y_mult2["timeFactor"] / alpha 
                         + self.Y_mult2["tripCount"]) 
                   * Y_factors[self.Y_mult2["YFactorIndex1"]] 
                   * Y_factors[self.Y_mult2["YFactorIndex2"]])
        Ypsilons = concatenate((Y_mult1, Y_mult2))
        """
        
        k = np.arange(truncationNo)[:,None]
        
        r = self.L_factorsBinom["timeFactor"] / alpha - 1
        # q3 = 1 / (1 + qFact*self.L_factorsBinom["localityFactor"])
        q3_part = qFact*self.L_factorsBinom["localityFactor"]
        # eta3 = (1-q3) * (1-nu_a)
        eta3 = (q3_part/(1+q3_part)) * (1-nu_a)
        L_Binom = agbinom(k+r, r) * power(eta3, k)
        
        r = self.L_factorsHypergeom["timeFactor"] / alpha
        tripCountHypergeom = self.L_factorsHypergeom["tripCount"]
        
        # q4 = 1 / (1 + qFact*self.L_factorsHypergeom["localityFactor"])
        q4_part = qFact*self.L_factorsHypergeom["localityFactor"]
        # eta4 = (1-q4) * (1-nu_a)
        eta4 = (q4_part/(1+q4_part)) * (1-nu_a)
        L_Hypergeom = (agbinom((tripCountHypergeom+r-1)+k, r-1) 
                       * hyp2f1(tripCountHypergeom, k+(tripCountHypergeom+r), 
                                k+(tripCountHypergeom+1), eta4)) * power(eta4, k)
        L_dists = concatenate((L_Binom.T, L_Hypergeom.T)) 
        
        L_convolve1 = L_dists[self.L_convolve1]
        L_convolve2 = convolve(L_dists[self.L_convolve2[:,0]], L_dists[self.L_convolve2[:,1]])
        L_convolve3 = convolve(convolve(L_dists[self.L_convolve3[:,0]],
                                        L_dists[self.L_convolve3[:,1]]),
                               L_dists[self.L_convolve3[:,2]])
        
        kPower1 = power(xi_0, np.arange(1, L_convolve1[0].size+1))
        kPower2 = power(xi_0, np.arange(1, L_convolve2[0].size+1))
        kPower3 = power(xi_0, np.arange(1, L_convolve3[0].size+1))
        
        Lambdas = concatenate((npsum(L_convolve1/L_convolve1.sum(1)[:,None] * kPower1, 1), 
                               npsum(L_convolve2/L_convolve2.sum(1)[:,None] * kPower2, 1),
                               npsum(L_convolve3/L_convolve3.sum(1)[:,None] * kPower3, 1),
                               [0])) 
        
        kappa = (self.angler_trip_kappa_0 * (xi_A + self.angler_trip_kappa1 * xi_R)) 
        kappa0 = (self.angler_trip_kappa_0 * xi_A) 
        
        iota = self.angler_trip_iota_0 - kappa
        iota0 = self.angler_trip_iota_0 - kappa0
        
        # combine the results and return
        YpsilonLambda = self.angler_trip_Lambda.pick(Lambdas) #* self.angler_trip_Ypsilon.pick(Ypsilons)
        
        summand1 = ((kappa+iota*YpsilonLambda).prod(2) * self.angler_regionCandidate_Probabilities).sum(1)
        summand2 = (kappa0+iota0*YpsilonLambda).prod(1) * self.angler_noRegionCandidateProbabilities
        
        # qq = 1 / (1 + alpha*mu_a_nu_a)
        qq_inverse = (1 + alpha*mu_a_nu_a)
        locality_qq_inverse = (1 + alpha*locality_mu_a_nu_a)
        result = (
             - npsum(self.locality_unobservedAnglers 
                    * log(1 + coverage * (locality_qq_inverse**(-nDay/alpha) - 1)))
             - np.log(coverage)*anglerNumber
             + npsum(log(qq_inverse)) * nDay/alpha
            #npsum(log(power(qq_inverse, nDay/alpha)-1)) 
              - npsum(gammaln(self.angler_dayTripCounts.data+angler_dayFactor))
              + npsum(gammaln(self.angler_dayTripCounts.data))
              + npsum(gammaln(angler_dayFactor)) 
              - npsum((log(alpha) + log(mu_a_nu_a) - log(qq_inverse)) * tripCounts)
              - npsum(log(summand1 + summand2))
              )
        """
        (-npsum(gammaln(counts+rt)) + npsum(gammaln(rt)) #+npsum(gammaln(counts+1)) 
                + r*self.dayCount*npsum(log_q_denominator) - counts.sum()*log(c)
                - npsum(counts*log(meanFactor[cities, subbasins])) 
                + npsum(counts*log_q_denominator[cities, subbasins])
                ) 
        """
        if np.isnan(result):
            return np.inf
        return result
    
    def negative_log_likelihood_independent(self, parameters, convertParameters=True):
        if convertParameters:
            parameters = self.convert_parameters(parameters)
        xi_0, xi_R, nu_a, alpha, coverage = parameters
        xi_A = 1 - xi_R
        
        power = np.power
        log = np.log
        npsum = np.sum
        convolve = convolve_positive
        concatenate = np.concatenate
        agbinom = binom
        mu_a_nu_a = self.anglerRateFactor / coverage
        qFact = alpha/(nu_a * coverage)
        
        nDay = self.localitySubbasinTrafficModel.endDate - self.localitySubbasinTrafficModel.startDate
        truncationNo = self.truncationNo
        angler_dayFactor = self.angler_dayFactor.data / alpha
        anglerNumber = angler_dayFactor.shape[0]
        tripCounts = self.tripCounts
        
        k = np.arange(truncationNo)[:,None]
        
        r = self.L_factorsBinom["timeFactor"] / alpha - 1
        # q3 = 1 / (1 + qFact*self.L_factorsBinom["localityFactor"])
        q3_part = qFact*self.L_factorsBinom["localityFactor"]
        # eta3 = (1-q3) * (1-nu_a)
        eta3 = (q3_part/(1+q3_part)) * (1-nu_a)
        L_Binom = agbinom(k+r, r) * power(eta3, k)
        
        r = self.L_factorsHypergeom["timeFactor"] / alpha
        tripCountHypergeom = self.L_factorsHypergeom["tripCount"]
        
        # q4 = 1 / (1 + qFact*self.L_factorsHypergeom["localityFactor"])
        q4_part = qFact*self.L_factorsHypergeom["localityFactor"]
        # eta4 = (1-q4) * (1-nu_a)
        eta4 = (q4_part/(1+q4_part)) * (1-nu_a)
        L_Hypergeom = (agbinom((tripCountHypergeom+r-1)+k, r-1) 
                       * hyp2f1(tripCountHypergeom, k+(tripCountHypergeom+r), 
                                k+(tripCountHypergeom+1), eta4)) * power(eta4, k)
        L_dists = concatenate((L_Binom.T, L_Hypergeom.T)) 
        
        L_convolve1 = L_dists[self.L_convolve1]
        L_convolve2 = convolve(L_dists[self.L_convolve2[:,0]], L_dists[self.L_convolve2[:,1]])
        L_convolve3 = convolve(convolve(L_dists[self.L_convolve3[:,0]],
                                        L_dists[self.L_convolve3[:,1]]),
                               L_dists[self.L_convolve3[:,2]])
        
        kPower1 = power(xi_0, np.arange(1, L_convolve1[0].size+1))
        kPower2 = power(xi_0, np.arange(1, L_convolve2[0].size+1))
        kPower3 = power(xi_0, np.arange(1, L_convolve3[0].size+1))
        
        Lambdas = concatenate((npsum(L_convolve1/L_convolve1.sum(1)[:,None] * kPower1, 1), 
                               npsum(L_convolve2/L_convolve2.sum(1)[:,None] * kPower2, 1),
                               npsum(L_convolve3/L_convolve3.sum(1)[:,None] * kPower3, 1),
                               [0])) 
        
        kappa = (self.angler_trip_kappa_0 * (xi_A + self.angler_trip_kappa1 * xi_R)) 
        kappa0 = (self.angler_trip_kappa_0 * xi_A) 
        
        iota = self.angler_trip_iota_0 - kappa
        iota0 = self.angler_trip_iota_0 - kappa0
        
        # combine the results and return
        YpsilonLambda = self.angler_trip_Lambda.pick(Lambdas) #* self.angler_trip_Ypsilon.pick(Ypsilons)
        
        summand1 = ((kappa+iota*YpsilonLambda).prod(2) * self.angler_regionCandidate_Probabilities).sum(1)
        summand2 = (kappa0+iota0*YpsilonLambda).prod(1) * self.angler_noRegionCandidateProbabilities
        
        # qq = 1 / (1 + alpha*mu_a_nu_a)
        qq_inverse = (1 + alpha*mu_a_nu_a)
        
        result = (anglerNumber * log(power(qq_inverse, nDay/alpha)-1)
              - npsum(gammaln(self.angler_dayTripCounts.data+angler_dayFactor))
              + npsum(gammaln(angler_dayFactor)) 
              - npsum((log(alpha) + log(mu_a_nu_a) - log(qq_inverse)) * tripCounts)
              - npsum(log(summand1 + summand2))
              )
        
        if np.isnan(result):
            return np.inf
        return result
        
    def convert_parameters(self, parameters, parametersConsidered=None):
        """#
        Converts an array of given parameters to an array of standard (maximal)
        length and in the parameter domain of the model
        
        See `BaseTrafficFactorModel.convert_parameters`
        """
        
        return (convert_R_0_1(parameters[0]),           # xi_0
                convert_R_0_1(parameters[1]),           # xi_R
                convert_R_0_1(parameters[2]),           # nu_a
                convert_R_pos(parameters[3]),           # alpha
                convert_R_pos(parameters[4]),           # mu_a
                )
        return (0,           # xi_0
                0,           # xi_R
                convert_R_0_1(parameters[2]),           # nu_a
                convert_R_pos(parameters[3]),           # alpha
                convert_R_pos(parameters[4]),           # mu_a
                )
        
    def get_nLL_functions(self):
        fun = self.negative_log_likelihood
        grad = nd.Gradient(fun) #, num_steps=10)
        hess = nd.Hessian(fun) #, num_steps=10)
        return fun, grad, hess
        
     
    def fit(self, refit=False, parameters=None, 
            continueOptimization=False, get_CI=True, plotFileName=None):
        """Fits the subbasin to subbasin traffic model.
        
        Parameters
        ----------
        refit : bool
            Whether to repeat the fitting procedure if the model has been 
            fitted earlier.
        parameters : dict
            Dictionary with the keys ``"parametersConsidered"`` and ``"parameters"`` 
            that provides an initial guess for the optimization or the 
            corresponding solution. ``"parametersConsidered"`` contains a `bool[]` with 
            the considered parameter combination (see :py:obj:`permutations`);
            ``"parameters"`` contains a `float[]` with the values for the 
            parameters where ``flowParameters["parametersConsidered"]`` is ``True``.
        continueOptimization : bool
            If ``True``, the :py:obj:`flowParameters` will be used as initial 
            guess. Otherwise, they will be considered as the optimal 
            parameters.
        get_CI : bool
            Whether confidence intervals shall be computed after the model
            has been fitted. Note that no confidence intervals will be computed,
            if ``continueFlowOptimization is False``.
        
        """
        self.prst("Fitting the subbasin to subbasin model.")
        self.increase_print_level()
        
        fittedModel = False
        if not refit and self.isFitted:
            self.prst("A model does already exist. I skip",
                      "this step. Enforce fitting with the argument",
                      "refit=True")
            return False
        
        if not hasattr(self, "anglerRateFactor"):
            self.prepare_survey_data()
        
        if parameters is None:
            result = self.maximize_likelihood()
            fittedModel = True
        else:
            if continueOptimization:
                result = self.maximize_likelihood(parameters)
                fittedModel = True
            else:
                result = self.maximize_likelihood(parameters, False)
        
        self.prst(result) 
        
        self.decrease_print_level()
        if fittedModel and get_CI: 
            self.investigate_profile_likelihood(result.x, disp=True)
        
        if plotFileName is None:
            plotFileName = self.fileName
        
        if (not self.isFitted or 
                (parameters is not None and not continueOptimization)):
            self.parameters = result.x
            self.plot_observed_predicted(plotFileName)
            
        return fittedModel
    
    
    def maximize_likelihood(self, parameters=None, continueOptimization=True):
        fun, jac, hess = self.get_nLL_functions()
        if not continueOptimization and parameters is not None:
            result = op.OptimizeResult(x=parameters, success=True, status=0,
                                       fun=self.negative_log_likelihood(parameters), 
                                       nfev=1, njev=1,
                                       nhev=0, nit=0,
                                       message="parameters checked")
            result.xOriginal = self.convert_parameters(result.x)
            #result.jacOriginal = jac(result.xOriginal)
            return result
        
        
        if parameters is None:
            bounds = self.PARAMETERS_BOUNDS
            np.random.seed()
            result = op.differential_evolution(fun, bounds, 
                                               popsize=30, maxiter=30, #300, 
                                               disp=True) #, workers=min(os.cpu_count(), 30))
            self.prst("Differential evolution result:", result)
            parameters = result.x.copy()
            result.xOriginal = self.convert_parameters(result.x)
            #result.jacOriginal = jac(result.xOriginal)
            
        result2 = op.minimize(fun, parameters, method="L-BFGS-B",
                              jac=jac, hess=hess,
                              bounds=None, options={"maxiter":1000,
                                                    "iprint":2})
        
        result2.xOriginal = self.convert_parameters(result2.x)
        #result2.jacOriginal = jac(result2.xOriginal)
        self.prst("L-BFGS-B result:", result)
        
        x0 = result2.x.copy()
        result = result2
        
        result2 = op.minimize(fun, x0, jac=jac, 
                              hess=hess, bounds=None, 
                              options={"maxiter":1000, 
                                       "iprint":2},
                              method="SLSQP")
        result2.xOriginal = self.convert_parameters(result2.x)
        #result2.jacOriginal = jac(result2.xOriginal)
        self.prst("SLSQP result:", result)
        if result2.fun < result.fun:
            parameters = result2.x.copy()
            result = result2
        
        result2 = op.minimize(fun, result.x, jac=jac, 
                              hess=hess, bounds=None, 
                              method="trust-exact",
                              options={"maxiter":1000, 
                                       "disp":True})
        result2.xOriginal = self.convert_parameters(result2.x)
        #result2.jacOriginal = jac(result2.xOriginal)
        self.prst("Trust-exact result:", result)
        print("trust-exact", result2)         
        if result2.fun < result.fun:
            result = result2
            
        return result
        
        
    def _find_profile_CI(self, x0, index, direction, profile_LL_args={}):
        """Searches the profile likelihood confidence interval for a given
        parameter.
        
        Parameters
        ----------
        profile_LL_args : dict
            Keyword arguments to be passed to :py:meth:`find_CI_bound`.
        
        """
        
        fun, jac, hess = self.get_nLL_functions()
        
        fun_ = lambda x: -fun(x)   
        jac_ = lambda x: -jac(x)   
        hess_ = lambda x: -hess(x) 
        
        return find_CI_bound(x0, fun_, index, direction, jac_, hess_, 
                                     **profile_LL_args)
    
    def _find_result_profile_CI(self, direction, fromToIndex, 
                                sourcesConsidered=None,
                                relativeError=1e-4, profile_LL_args={}):
        """Searches the profile likelihood confidence interval for a given
        result.
        
        Parameters
        ----------
        profile_LL_args : dict
            Keyword arguments to be passed to :py:meth:`find_CI_bound`.
        
        """
        
        fun_, _, _ = self.get_nLL_functions()
        
        (rateFactor, subbasinProbabilities, 
         regionNormalization, regionCounts, sharedRegionProbabilities
         ) = self._computeMeanData
        
        if hasattr(fromToIndex, '__iter__'):
            if len(fromToIndex) > 1: 
                fromIndex, toIndex = fromToIndex[:2]
            else:
                fromIndex, toIndex = fromToIndex[0], None
        else:
            fromIndex, toIndex = fromToIndex, None
        
        
        pairMode = toIndex is not None
        if pairMode:
            sharedRegionProbabilities = sharedRegionProbabilities[:, fromIndex, toIndex]
            sameDestination = (fromIndex == toIndex) + 0
            regionCounts = regionCounts[fromIndex]+regionCounts[toIndex]
        else:
            sharedRegionProbabilities = sharedRegionProbabilities[:, fromIndex]
            regionCounts = regionCounts[fromIndex]+regionCounts[sourcesConsidered]
            if sourcesConsidered is not None:
                subbasinProbabilities2 = subbasinProbabilities[:,sourcesConsidered]
                sharedRegionProbabilities = sharedRegionProbabilities[:, sourcesConsidered]
            else:
                subbasinProbabilities2 = subbasinProbabilities
            
        if not sharedRegionProbabilities.any():
            sharedRegionProbabilities = 0
        
        def result_fun(parameters):
            
            xi_0, xi_R, nu_a, _, _ = self.convert_parameters(parameters)
            xi_A = 1 - xi_R
            
            if pairMode:
                return ((rateFactor * subbasinProbabilities[:,fromIndex]) * 
                          (subbasinProbabilities[:,toIndex] * (1-xi_0) 
                           * (xi_A**2 + regionNormalization*(
                               xi_A*xi_R*regionCounts
                               + xi_R**2 * sharedRegionProbabilities)
                               ) + xi_0 * sameDestination)
                          ).sum() / nu_a
            else:
                return ((rateFactor * subbasinProbabilities[:,fromIndex])[:,None] * 
                          (subbasinProbabilities2 * (1-xi_0) 
                           * (xi_A**2 + regionNormalization[:,None]*(
                               xi_A*xi_R*regionCounts
                               + xi_R**2 * sharedRegionProbabilities)))
                          ).sum() / nu_a
        
        x0 = self.parameters              
        error = result_fun(x0) * relativeError
        x0 = np.insert(x0, 0, result_fun(x0))
        
        fun_2 = lambda x: 0.5 * ((result_fun(x[1:])-x[0])/error)**2
        fun = lambda x: -fun_(x[1:]) - fun_2(x)
        
        jac_ = nd.Gradient(fun_) #, base_step=base_step)
        jac_2 = grad(fun_2)
        def jac(x):
            result = - jac_2(x)
            result[1:] -= jac_(x[1:]) 
            return result
        
        hess_ = nd.Hessian(fun_)
        hess_2 = hessian(fun_2)
        def hess(x):
            result = -hess_2(x)
            result[1:, 1:] -= hess_(x[1:])
            return result
        return find_CI_bound(x0, fun, 0, direction, jac, hess, 
                                     **profile_LL_args)
    
    
    def get_extreme_result_CIs(self, dates, relativeError=1e-4, **profile_LL_args):
        
        
        self.prst("Creating confidence intervals for some extreme results")
        
        self.increase_print_level()
        
        m = self.get_traffic_mean(dates=dates)
        
        np.fill_diagonal(m, 0)
        
        pairIndex = np.nonzero(m==m.max())
        pairIndex = pairIndex[0][0], pairIndex[1][0]
        
        sources = self.localitySubbasinTrafficModel.subbasinData['infested']
        sourcesConsidered = [sources, sources, None, None]
        risk = m[sources].sum(0) * (~sources)
        riskIndex = np.argmax(risk)
        
        
        estimate = [risk[riskIndex], m[pairIndex]]
        
        labels = ["Maximal traffic pair", "Maximal risk subbasin"]
        dim = len(labels)
        
        result = np.zeros((dim, 2))
        
        indices, directions = zip(*iterproduct(range(dim), (-1, 1)))
        
        fromToIndex = [riskIndex, riskIndex, pairIndex, pairIndex]
        self.truncationNo = 150
        self.prst("Creating confidence intervals")
        with ProcessPoolExecutor() as pool:
            mapObj = pool.map(self._find_result_profile_CI, 
                              directions, fromToIndex, sourcesConsidered, 
                              repeat(relativeError), repeat(profile_LL_args))
            
            
            for index, direction, r in zip(indices, directions, mapObj):
                result[index][(0 if direction==-1 else 1)] = r.x[0]
        
        self.prst("Printing confidence intervals and creating profile plots")
        self.increase_print_level()
        
        for index, intv in enumerate(result):
            start, end = intv
            self.prst("CI for {:<40}: [{:10.4g} --- {:10.4g} --- {:10.4g}]".format(
                labels[index], start, estimate[index], end))
            
        self.decrease_print_level()
        self.decrease_print_level()
        
    
    def investigate_profile_likelihood(self, x0, **profile_LL_args):
        """# Searches the profile likelihood confidence interval for a given
        parameter."""
        
        self.prst("Investigating the profile likelihood")
        
        self.increase_print_level()
        
        dim = len(x0)
        
        result = np.zeros((dim, 2))
        
        labels = self.PARAMETERS_LABELS
        
        indices, directions = zip(*iterproduct(range(dim), (-1, 1)))
        
        self.prst("Creating confidence intervals")
        with ProcessPoolExecutor() as pool:
            mapObj = pool.map(self._find_profile_CI, repeat(x0),
                              indices, directions, repeat(profile_LL_args))
            
            
            for index, direction, r in zip(indices, directions, mapObj):
                result[index][(0 if direction==-1 else 1)
                              ] = np.array(self.convert_parameters(r.x))[index]
        
        self.prst("Printing confidence intervals and creating profile plots")
        self.increase_print_level()
        
        x0Orig = np.array(self.convert_parameters(x0))
        
        for index, intv in enumerate(result):
            start, end = intv
            self.prst("CI for {:<40}: [{:10.4g} --- {:10.4g} --- {:10.4g}]".format(
                labels[index], start, x0Orig[index], end))
            
        self.decrease_print_level()
        self.decrease_print_level()
    
    
    def get_mean_days_out_with_CI(self, dates=None, relativeError=1e-4, nmax=400,
                                  disp=True, profile_LL_args={}):
        
        if dates is None:
            dates = np.arange(self.localitySubbasinTrafficModel.startDate, self.localitySubbasinTrafficModel.endDate)
            
        anglerNumbers = self.localitySubbasinTrafficModel.localityData["localityAnglers"]
        timeFactor = self.localitySubbasinTrafficModel.timeModel.get_time_factor(dates)
        rateFactor = self.localitySubbasinTrafficModel.get_trip_rate_factor()
        allAnglers = anglerNumbers.sum()
        
        def result_fun(parameters):
            nu_a, alpha, coverage = self.convert_parameters(parameters)[2:]
            return ((1-ag.power(1+(alpha/(nu_a*coverage))*rateFactor, -timeFactor[:,None]/alpha)).sum(0)
                    *anglerNumbers).sum() / allAnglers
        
        fun_, _, _ = self.get_nLL_functions()
        
        
        x0 = self.parameters
        result = result_fun(x0)
        error = result * relativeError
        x0 = np.insert(x0, 0, result)
        
        fun_2 = lambda x: 0.5 * ((result_fun(x[1:])-x[0])/error)**2
        fun = lambda x: -fun_(x[1:]) - fun_2(x)
        
        jac_ = nd.Gradient(fun_) 
        jac_2 = grad(fun_2)
        
        def jac(x):
            result = - jac_2(x)
            result[1:] -= jac_(x[1:]) 
            return result
        
        fun_2(x0+np.eye(1,6,1).ravel()*1e-5)
        hess_ = nd.Hessian(fun_)
        hess_2 = hessian(fun_2)
        
        def hess(x):
            result = -hess_2(x)
            result[1:, 1:] -= hess_(x[1:])
            return result
        
        print("Finding lower CI")
        lowerCI = find_CI_bound(x0, fun,  0, -1, jac, hess,nmax=nmax,
                                disp=disp, **profile_LL_args).x[0]
        print("lowerCI", lowerCI)
        print("Finding upper CI")
        upperCI = find_CI_bound(x0, fun, 0, 1, jac, hess, nmax=nmax,
                                disp=disp, **profile_LL_args).x[0]
        print("upperCI", upperCI)
        
        print("Mean days out: [{:8.2f}, {:8.2f}, {:8.2f}]".format(lowerCI,
                                                                  result,
                                                                  upperCI))
        
        
    def get_traffic_mean(self, parameters=None, dates=None, 
                         getReportedOnly=False, getByLocality=False):
        
        if parameters is None:
            parameters = self.parameters
        
        xi_0, xi_R, nu_a, alpha, coverage = self.convert_parameters(parameters)
        xi_A = 1 - xi_R
        
        if dates is None:
            dates = np.arange(self.localitySubbasinTrafficModel.startDate, self.localitySubbasinTrafficModel.endDate)
            
        anglerNumbers = self.localitySubbasinTrafficModel.localityData["localityAnglers"]
        timeFactor = np.sum(self.localitySubbasinTrafficModel.timeModel.get_time_factor(dates))
        
        rateFactor = self.localitySubbasinTrafficModel.get_trip_rate_factor()/coverage # = nu_a*mu_a
        
        self._computeMeanData = [rateFactor*timeFactor*anglerNumbers]
        
        if not getReportedOnly:
            rateFactor /= nu_a
            pLarger0 = 0
            coverage = 1
        else:
            pLarger0 = 1 - np.power(1+alpha*rateFactor, -timeFactor/alpha)
        
        subbasinProbabilities = self.localitySubbasinTrafficModel.get_subbasin_probabilities()
        
        rawRegionProbabilities = np.array([[probs[r].sum() for r in self.regionToSubbasin]
                                                for probs in subbasinProbabilities])
        rawRegionProbabilitiesInv = 1/rawRegionProbabilities
        regionNormalization = 1/rawRegionProbabilities.sum(1)
        regionCounts = np.array(self.subbasinToRegion.fullshape).ravel()
        
        subbasinNumber = self.subbasinToRegion.size
        localityNumber = subbasinProbabilities.shape[0]
        
        sharedRegionProbabilities = np.zeros((localityNumber, subbasinNumber, subbasinNumber))
        for i, toRegion1 in enumerate(self.subbasinToRegion):
            for j, toRegion2 in enumerate(self.subbasinToRegion):
                sharedRegionProbabilities[:,i,j] = rawRegionProbabilitiesInv[:,
                                np.intersect1d(toRegion1, toRegion2, True)].sum(1)
        
        
        trafficByLocality = ((anglerNumbers*coverage*(rateFactor*timeFactor - pLarger0))[:,None,None]
                          * subbasinProbabilities[:,None] * (np.rollaxis(subbasinProbabilities.T[:,None], 2) * (1-xi_0)
                         * (xi_A**2 + regionNormalization[:,None,None]*(
                             xi_A*xi_R*(regionCounts+regionCounts[:,None])
                             + xi_R**2 * sharedRegionProbabilities)
                             ) + np.diag(np.full(subbasinProbabilities.shape[1], xi_0))[None,:,:])
                         )
        
        self._computeMeanData.extend([subbasinProbabilities, regionNormalization, regionCounts, sharedRegionProbabilities])
        
        if getByLocality:
            traffic = trafficByLocality
        else:
            traffic = trafficByLocality.sum(0)
        
        return traffic
        
    
    def get_traffic_quantiles(self, p, parameters=None, parametersConsidered=None,
                             responseRate=None):
        raise NotImplementedError()
    
    def get_possibly_observed_traffic(self, fullDataFileName=None):
        
        subbasinNumber = self.subbasinToRegion.size
        result = np.zeros((subbasinNumber, subbasinNumber))
        
        if not fullDataFileName:
            for row in self.angler_destinations:
                np.add.at(result, (row[:-1], row[1:]), 1)
        else:
            lastDestination = {}
            anglerData = np.genfromtxt(fullDataFileName, delimiter=",",  
                                       skip_header = True, 
                                       dtype = self.localitySubbasinTrafficModel.ANGLER_DATA_DTYPE)
            for anglerId, _, subbasinId, _ in anglerData:
                destination = self.localitySubbasinTrafficModel.subbasinIdToSubbasinIndex[subbasinId]
                if anglerId in lastDestination:
                    result[lastDestination[anglerId], destination] += 1
                lastDestination[anglerId] = destination
                
        return result
                
        
    def plot_observed_predicted(self, saveFileName=None,
                                comparisonFileName=None,
                                fullDataFileName=None):
        if saveFileName:
            if not os.access(saveFileName, os.F_OK): os.makedirs(saveFileName)
            saveFileName = os.path.join(saveFileName, saveFileName)
        
        if comparisonFileName:
            comparisonFileName = os.path.join(comparisonFileName, comparisonFileName)
        
        self.prst("Creating quality plots.")
        
        predicted = self.get_traffic_mean(getReportedOnly=not fullDataFileName)
        """
        err = self.get_traffic_quantiles(np.array([0.025, 0.975]), 
                                         responseRate=1)
        err[0] = predicted.ravel()-err[0]
        err[1] = err[1]-predicted.ravel()
        """
        
        observed = self.get_possibly_observed_traffic(fullDataFileName)
        
        predicted_ = predicted.copy()
        observed_ = observed.copy()
        
        mask = np.eye(observed.shape[0]).astype(bool)
        predicted[mask] = 0
        observed[mask] = 0
        
        create_observed_predicted_mean_error_plot(predicted.sum(1), 
                                                  observed.sum(1), 
                                                  saveFileName=saveFileName+"_subbasin_out",
                                                  comparisonFileName=comparisonFileName
                                                  )
        create_observed_predicted_mean_error_plot(predicted.sum(0), 
                                                  observed.sum(0), 
                                                  saveFileName=saveFileName+"_subbasin_in",
                                                  comparisonFileName=comparisonFileName
                                                  )
        create_observed_predicted_mean_error_plot(predicted[~mask].ravel(), 
                                                  observed[~mask].ravel(), 
                                                  saveFileName=saveFileName+"_subbasin_pairs",
                                                  comparisonFileName=comparisonFileName
                                                  )
        predicted = (predicted_+predicted_.T)[np.tri(*predicted_.shape, 0, bool)]
        observed = (observed_+observed_.T)[np.tri(*observed_.shape, 0, bool)]
        create_observed_predicted_mean_error_plot(predicted, 
                                                  observed, 
                                                  saveFileName=saveFileName+"_subbasin_pairs_bi",
                                                  comparisonFileName=comparisonFileName
                                                  )
        
        plt.show()
        
    def save_subbasin_subbasin_predictions(self, fileName=None, parameters=None, dates=None, 
                                 cities=[None]):
        
        cities = list(OrderedDict((i,i) for i in cities))
        
        localityIndices = []
        localityStrs = []
        
        if not fileName:
            fileName = self.fileName
        
        for locality in cities:
            if locality is not None:
                localityIndices.append(self.localitySubbasinTrafficModel.localityIdToLocalityIndex[locality])
                localityStrs.append("predicted"+str(locality))
            else:
                localityIndices.append(None)
                localityStrs.append("predictedAll")
        
        traffic = self.get_traffic_mean(parameters, dates, False, True)
                
        dtype = ([("pairId", int), ("fromSubbasin", IDTYPE), ("toSubbasin", IDTYPE)] 
                 + [(locality, float) for locality in localityStrs])
        
        subbasinData = self.localitySubbasinTrafficModel.subbasinData
        
        
        toSubbasin, fromSubbasin = np.meshgrid(subbasinData["subbasinId"], subbasinData["subbasinId"])
        
        mask = np.tri(subbasinData.size, subbasinData.size, -1, bool).T
        
        result = np.zeros(mask.sum(), dtype=dtype)
        result["fromSubbasin"] = fromSubbasin[mask]
        result["toSubbasin"] = toSubbasin[mask]
        result["pairId"] = np.arange(result.size)
        
        for locality, localityIndex in zip(localityStrs, localityIndices):
            if localityIndex is None:
                result[locality] = traffic.sum(0)[mask]
            else:
                result[locality] = traffic[localityIndex][mask]
                
        df = pd.DataFrame(result)
        df.to_csv(fileName + "_subbasin_subbasin.csv", index=False)
    
    def save_subbasin_risk(self, fileName=None, parameters=None, dates=None, 
                      cities=[None]):
        
        cities = list(OrderedDict((i,i) for i in cities))
        
        localityIndices = []
        localityStrs = []
        
        if not fileName:
            fileName = self.fileName
        
        for locality in cities:
            if locality is not None:
                localityIndices.append(self.localitySubbasinTrafficModel.localityIdToLocalityIndex[locality])
                localityStrs.append("predicted"+str(locality))
            else:
                localityIndices.append(None)
                localityStrs.append("predictedAll")
        
        traffic = self.get_traffic_mean(parameters, dates, False, True)
                
        dtype = ([("subbasinId", IDTYPE)] + [(locality + "All", float) for locality in localityStrs]
                 + [(locality + "Infested", float) for locality in localityStrs])
        
        subbasinData = self.localitySubbasinTrafficModel.subbasinData
        
        
        result = np.zeros(subbasinData.size, dtype=dtype)
        result["subbasinId"] = subbasinData["subbasinId"]
        
        for locality, localityIndex in zip(localityStrs, localityIndices):
            if localityIndex is None:
                localityTraffic = traffic.sum(0)
            else:
                localityTraffic = traffic[localityIndex]
            np.fill_diagonal(localityTraffic, 0)
            result[locality+"All"] = localityTraffic.sum(1)
            result[locality+"Infested"] = (localityTraffic*subbasinData["infested"]).sum(1)
                
        df = pd.DataFrame(result)
        df.to_csv(fileName + "_subbasin_risk.csv", index=False)
    
    def save_locality_significance(self, fileName=None, parameters=None, dates=None):
        
        if not fileName:
            fileName = self.fileName
        
        traffic = self.get_traffic_mean(parameters, dates, False, True)
                
        dtype = [("localityId", IDTYPE), ("localityPopulation", float), 
                 ("localityAnglers", float), ("totalRiskTrips", float),
                 ("riskTripsPerInhabitant", float), 
                 ("riskTripsPerAngler", float)]
        
        subbasinData = self.localitySubbasinTrafficModel.subbasinData
        localityData = self.localitySubbasinTrafficModel.localityData
        
        result = np.zeros(localityData.shape[0], dtype=dtype)
        
        result["localityId"] = localityData["localityId"]
        result["localityPopulation"] = localityData["localityPopulation"]
        result["localityAnglers"] = localityData["localityAnglers"]
        
        relevantMask = np.ix_(np.nonzero(subbasinData["infested"])[0],
                              np.nonzero(~subbasinData["infested"])[0])
        
        for localityIndex, row in enumerate(result):
            localityTraffic = traffic[localityIndex]
            np.fill_diagonal(localityTraffic, 0)
            row["totalRiskTrips"] = localityTraffic[relevantMask].sum()
        
        result["riskTripsPerInhabitant"] = result["totalRiskTrips"] / result["localityPopulation"]
        result["riskTripsPerAngler"] = result["totalRiskTrips"] / result["localityAnglers"]
        
        df = pd.DataFrame(result)
        df.to_csv(fileName + "_locality_significance.csv", index=False)
    
    
    def simulate_count_data(self, fileName, parameters=None, dates=None,
                            independentCities=False, preserveAnglerIdentity=False):
        
        if parameters is None:
            parameters = self.parameters
        
        if dates is None:
            dates = np.arange(self.localitySubbasinTrafficModel.startDate, self.localitySubbasinTrafficModel.endDate)
        
        xi_0, xi_R, nu_a, alpha, coverage = self.convert_parameters(parameters)
            
        timeFactors = self.localitySubbasinTrafficModel.timeModel.get_time_factor(dates)
        
        anglerNumbers = self.localitySubbasinTrafficModel.localityData["localityAnglers"].astype(int)
        mu_a = self.localitySubbasinTrafficModel.get_trip_rate_factor() / (nu_a * coverage) 
        
        subbasinProbabilities = self.localitySubbasinTrafficModel.get_subbasin_probabilities()
        
        rawRegionProbabilities = np.array([[probs[r].sum() for r in self.regionToSubbasin]
                                                for probs in subbasinProbabilities])
        regionProbabilities = rawRegionProbabilities / rawRegionProbabilities.sum(1)[:,None]
        
        anglerOrigins = np.concatenate([np.full(n_i, i) for i, n_i in 
                                        enumerate(anglerNumbers)])
        
        subbasinProbabilitiesByRegion = np.empty(regionProbabilities.shape, dtype=object)
        for i, (probs, norms) in enumerate(zip(subbasinProbabilities, rawRegionProbabilities)):
            for j, (r, norm) in enumerate(zip(self.regionToSubbasin, norms)):
                subbasinProbabilitiesByRegion[i, j] = probs[r]/norm
                
        
        np.random.seed()
        allTrips = self._simulate_angler_trips(anglerOrigins, dates, 
                                               timeFactors, subbasinProbabilities, 
                                               regionProbabilities, 
                                               self.regionToSubbasin, 
                                               subbasinProbabilitiesByRegion, 
                                               xi_0, xi_R, alpha, mu_a,
                                               independentCities)
        
        allTripsTransformed = np.zeros(allTrips.shape[0], 
                               dtype=self.localitySubbasinTrafficModel.ANGLER_DATA_DTYPE)
        
        allTripsTransformed["anglerId"] = allTrips[:, 0]
        allTripsTransformed["localityId"] = self.localitySubbasinTrafficModel.localityData["localityId"][allTrips[:, 1]]
        allTripsTransformed["subbasinId"] = self.localitySubbasinTrafficModel.subbasinData["subbasinId"][allTrips[:, 2]]
        allTripsTransformed["date"] = allTrips[:, 3]
        
        isReported = np.random.rand(allTripsTransformed.size) < nu_a
        
        potentiallyReportedTrips = allTripsTransformed[isReported]
        potentiallyReportedTripsUntranformed = allTrips[isReported]
        if preserveAnglerIdentity:
            _, ind = np.unique(self.localitySubbasinTrafficModel.surveyData['anglerId'], return_index=True)
            cities, counts = np.unique(self.localitySubbasinTrafficModel.surveyData[ind]['localityId'], return_counts=True)
            appUsers = []
            _, ind2 = np.unique(allTripsTransformed['anglerId'], return_index=True)
            potentiallyIncludedAnglers = allTripsTransformed[ind2]
            for locality, count in zip(cities, counts):
                considered_indices = np.random.choice(np.nonzero(
                    potentiallyIncludedAnglers['localityId']==locality)[0], 
                    count, replace=False)
                appUsers.extend(potentiallyIncludedAnglers['anglerId'][considered_indices])
            appUsers = set(appUsers)
            isAppUser = np.array([i in appUsers for i in potentiallyReportedTrips['anglerId']])
        else:
            appUsers = set(np.random.choice(anglerOrigins.size, 
                                        int(coverage*anglerOrigins.size),
                                        replace=False))
        
            isAppUser = np.array([i in appUsers for i in potentiallyReportedTripsUntranformed[:,0]])
        reportedTrips = potentiallyReportedTrips[isAppUser]
        
        for resultArr, fname in [(allTripsTransformed, "_allTrips"), (reportedTrips, "_reportedTrips")]:
            df = pd.DataFrame(resultArr)
            df.to_csv(fileName + fname + ".csv", index=False)
            
            
    def _simulate_angler_trips(self, anglerOrigins, dates, timeFactors, subbasinProbabilities,
                               regionProbabilities, regionToSubbasin, subbasinProbabilitiesByRegion, 
                               xi_0, xi_R, alpha, mu_a, independentCities=False):
        result = []
        
        lastDestinations = np.full(anglerOrigins.size, -1)
        
        randGenerator = np.random.default_rng()
        
        localityNumber, subbasinNumber = subbasinProbabilities.shape
        
        if independentCities:
            tripRates = randGenerator.gamma(timeFactors[:,None]/alpha, scale=alpha*mu_a)
        else:
            tripRates = randGenerator.gamma(timeFactors[:,None]/alpha, scale=alpha)*mu_a
        
        choice = lambda p: (p.cumsum(1) > np.random.rand(p.shape[0])[:,None]).argmax(1)
        #choice = lambda p: randGenerator.choice(regionNumber, p=p)
        #preferredRegions = np.apply_along_axis(choice, 1, regionProbabilities[anglerOrigins])
        preferredRegions = choice(regionProbabilities[anglerOrigins])
        #anglerNumber = anglerOrigins.size
        xi_A = 1-xi_R
        xi_A_ = xi_A*(1-xi_0)
        xi_R_ = 1-xi_0
        xi_A2_ = xi_A
        
        for dayIndex in range(dates.size):
            print("Processing day number", dayIndex)
            day = dates[dayIndex]
            tripNumbers = randGenerator.poisson(tripRates[dayIndex][anglerOrigins])
            for anglerIndex in np.nonzero(tripNumbers)[0]:
                origin = anglerOrigins[anglerIndex]
                preferredRegion = preferredRegions[anglerIndex]
                for _ in range(tripNumbers[anglerIndex]):
                    if lastDestinations[anglerIndex] >= 0:
                        xi_A__ = xi_A_
                        xi_R__ = xi_R_
                    else:
                        xi_A__ = xi_A2_
                        xi_R__ = 1
                    behaviour = np.random.rand() 
                    
                    if behaviour < xi_A__:
                        destination = randGenerator.choice(subbasinNumber, 
                               p=regionProbabilities[origin])
                    elif behaviour < xi_R__:
                        destination = randGenerator.choice(regionToSubbasin[preferredRegion],
                                                           p=subbasinProbabilitiesByRegion[origin][preferredRegion])
                    else:
                        destination = lastDestinations[anglerIndex]
                        
                    lastDestinations[anglerIndex] = destination 
                    
                    result.append([anglerIndex, origin, destination, day])
        
        return np.array(result)
    
    @staticmethod
    def new(self):
        pass
        
        


