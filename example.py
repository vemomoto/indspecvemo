'''
Created on 20.05.2022

@author: Samuel
'''
import numpy as np
from indspecvemo.angler_movement_model import *

from vemomoto_core.tools.saveobject import save_object, load_object

if __name__ == '__main__':
    # The first command line argument specifies the output file to which all output
    # will be written.     
    from vemomoto_core.tools.tee import Tee                
    if len(sys.argv) > 1:
        teeObject = Tee(sys.argv[1])


def fit_model(useKnownParameters=False):

    # We assume that we have put all necessary datasets into 
    # the following directory. The data used in the paper
    # for which this software was written can be retrieved from 
    # Dryad Digital Repository: doi.org/10.5061/dryad.6m905qg3j  
    directory = "ModelInput"
    
    # Switch into the directory of the data
    os.chdir(directory)
    
    # Specify a file name to save the model
    localityToSubbasinModelFileName = "LocalityToSubbasinModel"
    subbasinToSubbasinModelFileName = "SubbasinToSubbasinModel"
    
    # Specify input file names
    localityDataFileName = "Locality_Data.csv"
    subbasinDataFileName = "Subbasin_Data.csv"
    anglerDataFileName = "Angler_Data.csv"
    localityDistanceDataFileName = "Subbasin_Localitiy_Distances.csv"
    subbasinDistanceDataFileName = "Subbasin_Subbasin_Distances.csv"
    
    # Specify which fraction of the data shall be used for fitting
    # and which should be used for validaiton
    # If we want to compute the correct scaling parameters, use
    # all data for fitting, since otherwise only a fraction of the traffic
    # will be predicted.
    dataFractionForFitting = 1
    dataFractionForValidation = 1 - dataFractionForFitting
    
    # Specify the study period
    start, end = date_to_int("2018-05-01"), date_to_int("2020-05-01")
    
    # Create a model for the daily traffic between localities and subbasins.
    # `LocalitySubbasinAnglerTrafficFactorModel` is the class of the submodel
    # for estimating the mean traffic between localities and subbasins,
    # `TimeFactorModel` is the type of the submodel for the relative 
    # mean traffic per day.
    localitySubbasinModel = DailyLocalitySubbasinAnglerTrafficModel(
        LocalitySubbasinAnglerTrafficFactorModel,
        TimeFactorModel, 
        (start, end)
    )
    
    # Read in the data sets
    localitySubbasinModel.read_locality_data(localityDataFileName)
    localitySubbasinModel.read_subbasin_data(subbasinDataFileName)
    localitySubbasinModel.read_angler_data(anglerDataFileName, dataFractionForFitting, dataFractionForValidation)
    
    # Fit the time submodel
    localitySubbasinModel.fit_time_model([[True]*6], True, get_CI=False)
    
    # Save the progress
    save_object(localitySubbasinModel, localityToSubbasinModelFileName+".amm")
    
    # Fit the submodel for the mean traffic from localities to subbasins
    localitySubbasinModel.read_distance_data(localityDistanceDataFileName)
    localitySubbasinModel.create_traffic_factor_model()
    
    # Save the progress
    save_object(localitySubbasinModel, localityToSubbasinModelFileName+".amm")
    
    # if we want to skip the expensive fitting stage, use the fitted parameters from the paper
    if useKnownParameters:
        covariates = np.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0])>0
        parameters = np.array([5.9915222824534675, -20.721442219373433, 1.0871432328780741, 27.737686315295644, 45998.990138846726, 0.1419756354366372, 908.0724223203457, -1.3351778511814874, 5.76111817493212, 3.2508056900480553, -0.9285310817320704, -1.3978003726969375, 0.001087840248331292, 0.7937269840488905, -0.5677766785854472])
        basePermutation = np.ma.array(covariates, mask=True)
        localitySubbasinModel.fit(plotFileName=localityToSubbasinModelFileName, 
                                  parameters={"parameters":parameters, "parametersConsidered":covariates},
                                  continueOptimization=True, refit=True, get_CI=False, limit=1000)
    else:
        # Otherwise, do a complete new fit
        localitySubbasinModel.fit(plotFileName=localityToSubbasinModelFileName, 
                                  refit=True, limit=1000)
    
    # Save the progress
    save_object(localitySubbasinModel, localityToSubbasinModelFileName+".amm")
    
    # Plot observed and predicted values
    localitySubbasinModel.plot_observed_predicted()
    
    # Create the model for subbasin to subbasin traffic
    subbasinSubbasinModel = SubbasinSubbasinAnglerTrafficModel(subbasinToSubbasinModelFileName)
    subbasinSubbasinModel.set_locality_subbasin_traffic_model(localitySubbasinModel)
    subbasinSubbasinModel.read_subbasin_distances(subbasinDistanceDataFileName)
    subbasinSubbasinModel.save()
    
    # Do not print zero division warnings etc. for cleaner output. 
    # Things are handled okay anyway.
    np.seterr(all='ignore')
    
    if useKnownParameters:
        # Set the radius of the regions of preference to the
        # value from the paper
        subbasinSubbasinModel.set_regions_by_radius(31)
        subbasinSubbasinModel.fit()
    else:
        # Fit the subbasin to subbasin model and determine the
        # optimal radius of the regions of preference
        subbasinSubbasinModel.fit_region_radius(10, 80, 71) 
    
    # Save the model again
    subbasinSubbasinModel.save()
    
    # Compute how many days per year an average angler goes fishing
    #subbasinSubbasinModel.get_mean_days_out_with_CI(dates=np.arange(start, start+365))
    
    # Print the likelihood
    print(subbasinSubbasinModel.negative_log_likelihood(subbasinSubbasinModel.parameters))
    print(subbasinSubbasinModel.localitySubbasinTrafficModel.negative_log_likelihood(subbasinSubbasinModel.localitySubbasinTrafficModel.parameters, subbasinSubbasinModel.localitySubbasinTrafficModel.covariates))
    
    # Save predictions for subbasin to subbasin traffic
    subbasinSubbasinModel.save_subbasin_subbasin_predictions(dates=np.arange(start, start+365), cities=[None, 28, 1281])
    
    # Save predictions for locality to subbasin traffic
    subbasinSubbasinModel.localitySubbasinTrafficModel.save_subbasin_predictions("AppLocalitySubbasinModel2_HUC.csv")
    
    # Save predictions for incoming infested anglers
    subbasinSubbasinModel.save_subbasin_risk(dates=np.arange(start, start+365), cities=[None, 28, 1281])
    
    # Maybe: Compute confidence intervals for some extreme results
    #subbasinSubbasinModel.get_extreme_result_CIs(dates=np.arange(start, start+365), nmax=500) #, apprxtol=0.9)
    
    # See how the predictions change, if another subbasin gets infested
    subbasinSubbasinModel.localitySubbasinTrafficModel.subbasinData[subbasinSubbasinModel.localitySubbasinTrafficModel.subbasinIdToSubbasinIndex[11020301]]["infested"] = True
    subbasinSubbasinModel.save_subbasin_risk(fileName="HUC_risk_scenario", dates=np.arange(start, start+365), cities=[None, 28, 1281])
    

if __name__ == '__main__':
    fit_model(useKnownParameters=True)
    
    
    