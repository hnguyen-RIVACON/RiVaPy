import json
import numpy as np
from rivapy.pricing.green_ppa_pricing import GreenPPADeepHedgingPricer

class Repo:
    def __init__(self, repo_dir):
        self.repo_dir = repo_dir
        self.results = {}
        with open(repo_dir+'/hedge_results.json','r') as f:
            self.results = json.load(f)
            
    def run_hedge_experiment(self, val_date, ppa_spec, model, **kwargs):
        params = {}
        params['ppa_spec'] = ppa_spec.to_dict()
        params['ppa_spec_hash'] = ppa_spec.hash()
        params['model'] = model.to_dict()
        params['model_hash'] = model.hash()
        params['pricing_param'] = kwargs
        hash_key = FactoryObject.hash_for_dict(params)
        #if hash_key in results:
        
        pricing_result =  GreenPPADeepHedgingPricer.price(val_date, 
                                      ppa_spec, 
                                      model,
                                    **kwargs)
        return pricing_result

from rivapy.tools.interfaces import FactoryObject

def compute_pnl_figures(pricing_results):
    pnl = pricing_results.hedge_model.compute_pnl(pricing_results.paths, pricing_results.payoff)
    return {'mean': pnl.mean(), 'var': pnl.var(), '1%':np.percentile(pnl,1), '99%': np.percentile(pnl,1)}

def compute_pnl(pricing_results):
    return pricing_results.hedge_model.compute_pnl(pricing_results.paths, pricing_results.payoff)