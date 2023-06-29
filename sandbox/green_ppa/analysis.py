import copy
import json
import numpy as np
from rivapy.tools.interfaces import _JSONEncoder, _JSONDecoder, FactoryObject
from rivapy.models.residual_demand_fwd_model import LinearDemandForwardModel
from rivapy.pricing.green_ppa_pricing import GreenPPADeepHedgingPricer, PPAHedgeModel

class Repo:
    def __init__(self, repo_dir):
        self.repo_dir = repo_dir
        self.results = {}
        try:
            with open(repo_dir+'/results.json','r') as f:
                self.results = json.load(f, cls=_JSONDecoder)
        except:
            pass
    
    @staticmethod
    def compute_pnl_figures(pricing_results):
        pnl = pricing_results.hedge_model.compute_pnl(pricing_results.paths, pricing_results.payoff)
        return {'mean': pnl.mean(), 'var': pnl.var(), '1%':np.percentile(pnl,1), '99%': np.percentile(pnl,99)}

    def run(self, val_date, ppa_spec, model, **kwargs):
        params = {}
        params['ppa_spec'] = ppa_spec.to_dict()
        params['model'] = model.to_dict()
        _kwargs = copy.deepcopy(kwargs)
        _kwargs.pop('tensorboard_logdir', None) #remove  parameters irrelevant for hashing before generating kashkey
        _kwargs.pop('verbose', None)
        params['pricing_param'] = _kwargs
        hash_key = FactoryObject.hash_for_dict(params)
        params['pricing_param'] = kwargs
        params['ppa_spec_hash'] = ppa_spec.hash()
        params['model_hash'] = model.hash()
        #params['pricing_params_hash'] = FactoryObject.hash_for_dict(kwargs)
        if hash_key in self.results.keys():
            return self.results[hash_key]
        pricing_result =  GreenPPADeepHedgingPricer.price(val_date, 
                                      ppa_spec, 
                                      model,
                                    **kwargs)
        params['result'] = Repo.compute_pnl_figures(pricing_result)
        self.results[hash_key] = params
        with open(self.repo_dir+'/results.json','w') as f:
            json.dump(params, f, cls=_JSONEncoder)
        pricing_result.hedge_model.save(self.repo_dir+'/'+hash_key+'/')
        return pricing_result
    
    def get_hedge_model(self, hashkey:str)->PPAHedgeModel:
        return PPAHedgeModel.load(self.repo_dir+'/'+hashkey+'/')

    def get_model(self, hashkey:str)->PPAHedgeModel:
        LinearDemandForwardModel.from_dict(self.results[hashkey]['model'])
        return PPAHedgeModel.load(self.repo_dir+'/'+hashkey+'/')

    def simulate_model(self, hashkey: str)->np.ndarray:
        GreenPPADeepHedgingPricer._compute_points(self.results[hashkey]['model'])
        return self.results[hashkey]['model'].simulate()
#from rivapy.tools.interfaces import FactoryObject

#def compute_pnl(pricing_results):
#    return pricing_results.hedge_model.compute_pnl(pricing_results.paths, pricing_results.payoff)