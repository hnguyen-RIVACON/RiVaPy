import numpy as np
import datetime as dt
from typing import Union, Tuple, Dict
from rivapy.tools.interfaces import FactoryObject
from rivapy.tools.enums import Currency, ESGRating, Rating, Sector, Country, SecuritizationLevel
from rivapy.marketdata import DiscountCurveParametrized, NelsonSiegel, LinearRate, ConstantRate
from rivapy.marketdata.factory import create as _create
from rivapy.instruments.components import Issuer
from rivapy.instruments import PlainVanillaCouponBondSpecification
from rivapy.sample_data._logger import logger

class SpreadCurveCollection(FactoryObject):

    @staticmethod
    def _create_curve_or_float(x: Union[FactoryObject, dict, float]):
        if isinstance(x, list) or isinstance(x, tuple):
            if len(x) != 2:
                raise NotImplementedError('All list and tuples must have length equal 2.')
            return [SpreadCurveCollection._create_curve_or_float(x[0]), SpreadCurveCollection._create_curve_or_float(x[1])]
        if isinstance(x,dict):
            return _create(x)
        return x
        
    @staticmethod
    def _dict_entry(x):
        if isinstance(x, list) or isinstance(x, tuple):
            if len(x) != 2:
                raise NotImplementedError('All list and tuples must have length equal 2.')
            return [SpreadCurveCollection._dict_entry(x[0]), SpreadCurveCollection._dict_entry(x[1])]
        if hasattr(x,'to_dict'):
            return x.to_dict()
        if isinstance(x,dict):
            return {k: SpreadCurveCollection._dict_entry(v) for k,v in x.items()}
        return x
        
    def __init__(self, ref_date: dt.datetime,
                    rating_curve:Tuple[Union[FactoryObject, dict, float], Union[FactoryObject, dict, float]],
                    currency_spread: Dict[str, Tuple[Union[FactoryObject, dict, float], Union[FactoryObject, dict, float]]],
                    esg_spreads: Dict[str, float],
                    rating_weights: Dict[str, float],
                    sector_spreads: Dict[str, Tuple[Union[FactoryObject, dict, float], Union[FactoryObject, dict, float]]],
                    country_curves: Dict[str, Tuple[Union[FactoryObject, dict, float], Union[FactoryObject, dict, float]]],
                    sec_level_spreads: Dict[str, Tuple[Union[FactoryObject, dict, float], Union[FactoryObject, dict, float]]]
                    ):
        self.ref_date = ref_date
        self.rating_curve = [SpreadCurveCollection._create_curve_or_float(rating_curve[0]), SpreadCurveCollection._create_curve_or_float(rating_curve[1])]
        self.currency_spread = { k:SpreadCurveCollection._create_curve_or_float(v) for k,v in currency_spread.items() }
        self.esg_spreads = esg_spreads
        self.rating_weights = rating_weights
        self.sector_spreads = { k:SpreadCurveCollection._create_curve_or_float(v) for k,v in sector_spreads.items() }
        self.country_curves = { k:SpreadCurveCollection._create_curve_or_float(v) for k,v in country_curves.items() }
        self.sec_level_spreads = { k:SpreadCurveCollection._create_curve_or_float(v) for k,v in sec_level_spreads.items() }

    def _to_dict(self) -> dict:
        tmp = {'ref_date': self.ref_date, 
               'rating_curve': SpreadCurveCollection._dict_entry(self.rating_curve),
               'currency_spread': SpreadCurveCollection._dict_entry(self.currency_spread),
               'esg_spreads': SpreadCurveCollection._dict_entry(self.esg_spreads),
               'rating_weights': SpreadCurveCollection._dict_entry(self.rating_weights),
               'sector_spreads': SpreadCurveCollection._dict_entry(self.sector_spreads),
               'country_curves': SpreadCurveCollection._dict_entry(self.country_curves),
               'sec_level_spreads': SpreadCurveCollection._dict_entry(self.sec_level_spreads),
               }
        return tmp
    
    def get_curve(self, issuer: Issuer, bond: PlainVanillaCouponBondSpecification):
        logger.info('computing curve for issuer ' + issuer.name + ' and bond ' + bond.obj_id)
        rating_weight = self.rating_weights[issuer.rating]
        w1 = 1.0-rating_weight
        w2 = rating_weight
        rating_curve = w1*self.rating_curve[0] + w2*self.rating_curve[1]
        country_spread = w1*self.country_curves[issuer.country][0] + w2*self.country_curves[issuer.country][1]
        esg_spread = w1*self.esg_spreads[issuer.esg_rating][0] +  w2*self.esg_spreads[issuer.esg_rating][1]
        sector_spread = w1*self.sector_spreads[issuer.sector][0] + w2*self.sector_spreads[issuer.sector][1]
        currency_spread = w1*self.currency_spread[bond.currency][0] + w2*self.currency_spread[bond.currency][1]
        securitization_spread = w1*self.sec_level_spreads[bond.securitization_level][0] + w2* self.sec_level_spreads[bond.securitization_level][1]
        curve = 0.5*rating_curve + 0.5*(0.3*country_spread + 0.3*securitization_spread + 0.2*esg_spread + 0.1*sector_spread+0.1*currency_spread)
        return curve


class SpreadCurveSampler:
    def __init__(self):
        """This class samples spreadcurves used to price bonds. It creates different curves according to
            
            * issuer rating (for all ratings defined in :class:`rivapy.tools.enums.Rating`)
            * currency (for all currencies defined in :class:`rivapy.tools.enums.Currency`)
            * country (for all countries defined in :class:`rivapy.tools.enums.Country`)
            * esg rating (for all ratings defined in :class:`rivapy.tools.enums.ESGRating`)
            * sector (for all sectors defined in :class:`rivapy.tools.enums.Sector`)
            * securitization level (only SENIOR_SECURED, SENIOR_UNSECURED and SUBORDINATED are currently handled)

            An object of this class provides the method :meth:`get_curve` that returns a spread curve that may be adequate to price
            a bond of the given specification and issuer.

            As basis for the curve creation this method uses the Nelson-Siegel Parametrization, see :class:`rivapy.marketdata.curves.NelsonSiegel` for a more detailed description 
            of this parametrization. Each curve is constructed so that for all of the features above fixed, the curve is consistet w.r.t. the issuer rating in 
            the sense that a curve of a higher rating is strictly below the curve of a lower rating.

            The construction is as follows:

            We create two Nelson-Siegel parameterized curves by sampling the Nelson-Siegel parameters
             
        """
        pass
    
    def sample(self, ref_date: dt.datetime)->dict:
        min_params = {'min_short_term_rate': -0.01, 
                          'max_short_term_rate': 0.02, 
                          'min_long_run_rate': 0.0,
                          'max_long_run_rate': 0.03,
                          'min_hump': -0.02,
                          'max_hump': 0.05,
                          'min_tau': 0.5,
                          'max_tau': 3.0}
        
        max_params = {'min_short_term_rate': 0.1, 
                          'max_short_term_rate': 0.25, 
                          'min_long_run_rate': 0.1,
                          'max_long_run_rate': 0.25,
                          'min_hump': 0.0,
                          'max_hump': 0.3,
                          'min_tau': 0.5,
                          'max_tau': 5.0}
        
        curve_best_rating = DiscountCurveParametrized('', ref_date, 
                                                           NelsonSiegel._create_sample(n_samples=1, 
                                                                                       seed=None,**min_params)[0])
        curve_worst_rating = curve_best_rating + DiscountCurveParametrized('', ref_date, 
                                                           NelsonSiegel._create_sample(n_samples=1, 
                                                                                     seed=None,**max_params)[0])
        self.rating_curve = [curve_best_rating, curve_worst_rating]
        self._sample_currency_spread()
        self._sample_esg_rating_spreads()
        self._sample_rating_weights()
        self._sample_sector_spreads()
        self._sample_country_curves(ref_date=ref_date)
        self._sample_sec_level_spreads()
        return {'rating_curves': self.rating_curve,'currency_spread': self.currency_spread,'esg_rating_spread': self.esg_rating_spread,
                  'rating_weights': self.rating_weights, 'sector_spreads': self.sector_spreads, 'country_curves': self.country_curves,
                  'securitization_spreads': self.securitization_spreads}

    def sample_new(self, ref_date: dt.datetime)->SpreadCurveCollection:
        min_params = {'min_short_term_rate': -0.01, 
                          'max_short_term_rate': 0.02, 
                          'min_long_run_rate': 0.0,
                          'max_long_run_rate': 0.03,
                          'min_hump': -0.02,
                          'max_hump': 0.05,
                          'min_tau': 0.5,
                          'max_tau': 3.0}
        
        max_params = {'min_short_term_rate': 0.1, 
                          'max_short_term_rate': 0.25, 
                          'min_long_run_rate': 0.1,
                          'max_long_run_rate': 0.25,
                          'min_hump': 0.0,
                          'max_hump': 0.3,
                          'min_tau': 0.5,
                          'max_tau': 5.0}
        
        curve_best_rating = DiscountCurveParametrized('', ref_date, 
                                                           NelsonSiegel._create_sample(n_samples=1, 
                                                                                       seed=None,**min_params)[0])
        curve_worst_rating = curve_best_rating + DiscountCurveParametrized('', ref_date, 
                                                           NelsonSiegel._create_sample(n_samples=1, 
                                                                                     seed=None,**max_params)[0])
        self.rating_curve = [curve_best_rating, curve_worst_rating]
        self._sample_currency_spread()
        self._sample_esg_rating_spreads()
        self._sample_rating_weights()
        self._sample_sector_spreads()
        self._sample_country_curves(ref_date=ref_date)
        self._sample_sec_level_spreads()
        return SpreadCurveCollection(ref_date, self.rating_curve, self.currency_spread, self.esg_rating_spread, self.rating_weights, 
                                     self.sector_spreads, self.country_curves, self.securitization_spreads )
  
    def _sample_currency_spread(self):
        self.currency_spread = {}
        low = np.random.uniform(0.005,0.01)
        high = low + np.random.uniform(0.0,0.1)
        for c in Currency:
            self.currency_spread[c.value] = [low, high]
        for c in [Currency.EUR, Currency.USD, Currency.GBP, Currency.JPY]:
            low = np.random.uniform(0.0,0.01)
            high = low + np.random.uniform(0.0,0.1)
            self.currency_spread[c.value] = [low, high]
        
    def _sample_esg_rating_spreads(self):
        self.esg_rating_spread = {}
        low = 0.0
        for i,s in enumerate(ESGRating):
            high = low + np.random.uniform(low=0.01, high=0.07)
            self.esg_rating_spread[s.value] = (low, high)
            low = high+0.01
        
    def _sample_rating_weights(self):
        rating_weights = np.random.uniform(low=1.0, high=4.0, size=len(Rating)).cumsum()
        rating_weights[0] = 0.0
        rating_weights[-1] = 4.0
        rating_weights = rating_weights/rating_weights.max()
        self.rating_weights = {}
        for i,k in enumerate(Rating):
            self.rating_weights[k.value] = rating_weights[i]
        
    def _sample_sector_spreads(self):
        result = {}
        for s in Sector:
            s_low = np.random.uniform(low=0.001, high=0.0025)
            result[s.value] = (s_low, s_low+np.random.uniform(low=0.001, high=0.0025))
        self.sector_spreads = result
        
    def _sample_country_curves(self, ref_date):
        self.country_curves = {}
        for c in Country:
            shortterm_rate = np.random.uniform(low=0.0, high=0.02)
            longterm_rate = shortterm_rate + np.random.uniform(low=-0.005, high=0.005)
            lower_curve = DiscountCurveParametrized('', ref_date, LinearRate(shortterm_rate, longterm_rate))
            self.country_curves[c.value] = (lower_curve, lower_curve + DiscountCurveParametrized('', ref_date,
                                                                    ConstantRate(np.random.uniform(0.05, 0.15))
                                                                                       )
                                  )
    
    def _sample_sec_level_spreads(self):
        result = {}
        spread = 0.0
        result[SecuritizationLevel.SENIOR_SECURED.value]=(0.0,0.001)
        low = np.random.uniform(0.001, 0.005)
        result[SecuritizationLevel.SENIOR_UNSECURED.value] = (low, low + 0.01)
        low = np.random.uniform(0.01, 0.025) + result[SecuritizationLevel.SENIOR_UNSECURED.value][1]
        result[SecuritizationLevel.SUBORDINATED.value] = (low, low + 0.03)
        self.securitization_spreads = result

    def set_params(self, params: dict):
        self.rating_curve = params['rating_curves']
        self.currency_spread = params['currency_spread']
        self.esg_rating_spread = params['esg_rating_spread']
        self.rating_weights = params['rating_weights']
        self.sector_spreads = params['sector_spreads']
        self.country_curves = params['country_curves']
        self.securitization_spreads = params['securitization_spreads']

        
        
    def get_curve(self, issuer: Issuer, bond: PlainVanillaCouponBondSpecification):
        logger.info('computing curve for issuer ' + issuer.name + ' and bond ' + bond.obj_id)
        rating_weight = self.rating_weights[issuer.rating]
        w1 = 1.0-rating_weight
        w2 = rating_weight
        rating_curve = w1*self.rating_curve[0] + w2*self.rating_curve[1]
        country_spread = w1*self.country_curves[issuer.country][0] + w2*self.country_curves[issuer.country][1]
        esg_spread = w1*self.esg_rating_spread[issuer.esg_rating][0] +  w2*self.esg_rating_spread[issuer.esg_rating][1]
        sector_spread = w1*self.sector_spreads[issuer.sector][0] + w2*self.sector_spreads[issuer.sector][1]
        currency_spread = w1*self.currency_spread[bond.currency][0] + w2*self.currency_spread[bond.currency][1]
        securitization_spread = w1*self.securitization_spreads[bond.securitization_level][0] + w2* self.securitization_spreads[bond.securitization_level][1]
        curve = 0.5*rating_curve + 0.5*(0.3*country_spread + 0.3*securitization_spread 
                        + 0.2*esg_spread + 0.1*sector_spread+0.1*currency_spread)
        #curve = 0.5*rating_curve + 0.5*esg_spread#(0.3*country_spread + 0.3*securitization_spread 
        #                #+ 0.2*esg_spread + 0.1*sector_spread+0.1*currency_spread)
        return curve