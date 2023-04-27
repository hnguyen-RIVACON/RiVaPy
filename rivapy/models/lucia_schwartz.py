from typing import Union, Callable
import numpy as np
from rivapy.tools.interfaces import FactoryObject
from rivapy.models.ornstein_uhlenbeck import OrnsteinUhlenbeck

class LuciaSchwartz(FactoryObject):

    def _eval_grid(f, timegrid):
        if f is None:
            return np.zeros(timegrid.shape)
        try:
            return f(timegrid)
        except:
            result = np.full(timegrid.shape, f)
            return result

    def __init__(self, 
                rho: float,
                kappa: Union[float, Callable]=None, 
                sigma1: Union[float, Callable]=None, 
                mu: Union[float, Callable]=None,
                sigma2:Union[float, Callable]=None,
                
                f: Callable[[Union[float, np.ndarray]],Union[float, np.ndarray]]=None):
        """Lucia Schwartz two factor model.

        The model may be used to simulate spot/forward prices via

        .. math:: 

            S(t) = f(t) + X_1(t) + X_2(t)

            dX_1(t) = -\\kappa X_1(t)+\sigma_1dW_1(t)

            dX_2(t) = \\mu dt + \sigma_2 dW_2
            
        where :math:`f(t)` is a deterministic function, :math:`\\kappa` the speed of mean reversion for 
        the first process that may be interpreted as the long-term factor and :math:`\\sigma_1` the respective volatility.
        The second factor :math:`X_2` may be interpreted as a short-term factor that is influenced by  :math:`W_2`
        and has drift :math:`\\mu`. :math:`X_1` and :math:`X_2` may be correlated with correlation :math:`\\rho`. Note that this class just simulates 

        
        Args:
            kappa (Union[float, Callable]): The speed of mean reversion for the first factor :math:`X_1`. Can be either constant or time dependent.
            sigma1 (Union[float, Callable]): The volatility of the first factor :math:`X_1`. Can be either constant or time dependent.
            mu (Union[float, Callable]): The drift of teh second factor :math:`X_2`. Can be either constant or time dependent.
            sigma2 (Union[float, Callable]): The volatility of the second factor :math:`X_2`. Can be either constant or time dependent.
            rho (float): Correlation between X1 and X2.
            f (Union[float, Callable], optional): Deterministic function of time. Defaults to 0.
        """
        self.X1 = OrnsteinUhlenbeck(kappa, sigma1, 0.0)
        self.mu = mu
        self.sigma2 = sigma2
        self.rho = rho
        self._timegrid = None
        self.f = f

    def _to_dict(self) -> dict:
        return {'kappa': self.X1.speed_of_mean_reversion, 'sigma1': self.X1.volatility,
                'mu': self.mu, 'sigma2': self.sigma2, 'f': self.f}

    def _set_timegrid(self, timegrid):
        
        self._mu_grid = LuciaSchwartz._eval_grid(self.mu, timegrid)
        self._sigma2_grid = LuciaSchwartz._eval_grid(self.sigma2, timegrid)
        self._f_grid = LuciaSchwartz._eval_grid(self.f, timegrid)

    def rnd_shape(self, n_sims: int, n_timepoints: int)->tuple:
        return (n_timepoints-1, n_sims, 2)


    def simulate(self, timegrid, start_value, rnd):
        """_summary_

        Args:
            timegrid (_type_): _description_
            start_value (_type_): _description_
            rnd (_type_): _description_

        Returns:
            _type_: _description_
        """
        self._set_timegrid(timegrid)
        rnd_ = np.copy(rnd)
        rnd_[:,:,1] = self.rho*rnd[:,:,0] + np.sqrt(1.0-self.rho**2)*rnd[:,:,1]
        X2 = np.empty((timegrid.shape[0],rnd.shape[1],))
        if len(start_value.shape)==2:
            start_X1 = start_value[:,0]
            X2[0,:] = start_value[:,1]
        else:
            start_X1 = start_value[0]
            X2[0,:] = start_value[1]
        X1 = self.X1.simulate(timegrid, start_value=start_X1, rnd=rnd_[:,:,0])
       
        for i in range(timegrid.shape[0]-1):
            X2[i+1,:] = X2[i,:]*self.X1._delta_t[i] + self._sigma2_grid[i]*rnd[i,:,1]*self.X1._sqrt_delta_t[i]
        return  X1 + X2  + self._f_grid[:,np.newaxis]

