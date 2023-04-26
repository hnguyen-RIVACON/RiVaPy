from typing import Union, Callable
import numpy as np
from rivapy.tools.interfaces import FactoryObject

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
                kappa: Union[float, Callable], 
                sigma1: Union[float, Callable], 
                mu: Union[float, Callable],
                sigma2:Union[float, Callable],
                f: Callable[[Union[float, np.ndarray]],Union[float, np.ndarray]]=None):
        """Lucia Schwartz two factor model.

        The model is defined by

        .. math:: 

            S(t) = f(t) + X_1(t) + X_2(t)

            dX_1(t) = -\\kappa X_1(t)+\sigma_1dW_1(t)

            dX_2(t) = \\mu dt + \sigma_2 dW_2
            
        where :math:`f(t)` is a deterministic function, :math:`\\kappa` the speed of mean reversion for 
        the first process that may be interpreted as the long-term factor and :math:`\\sigma_1` the respective volatility.
        The second factor :math:`X_2` may be interpreted as a short-term factor that is influenced by  :math:`W_2`
        and has drift :math:`\\mu`. :math:`X_1` and :math:`X_2` may be correlated with correlation :math:`\\rho`.
        
        Args:
            kappa (Union[float, Callable]): The speed of mean reversion for the first factor :math:`X_1`. Can be either constant or time dependent.
            sigma1 (Union[float, Callable]): The volatility of the first factor :math:`X_1`. Can be either constant or time dependent.
            mu (Union[float, Callable]): The drift of teh second factor :math:`X_2`. Can be either constant or time dependent.
            sigma2 (Union[float, Callable]): The volatility of the second factor :math:`X_2`. Can be either constant or time dependent.
            f (Union[float, Callable], optional): Deterministic function of time. Defaults to 0.
        """
        self.kappa = kappa
        self.sigma1 = sigma1
        self.mu = mu
        self.sigma2 = sigma2
        self.f = f
        self._timegrid = None

    def _to_dict(self) -> dict:
        return {'kappa': self.kappa, 'sigma1': self.sigma1,
                'mu': self.mu, 'sigma2': self.sigma2, 'f': self.f}

    def _set_timegrid(self, timegrid):
        self._timegrid = np.copy(timegrid)
        self._delta_t = self._timegrid[1:]-self._timegrid[:-1]
        self._sqrt_delta_t = np.sqrt(self._delta_t)

        self._kappa_grid = LuciaSchwartz._eval_grid(self.kappa, timegrid)
        self._sigma1_grid = LuciaSchwartz._eval_grid(self.sigma1, timegrid)
        self._mu_grid = LuciaSchwartz._eval_grid(self.mu, timegrid)
        self._sigma2_grid = LuciaSchwartz._eval_grid(self.sigma2, timegrid)
        self._mean_reversion_level_grid = LuciaSchwartz._eval_grid(self.mean_reversion_level, timegrid)
        
    def rnd_shape(self, n_sims: int, n_timepoints: int)->tuple:
        return (n_timepoints-1, n_sims, 2)


    def simulate(self, timegrid, start_value, rnd, S_only):
        """_summary_

        Args:
            timegrid (_type_): _description_
            start_value (_type_): _description_
            rnd (_type_): _description_

        Returns:
            _type_: _description_
        """
        self._set_timegrid(timegrid)
        result = np.empty((self._timegrid.shape[0], rnd.shape[1], 2))
        result[0,:] = start_value

        for i in range(self._timegrid.shape[0]-1):
            result[i+1,:] = (result[i, :] * np.exp(-self._speed_of_mean_reversion_grid[i]*self._delta_t[i])
                        + self._mean_reversion_level_grid[i]* (1 - np.exp(-self._speed_of_mean_reversion_grid[i]*self._delta_t[i])) 
                        + self._volatility_grid[i]* np.sqrt((1 - np.exp(-2*self._speed_of_mean_reversion_grid[i]*self._delta_t[i])) / (2*self._speed_of_mean_reversion_grid[i])) * rnd[i,:]
                        )
        return result

