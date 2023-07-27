from typing import List, Dict, Union
import json

try:
    import tensorflow as tf
    try:
        tf.config.run_functions_eagerly(False)
    except:
        pass
except:
    import warnings
    warnings.warn('Tensorflow is not installed. You cannot use the PPA Deep Hedging Pricer!')
    
import numpy as np
import sys
sys.path.append('C:/Users/doeltz/development/RiVaPy/')
import datetime as dt
from rivapy.models.base_model import BaseFwdModel
from rivapy.models import ResidualDemandForwardModel
from rivapy.instruments.ppa_specification import GreenPPASpecification
from rivapy.tools.datetools import DayCounter
from rivapy.tools.enums import DayCounterType
from rivapy.tools.datetime_grid import DateTimeGrid
from rivapy.tools.interfaces import FactoryObject

from sklearn.preprocessing import StandardScaler
#class PPAModel(Protocol):
#    def __init__(self, )


            
def _generate_lr_schedule(initial_learning_rate: float, decay_step: int, decay_rate: float,):
    return tf.keras.optimizers.schedules.InverseTimeDecay(
            initial_learning_rate=initial_learning_rate,#1e-3,
            decay_steps=decay_step,
            decay_rate=decay_rate)
    
class DeepHedgeModelTrainingParameter(FactoryObject):
    def __init__(self, epochs: int, batch_size: int, 
                 initial_learning_rate: float, 
                 decay_step: int, decay_rate: float, **kwargs):
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.kwargs = kwargs

class DeepHedgeModel(tf.keras.Model):
    def __init__(self, hedge_instruments:List[str], 
                        additional_states:List[str],
                        timegrid, 
                        regularization: float, 
                        depth: int,
                        n_neurons: int,
                        model=None,
                        **kwargs):
        super().__init__(**kwargs)
        self.hedge_instruments = hedge_instruments
        if additional_states is None:
            self.additional_states = []
        else:
            self.additional_states = additional_states
        if model is None:
            self.model = self._build_model(depth,n_neurons)
        else:
            self.model = model
        self.timegrid = timegrid
        self.regularization = regularization
        self._prev_q = None
        self._forecast_ids = None
        
    def __call__(self, x, training=True):
        return self._compute_pnl(x, training) #+ self.price
    
    def _build_model(self, depth: int, nb_neurons: int):
        inputs= [tf.keras.Input(shape=(1,),name = ins) for ins in self.hedge_instruments]
        if self.additional_states is not None:
            for state in self.additional_states:
                inputs.append(tf.keras.Input(shape=(1,),name = state))
        inputs.append(tf.keras.Input(shape=(1,),name = "ttm"))
        fully_connected_Input = tf.keras.layers.concatenate(inputs)         
        values_all = tf.keras.layers.Dense(nb_neurons,activation = "selu", 
                        kernel_initializer=tf.keras.initializers.GlorotUniform())(fully_connected_Input)       
        for _ in range(depth):
            values_all = tf.keras.layers.Dense(nb_neurons,activation = "selu", 
                        kernel_initializer=tf.keras.initializers.GlorotUniform())(values_all)            
        value_out = tf.keras.layers.Dense(len(self.hedge_instruments), activation="linear",
                        kernel_initializer=tf.keras.initializers.GlorotUniform())(values_all)
        model = tf.keras.Model(inputs=inputs, outputs = value_out)
        return model

    def _compute_pnl(self, x, training):
        pnl = tf.zeros((tf.shape(x[0])[0],))
        self._prev_q = tf.zeros((tf.shape(x[0])[0], len(self.hedge_instruments)), name='prev_q')
        for i in range(self.timegrid.shape[0]-2):
            t = [self.timegrid[-1]-self.timegrid[i]]*tf.ones((tf.shape(x[0])[0],1))/self.timegrid[-1]
            inputs = [v[:,i] for v in x]
            inputs.append(t)
            quantity = self.model(inputs, training=training)#tf.squeeze(self.model(inputs, training=training))
            for j in range(len(self.hedge_instruments)):
                pnl += tf.math.multiply((self._prev_q[:,j]-quantity[:,j]), tf.squeeze(x[j][:,i]))
            self._prev_q = quantity
        for j in range(len(self.hedge_instruments)):
            pnl += self._prev_q[:,j]* tf.squeeze(x[j][:,-1])#+ rlzd_qty[:,-1]*(tf.squeeze(power_fwd[:,-1])-self.fixed_price)
        return pnl

    def compute_delta(self, paths: Dict[str, np.ndarray],
                      t: Union[int, float]=None):
        if t is None:
            result = np.zeros((self.timegrid.shape[0], next(iter(paths.values())).shape[1], 
                              len(self.hedge_instruments)))
            for i in range(self.timegrid.shape[0]):
                result[i,:,:]=self.compute_delta(paths, i)
            return result
        if isinstance(t, int):
            inputs_ = self._create_inputs(paths, check_timegrid=True)
            inputs = [inputs_[i][:,t] for i in range(len(inputs_))]
            t = (self.timegrid[-1] - self.timegrid[t])/self.timegrid[-1]
        else:
            inputs_ = self._create_inputs(paths, check_timegrid=False)
            inputs = [inputs_[i] for i in range(len(inputs_))]
        #for k,v in paths.items():
        inputs.append(np.full(inputs[0].shape, fill_value=t))
        return self.model.predict(inputs)

    def _compute_delta_path(self, paths: Dict[str, np.ndarray]):     
        result = np.zeros((self.timegrid.shape[0], next(iter(paths.values())).shape[0]))
        for i in range(self.timegrid.shape[0]):
            result[i,:]=self.compute_delta(paths, i)
        return result
    
    def compute_pnl(self, 
                    paths: Dict[str, np.ndarray],
                    payoff: np.ndarray):
        inputs = self._create_inputs(paths)
        return payoff+self.predict(inputs)

    @tf.function
    def custom_loss(self, y_true, y_pred):
        return - self.regularization*tf.keras.backend.mean(y_pred+y_true) + tf.keras.backend.var(y_pred+y_true)
        #return tf.keras.backend.mean(tf.keras.backend.exp(-self.lamda*y_pred))

    def _create_inputs(self, paths: Dict[str, np.ndarray], check_timegrid: bool=True)->List[np.ndarray]:
        inputs = []
        if check_timegrid:
            for k in self.hedge_instruments:
                if paths[k].shape[1] != self.timegrid.shape[0]:
                    inputs.append(paths[k].transpose())
                else:
                    inputs.append(paths[k])
            for k in self.additional_states:
                if paths[k].shape[1] != self.timegrid.shape[0]:
                    inputs.append(paths[k].transpose())
                else:
                    inputs.append(paths[k])
        else:
            for k in self.hedge_instruments:
                inputs.append(paths[k])
            for k in self.additional_states:
                inputs.append(paths[k])
        return inputs

    def train(self, paths: Dict[str,np.ndarray], 
            payoff: np.ndarray, 
            lr_schedule, 
            epochs: int, batch_size: int, 
            tensorboard_log:str=None,
            verbose=0):
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule) #beta_1=0.9, beta_2=0.999)
        callbacks = []
        if tensorboard_log is not None:
            logdir = tensorboard_log#os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
            tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=0)
            callbacks.append(tensorboard_callback)
        self.compile(optimizer=optimizer, loss=self.custom_loss)
        inputs = self._create_inputs(paths)
        return self.fit(inputs, payoff, epochs=epochs, 
                            batch_size=batch_size, callbacks=callbacks, 
                            verbose=verbose, validation_split=0.1, 
                            validation_freq=5)

    def save(self, folder):
        self.model.save(folder+'/delta_model')
        params = {}
        params['regularization'] = self.regularization
        params['timegrid'] =[x for x in self.timegrid]
        params['additional_states'] = self.additional_states
        params['hedge_instruments'] = self.hedge_instruments
        with open(folder+'/params.json','w') as f:
            json.dump(params, f)

    @staticmethod
    def load(folder: str):
        with open(folder+'/params.json','r') as f:
            params = json.load(f)
        base_model = tf.keras.models.load_model(folder+'/delta_model')
        params['timegrid'] = np.array(params['timegrid'])
        return DeepHedgeModel(depth=None,n_neurons=None, model=base_model, **params)

# class PPAHedgeModel(tf.keras.Model):
#     def __init__(self, model, timegrid, 
#                         regularization: float, 
#                         fixed_price: float,  
#                         **kwargs):
#         super().__init__(**kwargs)
#         self.model = model
#         #self.price = tf.Variable([0], trainable=True, dtype ="float32")
#         self.timegrid = timegrid
#         self.lamda = regularization
#         self._prev_q = None
#         self._output_scaler = None
#         self.fixed_price = fixed_price
#         self._forecast_ids = None
        
#     def __call__(self, x, training=True):
#         return self._compute_pnl(x, training) #+ self.price
    
    
#     def _compute_pnl(self,x, training):
#         power_fwd = x[0]
#         forecast = x[1]
#         rlzd_qty = x[-1]
        
#         pnl = 0.0
#         self._prev_q = tf.zeros((tf.shape(power_fwd)[0]), name='prev_q')
#         for i in range(self.timegrid.shape[0]-2):
#             t = [self.timegrid[-1]-self.timegrid[i]]*tf.ones((tf.shape(power_fwd)[0],1))/self.timegrid[-1]
#             inputs = [v[:,i] for v in x[:-1]]
#             inputs.append(t)
#             #quantity = tf.squeeze(self.model([power_fwd[:,i], forecast[:,i], t], training=training))
#             quantity = tf.squeeze(self.model(inputs, training=training))
#             #power_fwd
#             pnl = pnl + tf.math.multiply((self._prev_q-quantity), tf.squeeze(power_fwd[:,i]))
#             self._prev_q = quantity
#         pnl = pnl + self._prev_q* tf.squeeze(power_fwd[:,-1]) + rlzd_qty[:,-1]*(tf.squeeze(power_fwd[:,-1])-self.fixed_price)
#             # (-self._prev_q+forecast[:,-1]) * tf.squeeze(power_fwd[:,-1])-forecast[:,-1]*self.strike
#         return pnl #+ self.price

#     def compute_delta(self, fwd_prices: np.ndarray, 
#                         forecasts: Dict[str, np.ndarray], 
#                         t: Union[int, float]):
#         if self._forecast_ids is None:
#             raise('Model does not contain any forecast ids. Please train model first')
#         inputs_ = self._create_inputs(fwd_prices, forecasts, rlzd_qty=None,)
#         if isinstance(t, int):
#             inputs = [inputs_[i][:,t] for i in range(len(inputs_)-1)]
#             t = self.timegrid[-1]-self.timegrid[t]
#         else:
#             inputs = [inputs_[i] for i in range(len(inputs_)-1)]
#         inputs.append(t*np.ones((tf.shape(fwd_prices)[0],1)))
#         return self.model.predict(inputs)

#     def compute_pnl(self, fwd_prices: np.ndarray, forecasts: Dict[str, np.ndarray], rlzd_qty: np.ndarray=None):
#         inputs = self._create_inputs(fwd_prices, forecasts, rlzd_qty=rlzd_qty,)
#         return self.predict(inputs)

#     @tf.function
#     def custom_loss(self, y_true, y_pred):
#         return - self.lamda*tf.keras.backend.mean(y_pred) + tf.keras.backend.var(y_true-y_pred)
#         #return tf.keras.backend.mean(tf.keras.backend.exp(-self.lamda*y_pred))

#     def _create_inputs(self,  paths_fwd_price, paths_forecasts, rlzd_qty):
#         inputs = [paths_fwd_price]
#         if self._forecast_ids is None:
#             self._forecast_ids = list(paths_forecasts.keys())
#         inputs = [paths_fwd_price]
#         self._forecast_ids = list(paths_forecasts.keys())
#         for k in self._forecast_ids:
#             inputs.append(paths_forecasts[k])
#         inputs.append(rlzd_qty)
#         return inputs

#     def train(self, paths_fwd_price: np.ndarray, paths_forecasts: np.ndarray, 
#             rlzd_qty: np.ndarray, lr_schedule, 
#             epochs: int, batch_size: int, tensorboard_log:str=None, verbose=0):
#         optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule) #beta_1=0.9, beta_2=0.999)
#         callbacks = []
#         if tensorboard_log is not None:
#             logdir = tensorboard_log#os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
#             tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
#             callbacks.append(tensorboard_callback)
#         self.compile(optimizer=optimizer, loss=self.custom_loss)
#         y = np.zeros((paths_fwd_price.shape[0],1))
#         inputs = self._create_inputs(paths_fwd_price, paths_forecasts, rlzd_qty,)
#         return self.fit(inputs, y, epochs=epochs, 
#                             batch_size=batch_size, callbacks=callbacks, verbose=verbose)

    
def _build_model(depth, nb_neurons, regions: List[str] = None):
    inputs= [tf.keras.Input(shape=(1,),name = "power_fwd_price")]
    if regions is None:
        inputs.append(tf.keras.Input(shape=(1,),name = "forecast"))
    else:
        for r in regions:
            inputs.append(tf.keras.Input(shape=(1,),name = "forecast_"+r))
    inputs.append(tf.keras.Input(shape=(1,),name = "t"))
    fully_connected_Input = tf.keras.layers.concatenate(inputs)         
    values_all = tf.keras.layers.Dense(nb_neurons,activation = "selu", 
                    kernel_initializer=tf.keras.initializers.GlorotUniform())(fully_connected_Input)       
    for _ in range(depth):
        values_all = tf.keras.layers.Dense(nb_neurons,activation = "selu", 
                    kernel_initializer=tf.keras.initializers.GlorotUniform())(values_all)            
    value_out = tf.keras.layers.Dense(1, activation="linear",
                    kernel_initializer=tf.keras.initializers.GlorotUniform())(values_all)
    model = tf.keras.Model(inputs=inputs, outputs = value_out)
    return model

# class PricingResults:
#     def __init__(self, hedge_model: DeepHedgeModel, timegrid: DateTimeGrid,
#                 fwd_prices: np.ndarray, forecasts: Dict[str, np.ndarray], rlzd_qty: np.ndarray):
#         self.hedge_model = hedge_model
#         self.timegrid = timegrid
#         self.fwd_prices = fwd_prices
#         self.forecasts = forecasts
#         self.rlzd_qty = rlzd_qty

def _validate(val_date: dt.datetime,
            green_ppa: GreenPPASpecification,
            power_wind_model: ResidualDemandForwardModel):
    #if green_ppa.technology != power_wind_model.get_technology():
    #    raise Exception('PPA technology ' + green_ppa.technology + 
    #                    ' does not equal residual demand technology model ' 
    #                    + power_wind_model.get_technology())
    if green_ppa.n_deliveries() > 1:
        raise Exception('Pricer for more than one delivery not yet implemented.')


# def price( val_date: dt.datetime,
#             green_ppa: GreenPPASpecification,
#             power_wind_model: ResidualDemandForwardModel, 
#             depth: int, nb_neurons: int, 
#             n_sims: int, regularization: float, 
#             epochs: int,
#             verbose: bool=0,
#             tensorboard_logdir: str=None, initial_lr: float = 1e-4, 
#             batch_size: int = 100, decay_rate: float=0.7, decay_step=100, seed: int = 42):
#     """Price a green PPA using deeep hedging

#     Args:
#         val_date (dt.datetime): Valuation date.
#         green_ppa (GreenPPASpecification): Specification of a green PPA.
#         power_wind_model (ResidualDemandForwardModel): The model modeling power prices and renewable quantities.
#         depth (int): Number of layers of neural network.
#         nb_neurons (int): Number of activation functions. 
#         n_sims (int): Number of paths used as input for network training.
#         regularization (float): The regularization term entering the loss: Loss is defined by -E[pnl] + regularization*Var(pnl)
#         epochs (int): Number of epochs for network training.
#         verbose (bool, optional): Verbosity level (0, 1 or 2). Defaults to 0.
#         tensorboard_logdir (str, optional): Pah to tensorboard log, if None, no log is written. Defaults to None.
#         initial_lr (float, optional): Initial learning rate. Defaults to 1e-4.
#         batch_size (int, optional): The batch size. Defaults to 100.
#         decay_rate (float, optional): Decay of learning rate after each epoch. Defaults to 0.7.
#         seed (int, optional): Seed that is set to make results reproducible. Defaults to 42.

#     Returns:
#         _type_: _description_
#     """
#     #print(locals())
#     tf.keras.backend.set_floatx('float32')

#     _validate(val_date, green_ppa,power_wind_model)
#     ppa_schedule = green_ppa.get_schedule()
#     if ppa_schedule[-1] <= val_date:
#         return None
#     tf.random.set_seed(seed)
#     timegrid = DateTimeGrid(start=val_date, end=ppa_schedule[-1], freq='1H', closed=None)
#     np.random.seed(seed+123)
#     rnd = np.random.normal(size=power_wind_model.rnd_shape(n_sims, timegrid.timegrid.shape[0]))
#     fwd_prices, forecasts = power_wind_model.simulate(timegrid, rnd)
#     rlzd_qty = power_wind_model.compute_rlzd_qty(green_ppa.location, forecasts)
#     fwd_prices = np.squeeze(fwd_prices.transpose())
    
#     #print(fwd_prices.mean(axis=0))
#     # dirty hack to test!!!
#     regions =  list(forecasts.keys())
#     forecasts =  {r: np.squeeze(forecasts[r].transpose()) for r in regions}
#     #######################
#     model = _build_model(depth, nb_neurons, regions)
    
#     hedge_model = PPAHedgeModel(model, timegrid.timegrid, regularization, fixed_price=green_ppa.fixed_price)
#     lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#             initial_learning_rate=initial_lr,#1e-3,
#             decay_steps=decay_step*fwd_prices.shape[0]/batch_size,
#             decay_rate=decay_rate)
#     hedge_model.train(fwd_prices, forecasts, rlzd_qty, lr_schedule, epochs, batch_size, 
#                         tensorboard_log=tensorboard_logdir, verbose=verbose)
#     return PricingResults(hedge_model, timegrid, fwd_prices, forecasts, rlzd_qty)


class GreenPPADeepHedgingPricer:
    class PricingResults:
        def __init__(self, hedge_model: DeepHedgeModel, paths: np.ndarray, sim_results, payoff):
            self.hedge_model = hedge_model
            self.paths = paths
            self.sim_results = sim_results
            self.payoff = payoff

    @staticmethod
    def _compute_points(val_date: dt.datetime, 
                        green_ppa: GreenPPASpecification, 
                        forecast_hours: List[int]):
        ppa_schedule = green_ppa.get_schedule()
        if ppa_schedule[-1] <= val_date:
            return None
        timegrid = DateTimeGrid(start=val_date, end=ppa_schedule[-1], freq='1H', closed=None, daycounter=DayCounterType.Act365Fixed)
        dc = DayCounter(DayCounterType.Act365Fixed)
        fwd_expiries = [dc.yf(val_date, d) for d in ppa_schedule if d>val_date]
        forecast_points = [i for i in range(len(timegrid.dates)) if timegrid.dates[i].hour in forecast_hours]
        return timegrid, fwd_expiries, forecast_points


    @staticmethod
    def compute_payoff(n_sims: int, 
                       hedge_ins: Dict[str, np.ndarray], 
                       additional_states: Dict[str, np.ndarray], 
                       green_ppa: GreenPPASpecification):
        payoff = np.zeros((n_sims,))
        for k,v in hedge_ins.items(): #TODO: We assume that each hedge instruments corresponds to the spot price at the last time step. Make this more explicit!
            expiry = k.split('_')[-1]
            forecast_key = green_ppa.location+'_'+expiry
            payoff += (v[-1,:] -green_ppa.fixed_price)*(additional_states[forecast_key][-1,:])
        return payoff

    @staticmethod
    def generate_paths(green_ppa: GreenPPASpecification,
                power_wind_model: ResidualDemandForwardModel, 
                 initial_forecasts: dict,
                power_fwd_prices: np.ndarray,
                n_sims: int, 
                timegrid: DateTimeGrid=None,):
        if timegrid is None:
             timegrid, expiries, _ = GreenPPADeepHedgingPricer._compute_points(val_date, green_ppa, [0])
        rnd = np.random.normal(size=power_wind_model.rnd_shape(n_sims, timegrid.timegrid.shape[0]))
        return power_wind_model.simulate(timegrid.timegrid, rnd, expiries=expiries, 
                                                       initial_forecasts=initial_forecasts,
                                                       power_fwd_prices=power_fwd_prices)

    @staticmethod
    def price( val_date: dt.datetime,
                green_ppa: GreenPPASpecification,
                power_wind_model: ResidualDemandForwardModel, 
                initial_forecasts: dict,
                power_fwd_prices: np.ndarray,
                forecast_hours: List[int],
                depth: int, 
                nb_neurons: int, 
                n_sims: int, 
                regularization: float, 
                epochs: int,
                timegrid: DateTimeGrid=None,
                verbose: bool=0,
                tensorboard_logdir: str=None, 
                initial_lr: float = 1e-4, 
                batch_size: int = 100, 
                decay_rate: float=0.7, 
                decay_steps: int = 100_000,
                seed: int = 42,
                additional_states=None, 
                paths: Dict[str, np.ndarray] = None):
        """Price a green PPA using deeep hedging

        Args:
            val_date (dt.datetime): Valuation date.
            green_ppa (GreenPPASpecification): Specification of a green PPA.
            power_wind_model (ResidualDemandForwardModel): The model modeling power prices and renewable quantities.
            depth (int): Number of layers of neural network.
            nb_neurons (int): Number of activation functions. 
            n_sims (int): Number of paths used as input for network training.
            regularization (float): The regularization term entering the loss: Loss is defined by -E[pnl] + regularization*Var(pnl)
            timegrid (DateTimeGrid, optional): Timegrid used for simulation and hedging. If None, an hourly timegrid is used. Defaults to None.
            epochs (int): Number of epochs for network training.
            verbose (bool, optional): Verbosity level (0, 1 or 2). Defaults to 0.
            tensorboard_logdir (str, optional): Pah to tensorboard log, if None, no log is written. Defaults to None.
            initial_lr (float, optional): Initial learning rate. Defaults to 1e-4.
            batch_size (int, optional): The batch size. Defaults to 100.
            decay_rate (float, optional): Decay of learning rate after each epoch. Defaults to 0.7.
            seed (int, optional): Seed that is set to make results reproducible. Defaults to 42.

        Returns:
            _type_: _description_
        """
        #print(locals())
        if paths is None and power_wind_model is None:
            raise Exception('Either paths or a power wind model must be specified.')
        tf.keras.backend.set_floatx('float32')

        #_validate(val_date, green_ppa,power_wind_model)
        if green_ppa.udl not in power_wind_model.udls():
            raise Exception('Underlying ' + green_ppa.udl + ' not in underlyings of the model ' + str(power_wind_model.udls()))
        tf.random.set_seed(seed)
        np.random.seed(seed+123)
        timegrid, expiries, forecast_points = GreenPPADeepHedgingPricer._compute_points(val_date, green_ppa, forecast_hours)
        if len(expiries) == 0:
            return None
        rnd = np.random.normal(size=power_wind_model.rnd_shape(n_sims, timegrid.timegrid.shape[0]))
        simulation_results = power_wind_model.simulate(timegrid.timegrid, rnd, expiries=expiries, 
                                                       initial_forecasts=initial_forecasts,
                                                       power_fwd_prices=power_fwd_prices)
        
        hedge_ins = {}
        for i in range(len(expiries)):
            key = BaseFwdModel.get_key(green_ppa.udl, i)
            hedge_ins[key] = simulation_results.get(key, forecast_points)
        additional_states_ = {}
        for i in range(len(expiries)):
            key = BaseFwdModel.get_key(green_ppa.location, i)
            additional_states_[key] = simulation_results.get(key, forecast_points)
        if additional_states is not None:
            for a in additional_states:
                for i in range(len(expiries)):
                    key = BaseFwdModel.get_key(a, i)
                    additional_states_[key] = simulation_results.get(key, forecast_points)
        
        hedge_model = DeepHedgeModel(list(hedge_ins.keys()), list(additional_states_.keys()), timegrid.timegrid, 
                                        regularization=regularization,depth=depth, n_neurons=nb_neurons)
        paths = {}
        paths.update(hedge_ins)
        paths.update(additional_states_)
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=initial_lr,#1e-3,
                decay_steps=decay_steps,
                decay_rate=decay_rate, 
                staircase=True)

        payoff = GreenPPADeepHedgingPricer.compute_payoff(n_sims, hedge_ins, additional_states_, green_ppa)  
        
        hedge_model.train(paths, payoff,lr_schedule, epochs=epochs, batch_size=batch_size, tensorboard_log=tensorboard_logdir, verbose=verbose)
        return GreenPPADeepHedgingPricer.PricingResults(hedge_model, paths=paths, sim_results=simulation_results, payoff=payoff)

if __name__=='__main__':
    from rivapy.models import OrnsteinUhlenbeck
    from rivapy.models.residual_demand_model import SmoothstepSupplyCurve
    from rivapy.models.residual_demand_fwd_model import WindPowerForecastModel, MultiRegionWindForecastModel, ResidualDemandForwardModel
    
    days = 2
    timegrid = np.linspace(0.0, days*1.0/365.0, days*24)
    forward_expiries = [timegrid[-10] +i/(365.0*24.0) for i in range(4)]#
    #forward_expiries = [timegrid[-1] + i for i in range(4)]
    n_sims = 10_000
    wind_onshore = WindPowerForecastModel(region='Onshore', speed_of_mean_reversion=0.001, volatility=3.30)
    wind_offshore = WindPowerForecastModel(region='Offshore', speed_of_mean_reversion=0.001, volatility=3.30)
    regions = [ MultiRegionWindForecastModel.Region( 
                                        wind_onshore,
                                        capacity=1000.0,
                                        rnd_weights=[0.7,0.3]
                                    ),
            MultiRegionWindForecastModel.Region( 
                                        wind_offshore,
                                        capacity=500.0,
                                        rnd_weights=[0.3,0.7]
                                    )
            
            ]
    wind = MultiRegionWindForecastModel('Wind_Germany', regions)
    highest_price = OrnsteinUhlenbeck(speed_of_mean_reversion=1.0, volatility=0.01, mean_reversion_level=1.0)
    supply_curve = SmoothstepSupplyCurve(1.0, 0)
    rd_model = ResidualDemandForwardModel(wind_power_forecast=wind, highest_price_ou_model= highest_price, 
                                      supply_curve=supply_curve, max_price=1.0, power_name= 'Power_Germany')
    val_date = dt.datetime(2023,1,1)
    strike = 0.3 #0.22
    spec = GreenPPASpecification(udl='Power_Germany',
                                technology = 'Wind',
                                location = 'Onshore',
                                schedule = [val_date + dt.timedelta(days=2)], 
                                fixed_price=strike,
                                max_capacity = 1.0)
    price(val_date, spec, rd_model, initial_forecasts={'Onshore': [0.8, 0.7,0.6,0.5],
                                                          'Offshore': [0.6,0.6,0.6,0.6]},
            forecast_hours=[6, 10, 14, 18],
          depth=3, nb_neurons=32, n_sims=10_000, regularization=0.0,
          epochs=100, verbose=1, tensorboard_logdir = None, initial_lr=1e-3, 
          batch_size=100, decay_rate=0.8, seed=42)