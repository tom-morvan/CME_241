import abc
import numpy as np
from scipy.stats import norm

class Stock:
    
    def __init__(self, 
                 prices: np.ndarray, 
                 timesteps: np.ndarray,
                 vol: np.ndarray,
                 rfr: float):
        self.timesteps = timesteps
        self.prices = prices
        self.vol = vol
        self.rfr = rfr
        
class Option:
    
    def __init__(self, 
                 stock: Stock, 
                 strike: float):
        self.stock = stock
        self.strike = strike
        

    @abc.abstractmethod
    def price_call(self, 
                   price_date_index: int,
                   maturity_index: int):
        pass
    
    @abc.abstractmethod
    def price_put(self, 
                   price_date_index: int,
                   maturity_index: int):
        pass
    
class European(Option):
    
    def __init__(self, 
                 stock: Stock, 
                 strike: float):
        super.__init__(stock, strike)
    
    def price_call(self, 
                   price_date_index: int,
                   maturity_index: int):
        stock = self.stock
        K = self.strike
        S = stock.prices[price_date_index]
        t = stock.timesteps[price_date_index]
        T = stock.timesteps[maturity_index]
        sigma = stock.vol[price_date_index]
        r = stock.rfr
        
        d1 = 1/(sigma*np.sqrt(T-t)) * (np.log(S/K) + (r + 0.5*sigma**2)*(T-t))
        d2 = d1 - sigma*np.sqrt(T)
        
        return(S*norm.cdf(d1) - K*np.exp(-r*(T-t))*norm.cdf(d2))
        
    def price_put(self,
                  price_date_index: int,
                  maturity_index: int):
        C = self.price_call(price_date_index, maturity_index)
        K = self.stock.strike
        S = self.stock.prices[price_date_index]
        t = self.stock.timesteps[price_date_index]
        T = self.stock.timesteps[maturity_index]
        r = self.stock.rfr
        
        return(C - S + K*np.exp(-r*(T-t)))
        
class American(Option):
    
    def __init__(self, 
                 stock: Stock, 
                 strike: float):
        super.__init__(stock, strike)
        
