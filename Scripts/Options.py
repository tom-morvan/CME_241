import abc
import numpy as np
from numpy.random import normal
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
                 stock: Stock):
        self.stock = stock

    @abc.abstractmethod
    def price(self, 
              price_date_index: int,
              maturity_index: int,
              strike: float):
        pass
    
    @abc.abstractmethod
    def delta(self,
              price_date_index: int,
              maturity_index: int,
              strike: float):
        pass
    
class European(Option):
    
    def __init__(self, 
                 stock: Stock):
        super().__init__(stock)
        
    @abc.abstractmethod
    def price(self, 
              price_date_index: int,
              maturity_index: int,
              strike: float):
        pass
    
    def delta(self,
              price_date_index: int,
              maturity_index: int,
              strike: float):
        pass
    
class European_Call(European):
    
    def __init__(self, 
                 stock: Stock):
        super().__init__(stock)
    
    def price(self, 
              price_date_index: int,
              maturity_index: int,
              strike: float):
        K = strike
        S = self.stock.prices[price_date_index]
        t = self.stock.timesteps[price_date_index]
        T = self.stock.timesteps[maturity_index]
        sigma = self.stock.vol[price_date_index]
        r = self.stock.rfr
        
        d1 = 1/(sigma*np.sqrt(T-t)) * (np.log(S/K) + (r + 0.5*sigma**2)*(T-t))
        d2 = d1 - sigma*np.sqrt(T)
        
        return(S*norm.cdf(d1) - K*np.exp(-r*(T-t))*norm.cdf(d2))
        
        
class European_Put(European):
    
    def __init__(self, 
                 stock: Stock):
        super().__init__(stock)
    
    def price(self,
              price_date_index: int,
              maturity_index: int,
              strike: float):
        #C = self.price_call(price_date_index, maturity_index)
        K = strike
        S = self.stock.prices[price_date_index]
        t = self.stock.timesteps[price_date_index]
        T = self.stock.timesteps[maturity_index]
        sigma = self.stock.vol[price_date_index]
        r = self.stock.rfr
        
        d1 = 1/(sigma*np.sqrt(T-t)) * (np.log(S/K) + (r + 0.5*sigma**2)*(T-t))
        d2 = d1 - sigma*np.sqrt(T)
        
        return(-S*norm.cdf(-d1) + K*np.exp(-r*(T-t))*norm.cdf(-d2))
        
        #return(C - S + K*np.exp(-r*(T-t))) #Call-put Parity
        
class American(Option):
    
    def __init__(self, 
                 stock: Stock):
        super().__init__(stock)
        
    @abc.abstractmethod
    def price(self, 
              price_date_index: int,
              maturity_index: int,
              strike: float):
        pass
    ##To Do

class American_call(American):
    
    def __init__(self, 
                 stock: Stock):
        super().__init__(stock)
        
    
    def price(self, 
              price_date_index: int,
              maturity_index: int,
              strike: float,
              N: int):
        """
        Model: CCR
        """
        K = strike
        S = self.stock.prices[price_date_index]
        t = self.stock.timesteps[price_date_index]
        T = self.stock.timesteps[maturity_index]
        sigma = self.stock.vol[price_date_index]
        r = self.stock.rfr
        #n = maturity_index - price_date_index
        #N = n*(n+1)//2
        deltaT = (T-t)/N
        
        
        # up and down factor will be constant for the tree so we calculate outside the loop
        u = np.exp(sigma * np.sqrt(deltaT))
        d = 1.0 / u
     
        #to work with vector we need to init the arrays using numpy
        fs =  np.asarray([0.0 for i in range(N + 1)])
            
        #we need the stock tree for calculations of expiration values
        fs2 = np.asarray([(S * u**j * d**(N - j)) for j in range(N + 1)])
        
        #we vectorize the strikes as well so the expiration check will be faster
        fs3 =np.asarray( [float(K) for i in range(N + 1)])
        
     
        #rates are fixed so the probability of up and down are fixed.
        #this is used to make sure the drift is the risk free rate
        a = np.exp(r * deltaT)
        p = (a - d)/ (u - d)
        oneMinusP = 1.0 - p
     
       
        # Compute the leaves, f_{N, j}
        fs[:] = np.maximum(fs2-fs3, 0.0)
              
        #calculate backward the option prices
        for i in range(N-1, -1, -1):
           fs[:-1]=np.exp(-r * deltaT) * (p * fs[1:] + oneMinusP * fs[:-1])
           fs2[:]=fs2[:]*u
          
           #Simply check if the option is worth more alive or dead
           fs[:]=np.maximum(fs[:],fs2[:]-fs3[:])
                               
        # print fs
        return fs[0]
        
        
# =============================================================================
#         up = np.exp(sigma*np.sqrt(delta_t))
#         p0 = - np.exp(-r * delta_t) / (up**2 - 1)
#         p1 = np.exp(-r * delta_t) - p0
#         
#         prices = []
#         for i in range(0,n):
#             price = max(0,S * up**(2*i - n) - K)
#             prices.append(price)
#         
#         for j in range(n-1,-1,-1):
#             for i in range(0,j):
#                   #binomial value
#                   prices[i] = p0 * prices[i+1] + p1 * prices[i] 
#                   #exercise value
#                   exercise = S * up**(2*i - j) - K
#                   prices[i] = max(prices[i], exercise)
#         return prices[0]
# =============================================================================
    ##To Do
    

    
    
if __name__ == '__main__':
    
    time = np.linspace(0,1,365)
    prices = 100 + normal(0,1,365) # random prices, no model
    vol = np.ones(365)
    rate = 0.02
    
    AAPL = Stock(prices, time, vol, rate)
    OC_AAPL = European_Call(AAPL)
    print(OC_AAPL.price(0,364,120))
    
    AC_AAPL = American_call(AAPL)
    print(AC_AAPL.price(0,364,120,2000))
    
    