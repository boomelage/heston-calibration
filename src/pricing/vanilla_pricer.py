import os
import numpy as np
import QuantLib as ql
from joblib import Parallel, delayed
from scipy.stats import norm

from pricing._quantlib_utils import _quantlib_utils  # one home for all QuantLib construction
from pricing._quantlib_config import MC_STEPS, MC_RNG, MC_NUM_PATHS, MC_SEED  # MC defaults

class vanilla_pricer:
	def __init__(self,day_count_name=None,steps=None,rng=None,numPaths=None,seed=None):
		# All QuantLib processes/engines/options are built by this helper, so a constructor-order or
		# convention change happens in pricing/_quantlib_utils.py, not in every method below.
		self.qu = _quantlib_utils(day_count_name)

		# MC engine settings: a None arg resolves to the default in _quantlib_config.
		self.steps = MC_STEPS if steps is None else steps
		self.rng = MC_RNG if rng is None else rng
		self.numPaths = MC_NUM_PATHS if numPaths is None else numPaths
		self.seed = MC_SEED if seed is None else seed

	def numpy_black_scholes(self, s, k, t, r, volatility,w):
		if w == 'call':
			w = 1
		elif w == 'put':
			w = -1
		else:
			raise ValueError("call/put flag w sould be either 'call' or 'put'")            
		d1 = (np.log(s/k)+(r+volatility**2/2)*t/365)/(volatility*np.sqrt(t/365))
		d2 = d1-volatility*np.sqrt(t/365)
		return max(0,w*(s*norm.cdf(w*d1)-k*np.exp(-r*t/365)*norm.cdf(w*d2)))
		
	def row_numpy_black_scholes(self, row):
		return self.numpy_black_scholes(
			row['spot_price'],
			row['strike_price'],
			row['days_to_maturity'],
			row['risk_free_rate'],
			row['volatility'],
			row['w']
		)
	
	def df_numpy_black_scholes(self, df):
		max_jobs = os.cpu_count() // 4
		max_jobs = max(1, max_jobs)
		return Parallel(n_jobs=max_jobs)(delayed(self.row_numpy_black_scholes)(row) for _, row in df.iterrows())


	# ------- Monte Carlo Heston model price (stochastic volatility) -------	
	def mc_heston_price(self,
		s,k,t,r,g,w,
		kappa,theta,rho,eta,v0,
		):
		calculation_date = ql.Date.todaysDate()
		engine, *_ = self.qu._mc_heston_engine(
			s, r, g, kappa, theta, rho, eta, v0,
			self.rng, self.steps, self.numPaths, self.seed, 
   			calculation_date=calculation_date)
		european_option = self.qu._european_option(w, k, t, calculation_date)
		european_option.setPricingEngine(engine)
		return max(european_option.NPV(), 0)

	def row_mc_heston_price(self,row):
		return self.mc_heston_price(
			row['spot_price'],
			row['strike_price'],
			row['days_to_maturity'],
			row['risk_free_rate'],
			row['dividend_rate'],
			row['w'],
			row['kappa'],
			row['theta'],
			row['rho'],
			row['eta'],
			row['v0']
			)

	def df_mc_heston_price(self,df):
		max_jobs = os.cpu_count() // 4
		max_jobs = max(1, max_jobs)
		return Parallel(n_jobs=max_jobs)(delayed(self.row_mc_heston_price)(row) for _, row in df.iterrows())


	# ------- Analytic Heston model price (stochastic volatility) -------
	def heston_price(self,
		s,k,t,r,g,w,
		kappa,theta,rho,eta,v0,
	):
		calculation_date = ql.Date.todaysDate()
		engine, *_ = self.qu._heston_engine(
			s, r, g, kappa, theta, rho, eta, v0, calculation_date=calculation_date)
		european_option = self.qu._european_option(w, k, t, calculation_date)
		european_option.setPricingEngine(engine)
		return max(european_option.NPV(),0)

	def row_heston_price(self,row):
		return self.heston_price(
			row['spot_price'],
			row['strike_price'],
			row['days_to_maturity'],
			row['risk_free_rate'],
			row['dividend_rate'],
			row['w'],
			row['kappa'],
			row['theta'],
			row['rho'],
			row['eta'],
			row['v0']
			)

	def df_heston_price(self, df):
		max_jobs = os.cpu_count() // 4
		max_jobs = max(1, max_jobs)
		return Parallel(n_jobs=max_jobs)(delayed(self.row_heston_price)(row) for _, row in df.iterrows())

	# ------- Bates model price (stochastic volatility & jumps) -------
	def bates_price(self,
		s,k,t,r,g,w,
		kappa,theta,rho,eta,v0,
		lambda_, nu, delta
	):
		calculation_date = ql.Date.todaysDate()
		engine, *_ = self.qu._bates_engine(
			s, r, g, kappa, theta, rho, eta, v0, lambda_, nu, delta,
			calculation_date=calculation_date)
		european_option = self.qu._european_option(w, k, t, calculation_date)
		european_option.setPricingEngine(engine)
		return max(european_option.NPV(), 0)

	def row_bates_price(self,row):
		return self.bates_price(
			row['spot_price'],
			row['strike_price'],
			row['days_to_maturity'],
			row['risk_free_rate'],
			row['dividend_rate'],
			row['w'],
			row['kappa'],
			row['theta'],
			row['rho'],
			row['eta'],
			row['v0'],
			row['lambda_'],
			row['nu'],
			row['delta']
			)

	def df_bates_price(self, df):
		max_jobs = os.cpu_count() // 4
		max_jobs = max(1, max_jobs)
		return Parallel(n_jobs=max_jobs)(delayed(self.row_bates_price)(row) for _, row in df.iterrows())