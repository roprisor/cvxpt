import os
import sys
sys.path.insert(0, os.path.abspath('..'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cvxportfolio as cp

plotdir='../../portfolio/plots/'
datadir='../../data/'

sigmas=pd.read_csv(datadir+'ff_sigmas.csv.gz',index_col=0,parse_dates=[0])
returns=pd.read_csv(datadir+'ff_returns.csv.gz',index_col=0,parse_dates=[0])
volumes=pd.read_csv(datadir+'ff_volumes.csv.gz',index_col=0,parse_dates=[0])

w_b = pd.Series(index=returns.columns, data=1)
w_b.USDOLLAR = 0.
w_b/=sum(w_b)

start_t="2016-04-01"
end_t="2018-12-31"

simulated_tcost = cp.TcostModel(half_spread=0.0005/2., nonlin_coeff=1., sigma=sigmas, volume=volumes)
simulated_hcost = cp.HcostModel(borrow_costs=0.0001)
simulator = cp.MarketSimulator(returns, costs=[simulated_tcost, simulated_hcost],
                               market_volumes=volumes, cash_key='USDOLLAR')

estimate_data = pd.HDFStore(datadir+'ff_model.h5')

optimization_tcost = cp.TcostModel(half_spread=0.0005/2., nonlin_coeff=1.,
                                   sigma=estimate_data.sigma_estimate,
                                   volume=estimate_data.volume_estimate)
optimization_hcost=cp.HcostModel(borrow_costs=0.0001)

risk_model = cp.FactorModelSigma(estimate_data.exposures, estimate_data.factor_sigma, estimate_data.idyos)

results={}

results_pareto={}

policies={}
gamma_risks_pareto=[100, 500]
gamma_tcosts_pareto=[6]
gamma_holdings=[1]
for gamma_risk in gamma_risks_pareto:
    for gamma_tcost in gamma_tcosts_pareto :
        for gamma_holding in gamma_holdings:
            policies[(gamma_risk, gamma_tcost, gamma_holding)] = \
          cp.SinglePeriodOpt(estimate_data.return_estimate,
                             [gamma_risk*risk_model,gamma_tcost*optimization_tcost,\
                                       gamma_holding*optimization_hcost],
                                [cp.LeverageLimit(3)])

import warnings
warnings.filterwarnings('ignore')
results_pareto.update(dict(zip(policies.keys(),
                               simulator.run_multiple_backtest(1E8*w_b, start_time=start_t, end_time=end_t,
                                                               policies=policies.values(), parallel=True))))