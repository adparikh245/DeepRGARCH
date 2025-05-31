# %%
import pandas as pd
import pickle
from collections import defaultdict
from src import model as mdl, utils as ut

import warnings
warnings.filterwarnings('ignore')

# %%
data_dict = ut.build_data_dict(ut.load_rv(filepath='data/rv.pkl'))
tickers = list(data_dict.keys())
model_dict = defaultdict(lambda: {})

# %%
for ticker in tickers:
    data = data_dict[ticker]['train']
    y_train, y_test, rv_train, rv_test = data['y_train'], data['y_test'], data['rv_train'], data['rv_test']
    
    garch = mdl.GARCH(prior=ut.garch_prior,cstr_fn=ut.garch_cstr_fn,data=y_train)
    garch.run()
    model_dict[ticker]['garch'] = mdl.GARCHD(pre=garch, data=[y_train, y_test])
    model_dict[ticker]['garch'].run() 
    with open('checkpoint/{}_{}_{}.pkl'.format(ticker, 'garch', n), 'wb') as f:
        pickle.dump(model_dict[ticker]['garch'], f)

    rech = mdl.RECH(prior=ut.rech_prior,data=y_train)
    rech.run() 
    model_dict[ticker]['rech'] = mdl.RECHD(pre=rech, data=[y_train, y_test])
    model_dict[ticker]['rech'].run()
    with open('checkpoint/{}_{}_{}.pkl'.format(ticker, 'rech', n), 'wb') as f:
        pickle.dump(model_dict[ticker]['rech'], f)
    
    realgarch = mdl.RealGARCH(prior=ut.realgarch_prior,data=[y_train, rv_train])
    realgarch.run()
    model_dict[ticker]['realgarch'] = mdl.RealGARCHD(pre=realgarch, data=[y_train, y_test, rv_train, rv_test])
    model_dict[ticker]['realgarch'].run()
    with open('checkpoint/{}_{}_{}.pkl'.format(ticker, 'realgarch', n), 'wb') as f:
        pickle.dump(model_dict[ticker]['realgarch'], f)

    realrech = mdl.RealRECH(prior=ut.realrech_prior,data=[y_train, rv_train])
    realrech.run()
    model_dict[ticker]['realrech'] = mdl.RealRECHD(pre=realrech, data=[y_train, y_test, rv_train, rv_test])
    model_dict[ticker]['realrech'].run()
    with open('checkpoint/{}_{}_{}.pkl'.format(ticker, 'realrech', n), 'wb') as f:
        pickle.dump(model_dict[ticker]['realrech'], f)


