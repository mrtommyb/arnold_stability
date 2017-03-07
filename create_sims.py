
# coding: utf-8

# In[227]:

from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats

get_ipython().magic(u'matplotlib inline')


# for each planet we need to calculate
# * mass
# * density
# * semimajor
# * eccentricity
# * inclination
# * omega 
# * OMEGA 
# * mean anomaly

# In[281]:

# specify star and planet parameters

# star
star = dict(
    mass_msun=[0.0802,0.0073],
    radius_rsun=[0.117,0.0036]
)

# planet b
planetb = dict(
    period_days=[1.51087081,0.00000060],
    t0=[7322.51736,0.00010],
    impact=[0.126,0.090],
    mass_mearth=[0.85,0.72],
    ecc_max=[0.2], # 5-sigma upper limit
    td_percent=[0.7266,0.0088],
)

# planet c
planetc = dict(
    period_days=[2.4218233,0.0000017],
    t0=[7282.80728,0.00019],
    impact=[0.161,0.080],
    mass_mearth=[1.38,0.61],
    ecc_max=[0.2], # 5-sigma upper limit
    td_percent=[0.687,0.010],
)

# planet d
planetd = dict(
    period_days=[4.049610,0.000063],
    t0=[7670.14165,0.00035],
    impact=[0.17,0.11],
    mass_mearth=[0.41,0.27],
    ecc_max=[0.175], # 5-sigma upper limit
    td_percent=[0.367,0.017],
)

# planet e
planete = dict(
    period_days=[6.099615,0.000011],
    t0=[7660.37859,0.00026],
    impact=[0.12,0.10],
    mass_mearth=[0.62,0.58],
    ecc_max=[0.2], # 5-sigma upper limit
    td_percent=[0.519,0.026],
)

# planet f
planetf = dict(
    period_days=[9.206690,0.000015],
    t0=[7671.39767,0.00023],
    impact=[0.382,0.035],
    mass_mearth=[0.68,0.18],
    ecc_max=[0.12], # 5-sigma upper limit
    td_percent=[0.673,0.023],
)

# planet g
planetg = dict(
    period_days=[12.35294,0.00012],
    t0=[7665.34937,0.00021],
    impact=[0.421,0.031],
    mass_mearth=[1.34,0.88],
    ecc_max=[0.12], # 5-sigma upper limit
    td_percent=[0.782,0.027],
)

# planet h
planeth = dict(
    period_days_uniform=[14,35],
    t0=[7662.55463,0.00056],
    impact=[0.45,0.3],
    mass_mearth=[0.4,1.0],
    ecc_max=[0.3], # 5-sigma upper limit
    td_percent=[0.353,0.0326],
)





# In[ ]:




# In[292]:

def calc_mercury_parameters(pdicts, sdict, size=1):

    # stellar radius
    sradius_rsun = _get_property(sdict['radius_rsun'][0], sdict['radius_rsun'][1], 0.0, 100.0, size=size)
    
    # stellar mass
    smass_msun = _get_property(sdict['mass_msun'][0], sdict['mass_msun'][1], 0.0, 100.0, size=size)
    
    nplanets = len(pdicts)
    mercury_params = []
    for pdict in pdicts:
        mercury_params.append(_calc_planet_parameters(pdict, sradius_rsun, smass_msun, size=1))
    
    if size == 1:
        mercury_params = np.reshape(mercury_params, [nplanets, 8])
    return mercury_params

def _calc_planet_parameters(pdict, sradius_rsun, smass_msun, size=1):
      
    # mass 
    pmass_mearth = _get_property(pdict['mass_mearth'][0], pdict['mass_mearth'][1], 0.0, 5.0, size=size)
    pmass_msun = pmass_mearth * 3.003467E-6
    
    # density
    rprs = (_get_property(pdict['td_percent'][0], pdict['td_percent'][1], 0.0, 50., size=size)/100.)**0.5
    pradius_rsun = (rprs * sradius_rsun)
    pdensity_cgs = (pmass_msun * 1.989E33) / ((4./3.) *np.pi* (pradius_rsun * 69.57E9)**3) 
    
    # semimajor
    sdensity_cgs = (smass_msun * 1.989E33) / ((4./3.) * np.pi* (sradius_rsun * 69.57E9)**3)
    
    if 'period_days' in pdict.keys():
        pperiod_days = _get_property(pdict['period_days'][0], pdict['period_days'][1], 0.0, 10000.0, size=size)
    elif 'period_days_uniform' in pdict.keys():
        pperiod_days = np.random.uniform(pdict['period_days_uniform'][0], pdict['period_days_uniform'][1], size=size)
    else:
        raise 'period is missing'
    ars = get_ar(sdensity_cgs, pperiod_days)
    semimajor_au = ars * sradius_rsun * 0.00464913034
    
    # ecc
    ecc = np.random.uniform(0.0, pdict['ecc_max'], size=size)
    
    # inclination
    b = _get_property(pdict['impact'][0], pdict['impact'][1], 0.0, 1.0, size=size)
    inc = np.degrees(np.arccos(b / ars))
    
    # omega
    omega = np.random.rand(size) * 360
    
    # OMEGA
    OMEGA = np.random.rand(size) * 360 
    
    # meananomaly
    t0 = np.random.normal(pdict['t0'][0], pdict['t0'][1], size=size)
    meananomaly = (t0 % pperiod_days) / pperiod_days * 360
    
    
    return pmass_msun, pdensity_cgs, semimajor_au, ecc, inc, omega, OMEGA, meananomaly

def _get_property(mu, sigma, lower, upper, size):
    X = stats.truncnorm.rvs(
    (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma, size=size)
    return X


def get_ar(rho,period):
    """ gets a/R* from period and mean stellar density"""
    G = 6.67E-11
    rho_SI = rho * 1000.
    tpi = 3. * np.pi
    period_s = period * 86400.
    part1 = period_s**2 * G * rho_SI
    ar = (part1 / tpi)**(1./3.)
    return ar



# In[297]:

pdicts = [planetb, planetc, planetd, planete, planetf, planetg, planeth]
q = calc_mercury_parameters(pdicts, star, size=1)


# In[298]:

outstr = r''')O+_06 Big-body initial data  (WARNING: Do not delete this line!!)
) Lines beginning with ) are ignored.
)---------------------------------------------------------------------
style (Cartesian, Asteroidal, Cometary) = Ast
epoch (in days) = 0
)---------------------------------------------------------------------
 PL1  m={}   d={}
 {} {} {} {} {} {}  0. 0. 0.
 PL2  m={}   d={}
 {} {} {} {} {} {}  0. 0. 0.
 PL3  m={}   d={}
 {} {} {} {} {} {}  0. 0. 0.
 PL4  m={}   d={}
 {} {} {} {} {} {}  0. 0. 0.
 PL5  m={}   d={}
 {} {} {} {} {} {}  0. 0. 0.
 PL6  m={}   d={}
 {} {} {} {} {} {}  0. 0. 0.
 PL7  m={}   d={}
 {} {} {} {} {} {}  0. 0. 0.'''.format(*q.flatten())
print(outstr)


# In[ ]:




# In[ ]:




# In[ ]:



