import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

import warnings
import re

from scipy import stats
from scipy.stats import norm, lognorm, beta,expon, gamma, weibull_min, kstest, anderson
from tqdm.notebook import tqdm


#################################################################################################################################################################"

#function for class results generation

#function for class results generation
def lognormal_class_stats2(i_class_n, data, d):
 
    N_data = len(data)
    data_min = np.min(data)
    data_max = np.max(data)

    step = (data_max - data_min) / (i_class_n - 0.2)
    class_rep_vals = np.zeros(i_class_n + 1)
    lnx = np.zeros(i_class_n + 1)
    mu_lnx = np.zeros(i_class_n + 1)
    fi_50 = np.zeros(i_class_n + 1)
    i_vals = np.zeros(i_class_n + 1)

    class_rep_vals[0] = data_min + 0.9 * step
    lnx[0] = np.log(class_rep_vals[0] - d)

    for i in range(1, i_class_n + 1):
        class_rep_vals[i] = class_rep_vals[i - 1] + step
        lnx[i] = np.log(class_rep_vals[i] - d)

    freq, _ = np.histogram(data, bins=[-np.inf] + class_rep_vals.tolist()[0:i_class_n] + [np.inf])

    i_vals[0] = freq[0] + 1
    fi_50[0] = (i_vals[0] - 0.3) / (N_data + 0.4)
    mu_lnx[0] = norm.ppf(fi_50[0])

    for i in range(1, i_class_n + 1):
        i_vals[i] = freq[i] + i_vals[i - 1]
        fi_50[i] = (i_vals[i] - 0.3) / (N_data + 0.4)
        mu_lnx[i] = norm.ppf(fi_50[i])

    df = pd.DataFrame({
        'Classes': class_rep_vals,
        'Freq_absol': freq,
        'i': i_vals,
        'Fi_50pct': fi_50,
        'lnx': lnx,
        'mu_lnx': mu_lnx
    })

    model = smf.ols('mu_lnx ~ lnx', data=df.iloc[0:i_class_n]).fit()
    slope = model.params[1]
    intercept = model.params[0]
    r2 = model.rsquared

    slnx = 1 / slope
    mulnx = -intercept * slnx
    scale = np.exp(mulnx)

    p1 = lognorm.ppf(0.01, s=slnx, scale=scale)
    p50 = lognorm.ppf(0.5, s=slnx, scale=scale)
    p99 = lognorm.ppf(0.99, s=slnx, scale=scale)

    return {
        i_class_n, r2,slnx,mulnx,p1,p50, p99
    }

def lognormal_class_stats(i_class_n,data):
    
    d=0
    data_max= np.max(data)
    data_min = np.min(data)
    
    N_data = len(data)
    step = (data_max - data_min) / (i_class_n - 0.2)
    class_rep_vals = np.zeros(i_class_n + 1)
    lnx = np.zeros(i_class_n + 1)
    mu_lnx = np.zeros(i_class_n + 1)
    fi_50 = np.zeros(i_class_n + 1)
    i_vals = np.zeros(i_class_n + 1)

    val0_weight = 0.9 #on the excel file the value is 0.9
    class_rep_vals[0] = data_min + val0_weight * step
    lnx[0] = np.log(class_rep_vals[0] - d)

    # Frequency calculation
    for i in range(1, i_class_n + 1):
        class_rep_vals[i] = class_rep_vals[i - 1] + step
        lnx[i] = np.log(class_rep_vals[i] - d)

    freq, _ = np.histogram(data, bins=[-np.inf] + class_rep_vals.tolist()[0:i_class_n] + [np.inf])

    i_vals[0] = freq[0] 
    fi_50[0] = (i_vals[0] - 0.3) / (N_data + 0.4)
    mu_lnx[0] = norm.ppf(fi_50[0])

    for i in range(1, i_class_n +1):
        i_vals[i] = freq[i] + i_vals[i - 1]
        fi_50[i] = (i_vals[i] - 0.3) / (N_data + 0.4)
        mu_lnx[i] = norm.ppf(fi_50[i])

    df = pd.DataFrame({
        'Classes': class_rep_vals,
        'Freq_absol': freq,
        'i': i_vals,
        'Fi_50pct': fi_50,
        'lnx': lnx,
        'mu_lnx': mu_lnx
    })
    #print(df)
    model = smf.ols('mu_lnx ~ lnx', data=df.iloc[0:i_class_n]).fit()
    #print(f"model is : {model.rsquared}")
    slope = model.params.iloc[1]
    intercept = model.params.iloc[0]
    r2=model.rsquared

    slnx = 1 / slope
    mulnx = -intercept * slnx
    scale = np.exp(mulnx)

    p1 = lognorm.ppf(0.01, s=slnx, scale=scale)
    p50 = lognorm.ppf(0.5, s=slnx, scale=scale)
    p99 = lognorm.ppf(0.99, s=slnx, scale=scale)

    # Define percentiles to compare
    percentiles = [0.01, 0.5, 0.99]

    # Empirical percentiles (from data)
    empirical_percentiles = np.percentile(data, [p * 100 for p in percentiles])

    # Fitted law percentiles (from lognormal)
    fitted_percentiles = lognorm.ppf(percentiles, s=slnx, scale=scale)
    relative_diff = 100*(fitted_percentiles - empirical_percentiles) / empirical_percentiles
    step_percentiles = 5
    upper_bounnd_perct= 100
    lower_bound_perct = 0
    #evaluating MSE of law_estimated vs input data array
    empirical = np.percentile(data, np.arange(lower_bound_perct, upper_bounnd_perct, step_percentiles)) #reference is the percentiles from 0 to 100 with a step of perct_step from input data array
    fitted_perc=lognorm.ppf(np.arange(lower_bound_perct, upper_bounnd_perct, step_percentiles) / 100, s=slnx, scale=scale) #estimated value at each point x corresponding to the class representative values
    mse = np.mean(( empirical- fitted_perc) ** 2)
    mean_relat_diff=np.mean(abs(100*(fitted_perc - empirical) / empirical))
    #print(mse)
    # Print results side-by-side
    return {
        'i_class_n': i_class_n,
        'R2': r2,
        'slnx': slnx,
        'mulnx': mulnx,
        'fitted_percentiles': fitted_percentiles,
        'empirical_percentiles': empirical_percentiles,
        'MSE':mse,
        'Mean relat diff' : mean_relat_diff,
        'relative_diff': relative_diff
    }

def normal_class_stats(i_class_n,data):
    N_data = len(data) 
    data_min = np.min(data)
    data_max = np.max(data)

    step = (data_max - data_min) / (i_class_n - 0.2)
    class_rep_vals = np.zeros(i_class_n + 1)
    x_vals = np.zeros(i_class_n + 1)
    mu_x = np.zeros(i_class_n + 1)
    fi_50 = np.zeros(i_class_n + 1)
    i_vals = np.zeros(i_class_n + 1)

    val0_weight = 0.9
    class_rep_vals[0] = data_min + val0_weight * step
    x_vals[0] = class_rep_vals[0] - d  # no log

    for i in range(1, i_class_n + 1):
        class_rep_vals[i] = class_rep_vals[i - 1] + step
        x_vals[i] = class_rep_vals[i] - d  # no log

    freq, _ = np.histogram(data, bins=[-np.inf] + class_rep_vals.tolist()[0:i_class_n] + [np.inf])

    i_vals[0] = freq[0]
    fi_50[0] = (i_vals[0] - 0.3) / (N_data + 0.4)
    mu_x[0] = norm.ppf(fi_50[0])

    for i in range(1, i_class_n + 1):
        i_vals[i] = freq[i] + i_vals[i - 1]
        fi_50[i] = (i_vals[i] - 0.3) / (N_data + 0.4)
        mu_x[i] = norm.ppf(fi_50[i])

    df = pd.DataFrame({
        'Classes': class_rep_vals,
        'Freq_absol': freq,
        'i': i_vals,
        'Fi_50pct': fi_50,
        'x_vals': x_vals,
        'mu_x': mu_x
    })

    model = smf.ols('mu_x ~ x_vals', data=df.iloc[0:i_class_n]).fit()
    slope = model.params.iloc[1]
    intercept = model.params.iloc[0]
    r2 = model.rsquared

    sigma = 1 / slope
    mu = -intercept * sigma

    p1 = norm.ppf(0.01, loc=mu, scale=sigma)
    p50 = norm.ppf(0.5, loc=mu, scale=sigma)
    p99 = norm.ppf(0.99, loc=mu, scale=sigma)

    percentiles = [0.01, 0.5, 0.99]
    empirical_percentiles = np.percentile(data, [p * 100 for p in percentiles])
    fitted_percentiles = norm.ppf(percentiles, loc=mu, scale=sigma)
    relative_diff = 100 * (fitted_percentiles - empirical_percentiles) / empirical_percentiles

    step_percentiles = 5
    upper_bounnd_perct = 100
    lower_bound_perct = 0

    empirical = np.percentile(data, np.arange(lower_bound_perct, upper_bounnd_perct, step_percentiles))
    fitted_perc = norm.ppf(np.arange(lower_bound_perct, upper_bounnd_perct, step_percentiles) / 100, loc=mu, scale=sigma)

    mse = np.mean((empirical - fitted_perc) ** 2)
    mean_relat_diff = np.mean(abs(100 * (fitted_perc - empirical) / empirical))

    print(mse)

    return {
        'i_class_n': i_class_n,
        'R2': r2,
        'sigma': sigma,
        'mu': mu,
        'fitted_percentiles': fitted_percentiles,
        'empirical_percentiles': empirical_percentiles,
        'MSE': mse,
        'Mean relat diff': mean_relat_diff,
        'relative_diff': relative_diff
    }

def beta_class_stats(i_class_n,data):
    N_data = len(data) 
    data_min = np.min(data)
    data_max = np.max(data)

    # Normalize data to [0,1]
    scaled_data = (data - data_min) / (data_max - data_min)
    step = 1 / (i_class_n - 0.2)
    class_rep_vals = np.zeros(i_class_n + 1)
    x_beta = np.zeros(i_class_n + 1)
    mu_beta = np.zeros(i_class_n + 1)
    fi_50 = np.zeros(i_class_n + 1)
    i_vals = np.zeros(i_class_n + 1)

    val0_weight = 0.9
    class_rep_vals[0] = val0_weight * step
    x_beta[0] = class_rep_vals[0]

    def safe_minmax_scale(data, epsilon=1e-6):
        data = np.asarray(data)
        min_val = np.min(data)
        max_val = np.max(data)
        scaled = (data - min_val) / (max_val - min_val)
        # ensure values are within (0, 1) strictly
        scaled = np.clip(scaled, epsilon, 1 - epsilon)
        return scaled, min_val, max_val
    
    scaled_data, data_min, data_max = safe_minmax_scale(data)

    for i in range(1, i_class_n + 1):
        class_rep_vals[i] = class_rep_vals[i - 1] + step
        x_beta[i] = class_rep_vals[i]

    freq, _ = np.histogram(scaled_data, bins=[-np.inf] + class_rep_vals.tolist()[0:i_class_n] + [np.inf])
    i_vals[0] = freq[0]
    fi_50[0] = (i_vals[0] - 0.3) / (N_data + 0.4)
    mu_beta[0] = norm.ppf(fi_50[0])

    for i in range(1, i_class_n + 1):
        i_vals[i] = freq[i] + i_vals[i - 1]
        fi_50[i] = (i_vals[i] - 0.3) / (N_data + 0.4)
        mu_beta[i] = norm.ppf(fi_50[i])

    df = pd.DataFrame({
        'Classes': class_rep_vals,
        'Freq_absol': freq,
        'i': i_vals,
        'Fi_50pct': fi_50,
        'x_beta': x_beta,
        'mu_beta': mu_beta
    })

    model = smf.ols('mu_beta ~ x_beta', data=df.iloc[0:i_class_n]).fit()
    slope = model.params.iloc[1]
    intercept = model.params.iloc[0]
    r2 = model.rsquared

    # Rough estimation of alpha and beta (not analytically perfect)
    alpha = (slope + intercept) ** 2
    beta_param = alpha / slope if slope != 0 else np.nan

    # Use scipy to refine with MLE
    from scipy.stats import beta

    
    a_mle, b_mle, loc, scale = beta.fit(scaled_data, floc=0, fscale=1)

    # Rescale percentiles back to original data space
    percentiles = [0.01, 0.5, 0.99]
    fitted_percentiles_scaled = beta.ppf(percentiles, a=a_mle, b=b_mle, loc=loc, scale=scale)
    fitted_percentiles = data_min + fitted_percentiles_scaled * (data_max - data_min)
    empirical_percentiles = np.percentile(data, [p * 100 for p in percentiles])
    relative_diff = 100 * (fitted_percentiles - empirical_percentiles) / empirical_percentiles

    # MSE comparison over empirical vs fitted percentiles
    percent_range = np.arange(0, 100, 5)
    empirical = np.percentile(data, percent_range)
    fitted_scaled = beta.ppf(percent_range / 100, a=a_mle, b=b_mle, loc=loc, scale=scale)
    fitted = data_min + fitted_scaled * (data_max - data_min)

    mse = np.mean((empirical - fitted) ** 2)
    mean_relat_diff = np.mean(abs(100 * (fitted - empirical) / empirical))

    return {
        'i_class_n': i_class_n,
        'R2': r2,
        'alpha': a_mle,
        'beta': b_mle,
        'fitted_percentiles': fitted_percentiles,
        'empirical_percentiles': empirical_percentiles,
        'MSE': mse,
        'Mean relat diff': mean_relat_diff,
        'relative_diff': relative_diff
    }

def weibull_class_stats(i_class_n, data):
    N_data = len(data) 
    data_min = np.min(data)
    data_max = np.max(data)

    step = (data_max - data_min) / (i_class_n - 0.2)
    class_rep_vals = np.zeros(i_class_n + 1)
    ln_x = np.zeros(i_class_n + 1)
    ln_ln_term = np.zeros(i_class_n + 1)
    fi_50 = np.zeros(i_class_n + 1)
    i_vals = np.zeros(i_class_n + 1)

    val0_weight = 0.9
    class_rep_vals[0] = data_min + val0_weight * step
    ln_x[0] = np.log(class_rep_vals[0])

    for i in range(1, i_class_n + 1):
        class_rep_vals[i] = class_rep_vals[i - 1] + step
        ln_x[i] = np.log(class_rep_vals[i])

    freq, _ = np.histogram(data, bins=[-np.inf] + class_rep_vals.tolist()[0:i_class_n] + [np.inf])
    i_vals[0] = freq[0]
    fi_50[0] = (i_vals[0] - 0.3) / (N_data + 0.4)
    ln_ln_term[0] = np.log(-np.log(1 - fi_50[0]))

    for i in range(1, i_class_n + 1):
        i_vals[i] = freq[i] + i_vals[i - 1]
        fi_50[i] = (i_vals[i] - 0.3) / (N_data + 0.4)
        ln_ln_term[i] = np.log(-np.log(1 - fi_50[i]))

    df = pd.DataFrame({
        'Classes': class_rep_vals,
        'Freq_absol': freq,
        'i': i_vals,
        'Fi_50pct': fi_50,
        'ln_x': ln_x,
        'ln_ln_term': ln_ln_term
    })

    model = smf.ols('ln_ln_term ~ ln_x', data=df.iloc[0:i_class_n]).fit()
    slope = model.params.iloc[1]
    intercept = model.params.iloc[0]
    r2 = model.rsquared

    beta_shape = slope
    lambda_scale = np.exp(-intercept / slope)

    # Percentile estimates
    percentiles = [0.01, 0.5, 0.99]
    fitted_percentiles = weibull_min.ppf(percentiles, c=beta_shape, scale=lambda_scale)
    empirical_percentiles = np.percentile(data, [p * 100 for p in percentiles])
    relative_diff = 100 * (fitted_percentiles - empirical_percentiles) / empirical_percentiles

    # Full curve comparison
    percent_range = np.arange(0, 100, 5)
    empirical = np.percentile(data, percent_range)
    fitted = weibull_min.ppf(percent_range / 100, c=beta_shape, scale=lambda_scale)

    mse = np.mean((empirical - fitted) ** 2)
    mean_relat_diff = np.mean(abs(100 * (fitted - empirical) / empirical))

    return {
        'i_class_n': i_class_n,
        'R2': r2,
        'beta_shape': beta_shape,
        'lambda_scale': lambda_scale,
        'fitted_percentiles': fitted_percentiles,
        'empirical_percentiles': empirical_percentiles,
        'MSE': mse,
        'Mean relat diff': mean_relat_diff,
        'relative_diff': relative_diff
    }

def gamma_class_stats(i_class_n,data):
    N_data = len(data) 
    data_min = np.min(data)
    data_max = np.max(data)

    step = (data_max - data_min) / (i_class_n - 0.2)
    class_rep_vals = np.zeros(i_class_n + 1)
    ln_x = np.zeros(i_class_n + 1)
    z_score = np.zeros(i_class_n + 1)
    fi_50 = np.zeros(i_class_n + 1)
    i_vals = np.zeros(i_class_n + 1)

    val0_weight = 0.9
    class_rep_vals[0] = data_min + val0_weight * step
    ln_x[0] = np.log(class_rep_vals[0])

    for i in range(1, i_class_n + 1):
        class_rep_vals[i] = class_rep_vals[i - 1] + step
        ln_x[i] = np.log(class_rep_vals[i])

    freq, _ = np.histogram(data, bins=[-np.inf] + class_rep_vals.tolist()[0:i_class_n] + [np.inf])
    i_vals[0] = freq[0]
    fi_50[0] = (i_vals[0] - 0.3) / (N_data + 0.4)
    z_score[0] = norm.ppf(fi_50[0])

    for i in range(1, i_class_n + 1):
        i_vals[i] = freq[i] + i_vals[i - 1]
        fi_50[i] = (i_vals[i] - 0.3) / (N_data + 0.4)
        z_score[i] = norm.ppf(fi_50[i])

    df = pd.DataFrame({
        'Classes': class_rep_vals,
        'Freq_absol': freq,
        'i': i_vals,
        'Fi_50pct': fi_50,
        'ln_x': ln_x,
        'z_score': z_score
    })

    model = smf.ols('z_score ~ ln_x', data=df.iloc[0:i_class_n]).fit()
    slope = model.params.iloc[1]
    intercept = model.params.iloc[0]
    r2 = model.rsquared

    k_shape = (1 / slope) ** 2
    theta_scale = np.exp(-intercept * slope)

    # Percentile estimates
    percentiles = [0.01, 0.5, 0.99]
    fitted_percentiles = gamma.ppf(percentiles, a=k_shape, scale=theta_scale)
    empirical_percentiles = np.percentile(data, [p * 100 for p in percentiles])
    relative_diff = 100 * (fitted_percentiles - empirical_percentiles) / empirical_percentiles

    # Full curve comparison
    percent_range = np.arange(0, 100, 5)
    empirical = np.percentile(data, percent_range)
    fitted = gamma.ppf(percent_range / 100, a=k_shape, scale=theta_scale)

    mse = np.mean((empirical - fitted) ** 2)
    mean_relat_diff = np.mean(abs(100 * (fitted - empirical) / empirical))

    return {
        'i_class_n': i_class_n,
        'R2': r2,
        'k_shape': k_shape,
        'theta_scale': theta_scale,
        'fitted_percentiles': fitted_percentiles,
        'empirical_percentiles': empirical_percentiles,
        'MSE': mse,
        'Mean relat diff': mean_relat_diff,
        'relative_diff': relative_diff
    }

def exponential_class_stats(i_class_n,data):
    N_data = len(data) 
    data_min = np.min(data)
    data_max = np.max(data)

    step = (data_max - data_min) / (i_class_n - 0.2)
    class_rep_vals = np.zeros(i_class_n + 1)
    exp_x = np.zeros(i_class_n + 1)
    mu_exp_x = np.zeros(i_class_n + 1)
    fi_50 = np.zeros(i_class_n + 1)
    i_vals = np.zeros(i_class_n + 1)

    val0_weight = 0.9
    class_rep_vals[0] = data_min + val0_weight * step
    exp_x[0] = class_rep_vals[0] - d

    for i in range(1, i_class_n + 1):
        class_rep_vals[i] = class_rep_vals[i - 1] + step
        exp_x[i] = class_rep_vals[i] - d

    freq, _ = np.histogram(data, bins=[-np.inf] + class_rep_vals.tolist()[0:i_class_n] + [np.inf])

    i_vals[0] = freq[0]
    fi_50[0] = (i_vals[0] - 0.3) / (N_data + 0.4)
    mu_exp_x[0] = -np.log(1 - fi_50[0])

    for i in range(1, i_class_n + 1):
        i_vals[i] = freq[i] + i_vals[i - 1]
        fi_50[i] = (i_vals[i] - 0.3) / (N_data + 0.4)
        mu_exp_x[i] = -np.log(1 - fi_50[i])

    df = pd.DataFrame({
        'Classes': class_rep_vals,
        'Freq_absol': freq,
        'i': i_vals,
        'Fi_50pct': fi_50,
        'x': exp_x,
        'mu_x': mu_exp_x
    })

    model = smf.ols('mu_x ~ x', data=df.iloc[0:i_class_n]).fit()
    slope = model.params.iloc[1]
    intercept = model.params.iloc[0]
    r2 = model.rsquared

    lambda_inv = 1 / slope  # scale parameter
    p1 = expon.ppf(0.01, scale=lambda_inv)
    p50 = expon.ppf(0.5, scale=lambda_inv)
    p99 = expon.ppf(0.99, scale=lambda_inv)

    percentiles = [0.01, 0.5, 0.99]
    empirical_percentiles = np.percentile(data, [p * 100 for p in percentiles])
    fitted_percentiles = expon.ppf(percentiles, scale=lambda_inv)
    relative_diff = 100 * (fitted_percentiles - empirical_percentiles) / empirical_percentiles

    step_percentiles = 5
    empirical = np.percentile(data, np.arange(0, 100, step_percentiles))
    fitted_perc = expon.ppf(np.arange(0, 100, step_percentiles) / 100, scale=lambda_inv)
    mse = np.mean((empirical - fitted_perc) ** 2)
    mean_relat_diff = np.mean(abs(100 * (fitted_perc - empirical) / empirical))

    return {
        'i_class_n': i_class_n,
        'R2': r2,
        'lambda_inv': lambda_inv,
        'fitted_percentiles': fitted_percentiles,
        'empirical_percentiles': empirical_percentiles,
        'MSE': mse,
        'Mean relat diff': mean_relat_diff,
        'relative_diff': relative_diff
    }

#lois estimee -----------------

def lognormal_estimee(data,d):

    mu_x = np.nanmean(data)  
    sigma_x = np.nanstd(data) 

    if mu_x <= 0 or len(data) == 0:
        
        return np.nan, np.nan

    slnx = np.sqrt(np.log(1 + (sigma_x / mu_x) ** 2))
    mulnx = np.log(mu_x / np.sqrt(1 + (sigma_x / mu_x) ** 2))
    scale = np.exp(mulnx)

    percentiles = [0.01, 0.5, 0.99]
    fitted_percentiles = lognorm.ppf(percentiles, s=slnx, scale=scale)
    empirical_percentiles = np.percentile(data, [p * 100 for p in percentiles])

    mse = np.mean((empirical_percentiles - fitted_percentiles) ** 2)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        relative_diff = np.where(
            empirical_percentiles != 0,
            100 * (fitted_percentiles - empirical_percentiles) / empirical_percentiles,
            np.where(fitted_percentiles != 0, 0, 0)  # If both are 0, diff is 0; if only denominator is 0 (force zero) #handle_zero_division
        )
             
        # For mean_relat_diff, exclude zeros from denominator
        valid_mask = empirical_percentiles != 0
        if np.any(valid_mask):
            relative_diffs = abs(100 * (fitted_percentiles[valid_mask] - empirical_percentiles[valid_mask]) / empirical_percentiles[valid_mask]) #handle_zero_division forced division by 3 points
            mean_relat_diff = np.sum(relative_diffs)/3 #forced 3 points division
        else:
            # All empirical values are zero - can't compute relative difference
            mean_relat_diff = np.nan

    return {
        'i_class_n': 'Estimation',
        'R2': 'NA',
        'slnx': slnx,
        'mulnx': mulnx,
        'fitted_percentiles': fitted_percentiles,
        'empirical_percentiles': empirical_percentiles,
        'MSE': mse,
        'Mean relat diff': mean_relat_diff,
        'relative_diff': relative_diff
    }

def normal_estimee(data, d=0):
    """
    Direct parameter estimation for Normal distribution using Method of Moments
    
    Parameters:
    -----------
    data : array-like
        Input data
    d : float
        Displacement parameter (typically 0 for normal)
    
    Returns:
    --------
    dict : Estimation results including parameters and quality metrics
    """
    try:
        # Shift data if needed
        shifted_data = data - d
        
        # Method of Moments estimators
        mu = np.mean(shifted_data)
        sigma = np.std(shifted_data, ddof=1)  # Sample standard deviation
        
        if sigma <= 0:
            return None
        
        # Calculate percentiles
        percentiles = [0.01, 0.5, 0.99]
        fitted_percentiles = norm.ppf(percentiles, loc=mu, scale=sigma) + d
        empirical_percentiles = np.percentile(data, [p * 100 for p in percentiles])
        
        # MSE calculation
        step_percentiles = 5
        empirical = np.percentile(data, np.arange(0, 100, step_percentiles))
        fitted_perc = norm.ppf(np.arange(0, 100, step_percentiles) / 100, loc=mu, scale=sigma) + d
        mse = np.mean((empirical - fitted_perc) ** 2)
        
        # Relative difference calculation
        with np.errstate(divide='ignore', invalid='ignore'):
            relative_diff = np.where(
                empirical_percentiles != 0,
                100 * (fitted_percentiles - empirical_percentiles) / empirical_percentiles,
                np.where(fitted_percentiles != 0, 0, 0)
            )
            
            valid_mask = empirical_percentiles != 0
            if np.any(valid_mask):
                relative_diffs = abs(100 * (fitted_percentiles[valid_mask] - empirical_percentiles[valid_mask]) / empirical_percentiles[valid_mask])
                mean_relat_diff = np.sum(relative_diffs) / 3
            else:
                mean_relat_diff = np.nan
        
        return {
            'i_class_n': 'Estimation',
            'R2': np.nan,  # Not applicable for estimation
            'mu': mu + d,  # Add back displacement
            'sigma': sigma,
            'fitted_percentiles': fitted_percentiles,
            'empirical_percentiles': empirical_percentiles,
            'MSE': mse,
            'Mean relat diff': mean_relat_diff,
            'relative_diff': relative_diff
        }
        
    except Exception as e:
        print(f"Error in normal_estimee: {e}")
        return None

def beta_estimee(data, d=0):
    """
    Direct parameter estimation for Beta distribution using Method of Moments
    
    Note: Beta distribution requires data in [0,1], so we scale the data
    """
    try:
        # Scale data to [0, 1]
        data_min = np.min(data)
        data_max = np.max(data)
        
        if data_max == data_min:
            return None
        
        scaled_data = (data - data_min) / (data_max - data_min)
        
        # Ensure values are strictly in (0, 1)
        epsilon = 1e-6
        scaled_data = np.clip(scaled_data, epsilon, 1 - epsilon)
        
        # Method of Moments estimators
        mean = np.mean(scaled_data)
        var = np.var(scaled_data, ddof=1)
        
        if var <= 0 or var >= mean * (1 - mean):
            # Use MLE instead
            from scipy.stats import beta as beta_dist
            a_mle, b_mle, loc, scale = beta_dist.fit(scaled_data, floc=0, fscale=1)
            alpha = a_mle
            beta_param = b_mle
        else:
            # Method of Moments
            common = mean * (1 - mean) / var - 1
            alpha = mean * common
            beta_param = (1 - mean) * common
        
        # Calculate percentiles (in original scale)
        percentiles = [0.01, 0.5, 0.99]
        fitted_percentiles_scaled = beta.ppf(percentiles, a=alpha, b=beta_param)
        fitted_percentiles = data_min + fitted_percentiles_scaled * (data_max - data_min)
        empirical_percentiles = np.percentile(data, [p * 100 for p in percentiles])
        
        # MSE calculation
        percent_range = np.arange(0, 100, 5)
        empirical = np.percentile(data, percent_range)
        fitted_scaled = beta.ppf(percent_range / 100, a=alpha, b=beta_param)
        fitted = data_min + fitted_scaled * (data_max - data_min)
        mse = np.mean((empirical - fitted) ** 2)
        
        # Relative difference
        with np.errstate(divide='ignore', invalid='ignore'):
            relative_diff = np.where(
                empirical_percentiles != 0,
                100 * (fitted_percentiles - empirical_percentiles) / empirical_percentiles,
                np.where(fitted_percentiles != 0, 0, 0)
            )
            
            valid_mask = empirical_percentiles != 0
            if np.any(valid_mask):
                relative_diffs = abs(100 * (fitted_percentiles[valid_mask] - empirical_percentiles[valid_mask]) / empirical_percentiles[valid_mask])
                mean_relat_diff = np.sum(relative_diffs) / 3
            else:
                mean_relat_diff = np.nan
        
        return {
            'i_class_n': 'Estimation',
            'R2': np.nan,
            'alpha': alpha,
            'beta': beta_param,
            'data_min': data_min,
            'data_max': data_max,
            'fitted_percentiles': fitted_percentiles,
            'empirical_percentiles': empirical_percentiles,
            'MSE': mse,
            'Mean relat diff': mean_relat_diff,
            'relative_diff': relative_diff
        }
        
    except Exception as e:
        print(f"Error in beta_estimee: {e}")
        return None

def weibull_estimee(data, d=0):
    """
    Direct parameter estimation for Weibull distribution using MLE
    (Method of Moments is complex for Weibull, so we use MLE)
    """
    try:
        shifted_data = data - d
        
        if np.any(shifted_data <= 0):
            # Adjust d to ensure all data is positive
            d = np.min(data) - 1e-6
            shifted_data = data - d
        
        # Use scipy's MLE fit
        from scipy.stats import weibull_min
        c, loc, scale = weibull_min.fit(shifted_data, floc=0)
        
        beta_shape = c
        lambda_scale = scale
        
        # Calculate percentiles
        percentiles = [0.01, 0.5, 0.99]
        fitted_percentiles = weibull_min.ppf(percentiles, c=beta_shape, scale=lambda_scale) + d
        empirical_percentiles = np.percentile(data, [p * 100 for p in percentiles])
        
        # MSE calculation
        percent_range = np.arange(0, 100, 5)
        empirical = np.percentile(data, percent_range)
        fitted = weibull_min.ppf(percent_range / 100, c=beta_shape, scale=lambda_scale) + d
        mse = np.mean((empirical - fitted) ** 2)
        
        # Relative difference
        with np.errstate(divide='ignore', invalid='ignore'):
            relative_diff = np.where(
                empirical_percentiles != 0,
                100 * (fitted_percentiles - empirical_percentiles) / empirical_percentiles,
                np.where(fitted_percentiles != 0, 0, 0)
            )
            
            valid_mask = empirical_percentiles != 0
            if np.any(valid_mask):
                relative_diffs = abs(100 * (fitted_percentiles[valid_mask] - empirical_percentiles[valid_mask]) / empirical_percentiles[valid_mask])
                mean_relat_diff = np.sum(relative_diffs) / 3
            else:
                mean_relat_diff = np.nan
        
        return {
            'i_class_n': 'Estimation',
            'R2': np.nan,
            'beta_shape': beta_shape,
            'lambda_scale': lambda_scale,
            'fitted_percentiles': fitted_percentiles,
            'empirical_percentiles': empirical_percentiles,
            'MSE': mse,
            'Mean relat diff': mean_relat_diff,
            'relative_diff': relative_diff
        }
        
    except Exception as e:
        print(f"Error in weibull_estimee: {e}")
        return None

def gamma_estimee(data, d=0):
    """
    Direct parameter estimation for Gamma distribution using Method of Moments
    """
    try:
        shifted_data = data - d
        
        if np.any(shifted_data <= 0):
            d = np.min(data) - 1e-6
            shifted_data = data - d
        
        # Method of Moments estimators
        mean = np.mean(shifted_data)
        var = np.var(shifted_data, ddof=1)
        
        if var <= 0 or mean <= 0:
            return None
        
        # k = mean^2 / variance
        # theta = variance / mean
        k_shape = (mean ** 2) / var
        theta_scale = var / mean
        
        # Calculate percentiles
        percentiles = [0.01, 0.5, 0.99]
        fitted_percentiles = gamma.ppf(percentiles, a=k_shape, scale=theta_scale) + d
        empirical_percentiles = np.percentile(data, [p * 100 for p in percentiles])
        
        # MSE calculation
        percent_range = np.arange(0, 100, 5)
        empirical = np.percentile(data, percent_range)
        fitted = gamma.ppf(percent_range / 100, a=k_shape, scale=theta_scale) + d
        mse = np.mean((empirical - fitted) ** 2)
        
        # Relative difference
        with np.errstate(divide='ignore', invalid='ignore'):
            relative_diff = np.where(
                empirical_percentiles != 0,
                100 * (fitted_percentiles - empirical_percentiles) / empirical_percentiles,
                np.where(fitted_percentiles != 0, 0, 0)
            )
            
            valid_mask = empirical_percentiles != 0
            if np.any(valid_mask):
                relative_diffs = abs(100 * (fitted_percentiles[valid_mask] - empirical_percentiles[valid_mask]) / empirical_percentiles[valid_mask])
                mean_relat_diff = np.sum(relative_diffs) / 3
            else:
                mean_relat_diff = np.nan
        
        return {
            'i_class_n': 'Estimation',
            'R2': np.nan,
            'k_shape': k_shape,
            'theta_scale': theta_scale,
            'fitted_percentiles': fitted_percentiles,
            'empirical_percentiles': empirical_percentiles,
            'MSE': mse,
            'Mean relat diff': mean_relat_diff,
            'relative_diff': relative_diff
        }
        
    except Exception as e:
        print(f"Error in gamma_estimee: {e}")
        return None

def exponential_estimee(data, d=0):
    """
    Direct parameter estimation for Exponential distribution using Method of Moments
    """
    try:
        shifted_data = data - d
        
        if np.any(shifted_data <= 0):
            d = np.min(data) - 1e-6
            shifted_data = data - d
        
        # Method of Moments estimator
        # For exponential: lambda = 1/mean
        mean = np.mean(shifted_data)
        
        if mean <= 0:
            return None
        
        lambda_inv = mean  # scale parameter = 1/lambda = mean
        
        # Calculate percentiles
        percentiles = [0.01, 0.5, 0.99]
        fitted_percentiles = expon.ppf(percentiles, scale=lambda_inv) + d
        empirical_percentiles = np.percentile(data, [p * 100 for p in percentiles])
        
        # MSE calculation
        percent_range = np.arange(0, 100, 5)
        empirical = np.percentile(data, percent_range)
        fitted = expon.ppf(percent_range / 100, scale=lambda_inv) + d
        mse = np.mean((empirical - fitted) ** 2)
        
        # Relative difference
        with np.errstate(divide='ignore', invalid='ignore'):
            relative_diff = np.where(
                empirical_percentiles != 0,
                100 * (fitted_percentiles - empirical_percentiles) / empirical_percentiles,
                np.where(fitted_percentiles != 0, 0, 0)
            )
            
            valid_mask = empirical_percentiles != 0
            if np.any(valid_mask):
                relative_diffs = abs(100 * (fitted_percentiles[valid_mask] - empirical_percentiles[valid_mask]) / empirical_percentiles[valid_mask])
                mean_relat_diff = np.sum(relative_diffs) / 3
            else:
                mean_relat_diff = np.nan
        
        return {
            'i_class_n': 'Estimation',
            'R2': np.nan,
            'lambda_inv': lambda_inv,
            'fitted_percentiles': fitted_percentiles,
            'empirical_percentiles': empirical_percentiles,
            'MSE': mse,
            'Mean relat diff': mean_relat_diff,
            'relative_diff': relative_diff
        }
        
    except Exception as e:
        print(f"Error in exponential_estimee: {e}")
        return None

#checking classes for good fit / switch to estimations
def check_mrd_threshold_and_fallback_old(model_df, data, d, mrd_threshold):
   
    min_mrd = model_df['Mean relat diff'].min()
    best_class_idx = model_df['Mean relat diff'].idxmin()
    
    if min_mrd > mrd_threshold:
        # All classes exceed threshold - use estimation fallback
        st.warning(f"⚠️ **All modeled classes have MRD > {mrd_threshold}%**")
        st.info(f"🔄 **Fallback activated:** Using direct parameter estimation (Method of Moments)")
        
        fallback_result = lognormal_estimee(data, d)
        
        if fallback_result is None:
            st.error("❌ Fallback estimation also failed. Data may not be suitable for lognormal modeling.")
            return {
                'status': 'failed',
                'min_mrd': min_mrd,
                'threshold': mrd_threshold,
                'use_fallback': False,
                'result': None
            }
        
        return {
            'status': 'fallback',
            'min_mrd': min_mrd,
            'threshold': mrd_threshold,
            'use_fallback': True,
            'result': fallback_result,
            'best_class_idx': None
        }
    else:
        # At least one class is acceptable
        return {
            'status': 'success',
            'min_mrd': min_mrd,
            'threshold': mrd_threshold,
            'use_fallback': False,
            'result': None,
            'best_class_idx': best_class_idx
        }

def check_mrd_threshold_and_fallback(model_df, data, d, mrd_threshold, distribution_type="lognormal"):
    """
    Check if MRD threshold is exceeded for ALL percentiles (P1%, P50%, P99%) 
    and trigger fallback to estimation if needed.
    
    Parameters:
    -----------
    model_df : DataFrame
        Model results from class-based fitting
    data : array-like
        Input data
    d : float
        Displacement parameter
    mrd_threshold : float
        MRD threshold percentage (e.g., 10 for 10%)
    distribution_type : str
        Type of distribution ('lognormal', 'normal', 'beta', 'weibull', 'gamma', 'exponential')
    
    Returns:
    --------
    dict : Status and results
        - 'status': 'success' (class modeling OK) or 'fallback' (estimation used) or 'failed' (error)
        - 'use_fallback': True if estimation was used
        - 'result': Estimation result (dict) if fallback was triggered, None otherwise
        - 'best_class_idx': Best class index from modeling (if not using fallback)
    """
    
    min_p1 = model_df['relat_diff_1%'].abs().min()
    min_p50 = model_df['relat_diff_50%'].abs().min()
    min_p99 = model_df['relat_diff_99%'].abs().min()
    
    best_class_idx = model_df['Mean relat diff'].idxmin()
    min_mrd = model_df['Mean relat diff'].min()
    
    # Check if ALL three percentiles exceed threshold
    if min_p1 > mrd_threshold and min_p50 > mrd_threshold and min_p99 > mrd_threshold:
        # All percentiles exceed threshold - trigger fallback
        print(f"⚠️ All modeled classes have MRD > {mrd_threshold}% for ALL percentiles")
        print(f"   P1%: {min_p1:.2f}% | P50%: {min_p50:.2f}% | P99%: {min_p99:.2f}%")
        print(f"🔄 Fallback activated: Using direct parameter estimation for {distribution_type}")
        
        fallback_result = None
        
        if distribution_type == "lognormal":
            fallback_result = lognormal_estimee(data, d)
            
        elif distribution_type == "normal":
            fallback_result = normal_estimee(data, d)
            
        elif distribution_type == "beta":
            fallback_result = beta_estimee(data, d)
            
        elif distribution_type == "weibull":
            fallback_result = weibull_estimee(data, d)
            
        elif distribution_type == "gamma":
            fallback_result = gamma_estimee(data, d)
            
        elif distribution_type == "exponential":
            fallback_result = exponential_estimee(data, d)
            
        else:
            print(f"❌ Unknown distribution type: {distribution_type}")
            return {
                'status': 'failed',
                'min_mrd': min_mrd,
                'threshold': mrd_threshold,
                'use_fallback': False,
                'result': None,
                'best_class_idx': best_class_idx,
                'distribution_type': distribution_type
            }
        
        # Check if estimation succeeded
        if fallback_result is None:
            print(f"❌ Fallback estimation failed for {distribution_type}")
            return {
                'status': 'failed',
                'min_mrd': min_mrd,
                'threshold': mrd_threshold,
                'use_fallback': False,
                'result': None,
                'best_class_idx': best_class_idx,
                'distribution_type': distribution_type
            }
        
        print(f"✓ Estimation successful for {distribution_type}")
        return {
            'status': 'fallback',
            'min_mrd': min_mrd,
            'threshold': mrd_threshold,
            'use_fallback': True,
            'result': fallback_result,
            'best_class_idx': None,
            'distribution_type': distribution_type
        }
    else:
        # At least one percentile is acceptable - use class modeling
        print(f"✓ Class modeling acceptable (at least one percentile within threshold)")
        print(f"   P1%: {min_p1:.2f}% | P50%: {min_p50:.2f}% | P99%: {min_p99:.2f}%")
        return {
            'status': 'success',
            'min_mrd': min_mrd,
            'threshold': mrd_threshold,
            'use_fallback': False,
            'result': None,
            'best_class_idx': best_class_idx,
            'distribution_type': distribution_type
        }

#integrate mrd_threshhold check in the classes computation 
def calculate_model_summary(data, nb_classes, mrd_threshold):
        """Calculate model summary with MRD threshold check"""
        if data is None:
            return None

        N_data = len(data)
        data_min = min(data)
        data_max = max(data)
        d = 0

        #switch case for model df computation based on input pdf name    
        summary_df = run_lognormal_analysis(data, 0, nb_classes)
        model_df = summary_df.iloc[2:]
        
        # Check MRD threshold
        fallback_check = check_mrd_threshold_and_fallback(model_df, data, d, mrd_threshold,distribution_type= "lognormal")
        
        # Store both model_df and fallback status
        return fallback_check, model_df
#############################
def run_lognormal_analysis(data, d, nb_classes):
    rows = []

    for i_class in range(2, nb_classes + 1):
        result = lognormal_class_stats(i_class,data)
        # Extract needed info and add class index explicitly
        row = {
            'class_index': i_class,
            'slnx': result['slnx'],
            'mulnx': result['mulnx'],
            'R2': result['R2'],
            'percentile_1%': result['fitted_percentiles'][0],
            'percentile_50%': result['fitted_percentiles'][1],
            'percentile_99%': result['fitted_percentiles'][2],
            'percentile_1%_emp': result['empirical_percentiles'][0],
            'percentile_50%_emp': result['empirical_percentiles'][1],
            'percentile_99%_emp': result['empirical_percentiles'][2],
            'MSE':result['MSE'],
            'Mean relat diff':result['Mean relat diff'],
            'relat_diff_1%': result['relative_diff'][0],
            'relat_diff_50%': result['relative_diff'][1],
            'relat_diff_99%': result['relative_diff'][2]
        }
        rows.append(row)

    df_summary = pd.DataFrame(rows)
    #df_summary.set_index('class_index', inplace=True)  # set class index as dataframe index

    return df_summary

def plot_lognormal_from_params(data,N_bins, i_class,d, ulnx, slnx,R2,perso_title):
    # Calculate shape and scale from μ and σ of ln(x)
    shape = slnx                  # shape = σ
    loc = d                       # loc = d (often 0)
    scale = np.exp(ulnx)         # scale = exp(μ)

    # Histogram (percentage scale)
    count, bins, _ = plt.hist(
        data, bins=N_bins, 
        alpha=0.4, color='gray', edgecolor='black',
        #weights=np.ones_like(data)* 10. / len(data),
        label='Sample histogram'
        ,density=True
    )

    # Plot lognormal PDF
    x = np.linspace(min(data), max(data), 1000)
    pdf = lognorm.pdf(x, shape, loc=loc, scale=scale) #* 100
    plt.plot(x, pdf, 'k-', lw=2, label='Probability density')

    # Compute percentiles
    p1 = lognorm.ppf(0.01, shape, loc=loc, scale=scale)
    p50 = lognorm.ppf(0.50, shape, loc=loc, scale=scale)
    p99 = lognorm.ppf(0.99, shape, loc=loc, scale=scale)

    plt.axvline(p1, color='blue', linestyle='-', label='1%: {:,.2f}'.format(p1))
    plt.axvline(p50, color='green', linestyle='-', label='50%: {:,.2f}'.format(p50))
    plt.axvline(p99, color='red', linestyle='-', label='99%: {:,.2f}'.format(p99))

    # Labels and legend
    plt.xlabel("Value")
    plt.ylabel("Effective (%)")
    plt.title(f"{perso_title}"  )
    plt.legend(loc="upper right")

     # Separate parameter box (cleaned up format)
    param_text = (
        f"Class n° = {i_class:.0f}\n"
        f"$d$ = {d:.3f}\n"
        f"R² = {R2*100:.2f} %\n"
        f"$\\mu_{{\\ln(x)}}$ = {ulnx:.3f}\n"
        f"$\\sigma_{{\\ln(x)}}$ = {slnx:.3f}"
    )
    plt.text(
        0.98, 0.40,
        param_text,
        ha='right', va='bottom',
        transform=plt.gca().transAxes,
        bbox=dict(boxstyle="round,pad=0.4", 
                  facecolor="lightyellow",
                  edgecolor="black", 
                  alpha=0.8)
    )
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_lognormal_streamlit001(data, N_bins, i_class, d, ulnx, slnx, R2, perso_title,show_hist,
                            show_model_pdf,
                            show_model_cdf,
                            show_estimated_pdf,
                            show_estimated_cdf,
                            show_scatter):
    fig, ax = plt.subplots(figsize=(8, 5))
    if show_hist:
        ax.hist(data, bins=N_bins, density=True, alpha=0.4, color='gray', edgecolor='black', label="Sample histogram")
     
    if show_model_pdf:
        # Plot modelled probability density function
        x = np.linspace(min(data), max(data), 100)
        # Example lognormal pdf with params from ulnx, slnx (adjust if needed)
        from scipy.stats import lognorm
        pdf = lognorm.pdf(x, s=slnx, scale=np.exp(ulnx))
        ax.plot(x, pdf, label="Modelled PDF")

    if show_model_cdf:
        x = np.linspace(min(data), max(data), 100)
        from scipy.stats import lognorm
        cdf = lognorm.cdf(x, s=slnx, scale=np.exp(ulnx))
        ax.plot(x, cdf, label="Modelled CDF")

    if show_estimated_pdf:
        # Add estimated PDF plotting here if applicable
        pass

    if show_estimated_cdf:
        # Add estimated CDF plotting here if applicable
        pass

    if show_scatter:
        y = np.zeros_like(data)
        ax.scatter(data, y, label="Sample data (scatter)", color="red", marker="|")
    # Calculate shape and scale from μ and σ of ln(x)
    shape = slnx
    loc = d
    scale = np.exp(ulnx)
    
    # Plot lognormal PDF
    x = np.linspace(min(data), max(data), 1000)
    pdf = lognorm.pdf(x, shape, loc=loc, scale=scale) #* 100
    plt.plot(x, pdf, 'k-', lw=2, label='Probability density')

    # Percentiles
    p1 = lognorm.ppf(0.01, shape, loc=loc, scale=scale)
    p50 = lognorm.ppf(0.50, shape, loc=loc, scale=scale)
    p99 = lognorm.ppf(0.99, shape, loc=loc, scale=scale)

    ax.axvline(p1, color='blue', linestyle='-', label='1%: {:,.2f}'.format(p1))
    ax.axvline(p50, color='green', linestyle='-', label='50%: {:,.2f}'.format(p50))
    ax.axvline(p99, color='red', linestyle='-', label='99%: {:,.2f}'.format(p99))

    ax.set_xlabel("Value")
    ax.set_ylabel("Effective (%)")
    ax.set_title(f"{perso_title}")
    ax.legend(loc="upper right")

    param_text = (
        f"Class n° = {i_class:.0f}\n"
        f"$d$ = {d:.3f}\n"
        f"R² = {R2*100:.2f} %\n"
        f"$\\mu_{{\\ln(x)}}$ = {ulnx:.3f}\n"
        f"$\\sigma_{{\\ln(x)}}$ = {slnx:.3f}"
    )

    ax.text(
        0.98, 0.40,
        param_text,
        ha='right', va='bottom',
        transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.4",
                  facecolor="lightyellow",
                  edgecolor="black",
                  alpha=0.8)
    )

    ax.grid(True)
    st.pyplot(fig)

def plot_lognormal_streamlit(data, N_bins, i_class, d, ulnx, slnx, R2, perso_title, show_hist,
                            show_model_pdf, show_model_cdf, show_estimated_pdf, show_estimated_cdf, show_scatter):
    fig, ax = plt.subplots(figsize=(8, 5))
    
    if show_hist:
        ax.hist(data, bins=N_bins, density=True, alpha=0.4, color='gray', edgecolor='black', label="Sample histogram")
    
    # Calculate shape and scale from μ and σ of ln(x)
    shape = slnx
    loc = d
    scale = np.exp(ulnx)
    
    x = np.linspace(min(data), max(data), 1000)
    
    if show_model_pdf:
        pdf = lognorm.pdf(x, shape, loc=loc, scale=scale)
        ax.plot(x, pdf, 'k-', lw=2, label='Probability density')
    
    if show_model_cdf:
        cdf = lognorm.cdf(x, shape, loc=loc, scale=scale)
        ax.plot(x, cdf, 'b-', lw=2, label='Cumulative distribution')

    if show_scatter:
        y = np.zeros_like(data)
        ax.scatter(data, y, color="red", marker="|", alpha=0.5, label="Data points")

    # Percentiles
    p1 = lognorm.ppf(0.01, shape, loc=loc, scale=scale)
    p50 = lognorm.ppf(0.50, shape, loc=loc, scale=scale)
    p99 = lognorm.ppf(0.99, shape, loc=loc, scale=scale)

    ax.axvline(p1, color='blue', linestyle='-', label='1%: {:,.2f}'.format(p1))
    ax.axvline(p50, color='green', linestyle='-', label='50%: {:,.2f}'.format(p50))
    ax.axvline(p99, color='red', linestyle='-', label='99%: {:,.2f}'.format(p99))

    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.set_title(f"{perso_title}")
    ax.legend(loc="upper right")

    param_text = (
        f"Class n° = {i_class:.0f}\n"
        f"$d$ = {d:.3f}\n"
        f"R² = {R2*100:.2f} %\n"
        f"$\\mu_{{\\ln(x)}}$ = {ulnx:.3f}\n"
        f"$\\sigma_{{\\ln(x)}}$ = {slnx:.3f}"
    )

    ax.text(
        0.98, 0.40,
        param_text,
        ha='right', va='bottom',
        transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.4",
                  facecolor="lightyellow",
                  edgecolor="black",
                  alpha=0.8)
    )

    ax.grid(True)
    st.pyplot(fig)

def plot_normal_streamlit(data, N_bins, i_class, mu, sigma, R2, perso_title, show_hist,
                         show_model_pdf, show_model_cdf, show_estimated_pdf, show_estimated_cdf, show_scatter):
    fig, ax = plt.subplots(figsize=(8, 5))
    
    if show_hist:
        ax.hist(data, bins=N_bins, density=True, alpha=0.4, color='gray', edgecolor='black', label="Sample histogram")
    
    x = np.linspace(min(data), max(data), 1000)
    
    if show_model_pdf:
        pdf = norm.pdf(x, loc=mu, scale=sigma)
        ax.plot(x, pdf, 'k-', lw=2, label='Probability density')
    
    if show_model_cdf:
        cdf = norm.cdf(x, loc=mu, scale=sigma)
        ax.plot(x, cdf, 'b-', lw=2, label='Cumulative distribution')

    if show_scatter:
        y = np.zeros_like(data)
        ax.scatter(data, y, color="red", marker="|", alpha=0.5, label="Data points")

    # Percentiles
    p1 = norm.ppf(0.01, loc=mu, scale=sigma)
    p50 = norm.ppf(0.50, loc=mu, scale=sigma)
    p99 = norm.ppf(0.99, loc=mu, scale=sigma)

    ax.axvline(p1, color='blue', linestyle='-', label='1%: {:,.2f}'.format(p1))
    ax.axvline(p50, color='green', linestyle='-', label='50%: {:,.2f}'.format(p50))
    ax.axvline(p99, color='red', linestyle='-', label='99%: {:,.2f}'.format(p99))

    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.set_title(f"{perso_title}")
    ax.legend(loc="upper right")

    param_text = (
        f"Class n° = {i_class:.0f}\n"
        f"R² = {R2*100:.2f} %\n"
        f"$\\mu$ = {mu:.3f}\n"
        f"$\\sigma$ = {sigma:.3f}"
    )

    ax.text(
        0.98, 0.40,
        param_text,
        ha='right', va='bottom',
        transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.4",
                  facecolor="lightyellow",
                  edgecolor="black",
                  alpha=0.8)
    )

    ax.grid(True)
    st.pyplot(fig)

def plot_beta_streamlit(data, N_bins, i_class, alpha, beta_param, R2, perso_title, show_hist,
                       show_model_pdf, show_model_cdf, show_estimated_pdf, show_estimated_cdf, show_scatter):
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Scale data to [0,1] for visualization
    data_min, data_max = min(data), max(data)
    scaled_data = (data - data_min) / (data_max - data_min)
    
    if show_hist:
        ax.hist(scaled_data, bins=N_bins, density=True, alpha=0.4, color='gray', edgecolor='black', label="Sample histogram (scaled)")
    
    x = np.linspace(0, 1, 1000)  # Beta is defined on [0,1]
    
    if show_model_pdf:
        pdf = beta.pdf(x, a=alpha, b=beta_param)
        ax.plot(x, pdf, 'k-', lw=2, label='Probability density')
    
    if show_model_cdf:
        cdf = beta.cdf(x, a=alpha, b=beta_param)
        ax.plot(x, cdf, 'b-', lw=2, label='Cumulative distribution')

    if show_scatter:
        y = np.zeros_like(scaled_data)
        ax.scatter(scaled_data, y, color="red", marker="|", alpha=0.5, label="Data points (scaled)")

    # Percentiles (scaled)
    p1 = beta.ppf(0.01, a=alpha, b=beta_param)
    p50 = beta.ppf(0.50, a=alpha, b=beta_param)
    p99 = beta.ppf(0.99, a=alpha, b=beta_param)

    ax.axvline(p1, color='blue', linestyle='-', label='1%: {:,.2f}'.format(p1 * (data_max - data_min) + data_min))
    ax.axvline(p50, color='green', linestyle='-', label='50%: {:,.2f}'.format(p50 * (data_max - data_min) + data_min))
    ax.axvline(p99, color='red', linestyle='-', label='99%: {:,.2f}'.format(p99 * (data_max - data_min) + data_min))

    ax.set_xlabel("Value (scaled to [0,1])")
    ax.set_ylabel("Density")
    ax.set_title(f"{perso_title}")
    ax.legend(loc="upper right")

    # Add a second x-axis with the original scale
    ax2 = ax.twiny()
    ax2.set_xlim(data_min, data_max)
    ax2.set_xlabel("Original Value")

    param_text = (
        f"Class n° = {i_class:.0f}\n"
        f"R² = {R2*100:.2f} %\n"
        f"$\\alpha$ = {alpha:.3f}\n"
        f"$\\beta$ = {beta_param:.3f}"
    )

    ax.text(
        0.98, 0.40,
        param_text,
        ha='right', va='bottom',
        transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.4",
                  facecolor="lightyellow",
                  edgecolor="black",
                  alpha=0.8)
    )

    ax.grid(True)
    st.pyplot(fig)

def plot_weibull_streamlit(data, N_bins, i_class, beta_shape, lambda_scale, R2, perso_title, show_hist,
                          show_model_pdf, show_model_cdf, show_estimated_pdf, show_estimated_cdf, show_scatter):
    fig, ax = plt.subplots(figsize=(8, 5))
    
    if show_hist:
        ax.hist(data, bins=N_bins, density=True, alpha=0.4, color='gray', edgecolor='black', label="Sample histogram")
    
    x = np.linspace(min(data), max(data), 1000)
    
    if show_model_pdf:
        pdf = weibull_min.pdf(x, c=beta_shape, scale=lambda_scale)
        ax.plot(x, pdf, 'k-', lw=2, label='Probability density')
    
    if show_model_cdf:
        cdf = weibull_min.cdf(x, c=beta_shape, scale=lambda_scale)
        ax.plot(x, cdf, 'b-', lw=2, label='Cumulative distribution')

    if show_scatter:
        y = np.zeros_like(data)
        ax.scatter(data, y, color="red", marker="|", alpha=0.5, label="Data points")

    # Percentiles
    p1 = weibull_min.ppf(0.01, c=beta_shape, scale=lambda_scale)
    p50 = weibull_min.ppf(0.50, c=beta_shape, scale=lambda_scale)
    p99 = weibull_min.ppf(0.99, c=beta_shape, scale=lambda_scale)

    ax.axvline(p1, color='blue', linestyle='-', label='1%: {:,.2f}'.format(p1))
    ax.axvline(p50, color='green', linestyle='-', label='50%: {:,.2f}'.format(p50))
    ax.axvline(p99, color='red', linestyle='-', label='99%: {:,.2f}'.format(p99))

    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.set_title(f"{perso_title}")
    ax.legend(loc="upper right")

    param_text = (
        f"Class n° = {i_class:.0f}\n"
        f"R² = {R2*100:.2f} %\n"
        f"$\\beta$ (shape) = {beta_shape:.3f}\n"
        f"$\\lambda$ (scale) = {lambda_scale:.3f}"
    )

    ax.text(
        0.98, 0.40,
        param_text,
        ha='right', va='bottom',
        transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.4",
                  facecolor="lightyellow",
                  edgecolor="black",
                  alpha=0.8)
    )

    ax.grid(True)
    st.pyplot(fig)

def plot_gamma_streamlit(data, N_bins, i_class, k_shape, theta_scale, R2, perso_title, show_hist,
                        show_model_pdf, show_model_cdf, show_estimated_pdf, show_estimated_cdf, show_scatter):
    fig, ax = plt.subplots(figsize=(8, 5))
    
    if show_hist:
        ax.hist(data, bins=N_bins, density=True, alpha=0.4, color='gray', edgecolor='black', label="Sample histogram")
    
    x = np.linspace(min(data), max(data), 1000)
    
    if show_model_pdf:
        pdf = gamma.pdf(x, a=k_shape, scale=theta_scale)
        ax.plot(x, pdf, 'k-', lw=2, label='Probability density')
    
    if show_model_cdf:
        cdf = gamma.cdf(x, a=k_shape, scale=theta_scale)
        ax.plot(x, cdf, 'b-', lw=2, label='Cumulative distribution')

    if show_scatter:
        y = np.zeros_like(data)
        ax.scatter(data, y, color="red", marker="|", alpha=0.5, label="Data points")

    # Percentiles
    p1 = gamma.ppf(0.01, a=k_shape, scale=theta_scale)
    p50 = gamma.ppf(0.50, a=k_shape, scale=theta_scale)
    p99 = gamma.ppf(0.99, a=k_shape, scale=theta_scale)

    ax.axvline(p1, color='blue', linestyle='-', label='1%: {:,.2f}'.format(p1))
    ax.axvline(p50, color='green', linestyle='-', label='50%: {:,.2f}'.format(p50))
    ax.axvline(p99, color='red', linestyle='-', label='99%: {:,.2f}'.format(p99))

    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.set_title(f"{perso_title}")
    ax.legend(loc="upper right")

    param_text = (
        f"Class n° = {i_class:.0f}\n"
        f"R² = {R2*100:.2f} %\n"
        f"$k$ (shape) = {k_shape:.3f}\n"
        f"$\\theta$ (scale) = {theta_scale:.3f}"
    )

    ax.text(
        0.98, 0.40,
        param_text,
        ha='right', va='bottom',
        transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.4",
                  facecolor="lightyellow",
                  edgecolor="black",
                  alpha=0.8)
    )

    ax.grid(True)
    st.pyplot(fig)

def plot_exponential_streamlit(data, N_bins, i_class, lambda_inv, R2, perso_title, show_hist,
                              show_model_pdf, show_model_cdf, show_estimated_pdf, show_estimated_cdf, show_scatter):
    fig, ax = plt.subplots(figsize=(8, 5))
    
    if show_hist:
        ax.hist(data, bins=N_bins, density=True, alpha=0.4, color='gray', edgecolor='black', label="Sample histogram")
    
    x = np.linspace(min(data), max(data), 1000)
    
    if show_model_pdf:
        pdf = expon.pdf(x, scale=lambda_inv)
        ax.plot(x, pdf, 'k-', lw=2, label='Probability density')
    
    if show_model_cdf:
        cdf = expon.cdf(x, scale=lambda_inv)
        ax.plot(x, cdf, 'b-', lw=2, label='Cumulative distribution')

    if show_scatter:
        y = np.zeros_like(data)
        ax.scatter(data, y, color="red", marker="|", alpha=0.5, label="Data points")

    # Percentiles
    p1 = expon.ppf(0.01, scale=lambda_inv)
    p50 = expon.ppf(0.50, scale=lambda_inv)
    p99 = expon.ppf(0.99, scale=lambda_inv)

    ax.axvline(p1, color='blue', linestyle='-', label='1%: {:,.2f}'.format(p1))
    ax.axvline(p50, color='green', linestyle='-', label='50%: {:,.2f}'.format(p50))
    ax.axvline(p99, color='red', linestyle='-', label='99%: {:,.2f}'.format(p99))

    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.set_title(f"{perso_title}")
    ax.legend(loc="upper right")

    param_text = (
        f"Class n° = {i_class:.0f}\n"
        f"R² = {R2*100:.2f} %\n"
        f"$\\lambda^{{-1}}$ (scale) = {lambda_inv:.3f}"
    )

    ax.text(
        0.98, 0.40,
        param_text,
        ha='right', va='bottom',
        transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.4",
                  facecolor="lightyellow",
                  edgecolor="black",
                  alpha=0.8)
    )

    ax.grid(True)
    st.pyplot(fig)

def find_best_class_CapgeminiLogic(model_df: pd.DataFrame) -> dict:        
    calc_df = model_df.copy()
    print(f"find_best_class_CapgeminiLogic diag checpoint 1: model df copied, schema : {calc_df.info()}")
    # FIT_P1 = 'percentile_1%'
    # FIT_P50 = 'percentile_50%'
    # FIT_P99 = 'percentile_99%'
    
    EMP_P1, EMP_P50, EMP_P99 = 'percentile_1%_emp', 'percentile_50%_emp', 'percentile_99%_emp'
    
    print("find_best_class_CapgeminiLogic diag checpoint 1: sample percetiles and empirical percentiles abs differences are computed")
    calc_df['AbsDiff_P1'] = np.abs(calc_df['percentile_1%'] - calc_df[EMP_P1])
    calc_df['AbsDiff_P50'] = np.abs(calc_df['percentile_50%'] - calc_df[EMP_P50])
    calc_df['AbsDiff_P99'] = np.abs(calc_df['percentile_99%'] - calc_df[EMP_P99])
    print("diag checpoint 1.2")
    # Scénario 1: 1% et 50% (Critère de la queue inférieure)
    calc_df['SumDiff_1_50'] = calc_df['AbsDiff_P1'] + calc_df['AbsDiff_P50']
                
    # Scénario 2: 50% et 99% (Critère de la queue supérieure)
    calc_df['SumDiff_50_99'] = calc_df['AbsDiff_P50'] + calc_df['AbsDiff_P99']
    
    # Scénario 3: Somme Globale (Critère Capgemini)
    calc_df['SumDiff_All'] = calc_df['AbsDiff_P1'] + calc_df['AbsDiff_P50'] + calc_df['AbsDiff_P99']

    row_pos_1_50 = calc_df['SumDiff_1_50'].idxmin()
    row_pos_50_99 = calc_df['SumDiff_50_99'].idxmin()
    row_pos_1_50_99 = calc_df['SumDiff_All'].idxmin()

    index_best_1_50 = int(calc_df.loc[row_pos_1_50, 'class_index'])
    index_best_50_99 = int(calc_df.loc[row_pos_50_99, 'class_index'])
    index_best_1_50_99 = int(calc_df.loc[row_pos_1_50_99, 'class_index'])
  
    bestClassIndexCapgemini = index_best_1_50_99 
  
    # --- 5. Retourner les résultats ---
    return {
        'index_best_1_50': index_best_1_50,
        'index_best_50_99': index_best_50_99,
        'index_best_1_50_99': index_best_1_50_99,
        'bestClassIndexCapgemini': bestClassIndexCapgemini
    }

def find_best_distribution(data):
    # Suppress warnings during distribution fitting
    warnings.filterwarnings('ignore')
    
    def fit_distribution(data, dist_name, dist_obj):
        """Helper function to fit a distribution to data"""
        try:
            params = dist_obj.fit(data)
            loglik = np.sum(dist_obj.logpdf(data, *params))
            k = len(params)
            n = len(data)
            aic = 2 * k - 2 * loglik
            bic = k * np.log(n) - 2 * loglik

            # Compute RMSE between histogram and fitted PDF
            hist_y, bin_edges = np.histogram(data, bins='auto', density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            fitted_pdf = dist_obj.pdf(bin_centers, *params)
            rmse = np.sqrt(np.mean((hist_y - fitted_pdf) ** 2))

            return {
                "Distribution": dist_name,
                "AIC": aic,
                "BIC": bic,
                "LogLik": loglik,
                "RMSE": rmse,
                "Parameters": params
            }
        except Exception as e:
            print(f"Failed to fit {dist_name}: {e}")
            return {
                "Distribution": dist_name,
                "AIC": np.inf,
                "BIC": np.inf,
                "LogLik": -np.inf,
                "RMSE": np.inf,
                "Parameters": None
            }

    # Define distributions to test
    distributions = {
        "normal": stats.norm,
        "lognormal": stats.lognorm,
        "beta": stats.beta,
        "gamma": stats.gamma,
        "exponential": stats.expon,
        "gumbel": stats.gumbel_r,
    }
    
    # Fit each distribution
    results = [fit_distribution(data, name, dist) for name, dist in distributions.items()]
    results_df = pd.DataFrame(results).sort_values("AIC")
    
    # Get the best distribution (lowest AIC)
    best_dist = results_df.iloc[0]["Distribution"]
    best_aic = results_df.iloc[0]["AIC"]
    
    print(f"Based on AIC, the best fitting distribution is: {best_dist} (AIC: {best_aic:.2f})")
    
    # Return the best distribution name and full results dataframe
    return best_dist, results_df

def plot_best_distribution(data, dist_name, results_df,best_mrd_class,df_classes_modeling_results):
    # Get distribution object and parameters
    distributions = {
        "normal": stats.norm,
        "lognormal": stats.lognorm,
        "beta": stats.beta,
        "gamma": stats.gamma,
        "exponential": stats.expon,
        "gumbel": stats.gumbel_r,
    }
    
    dist_row = results_df[results_df['Distribution'] == dist_name].iloc[0]
    dist_obj = distributions[dist_name]
    params = dist_row['Parameters']
    
    # Calculate sample data percentiles
    p1_emp = np.percentile(data, 1)
    p50_emp = np.percentile(data, 50)
    p99_emp = np.percentile(data, 99)

    #Get percentiles of best class index from summary df
    p1_class=df_classes_modeling_results.loc[best_mrd_class, 'percentile_1%']
    p50_class=df_classes_modeling_results.loc[best_mrd_class, 'percentile_50%']
    p99_class=df_classes_modeling_results.loc[best_mrd_class, 'percentile_99%']


    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot histogram
    plt.hist(data, bins='auto', density=True, alpha=0.6, color='g', label='Data Histogram')
    
    # Plot PDF
    x = np.linspace(min(data), max(data), 1000)
    try:
        y = dist_obj.pdf(x, *params)
        plt.plot(x, y, 'r-', lw=2, label=f'Best Fit: {dist_name}')
        
        # PDF percentile lines
        plt.axvline(p1_class, color='blue', linestyle=':', alpha=0.7, label=f'1% PDF Percentile: {p1_class:.2f}')
        plt.axvline(p50_class, color='black', linestyle=':', alpha=0.7, label=f'50% PDF Percentile: {p50_class:.2f}')
        plt.axvline(p99_class, color='purple', linestyle=':', alpha=0.7, label=f'99% PDF Percentile: {p99_class:.2f}')
      
        # sample data percentile lines
        plt.axvline(p1_emp, color='blue', linestyle='-', linewidth=2, alpha=0.9,label=f'1% from sample {best_mrd_class_index}: {p1_emp:.2f}')
        plt.axvline(p50_emp, color='black', linestyle='-', linewidth=2, alpha=0.9,label=f'50% from sample {best_mrd_class_index}: {p50_emp:.2f}')
        plt.axvline(p99_emp, color='purple', linestyle='-', linewidth=2, alpha=0.9,label=f'99% from sample {best_mrd_class_index}: {p99_emp:.2f}')
    except Exception as e:
        print(f"Error plotting distribution: {e}")
    
    plt.title(f'Data Histogram with Best Fit Distribution: {dist_name} & Best Class, index n°: {best_mrd_class_index}')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


#################################################################################################################################################################"
# --- Function to plot best distribution ---
def plot_best_distribution2(study_name_folder, col_name, data, dist_name, results_df, best_mrd_class_index, df_classes_modeling_results, fallback_used=False, non_modelable=False):
    # Calculate sample percentiles
    p1_emp, p50_emp, p99_emp = np.percentile(data, [1, 50, 99])
    
    # ========================================
    # HANDLE NON-MODELABLE CASE
    # ========================================
    if non_modelable:
        plt.figure(figsize=(10, 6))
        plt.hist(data, bins=30, density=True, alpha=0.6, color='g', label='Data Histogram')
        
        # Only plot empirical percentiles (no PDF)
        plt.axvline(p1_emp, color='blue', linestyle='-', linewidth=2, alpha=0.9, label=f'1% Sample: {p1_emp:.2f}')
        plt.axvline(p50_emp, color='black', linestyle='-', linewidth=2, alpha=0.9, label=f'50% Sample: {p50_emp:.2f}')
        plt.axvline(p99_emp, color='purple', linestyle='-', linewidth=2, alpha=0.9, label=f'99% Sample: {p99_emp:.2f}')
        
        # Non-modelable warning box
        sample_size = len(data)
        plt.text(0.02, 0.98, f"⚠️ NON MODELISABLE\n(Taille échantillon: {sample_size} < 30)", 
                 transform=plt.gca().transAxes, 
                 fontsize=12, 
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        plt.suptitle(f"{col_name} - Non modelisable (Échantillon insuffisant)", fontsize=14, fontweight='bold')
        plt.title(f'Data Histogram (Sample size: {sample_size})', fontsize=12)
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.legend(loc='upper right')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        # Save plots
        import os 
        from pathlib import Path
        local_parent_folder = os.path.join(os.getcwd(), "Modeling Results")
        os.makedirs(local_parent_folder, exist_ok=True)
        local_output_folder = os.path.join(local_parent_folder, study_name_folder)
        os.makedirs(local_output_folder, exist_ok=True)
        local_save_path = os.path.join(local_output_folder, f"{col_name}.png")
        plt.savefig(local_save_path, bbox_inches='tight')
        print(f"Plot saved locally at: {local_save_path}")

        dbfs_base_path = "/Volumes/dafe_dev/customer_knowledge_utils/modeling_results"
        dbfs_output_folder = os.path.join(dbfs_base_path, study_name_folder)
        os.makedirs(dbfs_output_folder, exist_ok=True)
        dbfs_save_path = os.path.join(dbfs_output_folder, f"{col_name}.png")
        plt.savefig(dbfs_save_path, bbox_inches='tight')
        print(f"Plot saved on DBFS at: {dbfs_save_path}")
        
        plt.show()
        plt.close()
        return
    
    # ========================================
    # NORMAL PLOTTING (sample size >= 30)
    # ========================================
    distributions = {
        "normal": stats.norm,
        "lognormal": stats.lognorm,
        "beta": stats.beta,
        "gamma": stats.gamma,
        "exponential": stats.expon,
        "gumbel": stats.gumbel_r,
        "weibull": stats.weibull_min,
    }
    
    dist_row = results_df[results_df['Distribution'] == dist_name].iloc[0]
    dist_obj = distributions.get(dist_name, None)
    params = dist_row['Parameters'] if 'Parameters' in dist_row else []

    # Get best class row - HANDLE FALLBACK HERE
    if fallback_used or best_mrd_class_index is None or best_mrd_class_index == 'Estimation':
        best_class_row = df_classes_modeling_results.iloc[0]
        display_class_text = "LOI ESTIMÉE"
    else:
        results_best_class = find_best_class_CapgeminiLogic(df_classes_modeling_results)
        best_class_idx = results_best_class['index_best_1_50_99']
        best_class_row = df_classes_modeling_results[df_classes_modeling_results['class_index'] == best_class_idx].iloc[0]
        display_class_text = f"Class n° = {best_mrd_class_index}"
    
    p1_class = best_class_row['percentile_1%']
    p50_class = best_class_row['percentile_50%']
    p99_class = best_class_row['percentile_99%']
    
    # Convert R2 to float, handle string/NA cases
    R2 = best_class_row['R2']
    if isinstance(R2, str):
        if R2.upper() == 'NA':
            R2 = np.nan
        else:
            try:
                R2 = float(R2)
            except:
                R2 = np.nan
    
    R2_display = f"{R2*100:.2f}" if not np.isnan(R2) else "NA"

    # Create plot
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=30, density=True, alpha=0.6, color='g', label='Data Histogram')
    x = np.linspace(min(data), max(data), 1000)

    try:
        # --- Distribution-specific plotting ---
        if dist_name == "lognormal":
            shape = best_class_row['slnx']
            scale = np.exp(best_class_row['mulnx'])
            loc = 0
            pdf = stats.lognorm.pdf(x, s=shape, loc=loc, scale=scale)
            param_text = (
                f"{display_class_text}\n"
                f"R² = {R2_display} %\n"
                f"$\\mu_{{ln(x)}}$ = {best_class_row['mulnx']:.3f}\n"
                f"$\\sigma_{{ln(x)}}$ = {shape:.3f}\n"
                f"$d$ = {loc:.3f}"
            )

        elif dist_name == "normal":
            mu = best_class_row['mu']
            sigma = best_class_row['sigma']
            pdf = stats.norm.pdf(x, loc=mu, scale=sigma)
            param_text = (
                f"{display_class_text}\n"
                f"R² = {R2_display} %\n"
                f"$\\mu$ = {mu:.3f}\n"
                f"$\\sigma$ = {sigma:.3f}"
            )

        elif dist_name == "beta":
            alpha = best_class_row['alpha']
            beta_param = best_class_row['beta']
            data_min, data_max = min(data), max(data)
            scaled_x = np.linspace(0, 1, 1000)
            pdf = stats.beta.pdf(scaled_x, a=alpha, b=beta_param)
            x = data_min + scaled_x * (data_max - data_min)
            pdf = pdf / (data_max - data_min)
            param_text = (
                f"{display_class_text}\n"
                f"R² = {R2_display} %\n"
                f"$\\alpha$ = {alpha:.3f}\n"
                f"$\\beta$ = {beta_param:.3f}"
            )

        elif dist_name == "gamma":
            k_shape = best_class_row['k_shape']
            theta_scale = best_class_row['theta_scale']
            pdf = stats.gamma.pdf(x, a=k_shape, scale=theta_scale)
            param_text = (
                f"{display_class_text}\n"
                f"R² = {R2_display} %\n"
                f"$k$ (shape) = {k_shape:.3f}\n"
                f"$\\theta$ (scale) = {theta_scale:.3f}"
            )

        elif dist_name in ["weibull", "weibull_min"]:
            beta_shape = best_class_row['beta_shape']
            lambda_scale = best_class_row['lambda_scale']
            pdf = stats.weibull_min.pdf(x, c=beta_shape, scale=lambda_scale)
            param_text = (
                f"{display_class_text}\n"
                f"R² = {R2_display} %\n"
                f"$\\beta$ (shape) = {beta_shape:.3f}\n"
                f"$\\lambda$ (scale) = {lambda_scale:.3f}"
            )

        elif dist_name in ["exponential", "expon"]:
            lambda_inv = best_class_row['lambda_inv']
            pdf = stats.expon.pdf(x, scale=lambda_inv)
            param_text = (
                f"{display_class_text}\n"
                f"R² = {R2_display} %\n"
                f"$\\lambda^{{-1}}$ = {lambda_inv:.3f}"
            )

        else:
            pdf = dist_obj.pdf(x, *params) if dist_obj else np.zeros_like(x)
            param_text = (
                f"{display_class_text}\n"
                f"R² = {R2_display} %\n"
                f"Parameters: {[f'{p:.3f}' for p in params]}"
            )

        plt.plot(x, pdf, 'r-', lw=2, label=f'Best Fit: {dist_name}')

        # Percentile lines
        plt.axvline(p1_class, color='blue', linestyle=':', alpha=0.7, label=f'1% PDF: {p1_class:.2f}')
        plt.axvline(p50_class, color='black', linestyle=':', alpha=0.7, label=f'50% PDF: {p50_class:.2f}')
        plt.axvline(p99_class, color='purple', linestyle=':', alpha=0.7, label=f'99% PDF: {p99_class:.2f}')

        plt.axvline(p1_emp, color='blue', linestyle='-', linewidth=2, alpha=0.9, label=f'1% Sample: {p1_emp:.2f}')
        plt.axvline(p50_emp, color='black', linestyle='-', linewidth=2, alpha=0.9, label=f'50% Sample: {p50_emp:.2f}')
        plt.axvline(p99_emp, color='purple', linestyle='-', linewidth=2, alpha=0.9, label=f'99% Sample: {p99_emp:.2f}')

        # Parameter text box
        plt.text(
            0.98, 0.40,
            param_text,
            ha='right', va='bottom',
            transform=plt.gca().transAxes,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", edgecolor="black", alpha=0.8)
        )

    except Exception as e:
        print(f"Error plotting distribution: {e}")
        import traceback
        traceback.print_exc()

    # Title and labels
    lowest_mrd = best_class_row['Mean relat diff']
    
    if fallback_used:
        plt.suptitle(f"{col_name} - {dist_name} (LOI ESTIMÉE - Direct Estimation)", fontsize=14, fontweight='bold')
        plt.text(0.02, 0.98, "⚠️ LOI ESTIMÉE utilisée", 
                 transform=plt.gca().transAxes, 
                 fontsize=12, 
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    else:
        plt.suptitle(f"{col_name} - {dist_name} (Class {best_mrd_class_index})", fontsize=14, fontweight='bold')

    subtitle_text = f'Data Histogram with Best Fit: {dist_name}'
    if not fallback_used:
        subtitle_text += f' & Best Class {best_mrd_class_index}'
    subtitle_text += f', Lowest M.R.D = {lowest_mrd:.3f}'
    plt.title(subtitle_text, fontsize=12)

    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend(loc='upper right')
    plt.grid(alpha=0.3)
    plt.tight_layout()

    import os 
    from pathlib import Path
    local_parent_folder = os.path.join(os.getcwd(), "Modeling Results")
    os.makedirs(local_parent_folder, exist_ok=True)
    local_output_folder = os.path.join(local_parent_folder, study_name_folder)
    os.makedirs(local_output_folder, exist_ok=True)
    local_save_path = os.path.join(local_output_folder, f"{col_name}.png")
    plt.savefig(local_save_path, bbox_inches='tight')
    print(f"Plot saved locally at: {local_save_path}")

    dbfs_base_path = "/Volumes/dafe_dev/customer_knowledge_utils/modeling_results"
    dbfs_output_folder = os.path.join(dbfs_base_path, study_name_folder)
    os.makedirs(dbfs_output_folder, exist_ok=True)
    dbfs_save_path = os.path.join(dbfs_output_folder, f"{col_name}.png")
    plt.savefig(dbfs_save_path, bbox_inches='tight')
    print(f"Plot saved on DBFS at: {dbfs_save_path}")
    
    plt.show()
    plt.close()

def modeling_function(df_to_model, list_of_normalized_column_to_model, max_nbr_classes, model_choice="auto", study_name=None):
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.getOrCreate()
    
    import numpy as np
    import pandas as pd
    from tqdm import tqdm
    from scipy import stats
    from datetime import datetime
    import os

    # Create study folder
    today_str = datetime.today().strftime('%Y-%m-%d')

    if study_name is None or study_name.strip() == "":
        study_name_folder = today_str
    else:
        study_name_folder = f"{study_name}_{today_str}"

    local_output_folder = os.path.join(os.getcwd(), "Modeling Results", study_name_folder)
    os.makedirs(local_output_folder, exist_ok=True)

    dbfs_base_path = "/Volumes/dafe_dev/customer_knowledge_utils/modeling_results"
    dbfs_output_folder = os.path.join(dbfs_base_path, study_name_folder)
    os.makedirs(dbfs_output_folder, exist_ok=True)

    print(f"Plots and outputs will be saved in:\nLocal: {local_output_folder}\nDBFS: {dbfs_output_folder}")

    global data

    all_results = {}
    nb_c = max_nbr_classes
    mrd_threshold = 0.1

    for col_name in list_of_normalized_column_to_model:
        print(f"Processing column: {col_name}")

        local_data = df_to_model.filter(f"{col_name} > 0").select(col_name).rdd.flatMap(lambda x: x).collect()
        local_data = [x for x in local_data if x is not None]

        if len(local_data) == 0:
            print(f"Skipping {col_name}: empty or null column")
            continue

        data = local_data
        N_data = len(data)
        data_min, data_max = np.min(data), np.max(data)
        data_mean = np.mean(data)
        
        # ========================================
        # CHECK SAMPLE SIZE - NEW FEATURE
        # ========================================
        if N_data < 30:
            print(f"⚠️ Sample size ({N_data}) < 30 - No modeling performed, using empirical percentiles only")
            
            # Calculate empirical percentiles
            p1_emp = np.percentile(data, 1)
            p50_emp = np.percentile(data, 50)
            p99_emp = np.percentile(data, 99)
            
            # Create non-modelable result
            non_modelable_dict = {
                'criteria': col_name,
                'data_min': data_min,
                'data_max': data_max,
                'data_mean': data_mean,
                'loi': 'Non modelisable',
                'class_index': 'N/A',
                'R2': np.nan,
                'percentile_1%': p1_emp,  # PDF percentiles = empirical percentiles
                'percentile_50%': p50_emp,
                'percentile_99%': p99_emp,
                'percentile_1%_emp': p1_emp,
                'percentile_50%_emp': p50_emp,
                'percentile_99%_emp': p99_emp,
                'MSE': np.nan,
                'Mean relat diff': 0.0,  # Perfect match since they're identical
                'relat_diff_1%': 0.0,
                'relat_diff_50%': 0.0,
                'relat_diff_99%': 0.0,
                'sample_size': N_data
            }
            
            df_result = pd.DataFrame([non_modelable_dict])
            
            # Plot with non-modelable flag
            try:
                plot_best_distribution2(
                    study_name_folder, col_name, data, 'Non modelisable', 
                    None,  # No PDF results
                    None,  # No class
                    df_result, 
                    fallback_used=False,
                    non_modelable=True
                )
            except Exception as e:
                print(f"Error in displaying plots for column {col_name}: {e}")
            
            all_results[col_name] = {
                'data': data,
                'best_dist': 'Non modelisable',
                'PDF_selection_results': None,
                'best_mrd_class': None,
                # 'fallback_used': False,
                # 'non_modelable': True,
                # 'sample_size': N_data,
                'df_classes_modeling_results_best_classe': df_result,
                'summary_df': df_result,
            }
            
            print(f"END OF Processing column: {col_name} (Non modelisable) ##################################################################")
            continue
        
        # ========================================
        # NORMAL MODELING FLOW (sample size >= 30)
        # ========================================
        best_dist_name, results_df = find_best_distribution(data)

        # Distribution selection
        if model_choice == "auto":
            print("Finding best distribution fit...")
            PDF_selection_results = results_df
            best_dist_info = results_df[results_df['Distribution'] == best_dist_name].iloc[0].to_dict()
            display(f"The modeling is being performed using this PDF: {best_dist_name}")
        else:
            best_dist_name = model_choice
            display(f"Using user-specified distribution: {best_dist_name}")
            PDF_selection_results = results_df
            best_dist_info = {"Distribution": best_dist_name}

        # Modeling loop over classes
        rows = []
        print(f"Performing class modeling for {best_dist_name} distribution...")
        progress_bar = tqdm(range(2, nb_c + 1), desc="Modeling Progress", unit="class")

        if best_dist_name == "lognormal":
            for i_class in progress_bar:
                result = lognormal_class_stats(i_class, data)
                rows.append({
                    'criteria': col_name, 'data_min': data_min, 'data_max': data_max, 'data_mean': data_mean, 'loi': best_dist_name,
                    'class_index': i_class, 'slnx': result['slnx'], 'mulnx': result['mulnx'], 'R2': result['R2'],
                    'percentile_1%': result['fitted_percentiles'][0], 'percentile_50%': result['fitted_percentiles'][1],
                    'percentile_99%': result['fitted_percentiles'][2], 'percentile_1%_emp': result['empirical_percentiles'][0],
                    'percentile_50%_emp': result['empirical_percentiles'][1], 'percentile_99%_emp': result['empirical_percentiles'][2],
                    'MSE': result['MSE'], 'Mean relat diff': result['Mean relat diff'],
                    'relat_diff_1%': result['relative_diff'][0], 'relat_diff_50%': result['relative_diff'][1],
                    'relat_diff_99%': result['relative_diff'][2]
                })
            columns_to_select = [
                'criteria', 'data_min', 'data_max', 'data_mean', 'loi', 'class_index', 'slnx', 'mulnx', 'R2',
                'percentile_1%', 'percentile_50%', 'percentile_99%',
                'percentile_1%_emp', 'percentile_50%_emp', 'percentile_99%_emp'
            ]

        elif best_dist_name == "normal":
            for i_class in progress_bar:
                result = normal_class_stats(i_class, data)
                rows.append({
                    'criteria': col_name, 'data_min': data_min, 'data_max': data_max, 'data_mean': data_mean, 'loi': best_dist_name,
                    'class_index': i_class, 'mu': result['mu'], 'sigma': result['sigma'], 'R2': result['R2'],
                    'percentile_1%': result['fitted_percentiles'][0], 'percentile_50%': result['fitted_percentiles'][1],
                    'percentile_99%': result['fitted_percentiles'][2], 'percentile_1%_emp': result['empirical_percentiles'][0],
                    'percentile_50%_emp': result['empirical_percentiles'][1], 'percentile_99%_emp': result['empirical_percentiles'][2],
                    'MSE': result['MSE'], 'Mean relat diff': result['Mean relat diff'],
                    'relat_diff_1%': result['relative_diff'][0], 'relat_diff_50%': result['relative_diff'][1],
                    'relat_diff_99%': result['relative_diff'][2]
                })
            columns_to_select = [
                'criteria', 'data_min', 'data_max', 'data_mean', 'loi', 'class_index', 'mu', 'sigma', 'R2',
                'percentile_1%', 'percentile_50%', 'percentile_99%',
                'percentile_1%_emp', 'percentile_50%_emp', 'percentile_99%_emp'
            ]

        elif best_dist_name == "beta":
            for i_class in progress_bar:
                result = beta_class_stats(i_class, data)
                rows.append({
                    'criteria': col_name, 'data_min': data_min, 'data_max': data_max, 'data_mean': data_mean, 'loi': best_dist_name,
                    'class_index': i_class, 'alpha': result['alpha'], 'beta': result['beta'], 'R2': result['R2'],
                    'percentile_1%': result['fitted_percentiles'][0], 'percentile_50%': result['fitted_percentiles'][1],
                    'percentile_99%': result['fitted_percentiles'][2], 'percentile_1%_emp': result['empirical_percentiles'][0],
                    'percentile_50%_emp': result['empirical_percentiles'][1], 'percentile_99%_emp': result['empirical_percentiles'][2],
                    'MSE': result['MSE'], 'Mean relat diff': result['Mean relat diff'],
                    'relat_diff_1%': result['relative_diff'][0], 'relat_diff_50%': result['relative_diff'][1],
                    'relat_diff_99%': result['relative_diff'][2]
                })
            columns_to_select = [
                'criteria', 'data_min', 'data_max', 'data_mean', 'loi', 'class_index', 'alpha', 'beta', 'R2',
                'percentile_1%', 'percentile_50%', 'percentile_99%',
                'percentile_1%_emp', 'percentile_50%_emp', 'percentile_99%_emp'
            ]

        elif best_dist_name == "gamma":
            for i_class in progress_bar:
                result = gamma_class_stats(i_class, data)
                rows.append({
                    'criteria': col_name, 'data_min': data_min, 'data_max': data_max, 'data_mean': data_mean, 'loi': best_dist_name,
                    'class_index': i_class, 'k_shape': result['k_shape'], 'theta_scale': result['theta_scale'], 'R2': result['R2'],
                    'percentile_1%': result['fitted_percentiles'][0], 'percentile_50%': result['fitted_percentiles'][1],
                    'percentile_99%': result['fitted_percentiles'][2], 'percentile_1%_emp': result['empirical_percentiles'][0],
                    'percentile_50%_emp': result['empirical_percentiles'][1], 'percentile_99%_emp': result['empirical_percentiles'][2],
                    'MSE': result['MSE'], 'Mean relat diff': result['Mean relat diff'],
                    'relat_diff_1%': result['relative_diff'][0], 'relat_diff_50%': result['relative_diff'][1],
                    'relat_diff_99%': result['relative_diff'][2]
                })
            columns_to_select = [
                'criteria', 'data_min', 'data_max', 'data_mean', 'loi', 'class_index', 'k_shape', 'theta_scale', 'R2',
                'percentile_1%', 'percentile_50%', 'percentile_99%',
                'percentile_1%_emp', 'percentile_50%_emp', 'percentile_99%_emp'
            ]

        elif best_dist_name == "weibull":
            for i_class in progress_bar:
                result = weibull_class_stats(i_class, data)
                rows.append({
                    'criteria': col_name, 'data_min': data_min, 'data_max': data_max, 'data_mean': data_mean, 'loi': best_dist_name,
                    'class_index': i_class, 'beta_shape': result['beta_shape'], 'lambda_scale': result['lambda_scale'], 'R2': result['R2'],
                    'percentile_1%': result['fitted_percentiles'][0], 'percentile_50%': result['fitted_percentiles'][1],
                    'percentile_99%': result['fitted_percentiles'][2], 'percentile_1%_emp': result['empirical_percentiles'][0],
                    'percentile_50%_emp': result['empirical_percentiles'][1], 'percentile_99%_emp': result['empirical_percentiles'][2],
                    'MSE': result['MSE'], 'Mean relat diff': result['Mean relat diff'],
                    'relat_diff_1%': result['relative_diff'][0], 'relat_diff_50%': result['relative_diff'][1],
                    'relat_diff_99%': result['relative_diff'][2]
                })
            columns_to_select = [
                'criteria', 'data_min', 'data_max','data_mean', 'loi', 'class_index', 'beta_shape', 'lambda_scale', 'R2',
                'percentile_1%', 'percentile_50%', 'percentile_99%',
                'percentile_1%_emp', 'percentile_50%_emp', 'percentile_99%_emp'
            ]

        elif best_dist_name == "exponential":
            for i_class in progress_bar:
                result = exponential_class_stats(i_class, data)
                rows.append({
                    'criteria': col_name, 'data_min': data_min, 'data_max': data_max, 'data_mean': data_mean, 'loi': best_dist_name,
                    'class_index': i_class, 'lambda_inv': result['lambda_inv'], 'R2': result['R2'],
                    'percentile_1%': result['fitted_percentiles'][0], 'percentile_50%': result['fitted_percentiles'][1],
                    'percentile_99%': result['fitted_percentiles'][2], 'percentile_1%_emp': result['empirical_percentiles'][0],
                    'percentile_50%_emp': result['empirical_percentiles'][1], 'percentile_99%_emp': result['empirical_percentiles'][2],
                    'MSE': result['MSE'], 'Mean relat diff': result['Mean relat diff'],
                    'relat_diff_1%': result['relative_diff'][0], 'relat_diff_50%': result['relative_diff'][1],
                    'relat_diff_99%': result['relative_diff'][2]
                })
            columns_to_select = [
                'criteria', 'data_min', 'data_max', 'data_mean', 'loi', 'class_index', 'lambda_inv', 'R2',
                'percentile_1%', 'percentile_50%', 'percentile_99%',
                'percentile_1%_emp', 'percentile_50%_emp', 'percentile_99%_emp'
            ]

        else:
            print(f"Warning: Distribution '{best_dist_name}' not supported. Using lognormal instead.")
            for i_class in progress_bar:
                result = lognormal_class_stats(i_class, data)
                rows.append({
                    'criteria': col_name, 'data_min': data_min, 'data_max': data_max,  'data_mean': data_mean, 'loi': "lognormal",
                    'class_index': i_class, 'param1': result['slnx'], 'mulnx': result['mulnx'], 'R2': result['R2'],
                    'percentile_1%': result['fitted_percentiles'][0], 'percentile_50%': result['fitted_percentiles'][1],
                    'percentile_99%': result['fitted_percentiles'][2], 'percentile_1%_emp': result['empirical_percentiles'][0],
                    'percentile_50%_emp': result['empirical_percentiles'][1], 'percentile_99%_emp': result['empirical_percentiles'][2],
                    'MSE': result['MSE'], 'Mean relat diff': result['Mean relat diff'],
                    'relat_diff_1%': result['relative_diff'][0], 'relat_diff_50%': result['relative_diff'][1],
                    'relat_diff_99%': result['relative_diff'][2]
                })
            
            columns_to_select = [
                'criteria', 'data_min', 'data_max', 'data_mean', 'loi', 'class_index', 'slnx', 'mulnx', 'R2',
                'percentile_1%', 'percentile_50%', 'percentile_99%',
                'percentile_1%_emp', 'percentile_50%_emp', 'percentile_99%_emp'
            ]
            
        # Results aggregation
        summary_df = pd.DataFrame(rows)
        df_classes = summary_df.iloc[3:] if len(summary_df) > 3 else summary_df

        display(df_classes)

        results_best_class = find_best_class_CapgeminiLogic(df_classes)
        best_class_idx = results_best_class['index_best_1_50_99']
        
        fallback_result = check_mrd_threshold_and_fallback(
            model_df=df_classes,
            data=data,
            d=None,
            mrd_threshold=mrd_threshold,
            distribution_type=best_dist_name
        )
        
        print(f"Fallback status: {fallback_result['status']}")
        print(f"Use fallback: {fallback_result['use_fallback']}")
        
        if fallback_result['use_fallback']:
            print(f"⚠️ Fallback activated - using direct parameter estimation")
            fallback_dict = fallback_result['result']
            
            # Transform fallback result to match class-based structure
            if isinstance(fallback_dict, dict):
                # Extract percentiles from lists
                fitted_percentiles = fallback_dict.get('fitted_percentiles', [np.nan, np.nan, np.nan])
                empirical_percentiles = fallback_dict.get('empirical_percentiles', [np.nan, np.nan, np.nan])
                relative_diff = fallback_dict.get('relative_diff', [np.nan, np.nan, np.nan])
                
                # Build new dict with proper column structure
                transformed_dict = {
                    'criteria': col_name,
                    'data_min': data_min,
                    'data_max': data_max,
                    'data_mean': data_mean,
                    'loi': best_dist_name,
                    'class_index': 'Estimation',
                    'R2': fallback_dict.get('R2', np.nan),
                    'percentile_1%': fitted_percentiles[0],
                    'percentile_50%': fitted_percentiles[1],
                    'percentile_99%': fitted_percentiles[2],
                    'percentile_1%_emp': empirical_percentiles[0],
                    'percentile_50%_emp': empirical_percentiles[1],
                    'percentile_99%_emp': empirical_percentiles[2],
                    'MSE': fallback_dict.get('MSE', np.nan),
                    'Mean relat diff': fallback_dict.get('Mean relat diff', np.nan),
                    'relat_diff_1%': relative_diff[0],
                    'relat_diff_50%': relative_diff[1],
                    'relat_diff_99%': relative_diff[2]
                }
                
                # Add distribution-specific parameters
                if best_dist_name == "lognormal":
                    transformed_dict['slnx'] = fallback_dict.get('slnx', np.nan)
                    transformed_dict['mulnx'] = fallback_dict.get('mulnx', np.nan)
                elif best_dist_name == "normal":
                    transformed_dict['mu'] = fallback_dict.get('mu', np.nan)
                    transformed_dict['sigma'] = fallback_dict.get('sigma', np.nan)
                elif best_dist_name == "beta":
                    transformed_dict['alpha'] = fallback_dict.get('alpha', np.nan)
                    transformed_dict['beta'] = fallback_dict.get('beta', np.nan)
                elif best_dist_name == "gamma":
                    transformed_dict['k_shape'] = fallback_dict.get('k_shape', np.nan)
                    transformed_dict['theta_scale'] = fallback_dict.get('theta_scale', np.nan)
                elif best_dist_name == "weibull":
                    transformed_dict['beta_shape'] = fallback_dict.get('beta_shape', np.nan)
                    transformed_dict['lambda_scale'] = fallback_dict.get('lambda_scale', np.nan)
                elif best_dist_name == "exponential":
                    transformed_dict['lambda_inv'] = fallback_dict.get('lambda_inv', np.nan)
                
                df_classes_final = pd.DataFrame([transformed_dict])
            else:
                df_classes_final = fallback_dict
            
            best_mrd_class = None
            fallback_used = True
        else:
            print(f"✓ Using class-based modeling")
            df_classes_final = df_classes
            best_mrd_class = best_class_idx
            fallback_used = False
        
        # Select best class row
        if fallback_used:
            best_class_row = df_classes_final.iloc[0]
        else:
            best_class_row = df_classes_final[df_classes_final['class_index'] == best_mrd_class].iloc[0]
        
        df_classes_modeling_results_best_classe = df_classes_final[columns_to_select] if not fallback_used else df_classes_final

        R2 = best_class_row['R2']
        display(f"Best class: {best_mrd_class if not fallback_used else 'Fallback (direct estimation)'}")
        display(df_classes_modeling_results_best_classe)

        # Plot
        try:
            plot_best_distribution2(study_name_folder, col_name, data, best_dist_name, PDF_selection_results, best_mrd_class, df_classes_final, fallback_used=fallback_used, non_modelable=False)
        except Exception as e:
            print(f"Error in displaying plots for column {col_name}: {e}")

        graphs = {}

        all_results[col_name] = {
            'data': data,
            'best_dist': best_dist_name,
            'PDF_selection_results': PDF_selection_results,
            'best_mrd_class': best_mrd_class,
            # 'fallback_used': fallback_used,
            # 'non_modelable': False,
            # 'sample_size': N_data,
            'df_classes_modeling_results_best_classe': df_classes_modeling_results_best_classe,
            'summary_df': df_classes_final,
        }

        print(f"END OF Processing column: {col_name} ##################################################################")

    return all_results

import pandas as pd
import json
from pyspark.sql import SparkSession

def aggregate_modeling_results(results, list_of_columns, study, sample_description):
    aggregated_df = []

    # Convert sample_description dict to JSON string for Spark
    sample_description_str = json.dumps(sample_description)

    for column in list_of_columns:
        try:
            df_classes = results[column]['df_classes_modeling_results_best_classe']
            df_classes = df_classes.copy()
            
            # Define columns that should NOT be included in parameters
            columns_without_parameters = [
                'criteria', 'data_min', 'data_max', 'data_mean', 'loi', 'class_index', 'R2',
                'percentile_1%', 'percentile_50%', 'percentile_99%',
                'percentile_1%_emp', 'percentile_50%_emp', 'percentile_99%_emp',
                'study_title', 'sample_description'
            ]
            
            # Create parameters dictionary for each row
            df_classes['parameters'] = df_classes.apply(
                lambda row: {
                    col: row[col] 
                    for col in row.index 
                    if col not in columns_without_parameters and pd.notna(row[col])
                }, 
                axis=1
            )
            
            # Convert dictionary to string for PySpark compatibility
            df_classes['parameters'] = df_classes['parameters'].apply(json.dumps)
            
            # Keep only relevant columns
            columns_to_keep = [
                'criteria', 'data_min', 'data_max','data_mean', 'loi', 'parameters', 'class_index', 'R2',
                'percentile_1%', 'percentile_50%', 'percentile_99%',
                'percentile_1%_emp', 'percentile_50%_emp', 'percentile_99%_emp'
            ]
            df_classes = df_classes[[col for col in columns_to_keep if col in df_classes.columns]]
            
            # Rename percentile columns
            df_classes.rename(columns={
                'percentile_1%': 'client1%',
                'percentile_50%': 'client50%',
                'percentile_99%': 'client99%',
                'percentile_1%_emp': 'sample_client1%',
                'percentile_50%_emp': 'sample_client50%',
                'percentile_99%_emp': 'sample_client99%'
            }, inplace=True)
            
            # Add study and sample_description (as JSON string) at the beginning
            df_classes.insert(0, 'study_title', study)
            df_classes.insert(1, 'sample_description', sample_description_str)
            
            aggregated_df.append(df_classes)
            
        except Exception as e:
            print(f"Error processing column {column}: {e}")

    if aggregated_df:
        combined_df = pd.concat(aggregated_df, ignore_index=True)
    else:
        combined_df = pd.DataFrame()

    # Convert to PySpark DataFrame
    spark = SparkSession.builder.getOrCreate()
    combined_df = spark.createDataFrame(combined_df)
    return combined_df
