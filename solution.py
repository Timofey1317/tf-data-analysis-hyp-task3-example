import pandas as pd
import numpy as np
import scipy.stats as stats

chat_id = 1028099632 # Ваш chat ID, не меняйте название переменной

def solution(c_npv: np.array, t_npv: np.array) -> bool:# Одна или две выборке на входе, заполняется исходя из условия
    
    alpha = 0.08
    
    _, p_value_c = stats.shapiro(c_npv)
    _, p_value_t = stats.shapiro(t_npv)
    
    if p_value_c > alpha and p_value_t > alpha:
        statistic, p_value = stats.ttest_ind(c_npv, t_npv, equal_var=False)
    else:
        statistic, p_value = stats.mannwhitneyu(c_npv, t_npv, alternative='two-sided')
    
    # Измените код этой функции
    # Это будет вашим решением
    # Не меняйте название функции и её аргументы
    return p_value < alpha # Ваш ответ, True или False
