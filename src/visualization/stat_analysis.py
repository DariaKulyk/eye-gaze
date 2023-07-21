""" Contains functions for statistical analysis """

import scipy.stats as stats
import pandas as pd
import statsmodels.api as sm
from scipy.stats import shapiro, levene, kruskal

""" This function checks for normality assumption and prints whether 
the participants spent more time on areas with meaningful objects """
def calculate_significance(obj_arr, non_obj_arr):
    differences = [obj_arr - non_obj_arr for obj_arr, non_obj_arr in zip(obj_arr, non_obj_arr)]
    # Perform Shapiro-Wilk test for normality
    _, p_value_shapiro = stats.shapiro(differences)
    print("Shapiro-Wilk's test p-value (normality assumption):", p_value_shapiro)
    # Check the normality assumption
    alpha = 0.05  # Significance level
    if p_value_shapiro > alpha:
        # Sample passes the normality assumption
        # Perform paired t-test
        t_statistic, p_value = stats.ttest_rel(obj_arr, non_obj_arr, alternative="greater")
        print("Paired t-test")
        print("t-statistic:", t_statistic)
        print("p-value:", p_value)
        if p_value > alpha:
            print("Confirmed H0")
        else:
            print("Rejected H0: the average time spent on areas containing objects is significantly greater than the average time spent on areas not containing objects")
    else:
         # Perform Wilcoxon signed-rank test
        w_statistic, p_value = stats.wilcoxon(obj_arr, non_obj_arr, alternative="greater")
        print("The sample does not follow normal distribution. Conducting Wilcoxon signed-rank test...")
        print("W statistic:", w_statistic)
        print("p-value:", p_value)
        if p_value > alpha:
            print("Confirmed H0")
        else:
            print("Rejected H0: the median time spent on areas containing the objects is significantly greater than the median time spent on areas not containing objects")
        
 
""" This function checks for normality and homogeniety assumptions and prints
results to determine whether there is a difference in the average time spent
on the painting between the different object categories """ 
def perform_one_way_anova(anova_file):
    # Read the CSV file
    df = pd.read_csv(anova_file)
    # df = pd.read_csv('../../data/processed/anova-data-filtered-time.csv')
    list_of_column_names = list(df.columns)
    interest_column = list_of_column_names[0] # Column containing the time spent on the painting
    group_column = list_of_column_names[1] # Column containing the object type
    # Fit the one-way ANOVA model
    model = sm.formula.ols(f'{interest_column} ~ {group_column}', data=df).fit()
    # Calculate the residuals
    residuals = model.resid
    # Create a DataFrame with the residuals and the group column
    residuals_df = pd.DataFrame({group_column: df[group_column], 'Residuals': residuals})
    # Calculate the group residuals
    group_resid = residuals_df.groupby(group_column)['Residuals'].apply(list)
    # Check for normality using Shapiro-Wilk test
    _, p_value_shapiro = shapiro(residuals)
    # Check for homogeneity of variances using Levene's test
    _, p_value_levene = levene(*group_resid)
    alpha = 0.05  # Significance level
    print("Shapiro-Wilk's p-value (normality assumption):", p_value_shapiro)
    print("Levene's p-value (homogeniety of variances assumption):", p_value_levene)
  
    if p_value_shapiro > alpha and p_value_levene > alpha:
        # Perform the ANOVA
        anova_table = sm.stats.anova_lm(model)
        # Print the ANOVA table
        print("One-way ANOVA test")
        print(anova_table)
    else:
        # Perform the Kruskal-Wallis test
        stat_kruskal, p_value_kruskal = kruskal(*group_resid)
        print("The sample does not meet one-way ANOVA test assumptions. Conducting Kruskal-Wallis test...")
        print("Kruskal-Wallis test statistic:", stat_kruskal)
        print("P-value:", p_value_kruskal)

