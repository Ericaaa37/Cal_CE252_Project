import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import os

#data
household = pd.read_csv('/Users/mayunruo/Desktop/252/final_project/data/survey_household.csv')
vehicle = pd.read_csv('/Users/mayunruo/Desktop/252/final_project/data/survey_vehicle.csv')
trip = pd.read_csv('/Users/mayunruo/Desktop/252/final_project/data/survey_trip.csv')

# hh income
income_map = {
    1: 8000, 2: 12500, 3: 20000, 4: 30000, 5: 42500, 6: 62500,
    7: 87500, 8: 112500, 9: 137500, 10: 175000, 11: 200000
}
household['INC'] = household['hhfaminc'].map(income_map)

# CPM
fe_map = {1: 24.4, 2: 17.8, 3: 17.8, 4: 17.8, 5: 5.7, 6: 17.8, 7: 44}
vehicle['fuel_economy'] = vehicle['vehtype'].map(fe_map)
fuel_price = 4.642
vehicle['CPM'] = fuel_price / vehicle['fuel_economy']

# VMT
trip_vmt = trip.groupby('sampno')['trpmiles'].sum().reset_index()
trip_vmt.rename(columns={'trpmiles': 'VMT'}, inplace=True)
vehicle_mean = vehicle.groupby('sampno', as_index=False)['CPM'].mean()

df = (trip_vmt
      .merge(vehicle_mean, on='sampno', how='left')
      .merge(household[['sampno', 'INC', 'urbrur', 'hhvehcnt', 'wrkcount']], on='sampno', how='left'))

# urban/rural, hh worker, hh veh
df['U'] = df['urbrur']
df['W'] = df['wrkcount']
df['V'] = df['hhvehcnt']

# log
df = df[['VMT', 'CPM', 'INC', 'U','W','V']].dropna()
df = df[(df > 0).all(axis=1)]
df['lnVMT'] = np.log(df['VMT'])
df['lnCPM'] = np.log(df['CPM'])
df['lnINC'] = np.log(df['INC'])
df['lnW'] = np.log(df['W'])
df['lnV'] = np.log(df['V'])

# Separate regressions: Urban / Rural
df_urban = df[df['U'] == 1]
df_rural = df[df['U'] == 2]

formula = 'lnVMT ~ lnCPM + lnINC + I(0.5 * lnCPM**2) + I(0.5 * lnINC**2) + lnCPM:lnINC + lnW + lnV'

# Urban model
model_urban = smf.ols(formula, data=df_urban).fit()

# Rural model
model_rural = smf.ols(formula, data=df_rural).fit()

print("\n===== Urban households results =====")
print(model_urban.summary())

print("\n===== Rural households results =====")
print(model_rural.summary())

# Scenario
def calculate_scenario_avg(subset, vehicle):

    # regression
    subset["CPM_sq"] = 0.5 * subset["lnCPM"]**2    
    subset["INC_sq"] = 0.5 * subset["lnINC"]**2      

    formula = (
        'lnVMT ~ lnCPM + lnINC + CPM_sq + INC_sq + lnCPM:lnINC + lnW + lnV'
    )

    model = smf.ols(formula, data=subset).fit()

    b1  = model.params['lnCPM']
    b11 = model.params['CPM_sq']
    b12 = model.params['lnCPM:lnINC']

    # elasticity
    subset['elasticity'] = b1 + b11 * subset['lnCPM'] + b12 * subset['lnINC']

    # cost
    fee_per_mile = 0.02
    fuel_tax = 0.417
    subset['current_cost'] = fuel_tax / vehicle['fuel_economy'].mean()
    subset['new_cost'] = fee_per_mile
    subset['delta_price'] = -0.0349

    # ---- Scenario 1 ---- 
    subset['tax1'] = subset['VMT'] * subset['current_cost']
    subset['tax2'] = subset['VMT'] * subset['new_cost']

    # ---- Scenario 2 ----
    subset['elasticity_neg'] = subset['elasticity'].clip(upper=0)
    subset['VMT_s2'] = subset['VMT'] * (1 + subset['elasticity_neg'] * subset['delta_price'])
    subset['VMT_s2'] = subset['VMT_s2'].clip(lower=0)
    subset['tax2'] = subset['VMT_s2'] * subset['new_cost']

    # ---- Scenario 3 ----
    subset['VMT_s3'] = subset['VMT'] * (1 + subset['elasticity'] * subset['delta_price'])
    subset['VMT_s3'] = subset['VMT_s3'].clip(lower=0)
    subset['tax3'] = subset['VMT_s3'] * subset['new_cost']

    # income group 
    def income_group(inc):
        if inc <= 30000:
            return 'lowest'
        elif inc <= 62500:
            return 'middle-low'
        elif inc <= 137500:
            return 'middle-high'
        else:
            return 'highest'

    subset['income_group'] = subset['INC'].apply(income_group)

    # ---- tax table ----
    scenarios = ['tax1','tax2','tax3']
    records = {}
    for s in scenarios:
        grp = subset.groupby('income_group')[s].mean()
        grp['average'] = subset[s].mean()
        records[s] = grp
    scenario_table = pd.DataFrame(records)

    # ---- S1, S2, S3 ----
    vmt_avg = subset.groupby('income_group')['VMT'].mean()
    vmt_avg['average'] = subset['VMT'].mean()

    vmt_s2_avg = subset.groupby('income_group')['VMT_s2'].mean()
    vmt_s2_avg['average'] = subset['VMT_s2'].mean()

    vmt_s3_avg = subset.groupby('income_group')['VMT_s3'].mean()
    vmt_s3_avg['average'] = subset['VMT_s3'].mean()

    return {
        "scenario_table": scenario_table,
        "vmt_avg": vmt_avg,
        "vmt_s2_avg": vmt_s2_avg,
        "vmt_s3_avg": vmt_s3_avg
    }

result = calculate_scenario_avg(df, vehicle)

# Urban & Rural
df_urban = df[df['U'] == 1].copy()
df_rural = df[df['U'] == 2].copy()

# sample number
num_urban = df_urban.shape[0]
num_rural = df_rural.shape[0]

urban_avg = calculate_scenario_avg(df_urban, vehicle)
rural_avg = calculate_scenario_avg(df_rural, vehicle)

# Original VMT 
urban_orig = urban_avg['vmt_avg'].reset_index().rename(columns={'index':'income_group', 'VMT':'Original_VMT'})
urban_orig['area'] = 'Urban'

rural_orig = rural_avg['vmt_avg'].reset_index().rename(columns={'index':'income_group', 'VMT':'Original_VMT'})
rural_orig['area'] = 'Rural'

combined_orig = pd.concat([urban_orig, rural_orig], axis=0).reset_index(drop=True)

# Scenario 2 VMT
urban_s2 = urban_avg['vmt_s2_avg'].reset_index().rename(columns={'index':'income_group'})
urban_s2['area'] = 'Urban'

rural_s2 = rural_avg['vmt_s2_avg'].reset_index().rename(columns={'index':'income_group'})
rural_s2['area'] = 'Rural'

combined_s2 = pd.concat([urban_s2, rural_s2], axis=0).reset_index(drop=True)

# Scenario 3 VMT
urban_s3 = urban_avg['vmt_s3_avg'].reset_index().rename(columns={'index':'income_group'})
urban_s3['area'] = 'Urban'

rural_s3 = rural_avg['vmt_s3_avg'].reset_index().rename(columns={'index':'income_group'})
rural_s3['area'] = 'Rural'

combined_s3 = pd.concat([urban_s3, rural_s3], axis=0).reset_index(drop=True)

# Scenario tax burden by income group & area
urban_table = urban_avg["scenario_table"].copy()
urban_table['area'] = 'Urban'
rural_table = rural_avg["scenario_table"].copy()
rural_table['area'] = 'Rural'
combined_tax = pd.concat([urban_table, rural_table]).reset_index().rename(columns={'index':'income_group'})

# Save Excel
with pd.ExcelWriter('scenario_results_translog.xlsx') as writer:
    combined_s2.to_excel(writer, sheet_name='Scenario2_VMT', index=False)
    combined_s3.to_excel(writer, sheet_name='Scenario3_VMT', index=False)
    combined_tax.to_excel(writer, sheet_name='Tax_Burden', index=False)

print("✅ Excel 'scenario_results_translog.xlsx' saved")