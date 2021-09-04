"""
Python script where are stored constants
author : Damien Michelle
date: 03/09/2021
"""
import os

CURRENT_DIR = os.path.dirname(__file__)
MODEL_OUTPUT_DIR = 'models'
RESULTS_OUTPUT_DIR = 'results'

COLS = {
    'age': int,
    'class_of_worker': object,
    'industry_code': object,
    'occupation_code': object,
    'education': object,
    'wage_per_hour': int,
    'enrolled_in_edu_inst_last_wk': object,
    'marital_status': object,
    'major_industry_code': object,
    'major_occupation_code': object,
    'race': object,
    'hispanic_Origin': object,
    'sex': object,
    'member_of_a_labor_union': object,
    'reason_for_unemployment': object,
    'full_or_part_time_employment_stat': object,
    'capital_gains': int,
    'capital_losses': int,
    'divdends_from_stocks': int,
    'tax_filer_status': object,
    'region_of_previous_residence': object,
    'state_of_previous_residence': object,
    'detailed_household_and_family_stat': object,
    'detailed_household_summary_in_household': object,
    'instance_weight': float,
    'migration_code_change_in_msa': object,
    'migration_code_change_in_reg': object,
    'migration_code_move_within_reg': object,
    'live_in_this_house_1_year_ago': object,
    'migration_prev_res_in_sunbelt': object,
    'num_persons_worked_for_employer': int,
    'family_members_under_18': object,
    'country_of_birth_father': object,
    'country_of_birth_mother': object,
    'country_of_birth_self': object,
    'citizenship': object,
    'own_business_or_self_employed': object,
    'fill_inc_questionnaire_for_veterans_admin': object,
    'veterans_benefits': object,
    'weeks_worked_in_year': int,
    'year': object,
    'label': object
}

USELESS_COLS = [
    'instance_weight'
]

LABEL_COL = [
    'label'
]

ENCODING_LABEL = {
    ' - 50000.': 0,
    ' 50000+.': 1
}

PARAMS_RANDOMFOREST = [
    'n_estimators',
    'max_depth',
    'class_weight',
    'bootstrap'
]
