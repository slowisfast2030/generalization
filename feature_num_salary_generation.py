import re
import numpy as np
import pandas as pd
from zhconv import convert

def load_csv_data(data_path, test_path):
    '''
    读取csv文件
    '''
    all_data = pd.read_csv(data_path).drop(['Unnamed: 0'], axis=1).drop_duplicates(subset=['cv_id', 'jd_id'], keep='first').reset_index(drop=True)
    test_id = pd.read_pickle(test_path)
    df = all_data.join(test_id.set_index(['cv_id', 'jd_id']), on=['cv_id', 'jd_id'], how='inner')
    
    return df

def get_salary_year_cv(series, salary_col):
    """
    获取年薪的数字,以k为单位
    """
    salary = series[salary_col]

    # 转中文简体
    salary = convert(salary, "zh-hans")
    # 转英文小写
    salary = salary.lower()

    # case 10.5k*15.5
    res = re.search(r'(\d+\.*\d*)([k,w])[\*,x,·, ]+(\d+\.*\d*)', salary)
    if res:
        num1 = float(res.group(1))
        unit = res.group(2)
        num3 = float(res.group(3))
        #print(num1, unit, num3)
        salary_year = int(num1 * num3) 
        return salary_year if unit == "k" else 10 * salary_year
    
    # case 10.5*15.5k
    res = re.search(r'(\d+\.*\d*)[\*,x,·, ]+(\d+\.*\d*)([k,w])', salary)
    if res:
        num1 = float(res.group(1))
        num2 = float(res.group(2))
        unit = res.group(3)
        #print(num1, num2, unit)
        salary_year = int(num1 * num2) 
        return salary_year if unit == "k" else 10 * salary_year

    # case 10万/年
    res = re.search(r'(\d+)([万,w,元])/年', salary)
    if res:
        num = float(res.group(1))
        unit = res.group(2)
        salary_year = int(num * 10) if unit in ['万', 'w'] else int(num / 1000)
        return salary_year
    
    # case 10w/月
    res = re.search(r'(\d+)([万,w,元])/月', salary)
    if res:
        num = float(res.group(1))
        unit = res.group(2)
        # 默认12个月
        salary_year = int(num * 10) * 12 if unit in ['万', 'w'] else int(num / 1000) * 12
        return salary_year
 
    # 30k
    res = re.search(r'(\d+)k', salary)
    if res:
        num = float(res.group(1))
        # 默认12个月
        salary_year = int(num * 12)
        return salary_year
    
    # 42万
    res = re.search(r'(\d+)[万,w]', salary)
    if res:
        num = float(res.group(1))
        salary_year = int(num * 10)
        return salary_year

    # 9000元/月*12月
    res = re.search(r'(\d+)元/月[\*,x,·, ]+(\d+)', salary)
    if res:
        num1 = float(res.group(1))
        num2 = float(res.group(2))
        salary_year = int(num1 * num3 / 1000) 
        return salary_year
    
    # 都没匹配上
    return -1

def get_salary(data_path, test_path):
    '''
    给定数据路径，得到解析后的currentSalary和desiredSalary
    '''
    all_data = load_csv_data(data_path, test_path)

    all_data['desiredSalary'].fillna('', inplace=True)
    all_data['currentSalary'].fillna('', inplace=True)

    all_data['parsed_desiredSalary'] = all_data.apply(get_salary_year_cv, axis=1, args=('desiredSalary', ))
    all_data['parsed_currentSalary'] = all_data.apply(get_salary_year_cv, axis=1, args=('currentSalary', ))
    
    return all_data

if __name__ == "__main__":
    print("running...")

    data_path = '../data_20220831/raw_cvjd_20220831_spark.csv'
    test_path = '../generalization_data/cvjd_202208_6108.pkl'

    all_data = get_salary(data_path, test_path)

    print(all_data[['desiredSalary', 'currentSalary', 'parsed_desiredSalary', 'parsed_currentSalary']])
    print(all_data[['desiredSalary', 'currentSalary', 'parsed_desiredSalary', 'parsed_currentSalary']].info())

    all_data[['cv_id', 'jd_id', 'parsed_desiredSalary', 'parsed_currentSalary']].to_pickle('../generalization_data/gen_salary.pkl')

    print('all is well')

    