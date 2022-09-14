import json
import jieba
import re
import numpy as np
import pandas as pd
from zhconv import convert

cv_columns = ['cv_id', 'currentPosition', 'desiredPosition', 'skills']
jd_columns = ['jd_id', 'title', 'category_name', 'tags']

def load_csv_data(data_path, train_path):
    '''
    读取csv文件
    '''
    all_data = pd.read_csv(data_path).drop(['Unnamed: 0'], axis=1).drop_duplicates(subset=['cv_id', 'jd_id'], keep='first').reset_index(drop=True)
    train_id = pd.read_pickle(train_path)
    df = all_data.join(train_id.set_index(['cv_id', 'jd_id']), on=['cv_id', 'jd_id'], how='inner')
    
    return df

def col_merge_fun(series, col_list):
    '''
    合并多个文本列
    '''
    merge = ''
    for col in col_list:
        merge = merge + series[col] + ' '
    return merge.strip(' ')

def col_jieba_fun(series, col_name):
    '''
    将文本字符串切词成列表
    '''
    text = series[col_name]
    
    # 切词，精确模式
    col_list = jieba.lcut(text, cut_all=False)
    return col_list

def col_jieba_filter_fun(series, col_name_jieba):
    '''
    对切词后的列表进行过滤
    '''
    col_list_filter = []
    
    # 得到切词后的文本列表
    col_list = series[col_name_jieba]

    pun_masks_english = [",", ".", "/", "[", "]", "{", "}", "(", ")", ":", "*", "#", "!", " ", "\"", "\\"]
    pun_masks_chinese = ["，", "。", "、", "（", "）", "：", "！", "”", "“"]
    pun_masks = pun_masks_english + pun_masks_chinese

    # 过滤
    for tag in col_list:
        # 转中文简体
        tag = convert(tag, "zh-hans")
        # 转英文小写
        tag = tag.lower()

        # 过滤数字
        if tag.isdigit():
            continue
        
        # 过滤单个字符
        if len(tag) <= 1:
            continue
        
        # 过滤标点
        flag = 1
        for pun in pun_masks:
            if pun in tag:
                flag = 0
                break
        if flag == 1:
            col_list_filter.append(tag)
    return col_list_filter

def get_text_jieba_filter(data_path, train_path):
    '''
    给定csv数据路径，分别将cv和jd的文本列合并、分词、过滤
    '''
    all_data = load_csv_data(data_path, train_path)

    # 空值填充
    for col in cv_columns[1:]:
        all_data[col].fillna('', inplace=True)
    for col in jd_columns[1:]:
        all_data[col].fillna('', inplace=True)

    cv_jd = ['cv', 'jd']
    cv_jd_columns = [cv_columns, jd_columns]

    col_jieba_filter_list = []
    for col, col_columns in zip(cv_jd, cv_jd_columns):
        col_text = col + '_text'
        col_text_jieba = col_text + '_jieba'
        col_text_jieba_filter = col_text_jieba + '_filter'
        col_jieba_filter_list.append(col_text_jieba_filter)
        
        all_data[col_text] = all_data.apply(col_merge_fun, axis=1, args=(col_columns[1:], ))
        all_data[col_text_jieba] = all_data.apply(col_jieba_fun, axis=1, args=(col_text, ))
        all_data[col_text_jieba_filter] = all_data.apply(col_jieba_filter_fun, axis=1, args=(col_text_jieba, ))

    all_data["equal_job"] = all_data.apply(get_equal_word_num, axis=1, args=(col_jieba_filter_list, ))
    return all_data

def get_equal_word_num(series, col_list):
    cv_text_jieba_filter = series[col_list[0]]
    jd_text_jieba_filter = series[col_list[1]]
    # 这里不能去重吧？！
    # 如何计算这里的重复词的数目呢？
    # 有一个隐患：这里应该找到关键词！！！
    res = set(cv_text_jieba_filter).intersection(set(jd_text_jieba_filter))
    return len(res)


if __name__ == "__main__":
    print("running...")

    data_path = '../data_20220831/raw_cvjd_20220831_spark.csv'
    train_path = '../generalization_data/cvjd_train_filter_54339.pkl'

    all_data = get_text_jieba_filter(data_path, train_path)
    print(all_data["equal_job"])
    #print(all_data.info())

    all_data[['cv_id', 'jd_id', 'equal_job']].to_pickle('../generalization_data/train_equal_job.pkl')
    print("all is well!")

    