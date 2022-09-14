import json
import jieba
import re
import numpy as np
import pandas as pd
from zhconv import convert
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import pickle

'''
1.不论是单列文本还是多列文本都可以通过这个函数统一获取tfidf_lsa向量
2.给apply函数传入了参数，免去了global申明
'''

def load_csv_data(data_path, train_path):
    '''
    读取csv文件
    '''
    all_data = pd.read_csv(data_path).drop(['Unnamed: 0'], axis=1).drop_duplicates(subset=['cv_id', 'jd_id'], keep='first').reset_index(drop=True)
    train_id = pd.read_pickle(train_path)
    df = all_data.join(train_id.set_index(['cv_id', 'jd_id']), on=['cv_id', 'jd_id'], how='inner')
    # 用来拼接lsa向量
    df.to_pickle('../generalization_data/train_data_lsa_cvjd_54339.pkl')
    return df

def col_jieba_fun(series, col_name):
    '''
    将文本字符串切词成列表
    '''
    col = series[col_name]
    #print(col)
    # 加入特例判断 *Tracks。'[{},{}]', json无法解析。
    if col_name.endswith("Tracks"):
        col_list = jieba.lcut(col, cut_all=False)
        return col_list

    # 字符串变列表
    if col.startswith("[") and col.endswith("]"):
        col = json.loads(col)
    else:
        col = re.split(",|，|/| ", col)

    # 列表变字符串
    # 对于中文，进入jieba前不需要添加空格；不过，如果是中英文混合，就必须空格了
    col_str = " ".join(col)

    # 切词
    col_list = jieba.lcut(col_str, cut_all=False)
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
    return " ".join(col_list_filter)

def get_tfidf(df, col_name):
    '''
    将文本列转成tfidf向量
    '''
    text = df[col_name]
    
    vectorizer = TfidfVectorizer()
    vector = vectorizer.fit_transform(text)

    # 作为模型的名字
    col_name = col_name.strip('_jieba_filter') + '_tfidf_model.pkl'
    tfidf_pkl_filename = col_name
    with open(tfidf_pkl_filename, 'wb') as file:
        pickle.dump(vectorizer, file)

    return pd.DataFrame(vector.toarray()), vectorizer

def get_tfidf_lsa(tfidf, n, col_name):
    '''
    将tfidf向量降维
    '''
    # pca = PCA(n_components=n)
    lsa = TruncatedSVD(n_components=n)

    # tfidf_pca = pca.fit_transform(tfidf)
    tfidf_lsa = lsa.fit_transform(tfidf)

    # tfidf_pca = pd.DataFrame(tfidf_pca)
    tfidf_lsa = pd.DataFrame(tfidf_lsa)
    
    # 作为模型的名字
    col_name = col_name.strip('_jieba_filter') + '_lsa_model.pkl'
    lsa_pkl_filename = col_name
    with open(lsa_pkl_filename, 'wb') as file:
        pickle.dump(lsa, file)


    return tfidf_lsa

def col_merge_fun(series, col_name_jieba_filter_list):
    '''
    合并多个文本列
    '''
    merge = ''
    for col in col_name_jieba_filter_list:
        merge = merge + series[col] + ' '
    return merge.strip(' ')

def get_tfidf_lsa_from_text_cols(data_path, train_path, col_name_list, dimension):
    '''
    从多个文本列计算tfidf_lsa

    :param data_path csv数据路径
    :param col_name_list 文本列列名列表
    :param dimension tfidf经过lsa降维后的维度
    :returns: tfidf_lsa向量
    '''
    # 读取csv文件
    df = load_csv_data(data_path, train_path)
    print(df.shape)

    # 存储经过分词和过滤后的列名
    col_name_jieba_filter_list = []

    for col_name in col_name_list:

        col_name_jieba = col_name + '_jieba'
        col_name_jieba_filter = col_name_jieba + '_filter'
        col_name_jieba_filter_list.append(col_name_jieba_filter)

        # step1 空值填充
        df[col_name].fillna('', inplace=True)

        # step2 jieba分词
        df[col_name_jieba] = df.apply(col_jieba_fun, axis=1, args=(col_name, ))

        # step3 分词过滤
        df[col_name_jieba_filter] = df.apply(col_jieba_filter_fun, axis=1, args=(col_name_jieba, ))

        print("\n=================================={}==================================".format(col_name))
        print(df[[col_name, col_name_jieba, col_name_jieba_filter]])

    print(col_name_jieba_filter_list)
    
    merge_col_jieba_filter = "_".join(col_name_list) + '_jieba_filter'
    df[merge_col_jieba_filter] = df.apply(col_merge_fun, axis=1, args=(col_name_jieba_filter_list, ))

    print("\n=================================={}==================================".format('以上各列分词过滤后合并的新列'))
    print(df[[merge_col_jieba_filter]])

    # step4 得到tfidf
    tfidf, vectorizer = get_tfidf(df, merge_col_jieba_filter)
    print("\n=================================={}==================================".format('tfidf向量'))
    print(tfidf)

    # step5 得到tfidf_lsa
    tfidf_lsa = get_tfidf_lsa(tfidf, dimension, merge_col_jieba_filter)
    print("\n=================================={}==================================".format('tfidf_lsa向量'))
    print(tfidf_lsa)

    return tfidf_lsa


if __name__ == "__main__":
    print("running...")

    data_path = '../data_20220831/raw_cvjd_20220831_spark.csv'
    train_path = '../generalization_data/cvjd_train_filter_54339.pkl'
    
    print("\n从文本列获取tfidf_lsa向量\n")
    col_name_list1 = ['title', 'category_name', 'tags']
    col_name_list2 = ['description']
    col_name_list3 = ['requirement']

    col_name_list4 = ['currentPosition', 'desiredPosition']
    col_name_list5 = ['skills']
    col_name_list6 = ['jobTracks'] # 太吃内存，放弃
   
    tfidf_lsa1 = get_tfidf_lsa_from_text_cols(data_path, train_path, col_name_list1, dimension=30)
    tfidf_lsa1.to_pickle('../generalization_data/train_title_category_tags_tfidf_lsa.pkl')

    tfidf_lsa2 = get_tfidf_lsa_from_text_cols(data_path, train_path, col_name_list2, dimension=70)
    tfidf_lsa2.to_pickle('../generalization_data/train_description_tfidf_lsa.pkl')

    tfidf_lsa3 = get_tfidf_lsa_from_text_cols(data_path, train_path, col_name_list3, dimension=70)
    tfidf_lsa3.to_pickle('../generalization_data/train_requirement_tfidf_lsa.pkl')

    tfidf_lsa4 = get_tfidf_lsa_from_text_cols(data_path, train_path, col_name_list4, dimension=40)
    tfidf_lsa4.to_pickle('../generalization_data/train_position_tfidf_lsa.pkl')

    tfidf_lsa5 = get_tfidf_lsa_from_text_cols(data_path, train_path, col_name_list5, dimension=30)
    tfidf_lsa5.to_pickle('../generalization_data/train_skills_tfidf_lsa.pkl')
    
    print("all is well")
