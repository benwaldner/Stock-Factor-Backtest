import pandas as pd  
import numpy as np
from decimal import Decimal, ROUND_HALF_UP
import os
from datetime import datetime
from multiprocessing import pool, cpu_count, freeze_support
import platform
import glob

from function import *
from config import *
import warnings
warnings.filterwarnings('ignore') 
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 5000) 

# 因子回测框架
# 因子可以分两类添加至本框架中：
# 1. 需要先计算好再aggregate的因子放在本文的函数中计算（例如量价相关性）
# 2. 需要先aggregate之后再计算的因子要在function.py中修改（例如换手率，其分子的成交额需要先在aggregate中sum才能计算）

# ===读取所有股票代码的列表
stock_code_list = get_stock_code_list_in_one_dir(stock_data_path)
# print('股票数量：', len(stock_code_list))


# ===循环读取并且合并
# 导入上证指数，必须保证指数数据和股票数据在同一天结束，不然merge的时候会出问题，merge的时候是按照how='right'去合并的
# 如果时间对不齐，那就去real_time_stock_update里面跑最新的指数数据然后删去末尾的使时间同步
index_data = import_index_data(index_data_path + "\\sh000001.csv", start=back_test_start, end=back_test_end)


def calculate_by_stock(code):
    """
    整理数据核心函数
    :param code: 股票代码
    :return: 一个包含该股票所有历史数据的DataFrame
    """
    print("因子计算开始：",code)

    # =读入股票数据
    df = pd.read_csv(stock_data_path + '\\%s.csv' % code, encoding='gbk', skiprows=1, parse_dates=['交易日期'])

    # =计算涨跌幅
    df['涨跌幅'] = df['收盘价'] / df['前收盘价'] - 1
    df['开盘买入涨跌幅'] = df['收盘价'] / df['开盘价'] - 1  # 为之后开盘买入做好准备

    # 计算复权因子：假设你一开始有1元钱，投资到这个股票，最终会变成多少钱
    # 我们这几个指标，只有振幅需要用到复权后的最高价和最低价，而涨跌幅的计算不会用到复权后的开盘和收盘价
    # 所以只取最高和最低价的后复权，前复权应该也可以
    df['复权因子'] = (1 + df['涨跌幅']).cumprod() 
    df['收盘价_后复权'] = df['复权因子'] * (df.iloc[0]['收盘价'] / df.iloc[0]['复权因子'])
    df['开盘价_后复权'] = df['开盘价'] / df['收盘价'] * df['收盘价_后复权']
    df['最高价_后复权'] = df['最高价'] / df['收盘价'] * df['收盘价_后复权']
    df['最低价_后复权'] = df['最低价'] / df['收盘价'] * df['收盘价_后复权']

    # 计算交易天数
    df['上市至今交易天数'] = df.index + 1

    # 需要额外保存的字段
    extra_fill_0_list = []  # 在和上证指数合并时使用。
    extra_agg_dict = {}  # 在转换周期时使用。


    # =将股票和上证指数合并，补全停牌的日期，新增数据"是否交易"、"指数涨跌幅"
    df = merge_with_index_data(df, index_data, extra_fill_0_list)
    if df.empty:
        return pd.DataFrame()

    # =计算涨跌停价格，新增涨停价、跌停价、开盘涨停、开盘跌停、一字涨停、一字跌停
    # 我们不关心周期最后一天是否是涨跌停，只关心周期下一天开盘能不能买入，也就是下一天是不是开盘涨跌停或者一字涨跌停，而下一交易日本身涨跌停我们也不关心
    df = cal_zdt_price(df)

    # ==== 计算第一类因子（aggregate前计算）
    # 计算需要在resample前计算、resample过程中选择的因子
    df['每日振幅'] = df['最高价']/df['最低价']- 1
    df['20日振幅_1'] = df['最高价'].rolling(20,min_periods=1).max()/df['最低价'].rolling(20,min_periods=1).min() - 1
    df['20日振幅_2'] = df['每日振幅'].rolling(20,min_periods=1).mean()
    # 注意公式，换手率定义应该是成交量除以总流通股本，那分子分母同时乘上价格，就相当于成交额除以流通市值
    df['换手率'] = df['成交额']/df['流通市值']
    # 计算量价相关性，使用每日换手率和后复权的收盘价/复权因子的滑动相关性
    # 为什么“量”使用的是换手率而不是成交额，因为成交额取决于股票本身，有的股票本来盘就大，有的小，所以衡量“量”的流动性还是使用成交额除以流通市值比较好
    # 当然，rolling的period是超参数
    # 注意这里由于corr()的参数现在只能是series/dataframe，因此只能写一次rolling而不能在corr函数中再写一次rolling
    df['量价相关系数'] = df['复权因子'].rolling(10).corr(df['换手率'])

    # 添加因子到aggregate中
    extra_agg_dict['20日振幅_1'] = 'last'
    extra_agg_dict['20日振幅_2'] = 'last'
    extra_agg_dict['量价相关系数'] = 'last'


    # =计算下个交易的相关情况
    df['下日_是否交易'] = df['是否交易'].shift(-1)
    df['下日_一字涨停'] = df['一字涨停'].shift(-1)
    df['下日_开盘涨停'] = df['开盘涨停'].shift(-1)
    df['下日_一字跌停'] = df['一字跌停'].shift(-1)
    df['下日_开盘跌停'] = df['开盘跌停'].shift(-1)
    df['下日_是否ST'] = df['股票名称'].str.contains('ST').shift(-1)
    df['下日_是否S'] = df['股票名称'].str.contains('S').shift(-1)
    df['下日_是否退市'] = df['股票名称'].str.contains('退').shift(-1)
    # 计算日内涨跌幅，因为我们是开盘买入
    df['下日_开盘买入涨跌幅'] = df['开盘买入涨跌幅'].shift(-1)

    # =将日线数据转化为月线或者周线
    df = transfer_to_period_data(df, period, extra_agg_dict)

    # =对数据进行整理
    # 删除上市的第一个周期
    df.drop([0], axis=0, inplace=True)  # 删除第一行数据

    # # 删除2007年之前的数据
    # df = df[df['交易日期'] > pd.to_datetime('20061215')]

    # 计算下周期每天涨幅
    df['下周期每天涨跌幅'] = df['每天涨跌幅'].shift(-1)
    df['下周期涨跌幅'] = df['涨跌幅'].shift(-1)
    del df['每天涨跌幅']

    # =删除不能交易的周期数
    # 删除月末为st状态的周期数，只要今天选股是ST，就不要
    df = df[df['股票名称'].str.contains('ST') == False]
    # 删除月末为s状态的周期数
    df = df[df['股票名称'].str.contains('S') == False]
    # 删除月末有退市风险的周期数
    df = df[df['股票名称'].str.contains('退') == False]
    # 删除月末不交易的周期数
    df = df[df['是否交易'] == 1]
    # 删除交易天数过少的周期数，比如强行要交易天数大于等于10的，但有些月交易时间不多，所以看比例更合理一些
    df = df[df['交易天数'] / df['市场交易天数'] >= 0.8]
    df.drop(['交易天数', '市场交易天数'], axis=1, inplace=True)

    print("因子计算结束：", code)
    return df  # 返回计算好的数据

# 标记开始时间
if __name__ == '__main__':
    start_time = datetime.now()

    # 测试
    # print(calculate_by_stock('sz300479'))
    # exit()

    # 标记开始时间
    start_time = datetime.now()

    # 并行处理
    multiply_process = True
    if multiply_process:
        with pool.Pool(processes=2*cpu_count()) as my_pool:
            df_list = my_pool.map(calculate_by_stock, sorted(stock_code_list))
    # 串行处理
    else:
        df_list = []
        for stock_code in stock_code_list:
            data = calculate_by_stock(stock_code)
            df_list.append(data)
    print('读入完成, 开始合并，消耗时间', datetime.now() - start_time)

    # 合并为一个大的DataFrame
    all_stock_data = pd.concat(df_list, ignore_index=True)
    all_stock_data.sort_values(['交易日期', '股票代码'], inplace=True)  # ===将数据存入数据库之前，先排序、reset_index
    all_stock_data.reset_index(inplace=True, drop=True)

    # 检验效果
    # all_stock_data.to_csv('./processed_data/test1.csv', encoding='gbk')

    # 将数据存储到hdf文件
    # 这里做一个区分，把复权后的数据重新存储新文件
    # all_stock_data.to_hdf('./processed_data/all_stock_data_hfq_'+period_type+'.h5', key='df', mode='w')


    # 将数据存储到pickle文件
    all_stock_data.to_pickle(root_path + '/processed_data/all_stock_data_' + period + '.pkl')

    print(datetime.now() - start_time)
