# 导入模块
import pandas as pd  
import numpy as np
from decimal import Decimal, ROUND_HALF_UP
import os
from datetime import datetime
import itertools
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.offline import plot
from plotly.subplots import make_subplots

from config import *

pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', None)

# 导入某文件夹下所有股票的代码
def get_stock_code_list_in_one_dir(path):
    """
    从指定文件夹下，导入所有csv文件的文件名
    :param path:
    :return:
    """
    stock_list = []

    # 系统自带函数os.walk，用于遍历文件夹中的所有文件
    for root, dirs, files in os.walk(path):
        if files:  # 当files不为空的时候
            for f in files:
                if f.endswith('.csv'):
                    stock_list.append(f[:8])

    return sorted(stock_list)


# 导入指数
def import_index_data(path, start=None, end=None):
    """
    从指定位置读入指数数据
    :param end: 结束时间
    :param start: 开始时间
    :param path:
    :return:
    """
    # 导入指数数据
    df_index = pd.read_csv(path, parse_dates=['交易日期'], encoding='gbk')
    df_index['指数涨跌幅'] = df_index['收盘价'].pct_change()
    df_index = df_index[['交易日期', '指数涨跌幅']]
    df_index.dropna(subset=['指数涨跌幅'], inplace=True)

    if start:
        df_index = df_index[df_index['交易日期'] >= pd.to_datetime(start)]
    if end:
        df_index = df_index[df_index['交易日期'] <= pd.to_datetime(end)]
    df_index.sort_values(by=['交易日期'], inplace=True)
    df_index.reset_index(inplace=True, drop=True)

    return df_index

def create_empty_data(index_data, period):
    empty_df = index_data[['交易日期']].copy()
    empty_df['涨跌幅'] = 0.0
    empty_df['周期最后交易日'] = empty_df['交易日期']
    empty_df.set_index('交易日期', inplace=True)
    agg_dict = {'周期最后交易日': 'last'}
    empty_period_df = empty_df.resample(rule=period).agg(agg_dict)

    empty_period_df['每天涨跌幅'] = empty_df['涨跌幅'].resample(period).apply(lambda x: list(x))
    # 删除没交易的日期
    empty_period_df.dropna(subset=['周期最后交易日'], inplace=True)

    empty_period_df['选股下周期每天涨跌幅'] = empty_period_df['每天涨跌幅'].shift(-1)
    empty_period_df.dropna(subset=['选股下周期每天涨跌幅'], inplace=True)

    # 填仓其他列
    empty_period_df['股票数量'] = 0
    empty_period_df['买入股票代码'] = 'empty'
    empty_period_df['买入股票名称'] = 'empty'
    empty_period_df['选股下周期涨跌幅'] = 0.0
    empty_period_df.rename(columns={'周期最后交易日': '交易日期'}, inplace=True)
    empty_period_df.set_index('交易日期', inplace=True)

    empty_period_df = empty_period_df[['股票数量', '买入股票代码', '买入股票名称', '选股下周期涨跌幅', '选股下周期每天涨跌幅']]
    return empty_period_df    

# 将股票数据和指数数据合并
def merge_with_index_data(df, index_data, extra_fill_0_list=[]):
    """
    原始股票数据在不交易的时候没有数据。
    将原始股票数据和指数数据合并，可以补全原始股票数据没有交易的日期。
    :param df: 股票数据
    :param index_data: 指数数据
    :return:
    """
    df = pd.merge(left=df, right=index_data, on='交易日期', how='right', sort=True, indicator=True)

    # ===对开、高、收、低、前收盘价价格进行补全处理
    # 用前一天的收盘价，补全收盘价的空值
    df['收盘价'].fillna(method='ffill', inplace=True)
    df['开盘价'].fillna(value=df['收盘价'], inplace=True)
    df['最高价'].fillna(value=df['收盘价'], inplace=True)
    df['最低价'].fillna(value=df['收盘价'], inplace=True)
    # 补全前收盘价
    df['前收盘价'].fillna(value=df['收盘价'].shift(), inplace=True)
    # 注意，以上fill的原因在于，我想得到和指数齐平的每个交易日的数据，fillna之后我可以计算每天的涨跌停价格，尽管可能这天股票并没有交易
    # 既然如此，复权后的开高收低并不用于计算涨跌停价，那就不用填充了，下面计算因子需要rolling这里的复权后的开高收低，填充了反而不对

    # df['收盘价_后复权'].fillna(method='ffill', inplace=True)
    # df['开盘价_后复权'].fillna(value=df['收盘价_后复权'], inplace=True)
    # df['最高价_后复权'].fillna(value=df['收盘价_后复权'], inplace=True)
    # df['最低价_后复权'].fillna(value=df['收盘价_后复权'], inplace=True)

    # ===将停盘时间的某些列，数据填补为0
    fill_0_list = ['成交量', '成交额', '涨跌幅', '开盘买入涨跌幅'] + extra_fill_0_list
    df.loc[:, fill_0_list] = df[fill_0_list].fillna(value=0)

    # ===用前一天的数据，补全其余空值
    df.fillna(method='ffill', inplace=True)

    # ===去除上市之前的数据
    df = df[df['股票代码'].notnull()]

    # ===判断计算当天是否交易
    df['是否交易'] = 1
    df.loc[df['_merge'] == 'right_only', '是否交易'] = 0
    del df['_merge']

    df.reset_index(drop=True, inplace=True)

    return df

# 将日线数据转换为其他周期的数据，我在之前的大作业里日线数据转月线用的是groupby，不过真的还应该用resample，严谨的多
def transfer_to_period_data(df, period_type='m', extra_agg_dict={}):
    """
    将日线数据转换为相应的周期数据
    :param df:
    :param period_type:
    :return:
    """
    # 将交易日期设置为index
    df['周期最后交易日'] = df['交易日期']
    df.set_index('交易日期', inplace=True)

    agg_dict = {
            # 必须列
            '周期最后交易日': 'last',
            '股票代码': 'last',
            '股票名称': 'last',
            '是否交易': 'last',

            '前收盘价': 'first',
            '开盘价': 'first',
            '最高价': 'max',
            '最低价': 'min',
            '收盘价': 'last',
            '成交量': 'sum',
            '成交额': 'sum',

            '上市至今交易天数': 'last',
            '下日_是否交易': 'last',
            '下日_开盘涨停': 'last',
            '下日_一字涨停': 'last',
            '下日_一字跌停': 'last',
            '下日_开盘跌停': 'last',
            '下日_是否ST': 'last',
            '下日_是否S': 'last',
            '下日_是否退市': 'last',
            '下日_开盘买入涨跌幅': 'last',

            # 因子列（可以在这里加也可以在data_processing里加）
            '总市值': 'last',
            '流通市值': 'last',
        }

    #设置close为right，比如一周持仓周期，那么这周日就是aggregate后的index，正好我们是在下周一交易
    #运行完后，我们的index是每个周日（如果是周，月的话就是每个自然月最后一日），但“周期最后交易日”是该段时间周期最后一个交易日
    agg_dict = dict(agg_dict, **extra_agg_dict)
    period_df = df.resample(offset=0,closed='right',rule=period_type).agg(agg_dict)

    # 计算其余所需检验的因子

    # # 计算振幅时这个rolling我感觉问题很大。此时的df已经和index合并了，所以可能有20天压根就没交易，但是最高价最低价的复权价格已经被上面的merge函数给fillna了。此时rolling20天就并非是该股票20天个交易日的平均
    # period_df['振幅1'] = df['最高价_后复权'].rolling(20,min_periods=1).max()/df['最低价_后复权'].rolling(20,min_periods=1).min() - 1
    # df['每日振幅'] = df['最高价_后复权']/df['最低价_后复权'] - 1
    # period_df['振幅2'] = df['每日振幅'].rolling(20,min_periods=1).mean()
    # 所以要解决上面的问题，求振幅2不应该再period_df的基础上求，因为period_df已经和指数合并了，有些没交易的日期也填充了ffill的值，所以应该在原来的df上算每日振幅然后rolling20天，再想办法合并到period_df里面
    # 我无语了，给的答案直接原有df里面操作之后resample，这相当于定义都改了，就是每周调仓那振幅2就是每周的每日振幅的均值，每月调仓振幅2就是每月的每日振幅的均值，这TM谁不会？？
    # period_df['振幅2'] = df['每日振幅'].resample(period_type).mean()
    # 把index换回来
    # period_df.reset_index(inplace=True, drop=False)
    # period_df.set_index('交易日期', inplace=True)

    # 计算必须额外数据
    period_df['交易天数'] = df['是否交易'].resample(period_type).sum()
    period_df['市场交易天数'] = df['股票代码'].resample(period_type).size()
    # 这里还没完！！一定要注意有时候升采样的时候整个周期都不交易，resample.size还不行，也可能等于0
    period_df = period_df[period_df['市场交易天数'] > 0]  # 有的时候整个周期不交易（例如春节、国庆假期），需要将这一周期删除
   

    # 计算其他因子
    # 动量因子（当然更严谨可以使用对数涨跌幅，或者华泰金工里的加权的涨跌幅
    # 注意区分这个本周期涨跌幅是我们的因子，下面的“每天涨跌幅”和“涨跌幅”是一个list，用来到data_processing里面平移计算下个周期的每天涨跌幅和geometric return
    # 当然，这里动量因子也可以按照那样使用geometric return，但是没必要
    period_df['本周期涨跌幅'] = period_df['收盘价']/period_df['前收盘价']-1

    # 两种计算换手率的方法结果有差别
    # 我们换手率一种算法是在resample之前算，然后aggregate的时候取last；另一种是在resample之后，然后按下面的计算，然后把换手率从aggregate里面去掉
    # resample之前算再取last相当于只用了周期最后一天的换手率，而我们下面resample之后计算的是用的成交额的求和
    period_df['换手率'] = df['成交额'].resample(period_type).sum()/period_df['流通市值']
    # period_df['换手率'] = period_df['成交额']/period_df['流通市值']

    # 如果偷懒，振幅1定义为每个调仓周期的最高价的最高价除以该周期的最低价的最低价减一，而不是过去20天，振幅2定义为每周期内每日振幅的均值，而不是过去20天每日振幅的均值，那下面两行就能搞定
    # period_df['振幅1'] = period_df['最高价']/period_df['最低价'] - 1
    # period_df['振幅2'] = df['每日振幅'].resample(period_type).mean()

    # 如果严格遵循振幅1和振幅2的过去20天的数据
    period_df['振幅1'] = period_df['20日振幅_1']
    period_df['振幅2'] = period_df['20日振幅_2']
    
    # 计算周期资金曲线
    period_df['每天涨跌幅'] = df['涨跌幅'].resample(period_type).apply(lambda x: list(x))
    # 新版代码新加，修正了涨跌幅的计算，更加严谨
    period_df['涨跌幅'] = df['涨跌幅'].resample(period_type).apply(lambda x: (x + 1).prod() - 1)

    # 重新设定index
    period_df.reset_index(inplace=True)
    period_df['交易日期'] = period_df['周期最后交易日']
    del period_df['周期最后交易日']

    return period_df


# 计算涨跌停，注意计算涨跌停、一字涨跌停、开盘涨跌停还是用正常的开高收低，复权后的值只是在计算收益率的时候使用
def cal_zdt_price(df):
    """
    计算股票当天的涨跌停价格。在计算涨跌停价格的时候，按照严格的四舍五入。
    包含st股，但是不包含新股
    涨跌停制度规则:
        ---2020年8月3日
        非ST股票 10%
        ST股票 5%

        ---2020年8月4日至今
        普通非ST股票 10%
        普通ST股票 5%

        科创板（sh68） 20%
        创业板（sz30） 20%
        科创板和创业板即使ST，涨跌幅限制也是20%

        北交所（bj） 30%

    :param df: 必须得是日线数据。必须包含的字段：前收盘价，开盘价，最高价，最低价
    :return:
    """
    # 计算涨停价格
    # 普通股票
    cond = df['股票名称'].str.contains('ST')
    df['涨停价'] = df['前收盘价'] * 1.1
    df['跌停价'] = df['前收盘价'] * 0.9
    df.loc[cond, '涨停价'] = df['前收盘价'] * 1.05
    df.loc[cond, '跌停价'] = df['前收盘价'] * 0.95

    # 2020年8月3日之后涨跌停规则有所改变
    # 新规的科创板
    new_rule_kcb = (df['交易日期'] > pd.to_datetime('2020-08-03')) & df['股票代码'].str.contains('sh68')
    # 新规的创业板
    new_rule_cyb = (df['交易日期'] > pd.to_datetime('2020-08-03')) & df['股票代码'].str.contains('sz30')
    # 北交所条件
    cond_bj = df['股票代码'].str.contains('bj')

    # 科创板 & 创业板
    df.loc[new_rule_kcb | new_rule_cyb, '涨停价'] = df['前收盘价'] * 1.2
    df.loc[new_rule_kcb | new_rule_cyb, '跌停价'] = df['前收盘价'] * 0.8

    # 北交所
    df.loc[cond_bj, '涨停价'] = df['前收盘价'] * 1.3
    df.loc[cond_bj, '跌停价'] = df['前收盘价'] * 0.7

    # 四舍五入
    df['涨停价'] = df['涨停价'].apply(lambda x: float(Decimal(x * 100).quantize(Decimal('1'), rounding=ROUND_HALF_UP) / 100))
    df['跌停价'] = df['跌停价'].apply(lambda x: float(Decimal(x * 100).quantize(Decimal('1'), rounding=ROUND_HALF_UP) / 100))

    # 判断是否一字涨停
    df['一字涨停'] = False
    df.loc[df['最低价'] >= df['涨停价'], '一字涨停'] = True
    # 判断是否一字跌停
    df['一字跌停'] = False
    df.loc[df['最高价'] <= df['跌停价'], '一字跌停'] = True
    # 判断是否开盘涨停
    df['开盘涨停'] = False
    df.loc[df['开盘价'] >= df['涨停价'], '开盘涨停'] = True
    # 判断是否开盘跌停
    df['开盘跌停'] = False
    df.loc[df['开盘价'] <= df['跌停价'], '开盘跌停'] = True

    return df

# 计算策略评价指标
def strategy_evaluate(equity, select_stock):
    """
    :param equity:  每天的资金曲线
    :param select_stock: 每周期选出的股票
    :return:
    """

    # ===新建一个dataframe保存回测指标
    results = pd.DataFrame()

    # ===计算累积净值
    results.loc[0, '累积净值'] = round(equity['equity_curve'].iloc[-1], 2)

    # ===计算年化收益
    annual_return = (equity['equity_curve'].iloc[-1]) ** (
            '1 days 00:00:00' / (equity['交易日期'].iloc[-1] - equity['交易日期'].iloc[0]) * 365) - 1
    results.loc[0, '年化收益'] = str(round(annual_return * 100, 2)) + '%'

    # ===计算最大回撤，最大回撤的含义：《如何通过3行代码计算最大回撤》https://mp.weixin.qq.com/s/Dwt4lkKR_PEnWRprLlvPVw
    # 计算当日之前的资金曲线的最高点
    equity['max2here'] = equity['equity_curve'].expanding().max()
    # 计算到历史最高值到当日的跌幅，drowdwon
    equity['dd2here'] = equity['equity_curve'] / equity['max2here'] - 1
    # 计算最大回撤，以及最大回撤结束时间
    end_date, max_draw_down = tuple(equity.sort_values(by=['dd2here']).iloc[0][['交易日期', 'dd2here']])
    # 计算最大回撤开始时间
    start_date = equity[equity['交易日期'] <= end_date].sort_values(by='equity_curve', ascending=False).iloc[0]['交易日期']
    # 将无关的变量删除
    equity.drop(['max2here', 'dd2here'], axis=1, inplace=True)
    results.loc[0, '最大回撤'] = format(max_draw_down, '.2%')
    results.loc[0, '最大回撤开始时间'] = str(start_date)
    results.loc[0, '最大回撤结束时间'] = str(end_date)

    # ===年化收益/回撤比：我个人比较关注的一个指标
    results.loc[0, '年化收益/回撤比'] = round(annual_return / abs(max_draw_down), 2)

    # ===统计每个周期
    results.loc[0, '盈利周期数'] = len(select_stock.loc[select_stock['选股下周期涨跌幅'] > 0])  # 盈利笔数
    results.loc[0, '亏损周期数'] = len(select_stock.loc[select_stock['选股下周期涨跌幅'] <= 0])  # 亏损笔数
    results.loc[0, '胜率'] = format(results.loc[0, '盈利周期数'] / len(select_stock), '.2%')  # 胜率
    results.loc[0, '每周期平均收益'] = format(select_stock['选股下周期涨跌幅'].mean(), '.2%')  # 每笔交易平均盈亏
    results.loc[0, '盈亏收益比'] = round(select_stock.loc[select_stock['选股下周期涨跌幅'] > 0]['选股下周期涨跌幅'].mean() / \
                                    select_stock.loc[select_stock['选股下周期涨跌幅'] <= 0]['选股下周期涨跌幅'].mean() * (-1), 2)  # 盈亏比
    results.loc[0, '单周期最大盈利'] = format(select_stock['选股下周期涨跌幅'].max(), '.2%')  # 单笔最大盈利
    results.loc[0, '单周期大亏损'] = format(select_stock['选股下周期涨跌幅'].min(), '.2%')  # 单笔最大亏损

    # ===连续盈利亏损
    results.loc[0, '最大连续盈利周期数'] = max(
        [len(list(v)) for k, v in itertools.groupby(np.where(select_stock['选股下周期涨跌幅'] > 0, 1, np.nan))])  # 最大连续盈利次数
    results.loc[0, '最大连续亏损周期数'] = max(
        [len(list(v)) for k, v in itertools.groupby(np.where(select_stock['选股下周期涨跌幅'] <= 0, 1, np.nan))])  # 最大连续亏损次数

    # ===每年、每月收益率
    equity.set_index('交易日期', inplace=True)
    year_return = equity[['涨跌幅']].resample(rule='A').apply(lambda x: (1 + x).prod() - 1)
    monthly_return = equity[['涨跌幅']].resample(rule='M').apply(lambda x: (1 + x).prod() - 1)

    return results.T, year_return, monthly_return


# 绘制策略曲线
def draw_equity_curve_mat(df, data_dict, date_col=None, right_axis=None, pic_size=[16, 9], font_size=25,
                          log=False, chg=False, title=None, y_label='净值'):
    """
    绘制策略曲线
    :param df: 包含净值数据的df
    :param data_dict: 要展示的数据字典格式：｛图片上显示的名字:df中的列名｝
    :param date_col: 时间列的名字，如果为None将用索引作为时间列
    :param right_axis: 右轴数据 ｛图片上显示的名字:df中的列名｝
    :param pic_size: 图片的尺寸
    :param font_size: 字体大小
    :param chg: datadict中的数据是否为涨跌幅，True表示涨跌幅，False表示净值
    :param log: 是都要算对数收益率
    :param title: 标题
    :param y_label: Y轴的标签
    :return:
    """
    # 复制数据
    draw_df = df.copy()
    # 模块基础设置
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']  # 定义使用的字体，是个数组。
    plt.rcParams['axes.unicode_minus'] = False
    # plt.style.use('dark_background')

    plt.figure(figsize=(pic_size[0], pic_size[1]))
    # 获取时间轴
    if date_col:
        time_data = draw_df[date_col]
    else:
        time_data = draw_df.index
    # 绘制左轴数据
    for key in data_dict:
        if chg:
            draw_df[data_dict[key]] = (draw_df[data_dict[key]] + 1).fillna(1).cumprod()
        if log:
            draw_df[data_dict[key]] = np.log(draw_df[data_dict[key]].apply(float))
        plt.plot(time_data, draw_df[data_dict[key]], linewidth=2, label=str(key))
    # 设置坐标轴信息等
    plt.ylabel(y_label, fontsize=font_size)
    plt.legend(loc=2, fontsize=font_size)
    plt.tick_params(labelsize=font_size)
    plt.grid()
    if title:
        plt.title(title, fontsize=font_size)

    # 绘制右轴数据
    if right_axis:
        # 生成右轴
        ax_r = plt.twinx()
        # 获取数据
        key = list(right_axis.keys())[0]
        ax_r.plot(time_data, draw_df[right_axis[key]], 'y', linewidth=1, label=str(key))
        # 设置坐标轴信息等
        ax_r.set_ylabel(key, fontsize=font_size)
        ax_r.legend(loc=1, fontsize=font_size)
        ax_r.tick_params(labelsize=font_size)
    plt.show()


def draw_equity_curve_plotly(df, data_dict, date_col=None, right_axis=None, pic_size=[1500, 800], log=False, chg=False,
                             title=None, path=root_path + '/data/pic.html', show=True):
    """
    绘制策略曲线
    :param df: 包含净值数据的df
    :param data_dict: 要展示的数据字典格式：｛图片上显示的名字:df中的列名｝
    :param date_col: 时间列的名字，如果为None将用索引作为时间列
    :param right_axis: 右轴数据 ｛图片上显示的名字:df中的列名｝
    :param pic_size: 图片的尺寸
    :param chg: datadict中的数据是否为涨跌幅，True表示涨跌幅，False表示净值
    :param log: 是都要算对数收益率
    :param title: 标题
    :param path: 图片路径
    :param show: 是否打开图片
    :return:
    """
    draw_df = df.copy()

    # 设置时间序列
    if date_col:
        time_data = draw_df[date_col]
    else:
        time_data = draw_df.index

    # 绘制左轴数据
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for key in data_dict:
        if chg:
            draw_df[data_dict[key]] = (draw_df[data_dict[key]] + 1).fillna(1).cumprod()
        fig.add_trace(go.Scatter(x=time_data, y=draw_df[data_dict[key]], name=key, ))

    # 绘制右轴数据
    if right_axis:
        # for key in list(right_axis.keys()):
        key = list(right_axis.keys())[0]
        fig.add_trace(go.Scatter(x=time_data, y=draw_df[right_axis[key]], name=key + '(右轴)',
                                 marker=dict(color='rgba(220, 220, 220, 0.8)'), yaxis='y2'))  # 标明设置一个不同于trace1的一个坐标轴
    fig.update_layout(template="none", width=pic_size[0], height=pic_size[1], title_text=title, hovermode='x')
    # 是否转为log坐标系
    if log:
        fig.update_layout(yaxis_type="log")
    plot(figure_or_data=fig, filename=path, auto_open=False)

    # 打开图片的html文件，需要判断系统的类型
    if show:
        res = os.system('start ' + path)
        if res != 0:
            os.system('open ' + path)


# region 中性化相关的函数
def _factors_linear_regression(data, factor, neutralize_list, industry=None):
    """
    使用线性回归对目标因子进行中性化处理，此方法外部不可直接调用。
    :param data: 股票数据
    :param factor: 目标因子
    :param neutralize_list:中性化处理变量list
    :param industry: 行业字段名称，默认为None
    :return: 中性化之后的数据
    """
    # print(data['交易日期'].to_list()[0])
    lrm = LinearRegression(fit_intercept=True)  # 创建线性回归模型
    if industry:  # 如果需要对行业进行中性化，将行业的列名加入到neutralize_list中
        industry_cols = [col for col in data.columns if '所属行业' in col]
        for col in industry_cols:
            if col not in neutralize_list:
                neutralize_list.append(col)
    train = data[neutralize_list].copy()  # 输入变量
    label = data[[factor]].copy()  # 预测变量
    lrm.fit(train, label)  # 线性拟合
    predict = lrm.predict(train)  # 输入变量进行预测
    data[factor + '_中性'] = label.values - predict  # 计算残差
    return data

# endregion
