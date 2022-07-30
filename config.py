import os

# ===选股参数设定
period = 'M'  # W代表周，M代表月

# # ===获取项目根目录
# _ = os.path.abspath(os.path.dirname(__file__))  # 返回当前文件路径
# root_path = os.path.abspath(os.path.join(_, '..'))  # 返回根目录文件夹

root_path = os.getcwd()
stock_data_path = 'D:\\self_learning_python\\self_learning\\My_Strategies_Stocks\\real_time_stock_data_update\\all_stock_data\\stock_daily'
index_data_path = 'D:\\self_learning_python\\self_learning\\My_Strategies_Stocks\\real_time_stock_data_update\\all_stock_data\\index_daily'

back_test_start = '20070101'
back_test_end = '20220728'