import pandas as pd

# 读取lineitem.tbl文件，使用|作为分隔符，并忽略最后的多余分隔符
lineitem_df = pd.read_csv(r'C:/course/data management/Assignment 0/ADM_assignment1_V0/lineitem.tbl.60000', sep='|', header=None, usecols=range(16))

# 设置列名称（与TPC-H lineitem表结构匹配）
lineitem_df.columns = [
    'l_orderkey', 'l_partkey', 'l_suppkey', 'l_linenumber', 'l_quantity',
    'l_extendedprice', 'l_discount', 'l_tax', 'l_returnflag', 'l_linestatus',
    'l_shipdate', 'l_commitdate', 'l_receiptdate', 'l_shipinstruct',
    'l_shipmode', 'l_comment'
]

# 将DataFrame保存为CSV文件
lineitem_df.to_csv('lineitem.csv', index=False)
