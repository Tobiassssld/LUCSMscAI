import pandas as pd
import time

# read file
lineitem = pd.read_csv(r'lineitem.csv', parse_dates=['l_shipdate'])
# print("lineitem shape: ", lineitem.shape)

# Q6
start_time_q6 = time.time()

# filter data
filtered_data_q6 = lineitem[
    (lineitem['l_shipdate'] >= '1994-01-01') &
    (lineitem['l_shipdate'] < '1995-01-01') &
    (lineitem['l_discount'].between(0.05, 0.07)) &
    (lineitem['l_quantity'] < 24)
].copy()

# print("filtered_data_q6 shape", filtered_data_q6.shape)

# calc revenue
filtered_data_q6.loc[:, 'revenue'] = filtered_data_q6['l_extendedprice'] * filtered_data_q6['l_discount']

print("Q6 result total Revenue:", filtered_data_q6['revenue'].sum())

end_time_q6 = time.time()
elapsed_time_q6 = end_time_q6 - start_time_q6
print(f"Q6 Filtering and calculation time: {elapsed_time_q6:.4f} seconds")


# Q1
start_time_q1 = time.time()

filtered_data_q1 = lineitem[
    (lineitem['l_shipdate'] <= pd.Timestamp('1998-12-01') - pd.Timedelta(days=90))
]
# print("filtered_data_q1 shape: ", filtered_data_q1.shape)

grouped_data_q1 = filtered_data_q1.groupby(['l_returnflag', 'l_linestatus']).agg(
    sum_qty=pd.NamedAgg(column='l_quantity', aggfunc='sum'),
    sum_base_price=pd.NamedAgg(column='l_extendedprice', aggfunc='sum'),
    sum_disc_price=pd.NamedAgg(column='l_extendedprice',
                               aggfunc=lambda x: sum(x * (1 - filtered_data_q1.loc[x.index, 'l_discount']))),
    sum_charge=pd.NamedAgg(column='l_extendedprice', aggfunc=lambda x: sum(
        x * (1 - filtered_data_q1.loc[x.index, 'l_discount']) * (1 + filtered_data_q1.loc[x.index, 'l_tax']))),
    avg_qty=pd.NamedAgg(column='l_quantity', aggfunc='mean'),
    avg_price=pd.NamedAgg(column='l_extendedprice', aggfunc='mean'),
    avg_disc=pd.NamedAgg(column='l_discount', aggfunc='mean'),
    count_order=pd.NamedAgg(column='l_quantity', aggfunc='count')
).reset_index()

sorted_data_q1 = grouped_data_q1.sort_values(by=['l_returnflag', 'l_linestatus'])

sorted_data_q1.to_csv('q1.out', index=False, sep='|')

end_time_q1 = time.time()
elapsed_time_q1 = end_time_q1 - start_time_q1
print(f"Q1 Filtering and calculation time: {elapsed_time_q1:.4f} seconds")

print("Q1 result")
print(sorted_data_q1)
