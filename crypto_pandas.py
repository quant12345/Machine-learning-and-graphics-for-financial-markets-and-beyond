import pandas as pd


chunk_size = 1000000  # how many rows do we read on each chunk
Ncol_date, Ncol_time, Ncol_ask, Ncol_bid = 1, 2, 3, 4  # column numbers
skiprows = 0  # if there is a header then we transmit 1
name_file = 'BTCUSD.csv'

data = pd.read_csv(name_file,
                   header=None, skiprows=skiprows, chunksize=chunk_size)

df_minute = pd.DataFrame()  # empty dataframe for concatenation

i = 0
for chunk in data:
    print('chunk - ', i)
    i += 1
    # can replace the code below with your own!!!
    chunk[5] = chunk[Ncol_time].astype(str).str.zfill(6)# make time in format: 00:00:00
    chunk[5] = chunk[5].str[:2] + ":" + chunk[5].str[2:4] + ":" + chunk[5].str[4:]

    # combine date and time
    chunk[Ncol_date] = pd.to_datetime(pd.to_datetime({
        'year': chunk[Ncol_date] // 10000,
        'month': (chunk[Ncol_date] % 10000) // 100,
        'day': chunk[Ncol_date] % 100}).astype(str) + ' ' + chunk[5])

    chunk['Spread'] = ((chunk[Ncol_ask] - chunk[Ncol_bid]) / 0.001).astype('int64')

    # resample in 1 minute time
    df = chunk.resample(
        '1Min', on=Ncol_date).agg(
        OPEN=(Ncol_bid, 'first'), HIGH=(Ncol_bid, 'max'),
        LOW=(Ncol_bid, 'min'), CLOSE=(Ncol_bid, 'last'),
        TICKVOL=(Ncol_bid, 'count'), VOLUME=(Ncol_bid, 'count'),
        SPREAD=('Spread', 'first')).dropna()

    df.insert(0, 'DATETIME', df.index)
    df_minute = pd.concat([df_minute, df], ignore_index=True)


# resample again as there may be rows with the same datetime
dict_data = {'OPEN': 'first', 'HIGH': 'max', 'LOW': 'min', 'CLOSE': 'last',
             'TICKVOL': 'sum', 'VOLUME': 'sum', 'SPREAD': 'first'}
df_minute = df_minute.resample(
    '1Min', on='DATETIME').agg(dict_data).dropna().sort_index().reset_index()

# write data to file
df_minute.to_csv('BTCUSD_pandas.csv', header=False, index=False)
