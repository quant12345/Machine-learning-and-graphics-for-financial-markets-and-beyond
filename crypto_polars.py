import polars as pl

chunk_size = 1000000  # how many rows do we read on each chunk
skiprows = 0  # if there is a header then we transmit 1
name_file = 'BTCUSD.csv'

reader = pl.read_csv_batched(name_file, has_header=False,
                             skip_rows=skiprows, batch_size=chunk_size)
df_minute = pl.DataFrame()  # empty dataframe for concatenation

i = 0
while True:
    print('chunk - ', i)
    i += 1
    batches = reader.next_batches(1)
    if not batches:
        break

    for chunk in batches:
        # can replace the code below with your own!!!
        chunk = chunk.with_columns([ # change date(column_2) and time(column_3)
            pl.date(
                year=pl.col('column_2') // 10000,
                month=(pl.col('column_2') % 10000) // 100,
                day=pl.col('column_2') % 100
            ).alias('column_2').dt.to_string('%Y-%m-%d').alias('column_2'),
            (pl.col('column_3').cast(pl.Utf8).str.zfill(6).str.slice(0, 2) + ':' +
             pl.col('column_3').cast(pl.Utf8).str.zfill(6).str.slice(2, 2) + ':' +
             pl.col('column_3').cast(pl.Utf8).str.zfill(6).str.slice(4, 2))
            .alias('column_3')]
        ).with_columns( # combine date and time
            [(pl.col('column_2') + ' ' + pl.col('column_3')).str.to_datetime()
             .alias('DATETIME'),
             ((pl.col('column_4') - pl.col('column_5')) / 0.001).cast(pl.Int64)
             .alias('Spread')] # resample in 1 minute time
        ).sort('DATETIME').group_by_dynamic('DATETIME', every='1m').agg([
            pl.col('column_5').first().alias('OPEN'),
            pl.col('column_5').max().alias('HIGH'),
            pl.col('column_5').min().alias('LOW'),
            pl.col('column_5').last().alias('CLOSE'),
            pl.col('column_5').count().alias('TICKVOL'),
            pl.col('column_5').count().alias('VOLUME'),
            pl.col('Spread').first().alias('SPREAD')
        ])

        df_minute = pl.concat([df_minute, chunk])

# resample again as there may be rows with the same datetime
df_minute = (
    df_minute
    .sort('DATETIME')
    .group_by_dynamic('DATETIME', every='1m')
    .agg([
        pl.col('OPEN').first(),
        pl.col('HIGH').max(),
        pl.col('LOW').min(),
        pl.col('CLOSE').last(),
        pl.col('TICKVOL').sum(),
        pl.col('VOLUME').sum(),
        pl.col('SPREAD').first()
    ])
).with_columns(
    pl.col('DATETIME').dt.to_string('%Y-%m-%d %H:%M:%S')
)
# write data to file
df_minute.write_csv('BTCUSD_polars.csv', include_header=False)