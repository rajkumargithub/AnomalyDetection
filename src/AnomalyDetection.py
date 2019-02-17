from pyculiarity import detect_ts
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from datetime import time
import numpy as np

plt.style.use('ggplot')

__author__ = 'Raj Shanmuganathan'

if __name__ == '__main__':
    rawdata = pd.read_csv('/Users/rshanm200/Workbench/Anamoly_detection/data/newrawdata1.csv', usecols=['datetime','online'])
    rawdata['timestamp'] = pd.to_datetime(rawdata['datetime'],format='%Y-%m-%d %H:%M:%S')
    rawdata['timestamp'] = rawdata['timestamp'].astype(np.int64) // 10**9
    rawdata['value'] = rawdata['online'].apply(lambda x: 0 if pd.isna(x) else x)
    rawdata = rawdata.drop(['datetime','online'],axis=1)
    print(rawdata)

    results = detect_ts(rawdata, max_anoms=0.01, alpha=0.05, direction='both',piecewise_median_period_weeks=10,granularity='hr')
    print(results)
    
    # format the twitter data nicely
    results['timestamp'] = pd.to_datetime(rawdata['timestamp'])
    rawdata.set_index('timestamp', drop=True)

    # make a nice plot
    f, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(rawdata['timestamp'], rawdata['value'], 'b')
    ax[0].plot(results['anoms'].index, results['anoms']['anoms'], 'ro')
    ax[0].set_title('Detected Anomalies')
    ax[1].set_xlabel('Time Stamp')
    ax[0].set_ylabel('Count')
    ax[1].plot(results['anoms'].index, results['anoms']['anoms'], 'b')
    ax[1].set_ylabel('Anomaly Magnitude')
    plt.show()

    print(results['anoms'])
