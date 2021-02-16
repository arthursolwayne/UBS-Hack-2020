import alphien
import pandas as pd
import numpy as np

# Creates a list of Historical S&P 500 Tickers
tickersList = list(alphien.data.getTickersSP500()['ticker'])
print("Created a list of " + str(len(tickersList)) + " tickers.")

# High-Low-Adjusted Close data (HLC) dataframe for S&P 500 since 2007
df_HLC = alphien.data.getHistoryData(ticker=tickersList, field = ['high_price','low_price','close_price'])

# Returns a dataframe of portfolio allocations
def payout(df, freq=300, period=14):
    print("Setting up dataframes")
    # High-Low-Adjusted Close data (HLC) dataframe for S&P 500 since 2007
    df_HLC = alphien.data.getHistoryData(ticker=tickersList, field = ['high_price','low_price','close_price'])
    # Adjusted close data (bb live) dataframe for S&P 500 since 2007
    df_bblive = alphien.data.getHistoryData(ticker=tickersList,field = "bb_live")
    
    # Cleaning Data
    # Sets nan values to zero
    df_HLC = df_HLC.fillna(value=0)
    df_bblive = df_bblive.fillna(value=0)
    # Remove duplicate columns
    df_bblive = df_bblive.loc[:,~df_bblive.columns.duplicated()]
    df_HLC = df_HLC.loc[:,~df_HLC.columns.duplicated()]
    
    # Create a dataframe for allocations and scores with the rebalancing dates
    df_Allo = df_bblive.loc[[i for j, i in enumerate(df_bblive.index) if j % freq == 0]]
    df_Allo.add_suffix('.Allo')
    df_Scores = df_bblive.loc[[i for j, i in enumerate(df_bblive.index) if j % freq == 0]]
    df_Scores.add_suffix('.Scores')
    print("Starting to loop")

    # Loop over the rebalancing dates:
    for row in range(len(df_Scores)):
        row_HLC = row*freq
        if row != 0:
            # Loop over the companies
            for col in range(len(df_Scores.columns)):
                if df_Scores.iloc[row,col] != 0:
                    col_HLC = col*3
                    dxs = []
                    for j in range(period):
                        dx_date = row_HLC-j
                        # Directional Movement Positives
                        pos_DMS = []
                        # Directional Movement Negatives
                        neg_DMS = []
                        # True Ranges
                        trs = []
                            
                        # Calculate the Positive and Negative Directional Movement for the past period days
                        for i in range(period):
                            dm_date = dx_date-i
                            # Current High - Previous High
                            pos_DM = df_HLC.iat[dm_date,col_HLC] - df_HLC.iat[dm_date-1,col_HLC]
                            # Previous Low - Current Low
                            neg_DM = df_HLC.iat[dm_date-1,col_HLC+1] - df_HLC.iat[dm_date,col_HLC+1]
                            
                            if pos_DM > neg_DM:
                            	neg_DM = 0
                            elif pos_DM < 0 and neg_DM < 0:
                            	pos_DM = 0
                            	neg_DM = 0
                            else:
                            	pos_DM = 0

                            pos_DMS.append(pos_DM)
                            neg_DMS.append(neg_DM)
                            
                        # Calculate the Average Positive and Negative DMs
                        pos_DM_period = 0
                        for k in pos_DMS:
                            pos_DM_period += k
                        pos_DM_period = pos_DM_period / period 

                        neg_DM_period = 0
                        for l in neg_DMS:
                            neg_DM_period += k
                        neg_DM_period = neg_DM_period / period

                        # Calculate the True Range for the past period days
                        for i in range(period):
                            tr_date = dx_date-i
                            # Current high - current low
                            hl = df_HLC.iat[tr_date,col_HLC] - df_HLC.iat[tr_date,col_HLC+1]
                            # Current high - previous low (abs value)
                            hc = abs(df_HLC.iat[tr_date,col_HLC] - df_HLC.iat[tr_date-1,col_HLC+2])
                            # Current low - previous low (abs value)
                            lc = abs(df_HLC.iat[tr_date,col_HLC+1] - df_HLC.iat[tr_date-1,col_HLC+2])
                            trs.append(max(hl,hc,lc))
                            
                        # Calculate the ATR
                        atr = 0
                        for j in trs: 
                            atr += j
                        atr = atr/period

                        # Avoid division by zero errors in case ATR is 0
                        if atr == 0: atr = 0.0000000000001
                    
                        # Calculate the Positive and Negative Directional Indicators and DXScore
                        Pos_DI = 100*pos_DM_period/atr
                        Neg_DI = 100*neg_DM_period/atr
                        DXScore = Pos_DI - Neg_DI
                        #DX = 100*(abs(Pos_DI - Neg_DI)/abs(Pos_DI + Neg_DI))
                        dxs.append(DXScore)
                    
                    # Calculate ADXScore and input it into df_Scores
                    ADXScore = 0
                    for k in dxs:
                        ADXScore += k
                    ADXScore = ADXScore / period
                    print("row"+str(row)+"col"+str(col)+"ADXScore:" + str(ADXScore))
                    df_Scores.iloc[row,col] = ADXScore
    
    # Set all values in allocations DataFrame to zero to clear           
    for col in df_Allo.columns:
        df_Allo[col].values[:] = 0  
    
    print("Creating allocations")
    for row in range(len(df_Scores)):
        #copies a row into a list
        rowcopy = []
        for col in range(len(df_Scores.columns)):
            rowcopy.append(df_HLC.iloc[0,col])       
        
        #creates a list of the indexes of the top 50 scores in df_Scores
        top50_index = sorted(range(len(rowcopy)), key=lambda i: rowcopy[i], reverse=True)[:50]
        
        #stores 0.02 into the top 50 stocks of df_Allo
        for i in top50_index:
            df_Allo.iloc[row,i] = 0.02

    return df_Allo

port = alphien.portfolio.Portfolio(tickersList)
port.addFeatures()
port.payout(payout)
port.evaluate(zoom="2007::2016")
port.backtest()