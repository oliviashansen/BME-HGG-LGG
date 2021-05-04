import pandas as pd

def advancedStats(data, labels):
    '''Advanced stats should leverage pandas to calculate
    some relevant statistics on the data.

    data: numpy array of data
    labels: numpy array of labels
    '''
    # convert to dataframe
    df = pd.DataFrame(data)


    # print skew and kurtosis for every column
    
    # skew
    skew = df.skew(axis=0)
    
    # kurtosis
    kurt = df.kurt(axis=0)
    
    #len(skew) and len(kurt) are the same 
    # prints skew and kurtosis data for every column
    for i in range(len(skew)):
        print("Column " + str(i) + " statistics:\n\tSkewness: " + str(skew[i]) + 
              "\tKurtosis: " + str(kurt[i]))
        

    # assign in labels
    df["labels"] = labels

    print("\n\nDataframe statistics")

    # groupby labels into "benign" and "malignant"

    # collect means and standard deviations for columns,
    # grouped by label
    
    # grouped by label, aggregated by mean
    x = df.groupby("labels").mean()
    
    # grouped by label, aggreagated by standard deviation
    y = df.groupby("labels").std()
    

    # Print mean and stddev for Benign
    # benign will be 0th row because of alphabetical sorting of groupby
    print("Benign Stats:")
    print("Mean:")
    print(x.iloc[0])
    print("Std:")
    print(y.iloc[0])

    print("\n")

    # Print mean and stddev for Malignant
    # malignant will be 1st row
    print("Malignant Stats:")
    print("Mean:")
    print(x.iloc[1])
    print("Std:")
    print(y.iloc[1])

