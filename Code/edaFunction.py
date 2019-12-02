#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('classic')

class edaFunction:
    def __init__(self, dframe):
        self.dframe = dframe

    # check info
    def ckInfo(self):
        cnt = 1
        # info
        try:
            print(str(cnt) + ': info()')
            cnt += 1
            print(self.dframe.info())
        except:
            print("Please input a dataFrame")
        
        # describe
        print(str(cnt)+': describe(): ')
        cnt+=1
        print(self.dframe.describe())
        
        # dtype
        print(str(cnt)+': dtypes: ')
        cnt+=1
        print(self.dframe.dtypes)

        # columns
        try:
            print(str(cnt)+': columns: ')
            cnt+=1
            print(self.dframe.columns)
        except:
            print("error! No columns!")

        # head
        print(str(cnt)+': head() -- ')
        cnt+=1
        print(self.dframe.head())

        # shape
        print(str(cnt)+': shape: ')
        cnt+=1
        print(self.dframe.shape)

    # value
    def ckValues(self):
        cnt = 1
        for i in self.dframe.columns :
            print(str(cnt)+':', i, 'value_counts(): ')
            print(self.dframe[i].value_counts())
            cnt +=1

    # check NA
    def ckNA(self):
        print(self.dframe.isna().sum())
        pass

    # drop NA
    def dropNA(self):
        return self.dframe.dropna()

    # fill NA with 0
    def fillNA(self):
        return self.dframe.fillna(0)

    def seasonEDA(self):
        print("The count of season bike using shows below :\n", self.dframe.groupby('season')['cnt'].sum())
        self.dframe.groupby('season')['cnt'].sum().plot(kind='bar', figsize=(20, 4))

    def holidayEDA(self):
        print("The count of holiday bike using shows below :\n", self.dframe.groupby('holiday')['cnt'].sum())
        self.dframe.groupby('holiday')['cnt'].sum().plot(kind = 'bar', figsize = (20, 4))

    def workingdayEDA(self):
        print("The count of workingday bike using shows below :\n", self.dframe.groupby('workingday')['cnt'].sum())
        self.dframe.groupby('workingday')['cnt'].sum().plot(kind = 'bar', figsize = (20, 4))

    def weatherTypeEDA(self):
        print("The count of Weather Tpye bike using shows below :\n", self.dframe.groupby('weather')['cnt'].sum())
        self.dframe.groupby('weather')['cnt'].sum().plot(kind = 'bar', figsize = (20, 4))

    def weekendEDA(self):
        print("The count of weekend bike using shows below :\n", self.dframe.groupby('weekend')['cnt'].sum())
        self.dframe.groupby('weekend')['cnt'].sum().plot(kind = 'bar', figsize = (20, 4))

    def hourEDA(self):
        self.dframe.groupby('hour')['cnt'].sum().plot(kind = 'bar', figsize = (20, 4))

    def dayEDA(self):
        self.dframe.groupby('day')['cnt'].sum().plot(kind = 'bar', figsize = (20, 4))
    
    def monthEDA(self):
        self.dframe.groupby('month')['cnt'].sum().plot(kind = 'bar', figsize = (20, 4))

    def yearEDA(self):
        self.dframe.groupby('year')['cnt'].sum().plot(kind = 'bar', figsize = (20, 4))

    def hists(self):
        fig,axes = plt.subplots(2,2)
        axes[0,0].hist(x = "TF", data = self.dframe, edgecolor = "black", linewidth = 2, color = 'blue')
        axes[0,0].set_title("Variation of Temperature")
        axes[0,1].hist(x = "TFF", data = self.dframe, edgecolor = "black",linewidth = 2, color = 'blue')
        axes[0,1].set_title("Variation of Temperature Feels")
        axes[1,0].hist(x = "    ", data = self.dframe, edgecolor = "black",linewidth = 2, color = 'blue')
        axes[1,0].set_title("Variation of WindSpeed")
        axes[1,1].hist(x = "Humidity", data = self.dframe, edgecolor = "black", linewidth = 2, color = 'blue')
        axes[1,1].set_title("Variation of Humidity")
        fig.set_size_inches(10,10)
    
    def corelationMatrix(self):
        cor_mat= self.dframe[:].corr()
        mask = np.array(cor_mat)
        mask[np.tril_indices_from(mask)] = False
        fig=plt.gcf()
        fig.set_size_inches(40,20)
        sns.heatmap(data = cor_mat, mask = mask, square = True, annot = True, cbar = True)



# %%
