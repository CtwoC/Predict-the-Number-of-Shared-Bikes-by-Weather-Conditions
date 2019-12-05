#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# plt.style.use('classic')

class edaFunction:
    def __init__(self, dframe):
        self.dframe = dframe
        self.color = sns.color_palette()

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
        self.dframe.groupby('season')['cnt'].sum().plot(kind='bar', figsize=(6, 4), color = self.color)

    def holidayEDA(self):
        print("The count of holiday bike using shows below :\n", self.dframe.groupby('holiday')['cnt'].sum())
        self.dframe.groupby('holiday')['cnt'].mean().plot(kind = 'bar', figsize = (6, 4), color = self.color)

    def workingdayEDA(self):
        print("The count of workingday bike using shows below :\n", self.dframe.groupby('workingday')['cnt'].sum())
        self.dframe.groupby('workingday')['cnt'].mean().plot(kind = 'bar', figsize = (6, 4), color = self.color)

    def weatherTypeEDA(self):
        print("The count of Weather Tpye bike using shows below :\n", self.dframe.groupby('weather')['cnt'].sum())
        self.dframe.groupby('weather')['cnt'].sum().plot(kind = 'bar', figsize = (6, 4), color = self.color)

    # def weekendEDA(self):
    #     print("The count of weekend bike using shows below :\n", self.dframe.groupby('weekend')['cnt'].sum())
    #     self.dframe.groupby('weekend')['cnt'].sum().plot(kind = 'bar', figsize = (20, 4))

    def hourEDA(self):
        ax1 = self.dframe.groupby('hour')['cnt'].sum().plot(kind = 'line', figsize = (6, 4), color = 'red', label = 'line')
        self.dframe.groupby('hour')['cnt'].sum().plot(kind = 'bar', figsize = (6, 4), color = 'blue')
        plt.legend(loc = 'upper right')

    def register(self):
        # fig,axes = plt.subplots(2,1)
        registered = list(self.dframe.registered)
        casual = list(self.dframe.casual)
        plt.boxplot([registered,casual],positions=[1,2])
        plt.violinplot([registered,casual],positions=[1,2])
    
    def dayEDA(self):
        self.dframe.groupby('day')['cnt'].sum().plot(kind = 'bar', figsize = (6, 4), color = 'blue')
    
    def monthEDA(self):
        self.dframe.groupby('month')['cnt'].sum().plot(kind = 'bar', figsize = (6, 4), color = 'blue')

    def yearEDA(self):
        self.dframe.groupby('year')['cnt'].sum().plot(kind = 'bar', figsize = (6, 4), color = self.color)

    def hists(self):
        fig,axes = plt.subplots(2,2)
        axes[0,0].hist(x = "TF", data = self.dframe, edgecolor = "black", linewidth = 2)
        axes[0,0].set_title("Variation of Temperature")
        axes[0,1].hist(x = "TFF", data = self.dframe, edgecolor = "black",linewidth = 2)
        axes[0,1].set_title("Variation of Temperature Feels")
        axes[1,0].hist(x = "WindSpeed", data = self.dframe, edgecolor = "black",linewidth = 2)
        axes[1,0].set_title("Variation of WindSpeed")
        axes[1,1].hist(x = "Humidity", data = self.dframe, edgecolor = "black", linewidth = 2)
        axes[1,1].set_title("Variation of Humidity")
        fig.set_size_inches(10,10)
    
    def hists2(self):
        fig,axes = plt.subplots(2,2)
        axes[0,0].hist(x = "TF", data = self.dframe, edgecolor = "black", linewidth = 2)
        axes[0,0].set_title("Variation of Temperature")
        axes[1,0].hist(x = "WindSpeed", data = self.dframe, edgecolor = "black",linewidth = 2)
        axes[1,0].set_title("Variation of WindSpeed")
        axes[1,1].hist(x = "Humidity", data = self.dframe, edgecolor = "black", linewidth = 2)
        axes[1,1].set_title("Variation of Humidity")
        fig.set_size_inches(10,10)
    
    def corelationMatrix(self):
        cor_mat= self.dframe[:].corr()
        mask = np.array(cor_mat)
        mask[np.tril_indices_from(mask)] = False
        fig=plt.gcf()
        fig.set_size_inches(36,18)
        sns.heatmap(data = cor_mat, mask = mask, square = True, annot = True, cbar = True)



# %%
