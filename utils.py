import pandas as pd
import seaborn as sns
import scipy.stats as stats

class Dataset:

    @staticmethod
    def from_sheets():
        return Dataset(df=get_data())
        
    def __init__(self, df):
        self.df = df

    def fix_wing_span(self):
        ds.df.dropna(inplace=True)
        ds.df["Rozpiętość skrzydeł"] = ds.df["Rozpiętość skrzydeł"].astype(float)

    def basic_stats(self, key):
        return f"{key}: mean={self.df[key].mean():.2f}, std={self.df[key].std():.2f}"

    def _get_gender_dfs(self):
        category = "Płeć"
        f = self.df[x][self.df[category] == "W"]
        m = self.df[x][self.df[category] == "M"]
        return f, m
        
    
    def compare_by_gender(self, x):
        f, m = self._get_gender_dfs()
        u = stats.mannwhitneyu(f, m)
        print(f"{x} female: mean={f.mean():.2f}, std={f.std():.2f}")
        print(f"{x} male: mean={m.mean():.2f}, std={m.std():.2f}")
        print(f"Mann-Whitney 'u' {x} by gender: pvalue={u.pvalue:.2f}")
        # print("F", self.basic_stats(x))
        # print("M", self.basic_stats(y))
        return sns.boxplot(x=category, y=x, data=self.df)
        
    
    def linear_corr_pearson(self, x, y, **kwargs):
        print(self.basic_stats(x))
        print(self.basic_stats(y))
        corr = stats.pearsonr(self.df[x], self.df[y])
        f,m = self._get_gender_dfs()
        corr_f = stats.pearsonr(f[x], f[y])
        corr_m = stats.pearsonr(m[x], m[y])
        print(f"r^2={corr.statistic:.2f}, pvalue={corr.pvalue:.2f}")
        print(f"    F r^2={corr_f.statistic:.2f}, pvalue={corr_f.pvalue:.2f}")
        print(f"    M r^2={corr_m.statistic:.2f}, pvalue={corr_m.pvalue:.2f}")
        g = sns.scatterplot(data=self.df,x=x,  y=y, **kwargs)
        return g

    def corr_body_mass(self, y, **kwargs):
        return self.linear_corr_pearson(x="Masa ciała (kg)", y=y, **kwargs)

def init_notebook():
    sns.set(rc={"figure.figsize":(5, 3)})
    # %load_ext autoreload
    # %autoreload 2

def get_data():
    df = pd.read_csv("https://docs.google.com/spreadsheets/d/1Xeg_KqurQLsgOAikRfmxNsrqY1WkusTG/export?format=csv&gid=2112095053")
    df = df.replace("BRAK", pd.NA).replace(float("NaN"), pd.NA).replace("X", pd.NA)
    return df

"""
g = sns.jointplot(x="Masa ciała (kg)", y="Średnica stępu/skoku LEWEGO", data=df, kind="reg", truncate=False)
"""

    