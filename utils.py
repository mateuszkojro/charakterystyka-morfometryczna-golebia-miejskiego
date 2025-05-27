import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt

class Dataset:

    @staticmethod
    def from_sheets():
        return Dataset(df=get_data())
        
    def __init__(self, df):
        self.df = df

    def fix_wing_span(self):
        self.df.dropna(inplace=True)
        self.df["Rozpiętość skrzydeł"] = self.df["Rozpiętość skrzydeł"].astype(float)
        self.df["Rozpiętość skrzydeł (cm)"] = self.df["Rozpiętość skrzydeł"].astype(float)

    def fix_intestine(self):
        self.df.dropna(inplace=True)
        self.df["Długość jelita cieńkiego (cm)"] = self.df["Długość dwunastnicy"].astype(float) + self.df["Długość jelita czczego"].astype(float) + self.df["Długość jelita biodrowego"].astype(float)
        self.df["Długość jelita grubego (cm)"] = ((self.df["Długość jelita ślepiego P"].replace("BRAK", pd.NA).astype(float) + self.df["Długość jelita ślepiego P"].astype(float)).astype(float) / 2) + self.df["Okrężnica z odbytnicą"].astype(float)
        self.df["Stosunek C/G"] = self.df["Długość jelita cieńkiego (cm)"] / self.df["Długość jelita grubego (cm)"]
        self.df["Całkowita długość jelit (cm)"] = self.df["Długość jelita cieńkiego (cm)"] + self.df["Długość jelita grubego (cm)"]
        

    def basic_stats(self, key):
        f, m = self._get_gender_dfs(self.df)
        return f"{key}: mean={self.df[key].mean():.2f}, std={self.df[key].std():.2f}\n\tF"

    def _get_gender_dfs(self, df):
        category = "Płeć"
        f = df[self.df[category] == "Samice"]
        m = df[self.df[category] == "Samce"]
        return f, m

    def _save_plot(self, fig, name):
        path = "images/" + name.replace("/", "_")
        print(f"Saving plot to '{path}'")
        plt.savefig(path, dpi=300, bbox_inches='tight')
        
    def compare_by_gender(self, x):
        f, m = self._get_gender_dfs(self.df[x])
        u = stats.mannwhitneyu(f, m)
        cv_f = stats.variation(f)
        cv_m = stats.variation(m)
        print(f"{x} female: mean={f.mean():.2f}, std={f.std():.2f}, cv={cv_f:.2f}")
        print(f"{x} male: mean={m.mean():.2f}, std={m.std():.2f}, cv={cv_m:.2f}")
        print(f"Mann-Whitney 'u' {x} by gender: pvalue={u.pvalue:.2f}")
        # print("F", self.basic_stats(x))
        # print("M", self.basic_stats(y))
        g = sns.boxplot(x="Płeć", y=x, data=self.df)
        self._save_plot(g, f"{x} by gender.png")
        return g
        
    def histogram(self, x, bins=7, **kwargs):
        return sns.histplot(data=self.df, x=x, stat="count", bins=bins, **kwargs)

    def linear_corr_pearson(self, x, y, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()
            
        print(self.basic_stats(x))
        print(self.basic_stats(y))
    
        # Pearson correlation
        corr = stats.pearsonr(self.df[x], self.df[y])
        f, m = self._get_gender_dfs(self.df)
        corr_f = stats.pearsonr(f[x], f[y])
        corr_m = stats.pearsonr(m[x], m[y])
    
        print(f"r^2={corr.statistic:.2f}, pvalue={corr.pvalue:.2f}")
        print(f"    F r^2={corr_f.statistic:.2f}, pvalue={corr_f.pvalue:.2f}")
        print(f"    M r^2={corr_m.statistic:.2f}, pvalue={corr_m.pvalue:.2f}")
    
        # Linear regression for line of best fit
        slope, intercept, r_value, p_value, std_err = stats.linregress(self.df[x], self.df[y])
        line_eq = f"y = {slope:.2f}x + {intercept:.2f}\nR² = {r_value**2:.2f}"
    
        # Plot
        # plt.figure(figsize=(8, 6))
        g = sns.scatterplot(data=self.df, x=x, y=y, ax=ax, **kwargs)
        sns.lineplot(x=self.df[x], y=slope * self.df[x] + intercept, color='black', label='Linia regresji', ax=ax)

        # Annotate equation on the plot
        ax.text(0.05, 0.95, line_eq, transform=plt.gca().transAxes,
                 fontsize=12, verticalalignment='top')
    
        ax.legend()
        if ax is None or not hasattr(ax, 'figure'):
            fig = ax.get_figure()
            self._save_plot(fig, f"images/Correlation between {x} and {y}.png")
        return ax

    def corr_body_mass(self, y, **kwargs):
        return self.linear_corr_pearson(x="Masa ciała (g)", y=y, **kwargs)

def init_notebook():
    pass
    # sns.set(rc={"figure.figsize":(5, 3)})
    # %load_ext autoreload
    # %autoreload 2

def get_data():
    df = pd.read_csv("https://docs.google.com/spreadsheets/d/1Xeg_KqurQLsgOAikRfmxNsrqY1WkusTG/export?format=csv&gid=2112095053")
    df = df.replace("BRAK", pd.NA).replace(float("NaN"), pd.NA).replace("X", pd.NA)
    df["Płeć"] = df["Płeć"].replace("W", "Samice").replace("M", "Samce")
    df["Długość ciała (cm)"] = df["Długość ciała"]
    df["Masa ciała (g)"] = df["Masa ciała (kg)"] * 1000
    return df

"""
g = sns.jointplot(x="Masa ciała (kg)", y="Średnica stępu/skoku LEWEGO", data=df, kind="reg", truncate=False)
"""

    