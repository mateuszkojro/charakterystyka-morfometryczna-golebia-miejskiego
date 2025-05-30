import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np

class Dataset:

    @staticmethod
    def from_sheets(remove_outliers=True ,dropna=False):
        return Dataset(df=get_data(dropna=dropna, remove_outliers=remove_outliers))

    def __init__(self, df):
        self.df = df

    def fix_wing_span(self):
        self.df.dropna(inplace=True)
        self.df["Rozpiętość skrzydeł (cm)"] = self.df[
            "Rozpiętość skrzydeł (cm)"
        ].astype(float)

    @property
    def data_cols(self):
        return [
            "Masa ciała (g)",

            "Długość ciała (cm)",
            "Długość tułowia (cm)",
            "Długość skrzydła P (cm)",
            "Długość skrzydła L (cm)",
            "Rozpiętość skrzydeł (cm)",
            "Długość ogona (cm)",
            
            "Długość głowy (cm)",
            "Długość dzioba (cm)",
            "Grubość dzioba (cm)",
            "Szerokość głowy (cm)",

            "Średnica stępu L (cm)",
            
            "Szerokość klatki piersiowej (cm)",
            "Obwód klatki piersiowej (cm)",
            "Głębokość klatki piersiowej (cm)",
            
            "Długość przełyku z wolem (cm)",
            "Długość żołądka gruczołowego (cm)",
            "Masa żołądka mięśniowego (g)",
            "Obwód żołądka mięśniowego (cm)",
            "Objętość żołądka mięśniowego (ml)",
            
            "Długość dwunastnicy (cm)",
            "Długość jelita czczego (cm)",
            "Długość jelita biodrowego (cm)",
            "Długość jelita ślepego P (cm)",
            "Długość jelita ślepego L (cm)",
            "Okrężnica z odbytnicą (cm)",
            
            "Masa serca (g)",
            "Masa wątroby (g)",
            "Masa nerki P (g)",
            "Masa nerki L (g)",
        ]

    def fix_intestine(self):
        self.df.dropna(inplace=True)
        self.df["Długość jelita cieńkiego (cm)"] = (
            self.df["Długość dwunastnicy (cm)"].astype(float)
            + self.df["Długość jelita czczego (cm)"].astype(float)
            + self.df["Długość jelita biodrowego (cm)"].astype(float)
        )
        self.df["Długość jelita grubego (cm)"] = (
            (
                self.df["Długość jelita ślepego P (cm)"]
                .replace("BRAK", pd.NA)
                .astype(float)
                + self.df["Długość jelita ślepego P (cm)"].astype(float)
            ).astype(float)
            / 2
        ) + self.df["Okrężnica z odbytnicą (cm)"].astype(float)
        self.df["Stosunek C/G"] = (
            self.df["Długość jelita cieńkiego (cm)"]
            / self.df["Długość jelita grubego (cm)"]
        )
        self.df["Całkowita długość jelit (cm)"] = (
            self.df["Długość jelita cieńkiego (cm)"]
            + self.df["Długość jelita grubego (cm)"]
        )

    def basic_stats(self, key):
        f, m = self._get_gender_dfs(self.df)
        return (
            f"{key}: mean={self.df[key].mean():.2f}, std={self.df[key].std():.2f}\n\tF"
        )

    def _get_gender_dfs(self, df):
        category = "Płeć"
        f = df[self.df[category] == "Samice"]
        m = df[self.df[category] == "Samce"]
        return f, m

    def _save_plot(self, fig, name):
        path = "images/" + name.replace("/", "_") + ".png"
        print(f"Saving plot to '{path}'")
        plt.savefig(path, dpi=300, bbox_inches="tight")

    def compare_by_gender(self, x):
        f, m = self._get_gender_dfs(self.df[x])
        u = stats.ttest_ind(f, m, equal_var=False)
        cv_f = stats.variation(f)
        cv_m = stats.variation(m)
        print(f"{x} female: mean={f.mean():.2f}, std={f.std():.2f}, cv={cv_f:.2f}")
        print(f"{x} male: mean={m.mean():.2f}, std={m.std():.2f}, cv={cv_m:.2f}")
        print(f"t-test {x} by gender: pvalue={u.pvalue:.2f}")
        # print("F", self.basic_stats(x))
        # print("M", self.basic_stats(y))
        g = sns.boxplot(x="Płeć", y=x, data=self.df)
        self._save_plot(g, f"{x} by gender")
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
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            self.df[x], self.df[y]
        )
        line_eq = f"y = {slope:.2f}x + {intercept:.2f}\nR² = {r_value**2:.2f}"

        # Plot
        # plt.figure(figsize=(8, 6))
        g = sns.scatterplot(data=self.df, x=x, y=y, ax=ax, **kwargs)
        sns.lineplot(
            x=self.df[x],
            y=slope * self.df[x] + intercept,
            color="black",
            label="Linia regresji",
            ax=ax,
        )

        # Annotate equation on the plot
        ax.text(
            0.05,
            0.95,
            line_eq,
            transform=plt.gca().transAxes,
            fontsize=12,
            verticalalignment="top",
        )

        ax.legend()
        if ax is None or not hasattr(ax, "figure"):
            fig = ax.get_figure()
            self._save_plot(fig, f"Correlation between {x} and {y}")
        return ax

    def corr_body_mass(self, y, **kwargs):
        return self.linear_corr_pearson(x="Masa ciała (g)", y=y, **kwargs)
    
    def corr_heatmap(self, filter, columns, title, vmin=-1, vmax=1, **kwargs):
        plt.figure(figsize=(18, 10))
        corr = self.df[filter][self.data_cols].corr(numeric_only=True, method='pearson')
        ones = np.ones_like(corr)
        mask = ones - np.tril(ones) 
        g = sns.heatmap(
            corr[columns],
            annot=True,
            mask=mask.astype(bool),
            fmt=".2f",
            cmap="coolwarm",
            square=False,
            vmin=vmin,
            vmax=vmax,
            **kwargs,
        )
        self._save_plot(g, title)
        # g.set_title(title)
        return g



def init_notebook():
    pass
    # sns.set(rc={"figure.figsize":(5, 3)})
    # %load_ext autoreload
    # %autoreload 2


def get_data(remove_outliers, dropna):
    df = pd.read_csv(
        "https://docs.google.com/spreadsheets/d/1Xeg_KqurQLsgOAikRfmxNsrqY1WkusTG/export?format=csv&gid=2112095053"
    )
    df = df.replace("BRAK", pd.NA).replace(float("NaN"), pd.NA).replace("X", pd.NA)
    df["Płeć"] = df["Płeć"].replace("W", "Samice").replace("M", "Samce")
    df["Masa ciała (g)"] = df["Masa ciała (kg)"] * 1000
    del df["Masa ciała (kg)"]

    df.rename(
        columns={
            "Długość dzioba": "Długość dzioba (cm)",
            "Grubość dzioba": "Grubość dzioba (cm)",
            "Długość głowy": "Długość głowy (cm)",
            "Szerokość głowy": "Szerokość głowy (cm)",
            "Długość tułowia": "Długość tułowia (cm)",
            "Długość ciała": "Długość ciała (cm)",
            "Długość skrzydła P": "Długość skrzydła P (cm)",
            "Długość skrzydła L": "Długość skrzydła L (cm)",
            "Rozpiętość skrzydeł": "Rozpiętość skrzydeł (cm)",
            "Szerokość klatki piersiowej": "Szerokość klatki piersiowej (cm)",
            "Obwód klatki piersiowej": "Obwód klatki piersiowej (cm)",
            "Głebokość klatki piersiowej": "Głębokość klatki piersiowej (cm)",
            "Długość ogona": "Długość ogona (cm)",
            "Średnica stępu/skoku LEWEGO": "Średnica stępu L (cm)",
            "Masa ciała (kg)": "Masa ciała (g)",  # ⚠️ convert values too!
            "Długość przełyku z wolem (cm)": "Długość przełyku z wolem (cm)",
            "Długość żołądka gruczołowego (cm)": "Długość żołądka gruczołowego (cm)",
            "Masa żółądka mięśniowego (g)": "Masa żołądka mięśniowego (g)",
            "Obwód żołądka mięśniowego w naj.m. (cm)": "Obwód żołądka mięśniowego (cm)",
            "Objętość żołądka mięśniowego (ml)": "Objętość żołądka mięśniowego (ml)",
            "Długość dwunastnicy": "Długość dwunastnicy (cm)",
            "Długość jelita czczego": "Długość jelita czczego (cm)",
            "Długość jelita biodrowego": "Długość jelita biodrowego (cm)",
            "Długość jelita ślepiego P": "Długość jelita ślepego P (cm)",
            "Długość jelita ślepiego L": "Długość jelita ślepego L (cm)",
            "Okrężnica z odbytnicą": "Okrężnica z odbytnicą (cm)",
            "Masa serca (g)": "Masa serca (g)",
            "Masa wątroby (g)": "Masa wątroby (g)",
            "Nerki\tP": "Masa nerki P (g)",
            "Nerki L": "Masa nerki L (g)",
            "Płeć": "Płeć",
            "Długość ciała (cm)": "Długość ciała (cm)",
            "Masa ciała (g)": "Masa ciała (g)",
        },
        inplace=True,
    )
    # df["Rozpiętość skrzydeł (cm)"] = (
    #     df["Rozpiętość skrzydeł (cm)"].dropna().astype(float)
    # )


    if remove_outliers:
        print("Replacing outliers with na")
        df[outlier_indeces(df)] = pd.NA

    if dropna:
        print("Removing na")
        df.dropna(inplace=True)


    def to_float(number):
        try:
            return float(number)
        except TypeError as e:
            return float("nan")
            
    problematic_cols = ["Rozpiętość skrzydeł (cm)", 'Długość skrzydła L (cm)', 'Długość jelita biodrowego (cm)', 'Długość jelita ślepego P (cm)', 'Długość jelita ślepego L (cm)']
    for col in problematic_cols:
        df[col] = df[col].apply(to_float).astype(float)
    return df


"""
g = sns.jointplot(x="Masa ciała (kg)", y="Średnica stępu/skoku LEWEGO", data=df, kind="reg", truncate=False)
"""


def combined_sample_stats(groups):
    # groups = list of tuples: (n, mean, std)
    total_n = sum(n for n, _, _ in groups)
    mean = sum(n * x for n, x, _ in groups) / total_n

    within = sum((n - 1) * s**2 for n, _, s in groups)
    between = sum(n * (x - mean) ** 2 for n, x, _ in groups)

    var = (within + between) / (total_n - 1)
    return mean, var**0.5


def modified_z_score(series):
    series = series.copy().fillna(series.mean())
    median = np.median(series)
    mad = np.median(np.abs(series - median))
    if mad == 0:
        return np.zeros(len(series))  # Avoid division by zero
    return 0.6745 * (series - median) / mad


def outlier_indeces(df, zscore=3.5):
    z_scores = df.select_dtypes(include=[np.number]).apply(modified_z_score)
    outliers = (np.abs(z_scores) > 3.5)
    return outliers
