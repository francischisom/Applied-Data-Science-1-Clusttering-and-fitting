"""
This is the template file for the clustering and fitting assignment.
You will be expected to complete all the sections and
make this a fully working, documented file.
You should NOT change any function, file or variable names,
 if they are given to you here.
Make use of the functions presented in the lectures
and ensure your code is PEP-8 compliant, including docstrings.
Fitting should be done with only 1 target variable and 1 feature variable,
likewise, clustering should be done with only 2 variables.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LinearRegression


def plot_relational_plot(df):
    fig, ax = plt.subplots(figsize=()
    sns.scatterplot(data=df, x="bill_length_mm", y="flipper_length_mm",
                    hue="species", ax=ax)
    ax.set_title("Relational Plot: Bill Length vs Flipper Length")
    plt.savefig('relational_plot.png')
    return


def plot_categorical_plot(df):
    fig, ax = plt.subplots()
    sns.histplot(df["body_mass_g"].dropna(), kde=True, ax=ax)
    ax.set_title("Categorical Plot: Distribution of Body Mass")
    plt.savefig("categorical_plot.png")
    return


def plot_statistical_plot(df):
    fig, ax = plt.subplots(figsize=()
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Statistical Plot: Correlation Heatmap")
    plt.savefig('statistical_plot.png')
    return


def statistical_analysis(df, col: str):
    data = df[col].dropna()
    mean = np.mean(data)
    stddev = np.std(data, ddof=1)
    skew = ss.skew(data)
    excess_kurtosis = ss.kurtosis(data)
    return mean, stddev, skew, excess_kurtosis


def preprocessing(df):
    print("\nBasic Info:")
    print(df.info())
    print("\nSummary Statistics:")
    print(df.describe())
    print("\nCorrelation:")
    print(df.corr(numeric_only=True))

    df = df.dropna()
    return df


def writing(moments, col):
    print(f'For the attribute {col}:')
    print(f'Mean = {moments[0]:.2f}, '
          f'Standard Deviation = {moments[1]:.2f}, '
          f'Skewness = {moments[2]:.2f}, and '
          f'Excess Kurtosis = {moments[3]:.2f}.')

    if moments[2] > 0:
        skew_type = "right-skewed"
    elif moments[2] < 0:
        skew_type = "left-skewed"
    else:
        skew_type = "not skewed"

    if moments[3] > 0:
        kurt_type = "leptokurtic"
    elif moments[3] < 0:
        kurt_type = "platykurtic"
    else:
        kurt_type = "mesokurtic"

    print(f"The data is {skew_type} and {kurt_type}.")
    return


def perform_clustering(df, col1, col2):
    data = df[[col1, col2]].dropna()

    scaler = StandardScaler()
    X = scaler.fit_transform(data)

    def plot_elbow_method():
        inertias = []
        K = range(2, 8)
        for k in K:
            km = KMeans(n_clusters=k, random_state=42)
            km.fit(X)
            inertias.append(km.inertia_)
        fig, ax = plt.subplots()
        ax.plot(K, inertias, marker='o')
        ax.set_title("Elbow Plot")
        ax.set_xlabel("k")
        ax.set_ylabel("Inertia")
        plt.savefig("elbow_plot.png")
        return

    def one_silhouette_inertia():
        model = KMeans(n_clusters=3, random_state=42)
        labels = model.fit_predict(X)
        _score = silhouette_score(X, labels)
        _inertia = model.inertia_
        return labels, _score, _inertia, model

    plot_elbow_method()
    labels, _score, _inertia, model = one_silhouette_inertia()

    print(f"\nClustering Results: Silhouette = {_score:.3f}, Inertia = {_inertia:.2f}")

    centers = model.cluster_centers_
    xmodel, ymodel = centers[:, 0], centers[:, 1]
    cenlabels = [f"Cluster {i}" for i in range(len(centers))]

    return labels, data, xmodel, ymodel, cenlabels


def plot_clustered_data(labels, data, xkmeans, ykmeans, centre_labels):
    fig, ax = plt.subplots()
    sns.scatterplot(x=data.iloc[:, 0], y=data.iloc[:, 1],
                    hue=labels, palette="tab10", ax=ax)

    ax.scatter(xkmeans, ykmeans, color="black", s=150, marker="X")

    for i, label in enumerate(centre_labels):
        ax.text(xkmeans[i], ykmeans[i], label,
                fontsize=10, weight="bold", color="black")

    ax.set_title('Cluster Plot')
    plt.savefig('clustering.png')
    return


def perform_fitting(df, col1, col2):
    data = df[[col1, col2]].dropna()

    X = data[[col1]]
    y = data[col2]

    model = LinearRegression()
    model.fit(X, y)

    x_pred = pd.DataFrame(np.linspace(X.min(), X.max(), 100), columns=[col1])
    y_pred = model.predict(x_pred)
    

    print(f"\nFitting Results: y = {model.coef_[0]:.3f}x + {model.intercept_:.3f}")

    return data, x_pred, y_pred


def plot_fitted_data(data, x, y):
    fig, ax = plt.subplots()
    ax.scatter(data.iloc[:, 0], data.iloc[:, 1],
               color="gray", alpha=0.6, label="Data")

    ax.plot(x, y, color="red", linewidth=2, label="Fitted Line")
    ax.set_title("Fitting: Linear Regression")
    ax.set_xlabel(data.columns[0])
    ax.set_ylabel(data.columns[1])
    ax.legend()

    plt.savefig('fitting.png')
    return


def main():
    df = pd.read_csv('data.csv')
    df = preprocessing(df)

    col = 'body_mass_g'
    plot_relational_plot(df)
    plot_statistical_plot(df)
    plot_categorical_plot(df)

    moments = statistical_analysis(df, col)
    writing(moments, col)

    clustering_results = perform_clustering(df, 'flipper_length_mm', 'body_mass_g')
    plot_clustered_data(*clustering_results)

    fitting_results = perform_fitting(df, 'flipper_length_mm', 'body_mass_g')
    plot_fitted_data(*fitting_results)
    return


if __name__ == '__main__':
    main()
