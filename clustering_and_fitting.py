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


def plot_relational_plot(df):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x="flipper_length_mm",
        y="body_mass_g",
        hue="species",
        style="sex",
        ax=ax
    )
    ax.set_title("Relational Plot: Flipper Length vs Body Mass")
    ax.set_xlabel("Flipper Length (mm)")
    ax.set_ylabel("Body Mass (g)")
    plt.tight_layout()
    plt.savefig('relational_plot.png')
    return


def plot_categorical_plot(df):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=df, x="species", y="body_mass_g", hue="sex", ax=ax)
    ax.set_title("Categorical Plot: Average Body Mass by Species and Sex")
    ax.set_xlabel("Species")
    ax.set_ylabel("Average Body Mass (g)")
    plt.tight_layout()
    plt.savefig("categorical_plot.png")
    return
    


def plot_statistical_plot(df):
    fig, ax = plt.subplots(figsize=(8, 6))
    numeric_df = df.select_dtypes(include=[np.number])
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Statistical Plot: Correlation Heatmap")
    plt.tight_layout()
    plt.savefig('statistical_plot.png')
    return


def statistical_analysis(df, col: str):
    mean =  data.mean()
    stddev = data.std()
    skew = ss.skew(data)
    excess_kurtosis = ss.kurtosis(data)
    return mean, stddev, skew, excess_kurtosis


def preprocessing(df):
    # You should preprocess your data in this function and
    # make use of quick features such as 'describe', 'head/tail' and 'corr'.
    print("\nBasic Info:")
    print(df.info())
    print("\nSummary Statistics:")
    print(df.describe())
    print("\nCorrelation Matrix:")
    print(df.corr(numeric_only=True))

    # Drop missing rows
    df = df.dropna()
    return df


def writing(moments, col):
    print(f'\nFor the attribute {col}:')
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
    # Delete the following options as appropriate for your data.
    # Not skewed and mesokurtic can be defined with asymmetries <-2 or >2.
    print('The data was right/left/not skewed and platy/meso/leptokurtic.')
    return


def perform_clustering(df, col1, col2):
    data = df[[col1, co12]].dropna()
    X = StandardScaler().fit_transform(data)

    def plot_elbow_method():
     distortions = []
        K = range(2, 8)
        for k in K:
            km = KMeans(n_clusters=k, random_state=42)
            km.fit(X)
            distortions.append(km.inertia_)
        fig, ax = plt.subplots()
        ax.plot(K, distortions, 'bo-')
        ax.set_title("Elbow Method for Optimal k")
        ax.set_xlabel("Number of clusters")
        ax.set_ylabel("Inertia")
        plt.tight_layout()
        plt.savefig("elbow_plot.png")
        return

    def one_silhouette_inertia():
       kmeans = KMeans(n_clusters=3, random_state=42)
        labels = kmeans.fit_predict(X)
        _score = silhouette_score(X, labels)
        _inertia = kmeans.inertia_
        return labels, _score, _inertia, kmeans

    plot_elbow_method()
    labels, _score, _inertia, model = one_silhouette_inertia()

    print(f"\nClustering Results: Silhouette = {_score:.3f}, Inertia = {_inertia:.2f}")

    # Cluster centers
    centers = model.cluster_centers_
    xkmeans, ykmeans = centers[:, 0], centers[:, 1]
    cenlabels = [f"Cluster {i}" for i in range(len(centers))]

    return labels, data, xkmeans, ykmeans, cenlabels

def plot_clustered_data(labels, data, xkmeans, ykmeans, centre_labels):
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        data.iloc[:, 0],
        data.iloc[:, 1],
        c=labels,
        cmap="viridis",
        alpha=0.7,
        edgecolor="k"
    )
    ax.scatter(xkmeans, ykmeans, c="red", marker="X", s=200, label="Centers")
    for i, txt in enumerate(centre_labels):
    ax.text(xkmeans[i], ykmeans[i], txt, color="black", fontsize=10)
    ax.set_title("Clustering: K-Means Results")
    ax.set_xlabel(data.columns[0])
    ax.set_ylabel(data.columns[1])
    plt.legend()
    plt.tight_layout()
    plt.savefig("clustering.png")
    return


def perform_fitting(df, col1, col2):
    # Gather data and prepare for fitting
    fig, ax = plt.subplots(figsize=(8, 6))
    # Fit model
    data = df[[col1, col2]].dropna()
    X = data[[col1]]
    y = data[col2]
    model = LinearRegression()
    model.fit(X, y)

    # Predict across x
    x_pred = np.linspace(X.min(), X.max(), 100)
    y_pred = model.predict(x_pred)
    print(f"\nFitting Results: y = {model.coef_[0]:.3f}x + {model.intercept_:.3f}")
    
    return data, x, y


def plot_fitted_data(data, x, y):
    fig, ax = plt.subplots()
    ax.scatter(data.iloc[:, 0], data.iloc[:, 1], color="gray", alpha=0.6, label="Data")

    ax.plot(x, y, color="red", linewidth=2, label="Fitted Line")
    ax.set_title("Fitting: Linear Regression")
    ax.set_xlabel(data.columns[0])
    ax.set_ylabel(data.columns[1])
    ax.legend()
    plt.tight_layout()
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
