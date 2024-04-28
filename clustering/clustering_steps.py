import pandas as pd
import numpy as np
import os
from shutil import copytree
from sklearn.metrics import davies_bouldin_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.pyplot import figure

plt.rcParams.update(plt.rcParamsDefault)

# Disable warnings for chained assignments in pandas.
pd.options.mode.chained_assignment = None

plt.rcParams.update({'font.size': 11})
figure(figsize=(12, 10), dpi=100)


def load_data(usr_path):
    """
    Loads and combines training and testing data from specified directory,
    filters out entries with 'store_id' as 0, and resets the index.
    Args:
        usr_path (str): The file path to the user-specific data directory,
                        expected to contain 'train.csv' and 'test.csv'.
    Returns:
        pd.DataFrame: A DataFrame containing combined and filtered data
                      from the train and test datasets with a reset index
                      for further analysis.
    """
    train = pd.read_csv(usr_path + 'train.csv')  # Load training data
    test = pd.read_csv(usr_path + 'test.csv')  # Load testing data
    df = train.append(test)  # Combine train and test datasets
    df = df.loc[df['store_id'] != 0]  # Filter out entries where 'store_id' is 0
    return df.reset_index(drop=True)  # Reset the index of the DataFrame and return


def calculate_visit_fidelity_ratio(df, n_stores):
    """
    Calculates the visit fidelity ratio based on the most frequently visited store versus other stores.
    Args:
        df (pd.DataFrame): DataFrame containing user transactions.
        n_stores (int): Number of unique stores in the DataFrame.
    Returns:
        float: The computed visit fidelity ratio.
    """

    # Extract store IDs from the DataFrame into a list
    stores = df.store_id.to_list()

    # Count occurrences of each unique store ID
    occurences = [stores.count(x) for x in set(stores)]

    # Find the maximum occurrence, i.e., the most visited store
    xmax = max(occurences)

    # Total number of visits (or transactions) in the DataFrame
    xtotal = len(df)

    # Calculate the sum of visits to stores other than the most visited one
    sumx = xtotal - xmax

    # Compute the fidelity ratio based on whether there is more than one store
    # If there is only one store, the ratio is simply the proportion of visits to that store
    # If there are multiple stores, adjust the formula to factor in visits to other stores
    if n_stores != 1:
        fidelity_ratio = ((xmax - (1 / (n_stores - 1)) * sumx) / xtotal)
    elif n_stores == 1:
        fidelity_ratio = xmax / xtotal
    return fidelity_ratio


def calculate_price_fidelity_ratio(df, n_stores):
    """
    Calculates the price fidelity ratio based on spending at the most frequented store versus total spending.
    Args:
        df (pd.DataFrame): DataFrame containing user transactions.
        n_stores (int): Number of unique stores in the DataFrame.
    Returns:
        float: The computed price fidelity ratio.
    """

    # Aggregate total prices per store and convert the result into a list
    store_prices = df.groupby('store_id').sum()['price'].to_list()

    # Find the maximum total spent at any single store
    pmax = max(store_prices)

    # Calculate the total spent across all stores
    ptotal = sum(store_prices)

    # Compute the total spent at all other stores except the one with maximum spending
    sump = ptotal - pmax

    # Calculate the price fidelity ratio, which is a measure of spending concentration
    # If there is only one store, the ratio is the proportion of total spending at that store
    # If there are multiple stores, adjust the ratio to reflect the relative spending at the most frequented store
    if n_stores != 1:
        fidelity_ratio = ((pmax - (1 / (n_stores - 1)) * sump) / ptotal)
    elif n_stores == 1:
        fidelity_ratio = pmax / ptotal
    return fidelity_ratio


def vectorize_category(df):
    """
    Creates a vector representation of product categories purchased by a user.
    Args:
        df (pd.DataFrame): DataFrame containing user transactions.
    Returns:
        list: A vector with counts of each category and the user ID at the beginning.
    """

    # Initialize a vector of zeros for 24 categories.
    category_vector = 24 * [0]

    # Count occurrences of each category in the DataFrame.
    category_count = pd.value_counts(df.category)

    # Fill the category vector with counts for each corresponding category.
    for idx in category_count.index:
        category_vector[int(idx) - 1] = category_count[idx]

    # Insert the unique user ID at the beginning of the vector.
    category_vector.insert(0, df.user_id.unique()[0])

    return category_vector


def make_cluster_data(path, cluster_data, category_matrix):
    """
    Processes each user data to extract features and append to the cluster data.
    Args:
        path (str): Path to the user data directory.
        cluster_data (pd.DataFrame): DataFrame to store clusterable user attributes.
        category_matrix (pd.DataFrame): DataFrame to store category vectors for PCA.
    Returns:
        pd.DataFrame: Updated cluster data with new user features and embedded category vectors.
    """
    print("make_cluster_data")

    # Loop through each user directory within the provided path
    for user in os.listdir(path):
        usr_path = path + '/' + user + '/'
        df = load_data(usr_path)

        # Extract essential metrics and features from the user's data
        user_id = df.user_id.unique()[0]
        x_special = np.mean(df.special)  # Average special offer engagement
        x_price = np.mean(df.price)  # Average price of purchased items
        n_stores = len(df.store_id.unique())  # Count of unique stores visited
        x_basket_size = len(df) / len(df.order_id.unique())  # Average basket size per order
        group = df.group.unique()[0]  # User's group based on certain criteria

        # Calculate fidelity ratios for visit and price
        visit_fidelity_ratio = calculate_visit_fidelity_ratio(df, n_stores)
        price_fidelity_ratio = calculate_price_fidelity_ratio(df, n_stores)
        x_fidelity_ratio = (visit_fidelity_ratio + price_fidelity_ratio) / 2

        # Vectorize product categories
        category_vector = vectorize_category(df)

        # Compile extracted features into a series and append to the cluster data DataFrame
        row = [user_id, group, x_price, x_special, x_fidelity_ratio, x_basket_size]
        row = pd.Series(row, index=cluster_data.columns)
        cluster_data = cluster_data.append(row, ignore_index=True)

        # Append the category vector to the category matrix DataFrame
        category_vector = pd.Series(category_vector, index=category_matrix.columns)
        category_matrix = category_matrix.append(category_vector, ignore_index=True)
        cluster_data['user_id'] = cluster_data['user_id'].astype('int')

    # Reduce dimensionality of the category matrix using PCA
    reduced_dimension = PCA(n_components=1, random_state=1).fit_transform(category_matrix.iloc[:, -24:])
    embedded_category = pd.DataFrame(reduced_dimension, columns=['category_embbed'])

    # Join the embedded category vector with the cluster data DataFrame
    cluster_data = cluster_data.join(embedded_category)
    return cluster_data


def prep_data(path):
    """
    Prepares data for clustering by initializing data structures and processing user data from the specified path.

    The function initializes empty DataFrames for clustering and category data. It then calls
    `make_cluster_data` to populate these DataFrames based on the user data available in the specified
    directory. The result is a DataFrame ready for clustering algorithms.

    Args:
        path (str): Path to the directory containing user data folders.
    Returns:
        pd.DataFrame: A DataFrame containing preprocessed data for clustering, including user metrics
                      and category vectors.
    Note:
        The function could also shuffle the DataFrame before returning, though this is currently commented out.
    """

    print("prep_data")

    # Initialize DataFrame for user metrics
    cluster_data = pd.DataFrame(columns=['user_id', 'group', 'x_price',
                                         'x_special', 'x_fidelity_ratio', 'x_basket_size'])

    # Define columns for the category matrix
    columns = [*range(1, 25)]

    # Insert 'user_id' as the first column in the category matrix
    columns.insert(0, 'user_id')

    # Initialize DataFrame for category vectors
    category_matrix = pd.DataFrame(columns=columns)

    # Populate DataFrames with processed data
    cluster_data = make_cluster_data(path, cluster_data, category_matrix)

    # cluster_data = cluster_data.sample(frac=1) #random shuffle df

    return cluster_data


def make_clusters(cluster_data):
    """
    Performs clustering on preprocessed data and visualizes the clusters using t-SNE.

    This function scales the data using MinMax scaling, applies Agglomerative Clustering to categorize
    users into clusters, and then visualizes the clusters using t-SNE for dimensionality reduction.
    The clustering results are saved to a CSV file, and a scatter plot of the clusters is generated and saved.

    Args:
        cluster_data (pd.DataFrame): Preprocessed data containing user metrics necessary for clustering.

    Returns:
        pd.DataFrame: A DataFrame with user IDs and their respective cluster labels.

    Process:
        - Normalize the feature space.
        - Perform agglomerative clustering.
        - Map cluster labels back to the original data.
        - Save the labeled data to a CSV file.
        - Visualize the data using t-SNE and save the plot.
    """

    print("make_clusters")

    # Normalize feature columns
    norm = MinMaxScaler()
    cluster_data.iloc[:, 2:] = norm.fit_transform(cluster_data.iloc[:, 2:])

    # Prepare and execute clustering algorithm
    X = cluster_data.iloc[:, 2:].values
    clt_ = AgglomerativeClustering(n_clusters=4)
    labels = clt_.fit_predict(X)

    # Map cluster labels back to DataFrame
    cluster_data['label'] = labels

    # Extract relevant data for saving
    labeled_cluster_data = cluster_data[['user_id', 'label']]
    labeled_cluster_data.to_csv('labeled_cluster_data.csv')

    plt.clf()
    plt.close('all')

    # Perform t-SNE dimensionality reduction to visualize clusters in a 2D space
    dim = TSNE(n_components=2, random_state=1, perplexity=30, learning_rate=925, init='pca', early_exaggeration=1)
    reduced_dimension = dim.fit_transform(X)

    # Create DataFrame for visual plotting
    df = pd.DataFrame(reduced_dimension, columns=['pca1', 'pca2'])
    df['labels'] = labels
    colors = ['red', 'limegreen', 'dodgerblue', 'yellow']

    # Plot clusters using t-SNE output
    plt.scatter(df.values[:, 0], df.values[:, 1], c=df.labels, s=75,
                cmap=matplotlib.colors.ListedColormap(colors),
                alpha=0.95, edgecolors='black')

    for i, txt in enumerate(labels):
        plt.annotate(int(txt) + 1, (reduced_dimension[:, 0][i], reduced_dimension[:, 1][i]), fontsize=7)

    plt.axis([-50, 62, -30, 50])
    plt.savefig('clustering.png')
    plt.show()

    return labeled_cluster_data


def redistribute_users(labeled_cluster_data):
    """
    Distributes user data into directories based on their cluster labels for further processing.

    This function creates a directory for each cluster and moves the user data files
    into the corresponding cluster directory. It ensures that all user files are sorted
    into new directories representing their clusters, which can be useful for segmenting
    users for analysis or targeted strategies.

    Args:
        labeled_cluster_data (pd.DataFrame): A DataFrame containing user IDs and their corresponding cluster labels.
    Process:
        - Check for and create a main directory if it doesn't exist.
        - For each cluster, create a subdirectory.
        - Move each user's data to the respective cluster directory based on their label.
    """

    print("redistribute_users")
    df = labeled_cluster_data

    # Determine the number of unique clusters
    n_clusters = len(df.label.unique())

    # Create a main directory for all cluster directories if it does not exist
    if not os.path.exists('../postprocess_all_products/'):
        os.mkdir('../postprocess_all_products/')

    # print(df.label.unique())
    for i in range(n_clusters):
        try:
            os.mkdir('../postprocess/g' + str(i + 1))
        except:
            print('File already exists.')

        # Get the subset of data for the current cluster
        cluster = df.loc[df.label == i]

        # List of user IDs in the current cluster
        cluster_users = cluster.user_id.to_list()

        # Loop through the data directory
        for file in os.listdir('../data'):
            if int(file) in cluster_users:
                # Move each user's data to the appropriate cluster directory
                copytree('../data/' + str(file),
                         '../postprocess_all_products/g' + str(i + 1) + '/' + str(file))


def start_clustering_steps():
    """
    Coordinates the entire clustering process from data preparation to user redistribution.

    Sets the data directory, prepares data, performs clustering, and redistributes users
    based on cluster results. Outputs are saved to files and directories are updated.
    """
    path = '../data'  # Directory containing user data
    cluster_data = prep_data(path)  # Prepare data for clustering
    clusterized_data = make_clusters(cluster_data)  # Perform clustering
    redistribute_users(clusterized_data)  # Redistribute users by cluster


if __name__ == "__main__":
    start_clustering_steps()
