from sklearn.model_selection import train_test_split

# Constantes
SEED = 314

# Fonctions - Data loading, splitting and preprocessing
def show_nums_axes(ax, orient='v', fmt='.0g', stacked=False):
    """
    Affiche les valeurs numériques sur les barres d'un graphique à barres.

    Args:
        ax (matplotlib.axes.Axes): L'axe du graphique.
        fmt (str, optional): Format d'affichage des nombres
        orient (str, optional): L'orientation des barres. 'v' pour vertical (par défaut), 'h' pour horizontal.
        stacked (bool, optionnal): S'adapte pour un barstackedplot

    Returns:
        None
    """
    # Error management
    if orient not in ['h', 'v']:
        raise ValueError("orient doit être égal à 'h' ou 'v si spécifié")
    try:
        format(-10.5560, fmt)
    except ValueError as e:
        raise f"{e}: le format spécifié dans fmt n'est pas correct."
    if not isinstance(stacked, bool):
        raise ValueError("stacked doit être un booléen")
    # Body
    for p in ax.patches:
        width, height = p.get_width(), p.get_height()
        x, y = p.get_xy()
        if orient == 'v':
            if not stacked:
                ax.annotate(f'{height:{fmt}}' if height!=0 else '',
                            (x + width / 2., height), ha='center', va='bottom')
            else:
                ax.annotate(f'{height:{fmt}}', (x + width-4, y + height/2),
                            fontsize=10, fontweight='bold', ha='center', va='top')
        else:
            if not stacked:
                ax.annotate(f'{width:{fmt}}' if width!=0 else '',
                            (width, y + height / 2.), ha='left', va='center')
            else:
                ax.annotate(f'{width:{fmt}}', (x + width-1, y + height/2),
                            fontsize=10, fontweight='bold', ha='right', va='center')


# Fonctions - Data loading, splitting and preprocessing
def split_data(df, test_split=0.2, sampling=True, proportion=0.01, stratify=True):
    """
    Split the data into train and test sets
    :param test_split: Proportion of the data to include in the test split
    :param sampling: Whether to sample the data
    :param proportion: Proportion of the data to sample
    :return: X_train, X_test, y_train, y_test
    """
    assert df.columns[0] == "target", "The first column must be the target column"
    assert len(df.columns) > 1, "At least one feature column must be provided"
    # Sample the data if needed to reduce the size
    if sampling:
        df_sample = df.sample(frac=proportion, random_state=SEED)
    else:
        df_sample = df
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        df_sample.iloc[:, 1:],
        df_sample.iloc[:, 0],
        test_size=test_split,
        stratify=df_sample.iloc[:, 0] if stratify else None,
        random_state=SEED,
    )
    # Delete the sample dataframe to free up memory
    del df_sample
    # Return the train and test sets
    return X_train.squeeze(), X_test.squeeze(), y_train, y_test

def load_splits_from_parquet(X_train, X_test, cols, path):
    """
    Load and align train and test splits from a parquet file.
    This function reads a parquet file containing a corpus and a target column, 
    aligns the dataframes with the provided train and test splits, and reindexes 
    them to match the original indices.
    Parameters:
    X_train (pd.DataFrame): Training features dataframe.
    X_test (pd.DataFrame): Testing features dataframe.
    y_train (pd.Series): Training target series.
    y_test (pd.Series): Testing target series.
    col_name (str): The name of the column to be used as features.
    path (str): The path to the parquet file.
    Returns:
    tuple: A tuple containing the aligned and reindexed training features, 
           testing features, training target, and testing target.
    """
    # Define columns to read
    assert isinstance(cols, list)
    assert set(cols).issubset(set(X_train.columns)), f"Some column in {cols} not found in the dataframe"
    assert set(cols).issubset(set(X_test.columns)), f"Some column in {cols} not found in the dataframe"

    # Align the dataframes and reindex in the same order
    X_train = X_train.filter(cols)
    X_test = X_test.filter(cols)

    # Return the aligned data with features squeezed to remove the extra dimension if necessary
    return X_train, X_test