import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pathlib import Path


def load_data(input_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype in ["int64", "float64"]:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates()


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    categorical_columns = df.select_dtypes(include=["object"]).columns
    le = LabelEncoder()
    for col in categorical_columns:
        df[col] = le.fit_transform(df[col])
    return df


def handle_outliers_iqr(df: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns

    for col in numeric_columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

    return df


def split_and_scale(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
):

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


def save_preprocessed_data(
    X_train, X_test, y_train, y_test, output_dir: str
):
    pd.DataFrame(X_train).to_csv(output_dir / "X_train.csv", index=False)
    pd.DataFrame(X_test).to_csv(output_dir / "X_test.csv", index=False)
    y_train.to_csv(output_dir / "y_train.csv", index=False)
    y_test.to_csv(output_dir / "y_test.csv", index=False)



def main():
    BASE_DIR = Path(__file__).resolve().parent.parent

    input_path = BASE_DIR / "synthetic_employee_burnout_raw" / "synthetic_employee_burnout.csv"
    output_dir = BASE_DIR / "preprocessing" / "synthetic_employee_burnout_preprocessing"
    output_dir.mkdir(parents=True, exist_ok=True)

    target_column = "Burnout"

    df = load_data(input_path)

    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = encode_categorical_features(df)
    df = handle_outliers_iqr(df)

    X_train, X_test, y_train, y_test = split_and_scale(
        df, target_column
    )

    save_preprocessed_data(
        X_train, X_test, y_train, y_test, output_dir
    )

    print("Preprocessing selesai. Dataset siap digunakan.")


if __name__ == "__main__":
    main()
