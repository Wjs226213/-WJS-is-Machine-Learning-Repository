import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA


## 首先完成数据的读取和加载
def load_and_explore_data():
    print("="*50)   
    print("1. Data Loading and Exploratory Analysis")
    print("="*50)
    
    # 通过官方数据加载
    iris = load_iris()
    
    # Create DataFrame
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    df['target_names'] = df['target'].map({i: name for i, name in enumerate(iris.target_names)})  
    
    print(f"Dataset shape: {df.shape}")
    print("\nFirst 5 rows of the dataset:")
    print(df.head())
    
    print("\nBasic statistical information:")
    print(df.describe())
    
    print("\nClass distribution:")
    print(df['target_names'].value_counts())
    
    return df, iris


def visualize_data(df, iris):
    """Data visualization"""
    print("\n" + "="*50)
    print("2. Data Visualization")
    print("="*50)
    
    # Ensure the directory exists（创建同级目录）
    output_dir = './iris_fit_history_plot_show'  # 同级目录：./ 表示当前py文件所在目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    # Set plot style
    sns.set(style="whitegrid")
    
    # 1. Box plots for each feature
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    sns.boxplot(x='target_names', y='sepal length (cm)', data=df)
    plt.title('Sepal Length Distribution')
    
    plt.subplot(2, 2, 2)
    sns.boxplot(x='target_names', y='sepal width (cm)', data=df)
    plt.title('Sepal Width Distribution')
    
    plt.subplot(2, 2, 3)
    sns.boxplot(x='target_names', y='petal length (cm)', data=df)
    plt.title('Petal Length Distribution')
    
    plt.subplot(2, 2, 4)
    sns.boxplot(x='target_names', y='petal width (cm)', data=df)
    plt.title('Petal Width Distribution')
    
    plt.tight_layout()
    # 关键修改1：保存路径改为同级目录（去掉 ../）
    plt.savefig('./iris_fit_history_plot_show/boxplots.png')
    plt.close()  # 关闭图形，释放内存（避免多个图叠加）
    
    # 2. Pair plot of features
    plt.figure(figsize=(10, 8))
    sns.pairplot(df, hue='target_names', vars=iris.feature_names)
    # 关键修改2：保存路径改为同级目录
    plt.savefig('./iris_fit_history_plot_show/pairplot.png')
    plt.close()
    
    # 3. Correlation heatmap
    plt.figure(figsize=(8, 6))
    correlation_matrix = df[iris.feature_names].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Feature Correlation Heatmap')
    # 关键修改3：保存路径改为同级目录
    plt.savefig('./iris_fit_history_plot_show/correlation_heatmap.png')
    plt.close()
    
    # 4. Feature distribution histograms
    plt.figure(figsize=(12, 8))
    for i, feature in enumerate(iris.feature_names):
        plt.subplot(2, 2, i+1)
        sns.histplot(df, x=feature, hue='target_names', kde=True)
        plt.title(f'{feature} Distribution')
    
    plt.tight_layout()
    # 关键修改4：保存路径改为同级目录
    plt.savefig('./iris_fit_history_plot_show/histograms.png')
    plt.close()


def train_models(X_train, X_test, y_train, y_test):
    """Train multiple classification models and return performance metrics"""
    print("\n" + "="*50)
    print("3. Model Training")
    print("="*50)
    
    # Define models to use
    models = {
        "KNN": KNeighborsClassifier(n_neighbors=3),
        "Logistic Regression": LogisticRegression(max_iter=200),
        "SVM": SVC(kernel='rbf', probability=True),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(n_estimators=100)
    }
    
    # Train and evaluate each model
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f"\nTraining model: {name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        
        # Detailed classification report
        print(f"Classification report:\n{classification_report(y_test, y_pred)}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"Confusion matrix:\n{cm}")
        
        # Save results
        results[name] = {
            'accuracy': accuracy,
            'y_pred': y_pred,
            'cm': cm
        }
        trained_models[name] = model
    
    return results, trained_models


def visualize_results(results, X_test, y_test, iris):
    """Visualize model results"""
    print("\n" + "="*50)
    print("4. Result Visualization")
    print("="*50)
    
    # Ensure the directory exists（再次确认目录存在，避免异常）
    output_dir = './iris_fit_history_plot_show'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    # 1. Accuracy comparison
    plt.figure(figsize=(10, 6))
    model_names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in model_names]
    
    sns.barplot(x=model_names, y=accuracies)
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.ylim(0.8, 1.0)
    
    # Add accuracy value labels
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.01, f'{acc:.4f}', ha='center')
    
    # 关键修改5：保存路径改为同级目录
    plt.savefig('./iris_fit_history_plot_show/model_accuracy_comparison.png')
    plt.close()
    
    # 2. PCA dimensionality reduction visualization of test set
    # Standardize the test set
    scaler = StandardScaler()
    X_test_scaled = scaler.fit_transform(X_test)
    
    # PCA dimensionality reduction to 2D
    pca = PCA(n_components=2)
    X_test_pca = pca.fit_transform(X_test_scaled)
    
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, cmap='viridis', edgecolor='k', s=50)
    plt.colorbar(scatter, ticks=[0, 1, 2], label='Class')
    plt.title('Test Set Distribution after PCA Dimensionality Reduction')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    # 关键修改6：保存路径改为同级目录
    plt.savefig('./iris_fit_history_plot_show/pca_test_set.png')
    plt.close()
    
    # 3. Confusion matrix heatmaps
    plt.figure(figsize=(15, 10))
    for i, (name, result) in enumerate(results.items(), 1):
        plt.subplot(2, 3, i)
        sns.heatmap(result['cm'], annot=True, fmt='d', cmap='Blues', 
                    xticklabels=iris.target_names, yticklabels=iris.target_names)
        plt.title(f'{name} Confusion Matrix')
        plt.xlabel('Predicted Class')
        plt.ylabel('Actual Class')
    
    plt.tight_layout()
    # 关键修改7：保存路径改为同级目录
    plt.savefig('./iris_fit_history_plot_show/confusion_matrices.png')
    plt.close()


def main():
    """Main function to execute the complete machine learning workflow"""
    print("Iris Dataset Classification Analysis")
    print("="*50)
    
    # 1. Load and explore data
    df, iris = load_and_explore_data()
    
    # 2. Data visualization
    visualize_data(df, iris)
    
    # 3. Prepare training data
    X = df[iris.feature_names]
    y = df['target']
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print(f"\nTraining set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
    
    # Data standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Train models
    results, trained_models = train_models(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # 5. Result visualization
    visualize_results(results, X_test, y_test, iris)
    
    print("\n" + "="*50)
    print("Analysis completed!")
    print("Generated visualization files in ./iris_fit_history_plot_show/:")
    print("- boxplots.png: Feature box plots")
    print("- pairplot.png: Feature pair plots")
    print("- correlation_heatmap.png: Feature correlation heatmap")
    print("- histograms.png: Feature distribution histograms")
    print("- model_accuracy_comparison.png: Model accuracy comparison")
    print("- pca_test_set.png: PCA dimensionality reduction of test set")
    print("- confusion_matrices.png: Confusion matrices for all models")
    print("="*50)


if __name__ == "__main__":
    main()