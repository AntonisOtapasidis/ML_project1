#!/usr/bin/env python
# coding: utf-8

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import svd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load data set and check its structure and column names
cancer_data = pd.read_csv('data.csv')
cancer_data.shape
cancer_data.columns

# Check for missing values
print(cancer_data.isnull().sum())

# Drop empty column
cancer_data = cancer_data.drop(columns= ['Unnamed: 32'])
print(cancer_data)

# Check if attributes are of the correct data type
print(cancer_data.info())

# Summary statistics
print(cancer_data.iloc[:,1:].describe())

# Create violinplots to check distribution
columns_of_interest = ['radius', 'texture', 'perimeter', 'area', 'smoothness', 
                       'compactness', 'concavity', 'concave points', 'symmetry', 
                       'fractal_dimension']
# Create side-by-side violin plots
fig, axes = plt.subplots(nrows=10, ncols=1, figsize=(20, 40))  # Increase figsize height for better visualization

for i, column in enumerate(columns_of_interest):
    # Select columns for '_mean', '_se', and '_worst'
    columns_to_plot = [f"{column}_mean", f"{column}_se", f"{column}_worst"]
    # Extract data for each column to plot
    data = [cancer_data[col] for col in columns_to_plot]
    # Plot the violin plots with adjusted width
    axes[i].violinplot(data, widths=0.8)  # Adjust width as needed
    # Set x-axis labels
    axes[i].set_xticks(np.arange(1, len(columns_to_plot) + 1))
    axes[i].set_xticklabels(['Mean', 'SE', 'Worst'])
    # Set plot title
    axes[i].set_title(column)
    # Set plot labels
    axes[i].set_xlabel('Variables')
    axes[i].set_ylabel('Values')

plt.tight_layout()
plt.show()

#plt.savefig("violin_plot.png")

# Create boxplots to check distribution
columns_of_interest = ['radius', 'texture', 'perimeter', 'area', 'smoothness', 
                       'compactness', 'concavity', 'concave points', 'symmetry', 
                       'fractal_dimension']
# Create side-by-side boxplots
fig, axes = plt.subplots(nrows=10, ncols=1, figsize=(20, 15))
for i, column in enumerate(columns_of_interest):
    # Select columns for '_mean', '_se', and '_worst'
    columns_to_plot = [f"{column}_mean", f"{column}_se", f"{column}_worst"]    
    # Plot the boxplots
    cancer_data[columns_to_plot].boxplot(ax=axes[i], vert=False)
    # Set plot title
    axes[i].set_title(column)
    # Set plot labels
    axes[i].set_xlabel('Values')
    axes[i].set_ylabel('Variables')
    # Add legend
    axes[i].legend(['Mean', 'SE', 'Worst'], loc='upper right')
plt.tight_layout()
plt.show()
#plt.savefig("boxplot.png")

# Create histograms to check distribution
num_rows = len(columns_of_interest) // 2 + len(columns_of_interest) % 2
# Create a figure and axes for subplots
fig, axes = plt.subplots(nrows=num_rows, ncols=2, figsize=(15, 12))
for i, column in enumerate(columns_of_interest):
    # Calculate the row and column indices for the current subplot
    row_index = i // 2
    col_index = i % 2
    # Select columns for '_mean', '_se', and '_worst'
    columns_to_plot = [f"{column}_mean", f"{column}_se", f"{column}_worst"]
    # Plot histograms for each column in the current row
    for j, col in enumerate(columns_to_plot):
        axes[row_index, col_index].hist(cancer_data[col], bins=30, alpha=0.5, label=col)
       # axes[row_index, col_index].set_title(column)  # Set title for the subplot
        axes[row_index, col_index].set_xlabel(column)  # Set x-axis label as the column name
        axes[row_index, col_index].set_ylabel('Counts')  # Set y-axis label
        axes[row_index, col_index].legend()  # Show legend
plt.tight_layout()
plt.show()
#plt.savefig('subplots_figure.png')

# Further on, we proceed only with the mean group of the attributes
new_cancer_data = cancer_data.iloc[:, :12] # Also exclude the id attribute
print(new_cancer_data)

# Exclude the "_mean" string from attributes names since the other groups were dropped
for col in new_cancer_data.columns:
    # Check if column name contains '_mean'
    if '_mean' in col:
        # Replace '_mean' with an empty string
        new_col_name = col.replace('_mean', '')
        # Rename the column
        new_cancer_data.rename(columns={col: new_col_name}, inplace=True)

# Create a checkpoint
new_cancer_data_1 = new_cancer_data.copy()

# Create a barplot of diagnosis attribute
diagnosis_counts = new_cancer_data_1['diagnosis'].value_counts()
colors = {'B': 'skyblue', 'M': 'salmon'}
barplot_data = pd.DataFrame({'diagnosis': diagnosis_counts.index, 'count': diagnosis_counts.values})
sns.barplot(data=barplot_data, x='diagnosis', y='count', palette=colors)
plt.title('Diagnosis Distribution')
plt.ylabel('Count')
plt.xlabel('Diagnosis')
plt.show()
#plt.savefig("diagnosis_barplot.png")

# Create histograms to check distribution stratified on diagnosis
numerical_vars = ['radius', 'texture', 'perimeter', 'area', 'smoothness', 'compactness', 'concavity',
       'concave points', 'symmetry', 'fractal_dimension']
n_vars = len(numerical_vars)
# Calculate the number of rows and columns based on the number of variables
num_rows = (n_vars - 1) // 3 + 1
num_cols = min(n_vars, 3)
fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(20, 15))
# Flatten the axes for easy iteration
axes = axes.flatten()
# Plot histograms for each variable
for i, var in enumerate(numerical_vars):
    sns.histplot(data=new_cancer_data_1, x=var, hue='diagnosis', kde=True, ax=axes[i])
    axes[i].set_title(var)  # Set title for each subplot
# Remove empty subplots
for j in range(n_vars, num_rows * num_cols):
    fig.delaxes(axes[j])
# Adjust layout and show plot
plt.tight_layout()
plt.show()
#plt.savefig("histogram_diagnosis.png")

# Identify the outliers
# Calculate the number of rows and columns based on the number of variables
num_rows = (n_vars - 1) // 3 + 1
num_cols = min(n_vars, 3)

fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(20, 15))

# Flatten the axes for easy iteration
axes = axes.flatten()

# Plot bar plots for each variable
for i, var in enumerate(numerical_vars):
    if i >= len(axes):  # Check if all axes have been used
        break
    sns.boxplot(data=new_cancer_data_1, x='diagnosis', y=var, ax=axes[i])
    axes[i].set_xlabel('Diagnosis')  # Set x-axis label

# Hide unused subplots
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()



#Encode diagnosis attribute to a binary to further proceed with our analysis
new_cancer_diagnosis = {'B': 0, 'M': 1}
new_cancer_data_1['diagnosis'] = new_cancer_data_1['diagnosis'].map(new_cancer_diagnosis)
print(new_cancer_data_1)

# Create pairplots of all the numeric attributes stratified on diganosis
cancer_pairplot = sns.pairplot(new_cancer_data_1.iloc[:,1:], hue='diagnosis')
cancer_pairplot._legend.set_title('Diagnosis')
cancer_pairplot._legend.set_bbox_to_anchor((1.1, 0.5))  # Adjust the position of the legend
# Rename the labels
new_labels = {'0': 'Benign', '1': 'Malignant'}
for t, l in zip(cancer_pairplot._legend.texts, new_labels.values()):
    t.set_text(l)
    t.set_fontsize(12)
plt.show()
#plt.savefig("cancer_pairplot.png")

# Create and plot correlation matrix
correlation_matrix = new_cancer_data_1.iloc[:,2:].corr()
plt.figure(figsize=(10, 8))
# Generate a heatmap 
sns.heatmap(correlation_matrix, annot=True, cmap='viridis', fmt=".0%", linewidths=0.5)
plt.title('Correlation Matrix of Numerical Variables')
# Rotate y-axis labels for better readability
plt.yticks(rotation=0)
#plt.show()
plt.savefig("correlation_matrix.png")

# Data manipulation before performing PCA
# Exclusion of the ID (not to be used here) and the diagnosis (attribute to be predicted) 
X = new_cancer_data_1.iloc[:,2:] 
y = new_cancer_data_1['diagnosis'] # The attribute we want to predict
# Data standardization
scaler = StandardScaler() # Subtract the mean and divide with the standard deviation
scaled_data = scaler.fit_transform(X) # Transform the data for the PCA

# Performing the PCA
pca = PCA() # The PCA function performs Singular Value Decomposition (SVD) internally to perform the dimensionality reduction
pca_data = pca.fit_transform(scaled_data)
print(pca_data)
# Plot the variance explained by each principal component
plt.figure(figsize=(8, 6))
plt.bar(range(1, pca.n_components_ + 1), pca.explained_variance_ratio_ * 100, alpha=0.7, align='center')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained (%)')
plt.title('Variance Explained by Principal Components')
plt.xticks(range(1, pca.n_components_ + 1))
plt.grid(True)
plt.show()
#plt.savefig("variance_explained.png")

# Plot cumulative variance explained by principal components
threshold = 0.9
plt.figure()
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, "x-")
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), np.cumsum(pca.explained_variance_ratio_), "o-")
plt.plot([1, len(pca.explained_variance_ratio_)], [threshold, threshold], "k--")
plt.title("Variance explained by principal components")
plt.xlabel("Principal component")
plt.ylabel("Variance explained (%)")
plt.legend(["Individual", "Cumulative", "Threshold"])
plt.grid()
plt.show()
#plt.savefig("cumulative_variance.png")

# PCA plot  
plt.figure(figsize=(8, 6))
sns.scatterplot(x=pca_data[:, 0], y=pca_data[:, 1], hue=y, palette=['salmon', 'skyblue'], alpha=0.7)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA Plot')
plt.legend(title='Diagnosis')
plt.grid(True)
plt.show()
#plt.savefig("PCA_plot.png")

# Get information about the principal coefficients
pc_df = pd.DataFrame(data=pca_data)
# Display the first few rows of the dataframe
print(pc_df.head())

# Plot the principal directions of the first three principal components
pcs = [0, 1, 2]
legendStrs = ["PC" + str(e + 1) for e in pcs]
c = ["r", "g", "b"]
bw = 0.2
# Determine the number of attributes directly from the shape of the principal components
num_attributes = pca.components_.shape[1]
r = np.arange(1, num_attributes + 1)
# The actual plot
for i, pc_index in enumerate(pcs):
    plt.bar(r + i * bw, pca.components_[pc_index], width=bw, color=c[i], label=legendStrs[i])
plt.xlabel("Attributes")
plt.ylabel("Component coefficients")
plt.legend()
plt.grid()
plt.title("PCA Component Coefficients")
plt.show()
#plt.savefig("projections_first_three.png")

# Plot the covariance matrix (Σ)
cov_matrix = np.cov(scaled_data, rowvar=False)
plt.figure(figsize=(10, 8))
sns.heatmap(cov_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Covariance Matrix")
plt.show()
#plt.savefig("covariance_matrix.png")
