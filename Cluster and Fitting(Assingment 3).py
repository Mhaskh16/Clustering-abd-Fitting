
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import plotly.express as px
from IPython.display import display, HTML

def error_prop(x, func, parameter, covar):
    var = np.zeros_like(x)
    for i in range(len(parameter)):
        deriv1 = deriv(x, func, parameter, i)
        for j in range(len(parameter)):
            deriv2 = deriv(x, func, parameter, j)
            var = var + deriv1 * deriv2 * covar[i, j]
    sigma = np.sqrt(var)
    return sigma

def deriv(x, func, parameter, ip):
    scale = 1e-6
    delta = np.zeros_like(parameter, dtype=float)
    val = scale * np.abs(parameter[ip])
    delta[ip] = val
    diff = 0.5 * (func(x, *parameter + delta) - func(x, *parameter - delta))
    dfdx = diff / val
    return dfdx

def covar_to_corr(covar):
    sigma = np.sqrt(np.diag(covar))
    matrix = np.outer(sigma, sigma)
    corr = covar / matrix
    return corr

def map_corr(df, size=6):
    corr = df.corr()
    plt.figure(figsize=(size, size))
    plt.matshow(corr, cmap='coolwarm', location="bottom")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.colorbar()

def scaler(df):
    df_min = df.min()
    df_max = df.max()
    df_normalized = (df - df_min) / (df_max - df_min)
    return df_normalized, df_min, df_max

def backscale(arr, df_min, df_max):
    minima = df_min.to_numpy()
    maxima = df_max.to_numpy()
    for i in range(len(minima)):
        arr[:, i] = arr[:, i] * (maxima[i] - minima[i]) + minima[i]
    return arr

def get_diff_entries(df1, df2, column):
    df_out = pd.merge(df1, df2, on=column, how="outer")
    df_in = pd.merge(df1, df2, on=column, how="inner")
    df_in["exists"] = "Y"
    df_merge = pd.merge(df_out, df_in, on=column, how="outer")
    df_diff = df_merge[(df_merge["exists"] != "Y")]
    diff_list = df_diff[column].to_list()
    return diff_list

def plot_data(data, ylabel, title, save_path):
    data_df = data.transpose()
    data_df.columns = data_df.iloc[0]
    data_df = data_df.iloc[2:]

    # Plot for each country
    fig, ax = plt.subplots()
    for country in data_df.columns:
        color = "#" + ''.join([random.choice('0123456789ABCDEF') for _ in range(6)])
        ax.plot(data_df.index, data_df[country], color=color)

    # Customize the plot
    ax.legend(list(data_df.columns))
    ax.set_xlabel('Year', fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_title(title, fontsize=16, color='g')
    x = data_df.index
    ax.set_xticks(x[::5])
    ax.set_xticklabels(x[::5], rotation=25)
    current_values = ax.get_yticks()
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))

    # Save the figure
    fig.savefig(save_path)
    plt.close(fig)

# Read data from Excel file
all_data = pd.read_excel(r'API_Download_DS2_en_excel_v2_6240987.xls', sheet_name="Data")
all_data.drop(["Country Code", "Indicator Code"], axis=1, inplace=True)

# Choose a set of indicators for the scatter plot
scatter_indicators = [
    "Intentional homicides, female (per 100,000 female)",
    "Battle-related deaths (number of people)",
    "Voice and Accountability: Percentile Rank",
    "Transport services (% of commercial service exports)"
]

# Filter data for selected indicators
scatter_data = all_data[all_data['Indicator Name'].isin(scatter_indicators)]

# Reshape data for Plotly scatter plot
scatter_data_long = pd.melt(scatter_data, id_vars=["Country Name", "Indicator Name"],
                            var_name="Year", value_name="Value")

# Convert the 'Year' column to numeric (remove non-numeric entries)
scatter_data_long["Year"] = pd.to_numeric(scatter_data_long["Year"], errors="coerce")

# Remove rows with NaN values in any column
scatter_data_long = scatter_data_long.dropna()

# Create an interactive scatter plot using Plotly Express
scatter_fig = px.scatter(scatter_data_long, x="Year", y="Value", color="Country Name",
                         size="Value", facet_col="Indicator Name",
                         labels={"Value": "Indicator Value"},
                         title="Interactive Scatter Plot",
                         template="plotly_dark")

# Save the scatter plot
scatter_fig.write_html('scatter_plot.html')

# Display the scatter plot
display(scatter_fig)

# Simplified Scatter plot for a specific pair of indicators
indicator1 = "Imports of goods and services (constant 2015 US$)"
indicator2 = "CO2 emissions from electricity and heat production, total (% of total fuel combustion)"

scatter_ax = plt.figure().add_subplot(111)
scatter_ax.scatter(all_data.loc[all_data['Indicator Name'] == indicator1].iloc[2:].transpose(),
                   all_data.loc[all_data['Indicator Name'] == indicator2].iloc[2:].transpose())
scatter_ax.set_xlabel(indicator1)
scatter_ax.set_ylabel(indicator2)
scatter_ax.set_title(f'Scatter Plot: {indicator1} vs {indicator2}')

# Save the scatter plot
# scatter_ax.figure.savefig('scatter_plot.png')
# plt.close(scatter_ax.figure)

# Simplified Heatmap
corr = all_data.corr()
heatmap_ax = plt.figure().add_subplot(111)
sns.heatmap(corr, cmap='coolwarm', annot=True, fmt=".2f", linewidths=.5, ax=heatmap_ax)
heatmap_ax.set_title('Correlation Heatmap')

# Save the heatmap
heatmap_ax.figure.savefig('correlation_heatmap.png')
plt.close(heatmap_ax.figure)

# Plot for each indicator
indicators = [
    "Imports of goods and services (constant 2015 US$)",
    "CO2 emissions from electricity and heat production, total (% of total fuel combustion)",
    "CO2 emissions from residential buildings and commercial and public services (% of total fuel combustion)",
    "Manufactures exports (% of merchandise exports)"
]

filtered_data = [all_data.loc[all_data['Indicator Name'] == indicator].fillna(0) for indicator in indicators]

# Plot for each indicator
for i, data in enumerate(filtered_data, start=1):
    indicator_name = indicators[i-1]
    ylabel = ' '.join([word.capitalize() for word in indicator_name.split()[:-1]])
    title = f'{ylabel} Comparison by Country'

    # Save the plot for each indicator
    plot_data(data, ylabel, title, f'plot_{indicator_name}.png')

# Display all plots
HTML("<style>.output_png{display: flex; flex-wrap: wrap;}</style>")
