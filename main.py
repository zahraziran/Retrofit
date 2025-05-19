import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
import matplotlib.cm as cm
from sklearn.inspection import permutation_importance
import os

# Set style for plots
plt.style.use('ggplot')
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Create output directory for plots
if not os.path.exists('plots'):
    os.makedirs('plots')


# Function to load data
def load_data(file_path):
    """Load data from Excel file"""
    # Read the building characteristics sheet
    building_data = pd.read_excel(file_path, sheet_name="Foglio1")

    # Read monthly consumption data
    monthly_total = pd.read_excel(file_path, sheet_name="Monthly (TOTAL) (1)")
    monthly_heating = pd.read_excel(file_path, sheet_name="Monthly (HEATING) (2)")
    monthly_dhw = pd.read_excel(file_path, sheet_name="Monthly (DHW) (3)")
    monthly_cooling = pd.read_excel(file_path, sheet_name="Monthly (COOLING) (4)")

    # Clean data
    # 1. Remove any rows with all NaN values
    building_data = building_data.dropna(how='all')

    # 2. Fill missing numeric values with column means
    numeric_cols = building_data.select_dtypes(include=['float64', 'int64']).columns
    building_data[numeric_cols] = building_data[numeric_cols].fillna(building_data[numeric_cols].mean())

    # 3. Fill missing categorical values with the most common value
    categorical_cols = building_data.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        building_data[col] = building_data[col].fillna(building_data[col].mode()[0])

    return building_data, monthly_total, monthly_heating, monthly_dhw, monthly_cooling


# Phase 1: Data Exploration and Visualization
def data_exploration(building_data, monthly_total):
    """Explore and visualize the building characteristics and energy consumption data"""
    print("Phase 1: Data Exploration")
    print("-" * 50)

    # Basic data overview
    print(f"Number of buildings: {building_data.shape[0]}")
    print(f"Number of features: {building_data.shape[1]}")
    print("\nBuilding data columns:")
    print(building_data.columns.tolist())

    # Plot distribution of energy consumption
    plt.figure(figsize=(12, 6))
    sns.histplot(building_data["Total consumption [kWh/m2/ year]"], kde=True)
    plt.title("Distribution of Total Energy Consumption")
    plt.xlabel("Energy Consumption (kWh/m²/year)")
    plt.ylabel("Frequency")
    plt.savefig('plots/energy_consumption_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot energy consumption components
    consumption_cols = [
        "Heating consumption [kWh/m2/ year]",
        "DHW consumption [kWh/m2/ year]",
        "Cooling consumption [kWh/m2/ year]"
    ]

    # Calculate percentage of each component
    consumption_data = building_data[consumption_cols].copy()
    consumption_pct = consumption_data.div(consumption_data.sum(axis=1), axis=0) * 100

    # Plot average percentage of each component
    plt.figure(figsize=(10, 6))
    average_pct = consumption_pct.mean()
    plt.pie(average_pct, labels=["Heating", "DHW", "Cooling"],
            autopct='%1.1f%%', startangle=90, colors=sns.color_palette("Set2"))
    plt.title("Average Distribution of Energy Consumption Components")
    plt.savefig('plots/energy_consumption_components.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Visualize monthly patterns
    monthly_cols = [col for col in monthly_total.columns if 'kWh/m2' in str(col)]
    monthly_means = monthly_total[monthly_cols].mean()

    # Create a more readable index for the plot
    month_labels = [col.split('[')[0].strip() for col in monthly_cols]

    plt.figure(figsize=(12, 6))
    monthly_means.plot(kind='bar', color='skyblue')
    plt.title("Average Monthly Energy Consumption")
    plt.xlabel("Month")
    plt.ylabel("Energy Consumption (kWh/m²)")
    plt.xticks(range(len(month_labels)), month_labels, rotation=45)
    plt.savefig('plots/monthly_consumption_pattern.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Explore correlation between building characteristics and energy consumption
    # Select numerical columns for correlation analysis
    numeric_cols = building_data.select_dtypes(include=['float64', 'int64']).columns

    # Compute correlation matrix
    corr_matrix = building_data[numeric_cols].corr()

    # Plot correlation heatmap
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Correlation Matrix of Building Characteristics and Energy Consumption")
    plt.tight_layout()
    plt.savefig('plots/correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Explore relationship between key building characteristics and energy consumption
    key_features = [
        "Wall heat transfer coefficient [W/m2k]",
        "Glass heat transfer coefficient",
        "Glass surface [%]",
        "Surface / Volume ratio",
        "Year of construction"
    ]

    # Plot relationship between key features and energy consumption
    for feature in key_features:
        if feature == "Year of construction":
            # For categorical feature, use boxplot
            plt.figure(figsize=(14, 6))
            sns.boxplot(x=feature, y="Total consumption [kWh/m2/ year]", data=building_data)
            plt.title(f"Energy Consumption by {feature}")
            plt.xlabel(feature)
            plt.ylabel("Energy Consumption (kWh/m²/year)")
            plt.xticks(rotation=45)
        else:
            # For numerical features, use scatterplot
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=feature, y="Total consumption [kWh/m2/ year]",
                            hue="Year of construction", data=building_data)
            plt.title(f"Energy Consumption vs {feature}")
            plt.xlabel(feature)
            plt.ylabel("Energy Consumption (kWh/m²/year)")

        plt.tight_layout()
        plt.savefig(f'plots/energy_vs_{feature.split("[")[0].strip().replace(" ", "_").replace("/", "_")}.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    return building_data


# Phase 2: Feature Selection and Machine Learning Analysis
def machine_learning_analysis(building_data):
    """Use machine learning algorithms to identify important factors affecting energy consumption"""
    print("\nPhase 2: Machine Learning Analysis")
    print("-" * 50)

    # Prepare data for machine learning
    # Select relevant features
    features = [
        "Floors", "n° of dwellings", "Average size of dwellings [m2]",
        "Surface [m2]", "Total occupancy", "Wall heat transfer coefficient [W/m2k]",
        "Glass heat transfer coefficient", "Glass surface [%]",
        "Floor spacing [m]", "Surface / Volume ratio"
    ]

    # Create a copy of the dataframe with just the selected features
    X = building_data[features].copy()
    y = building_data["Total consumption [kWh/m2/ year]"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest Regressor to identify feature importance
    print("Training Random Forest model to identify important features...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # Make predictions
    y_pred = rf.predict(X_test)

    # Calculate performance metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Model performance:")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R² Score: {r2:.2f}")

    # Feature importance analysis
    # Use permutation importance which is more reliable than the built-in feature importance
    perm_importance = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42)

    # Create dataframe of feature importance
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': perm_importance.importances_mean
    }).sort_values(by='Importance', ascending=False)

    print("\nTop 5 important features:")
    print(importance_df.head(5))

    # Visualize feature importance
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title('Feature Importance for Energy Consumption')
    plt.tight_layout()
    plt.savefig('plots/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()

    return importance_df, rf


# Phase 3: Building Clustering Analysis
def building_clustering(building_data):
    """Cluster buildings based on energy consumption and characteristics"""
    print("\nPhase 3: Building Clustering Analysis")
    print("-" * 50)

    # Select features for clustering
    clustering_features = [
        "Total consumption [kWh/m2/ year]",
        "Heating consumption [kWh/m2/ year]",
        "DHW consumption [kWh/m2/ year]",
        "Cooling consumption [kWh/m2/ year]",
        "Wall heat transfer coefficient [W/m2k]",
        "Glass heat transfer coefficient",
        "Surface / Volume ratio"
    ]

    # Create a copy of the dataframe with just the selected features
    X_cluster = building_data[clustering_features].copy()

    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)

    # Determine optimal number of clusters using elbow method
    inertia = []
    k_range = range(2, 10)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)

    # Plot elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertia, 'o-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True)
    plt.savefig('plots/elbow_method.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Based on elbow curve, select number of clusters
    # This will be determined programmatically, but for now let's use 3 clusters
    # In a real scenario, you would want to analyze the elbow curve visually and select k
    n_clusters = 3

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    building_data['cluster'] = kmeans.fit_predict(X_scaled)

    # Analyze clusters
    cluster_summary = building_data.groupby('cluster')[clustering_features].mean()
    print("\nCluster centers (average values):")
    print(cluster_summary)

    # Visualize clusters using PCA for dimensionality reduction
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Create a dataframe with PCA results and cluster labels
    pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
    pca_df['cluster'] = building_data['cluster']

    # Plot clusters
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='PC1', y='PC2', hue='cluster', data=pca_df, palette='viridis', s=100)
    plt.title('Building Clusters Based on Energy Characteristics')
    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.savefig('plots/building_clusters.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Visualize characteristic differences between clusters
    for feature in clustering_features:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='cluster', y=feature, data=building_data)
        plt.title(f'{feature} by Cluster')
        plt.xlabel('Cluster')
        plt.ylabel(feature)
        plt.savefig(f'plots/cluster_{feature.split("[")[0].strip().replace(" ", "_").replace("/", "_")}.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    # Assign cluster labels based on energy consumption
    cluster_total_energy = cluster_summary['Total consumption [kWh/m2/ year]']
    cluster_labels = {
        cluster_total_energy.idxmax(): 'High Consumption',
        cluster_total_energy.idxmin(): 'Low Consumption'
    }

    # The middle cluster is moderate consumption
    for i in range(n_clusters):
        if i not in cluster_labels:
            cluster_labels[i] = 'Moderate Consumption'

    # Add cluster labels to the dataframe
    building_data['cluster_label'] = building_data['cluster'].map(cluster_labels)

    print("\nCluster labels based on energy consumption:")
    print(building_data[['cluster', 'cluster_label']].value_counts())

    return building_data, cluster_summary, cluster_labels


# Phase 4: Cost-Benefit Analysis for Retrofit Measures
def cost_benefit_analysis(building_data, importance_df, cluster_summary):
    """Perform cost-benefit analysis for different retrofit measures"""
    print("\nPhase 4: Cost-Benefit Analysis for Retrofit Measures")
    print("-" * 50)

    # Define retrofit measures and their costs
    # Note: These are example costs and should be replaced with real data
    retrofit_measures = {
        'Wall insulation': {
            'target_feature': 'Wall heat transfer coefficient [W/m2k]',
            'improvement': 0.4,  # Reduction in W/m²K
            'cost_per_m2': 60,  # € per m²
            'lifespan': 30,  # years
            'expected_energy_reduction': 0.15  # 15% reduction in total energy
        },
        'Window replacement': {
            'target_feature': 'Glass heat transfer coefficient',
            'improvement': 3.0,  # Reduction in W/m²K
            'cost_per_m2': 350,  # € per m²
            'lifespan': 25,  # years
            'expected_energy_reduction': 0.12  # 12% reduction
        },
        'HVAC system upgrade': {
            'target_feature': None,  # Affects overall efficiency
            'improvement': None,
            'cost_per_unit': 8000,  # € per unit
            'cost_per_m2': None,
            'lifespan': 20,  # years
            'expected_energy_reduction': 0.20  # 20% reduction
        },
        'Solar panels': {
            'target_feature': None,
            'improvement': None,
            'cost_per_kW': 1500,  # € per kW
            'energy_production': 1200,  # kWh per kW per year
            'lifespan': 25,  # years
            'expected_energy_reduction': 0.15  # 15% reduction via offset
        }
    }

    # Energy price assumption (€ per kWh)
    energy_price = 0.25
    energy_price_increase = 0.03  # 3% annual increase

    # Calculate retrofit costs and benefits for each cluster
    results = []

    for cluster_id, label in enumerate(cluster_summary.index):
        cluster_buildings = building_data[building_data['cluster'] == cluster_id]
        avg_consumption = cluster_summary.loc[label, 'Total consumption [kWh/m2/ year]']
        avg_surface = cluster_buildings['Surface [m2]'].mean()
        avg_glass_surface = avg_surface * (cluster_buildings['Glass surface [%]'].mean() / 100)

        for measure, details in retrofit_measures.items():
            # Calculate costs
            if measure == 'Wall insulation':
                cost = details['cost_per_m2'] * avg_surface
            elif measure == 'Window replacement':
                cost = details['cost_per_m2'] * avg_glass_surface
            elif measure == 'HVAC system upgrade':
                avg_units = cluster_buildings['n° of dwellings'].mean()
                cost = details['cost_per_unit'] * avg_units
            elif measure == 'Solar panels':
                # Assume solar system sized to cover 30% of average consumption
                required_kW = (avg_consumption * avg_surface * 0.3) / details['energy_production']
                cost = details['cost_per_kW'] * required_kW

            # Calculate annual savings
            annual_saving_kwh = avg_consumption * avg_surface * details['expected_energy_reduction']
            annual_saving_eur = annual_saving_kwh * energy_price

            # Calculate simple payback time
            simple_payback = cost / annual_saving_eur

            # Calculate NPV (Net Present Value)
            discount_rate = 0.04  # 4% discount rate
            npv = -cost
            for year in range(1, details['lifespan'] + 1):
                # Energy price increases each year
                year_saving = annual_saving_kwh * energy_price * (1 + energy_price_increase) ** (year - 1)
                npv += year_saving / (1 + discount_rate) ** year

            # Calculate ROI
            roi = (npv / cost) * 100

            results.append({
                'Cluster': cluster_id,
                'Cluster Label': building_data[building_data['cluster'] == cluster_id]['cluster_label'].iloc[0],
                'Measure': measure,
                'Cost (€)': cost,
                'Annual Savings (kWh)': annual_saving_kwh,
                'Annual Savings (€)': annual_saving_eur,
                'Payback Period (years)': simple_payback,
                'NPV (€)': npv,
                'ROI (%)': roi,
                'Lifespan (years)': details['lifespan']
            })

    # Create dataframe with results
    results_df = pd.DataFrame(results)

    # Print summary results
    print("\nCost-benefit analysis summary:")
    print(results_df[['Cluster Label', 'Measure', 'Cost (€)', 'Annual Savings (€)',
                      'Payback Period (years)', 'ROI (%)']].groupby(['Cluster Label', 'Measure']).mean())

    # Visualize payback periods by measure and cluster
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Measure', y='Payback Period (years)', hue='Cluster Label', data=results_df)
    plt.title('Payback Period by Retrofit Measure and Building Cluster')
    plt.xlabel('Retrofit Measure')
    plt.ylabel('Payback Period (years)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('plots/payback_periods.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Visualize ROI by measure and cluster
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Measure', y='ROI (%)', hue='Cluster Label', data=results_df)
    plt.title('ROI by Retrofit Measure and Building Cluster')
    plt.xlabel('Retrofit Measure')
    plt.ylabel('ROI (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('plots/roi_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Analyze combined measures (packages of interventions)
    # For simplicity, we'll define some common packages
    retrofit_packages = {
        'Basic package': ['Wall insulation', 'Window replacement'],
        'Advanced package': ['Wall insulation', 'Window replacement', 'HVAC system upgrade'],
        'Comprehensive package': ['Wall insulation', 'Window replacement', 'HVAC system upgrade', 'Solar panels']
    }

    # Calculate the effect of combining measures
    package_results = []

    for cluster_id, label in enumerate(cluster_summary.index):
        cluster_buildings = building_data[building_data['cluster'] == cluster_id]
        avg_consumption = cluster_summary.loc[label, 'Total consumption [kWh/m2/ year]']
        avg_surface = cluster_buildings['Surface [m2]'].mean()

        for package_name, measures in retrofit_packages.items():
            # Calculate total cost and energy reduction
            total_cost = 0
            total_energy_reduction = 0

            # Assume diminishing returns for combined measures
            # Each subsequent measure is 80% as effective as it would be alone
            efficiency_factor = 1.0

            for measure in measures:
                details = retrofit_measures[measure]

                # Calculate costs for each measure in the package
                if measure == 'Wall insulation':
                    cost = details['cost_per_m2'] * avg_surface
                elif measure == 'Window replacement':
                    avg_glass_surface = avg_surface * (cluster_buildings['Glass surface [%]'].mean() / 100)
                    cost = details['cost_per_m2'] * avg_glass_surface
                elif measure == 'HVAC system upgrade':
                    avg_units = cluster_buildings['n° of dwellings'].mean()
                    cost = details['cost_per_unit'] * avg_units
                elif measure == 'Solar panels':
                    required_kW = (avg_consumption * avg_surface * 0.3) / details['energy_production']
                    cost = details['cost_per_kW'] * required_kW

                total_cost += cost

                # Apply diminishing returns to energy reduction
                adjusted_reduction = details['expected_energy_reduction'] * efficiency_factor
                total_energy_reduction += adjusted_reduction

                # Reduce efficiency for next measure
                efficiency_factor *= 0.8

            # Calculate savings and payback for package
            annual_saving_kwh = avg_consumption * avg_surface * total_energy_reduction
            annual_saving_eur = annual_saving_kwh * energy_price
            simple_payback = total_cost / annual_saving_eur

            # Calculate NPV for package (using weighted average lifespan)
            weighted_lifespan = 0
            for measure in measures:
                weighted_lifespan += retrofit_measures[measure]['lifespan'] / len(measures)

            npv = -total_cost
            for year in range(1, int(weighted_lifespan) + 1):
                year_saving = annual_saving_kwh * energy_price * (1 + energy_price_increase) ** (year - 1)
                npv += year_saving / (1 + discount_rate) ** year

            # Calculate ROI
            roi = (npv / total_cost) * 100

            package_results.append({
                'Cluster': cluster_id,
                'Cluster Label': building_data[building_data['cluster'] == cluster_id]['cluster_label'].iloc[0],
                'Package': package_name,
                'Measures': ', '.join(measures),
                'Total Cost (€)': total_cost,
                'Annual Savings (kWh)': annual_saving_kwh,
                'Annual Savings (€)': annual_saving_eur,
                'Payback Period (years)': simple_payback,
                'NPV (€)': npv,
                'ROI (%)': roi,
                'Weighted Lifespan (years)': weighted_lifespan
            })

    # Create dataframe with package results
    package_df = pd.DataFrame(package_results)

    # Print package results
    print("\nRetrofit package analysis:")
    print(package_df[['Cluster Label', 'Package', 'Total Cost (€)',
                      'Annual Savings (€)', 'Payback Period (years)', 'ROI (%)']].groupby(
        ['Cluster Label', 'Package']).mean())

    # Visualize package comparison
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Package', y='Payback Period (years)', hue='Cluster Label', data=package_df)
    plt.title('Payback Period by Retrofit Package and Building Cluster')
    plt.xlabel('Retrofit Package')
    plt.ylabel('Payback Period (years)')
    plt.tight_layout()
    plt.savefig('plots/package_payback_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Compare ROI of packages
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Package', y='ROI (%)', hue='Cluster Label', data=package_df)
    plt.title('ROI by Retrofit Package and Building Cluster')
    plt.xlabel('Retrofit Package')
    plt.ylabel('ROI (%)')
    plt.tight_layout()
    plt.savefig('plots/package_roi_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    return results_df, package_df


# Phase 5: Decision Support Tool Development
def decision_support_tool(building_data, importance_df, results_df, package_df, rf_model):
    """Create a decision support tool to recommend optimal retrofit strategies"""
    print("\nPhase 5: Decision Support Tool Development")
    print("-" * 50)

    # Create a function to recommend retrofit measures for a specific building
    def recommend_retrofit(building_id, budget_constraint=None):
        """
        Recommend retrofit measures for a specific building based on characteristics and budget

        Parameters:
        -----------
        building_id : str
            The identifier of the building
        budget_constraint : float, optional
            Maximum budget in euros

        Returns:
        --------
        dict
            Recommendations and analysis
        """
        # Get building data
        building = building_data[building_data['Building'] == building_id].iloc[0]
        cluster = building['cluster']
        cluster_label = building['cluster_label']

        print(f"Analysis for building: {building_id}")
        print(f"Cluster: {cluster_label}")
        print(f"Current energy consumption: {building['Total consumption [kWh/m2/ year]']:.2f} kWh/m²/year")

        # Get retrofit measures for this cluster
        cluster_measures = results_df[results_df['Cluster'] == cluster].copy()

        # If budget constraint exists, filter measures
        if budget_constraint:
            affordable_measures = cluster_measures[cluster_measures['Cost (€)'] <= budget_constraint]
            print(f"\nMeasures within budget constraint of {budget_constraint} €:")
            if affordable_measures.empty:
                print("No individual measures are within the specified budget.")
            else:
                for _, measure in affordable_measures.iterrows():
                    print(f"- {measure['Measure']}: {measure['Cost (€)']:.2f} €, "
                          f"ROI: {measure['ROI (%)']:.2f}%, Payback: {measure['Payback Period (years)']:.2f} years")

            # Check packages within budget
            cluster_packages = package_df[package_df['Cluster'] == cluster].copy()
            affordable_packages = cluster_packages[cluster_packages['Total Cost (€)'] <= budget_constraint]
            print(f"\nPackages within budget constraint of {budget_constraint} €:")
            if affordable_packages.empty:
                print("No retrofit packages are within the specified budget.")
            else:
                for _, package in affordable_packages.iterrows():
                    print(f"- {package['Package']}: {package['Total Cost (€)']:.2f} €, "
                          f"ROI: {package['ROI (%)']:.2f}%, Payback: {package['Payback Period (years)']:.2f} years")

        # Get best measures by ROI
        best_roi_measure = cluster_measures.loc[cluster_measures['ROI (%)'].idxmax()]
        print(f"\nBest measure by ROI: {best_roi_measure['Measure']}")
        print(f"ROI: {best_roi_measure['ROI (%)']:.2f}%")
        print(f"Cost: {best_roi_measure['Cost (€)']:.2f} €")
        print(f"Payback period: {best_roi_measure['Payback Period (years)']:.2f} years")

        # Get best measure by payback period
        best_payback_measure = cluster_measures.loc[cluster_measures['Payback Period (years)'].idxmin()]
        print(f"\nBest measure by payback period: {best_payback_measure['Measure']}")
        print(f"Payback period: {best_payback_measure['Payback Period (years)']:.2f} years")
        print(f"Cost: {best_payback_measure['Cost (€)']:.2f} €")
        print(f"ROI: {best_payback_measure['ROI (%)']:.2f}%")

        # Get best package
        cluster_packages = package_df[package_df['Cluster'] == cluster].copy()
        if not cluster_packages.empty:
            best_roi_package = cluster_packages.loc[cluster_packages['ROI (%)'].idxmax()]
            print(f"\nBest package by ROI: {best_roi_package['Package']}")
            print(f"ROI: {best_roi_package['ROI (%)']:.2f}%")
            print(f"Cost: {best_roi_package['Total Cost (€)']:.2f} €")
            print(f"Payback period: {best_roi_package['Payback Period (years)']:.2f} years")
            print(f"Includes: {best_roi_package['Measures']}")

        # Create recommendations dictionary
        recommendations = {
            'building_id': building_id,
            'cluster': cluster_label,
            'current_consumption': building['Total consumption [kWh/m2/ year]'],
            'best_roi_measure': {
                'measure': best_roi_measure['Measure'],
                'roi': best_roi_measure['ROI (%)'],
                'cost': best_roi_measure['Cost (€)'],
                'payback': best_roi_measure['Payback Period (years)'],
                'annual_savings_eur': best_roi_measure['Annual Savings (€)']
            },
            'best_payback_measure': {
                'measure': best_payback_measure['Measure'],
                'roi': best_payback_measure['ROI (%)'],
                'cost': best_payback_measure['Cost (€)'],
                'payback': best_payback_measure['Payback Period (years)'],
                'annual_savings_eur': best_payback_measure['Annual Savings (€)']
            }
        }

        if not cluster_packages.empty:
            recommendations['best_roi_package'] = {
                'package': best_roi_package['Package'],
                'roi': best_roi_package['ROI (%)'],
                'cost': best_roi_package['Total Cost (€)'],
                'payback': best_roi_package['Payback Period (years)'],
                'annual_savings_eur': best_roi_package['Annual Savings (€)'],
                'measures': best_roi_package['Measures']
            }

        return recommendations

    # Create an example of the decision support tool in action
    # Select a sample building for demonstration
    sample_building = building_data['Building'].iloc[0]

    print(f"\nDEMONSTRATION OF DECISION SUPPORT TOOL")
    print(f"======================================")

    # Example 1: No budget constraint
    print("\nExample 1: Retrofit recommendations without budget constraint")
    recommendations = recommend_retrofit(sample_building)

    # Example 2: With budget constraint
    budget = 50000  # Example budget of 50,000 €
    print(f"\nExample 2: Retrofit recommendations with budget constraint of {budget} €")
    recommendations_with_budget = recommend_retrofit(sample_building, budget)

    # Create visual representation of recommendations
    selected_buildings = building_data['Building'].iloc[:3].tolist()  # Select first 3 buildings

    # Collect recommendations for selected buildings
    recommendations_list = []

    for building_id in selected_buildings:
        rec = recommend_retrofit(building_id)
        if 'best_roi_package' in rec:
            recommendations_list.append({
                'Building': building_id,
                'Current Consumption': rec['current_consumption'],
                'Recommended Package': rec['best_roi_package']['package'],
                'Cost (€)': rec['best_roi_package']['cost'],
                'ROI (%)': rec['best_roi_package']['roi'],
                'Payback (years)': rec['best_roi_package']['payback'],
                'Annual Savings (€)': rec['best_roi_package']['annual_savings_eur']
            })
        else:
            recommendations_list.append({
                'Building': building_id,
                'Current Consumption': rec['current_consumption'],
                'Recommended Measure': rec['best_roi_measure']['measure'],
                'Cost (€)': rec['best_roi_measure']['cost'],
                'ROI (%)': rec['best_roi_measure']['roi'],
                'Payback (years)': rec['best_roi_measure']['payback'],
                'Annual Savings (€)': rec['best_roi_measure']['annual_savings_eur']
            })

    # Create dataframe with recommendations
    recommendations_df = pd.DataFrame(recommendations_list)

    # Visualize recommendations
    # Plot costs vs ROI for different buildings
    plt.figure(figsize=(12, 8))
    for i, building in enumerate(recommendations_df['Building']):
        plt.scatter(
            recommendations_df.loc[recommendations_df['Building'] == building, 'Cost (€)'],
            recommendations_df.loc[recommendations_df['Building'] == building, 'ROI (%)'],
            s=100,
            label=building
        )

    plt.title('Cost vs ROI of Recommended Retrofit Strategies')
    plt.xlabel('Cost (€)')
    plt.ylabel('ROI (%)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/recommendations_cost_roi.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot payback period vs annual savings
    plt.figure(figsize=(12, 8))
    for i, building in enumerate(recommendations_df['Building']):
        plt.scatter(
            recommendations_df.loc[recommendations_df['Building'] == building, 'Payback (years)'],
            recommendations_df.loc[recommendations_df['Building'] == building, 'Annual Savings (€)'],
            s=100,
            label=building
        )

    plt.title('Payback Period vs Annual Savings of Recommended Strategies')
    plt.xlabel('Payback Period (years)')
    plt.ylabel('Annual Savings (€)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/recommendations_payback_savings.png', dpi=300, bbox_inches='tight')
    plt.close()

    return recommend_retrofit


# Main function to run the analysis pipeline
def main():
    """Main function to run the full analysis pipeline"""
    # Set file path
    file_path = "2023_Energy data ERP_COMPLETE 1.xlsx"

    # Phase 1: Load and explore data
    building_data, monthly_total, monthly_heating, monthly_dhw, monthly_cooling = load_data(file_path)
    building_data = data_exploration(building_data, monthly_total)

    # Phase 2: Machine learning analysis to identify important factors
    importance_df, rf_model = machine_learning_analysis(building_data)

    # Phase 3: Building clustering
    building_data, cluster_summary, cluster_labels = building_clustering(building_data)

    # Phase 4: Cost-benefit analysis
    results_df, package_df = cost_benefit_analysis(building_data, importance_df, cluster_summary)

    # Phase 5: Decision support tool
    recommend_retrofit = decision_support_tool(building_data, importance_df, results_df, package_df, rf_model)

    print("\nAnalysis complete! All visualizations have been saved to the 'plots' directory.")

    return building_data, importance_df, results_df, package_df, recommend_retrofit


if __name__ == "__main__":
    main()
