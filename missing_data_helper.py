import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional

class MissingDataHelper:
    """
    A helper class to analyze missing data and recommend appropriate imputation strategies.
    """
    
    def __init__(self, df: pd.DataFrame):
        """Initialize with a pandas DataFrame."""
        self.df = df
        self.analysis_results = {}
    
    def list_columns_with_missing(self) -> pd.DataFrame:
        """Display all columns with missing values and their statistics."""
        missing_stats = []
        
        for col in self.df.columns:
            missing_count = self.df[col].isna().sum()
            if missing_count > 0:
                missing_pct = (missing_count / len(self.df)) * 100
                dtype = self.df[col].dtype
                missing_stats.append({
                    'Column': col,
                    'Missing_Count': missing_count,
                    'Missing_Percent': round(missing_pct, 2),
                    'Data_Type': dtype
                })
        
        if missing_stats:
            return pd.DataFrame(missing_stats).sort_values('Missing_Percent', ascending=False)
        else:
            print("No missing values found in any column!")
            return pd.DataFrame()
    
    def analyze_column(self, column_name: str) -> Dict:
        """Perform detailed analysis on a specific column."""
        if column_name not in self.df.columns:
            raise ValueError(f"Column '{column_name}' not found in DataFrame")
        
        col = self.df[column_name]
        
        # Basic statistics
        total_rows = len(self.df)
        missing_count = col.isna().sum()
        missing_pct = (missing_count / total_rows) * 100
        present_count = total_rows - missing_count
        
        # Data type analysis
        dtype = col.dtype
        is_numeric = pd.api.types.is_numeric_dtype(col)
        is_categorical = pd.api.types.is_categorical_dtype(col) or dtype == 'object'
        
        # Cardinality (unique values)
        unique_values = col.nunique()
        cardinality = "High" if unique_values > 10 else "Low"
        
        analysis = {
            'column_name': column_name,
            'total_rows': total_rows,
            'missing_count': missing_count,
            'missing_percent': round(missing_pct, 2),
            'present_count': present_count,
            'data_type': str(dtype),
            'is_numeric': is_numeric,
            'is_categorical': is_categorical,
            'unique_values': unique_values,
            'cardinality': cardinality
        }
        
        # Value distribution for categorical
        if is_categorical and present_count > 0:
            value_counts = col.value_counts()
            analysis['top_5_values'] = value_counts.head(5).to_dict()
            if len(value_counts) > 0:
                analysis['mode'] = value_counts.index[0]
                analysis['mode_frequency'] = value_counts.iloc[0]
                analysis['mode_percent'] = round((value_counts.iloc[0] / present_count) * 100, 2)
        
        # Statistics for numeric
        if is_numeric and present_count > 0:
            analysis['mean'] = round(col.mean(), 2)
            analysis['median'] = round(col.median(), 2)
            analysis['std'] = round(col.std(), 2)
            analysis['min'] = round(col.min(), 2)
            analysis['max'] = round(col.max(), 2)
        
        self.analysis_results[column_name] = analysis
        return analysis
    
    def print_analysis_report(self, analysis: Dict):
        """Print a formatted analysis report."""
        print("\n" + "="*70)
        print(f"MISSING DATA ANALYSIS: {analysis['column_name']}")
        print("="*70)
        
        print(f"\n📊 BASIC STATISTICS:")
        print(f"   Total Rows:        {analysis['total_rows']:,}")
        print(f"   Missing Values:    {analysis['missing_count']:,} ({analysis['missing_percent']}%)")
        print(f"   Present Values:    {analysis['present_count']:,}")
        
        print(f"\n🔍 DATA CHARACTERISTICS:")
        print(f"   Data Type:         {analysis['data_type']}")
        print(f"   Unique Values:     {analysis['unique_values']}")
        print(f"   Cardinality:       {analysis['cardinality']}")
        
        if analysis['is_categorical'] and 'top_5_values' in analysis:
            print(f"\n📈 VALUE DISTRIBUTION (Top 5):")
            for value, count in analysis['top_5_values'].items():
                pct = (count / analysis['present_count']) * 100
                print(f"   '{value}': {count} ({pct:.1f}%)")
            
            if 'mode' in analysis:
                print(f"\n   Mode: '{analysis['mode']}' appears {analysis['mode_frequency']} times ({analysis['mode_percent']}%)")
        
        if analysis['is_numeric']:
            print(f"\n📈 NUMERICAL STATISTICS:")
            print(f"   Mean:              {analysis.get('mean', 'N/A')}")
            print(f"   Median:            {analysis.get('median', 'N/A')}")
            print(f"   Std Dev:           {analysis.get('std', 'N/A')}")
            print(f"   Range:             {analysis.get('min', 'N/A')} to {analysis.get('max', 'N/A')}")
    
    def recommend_strategy(self, analysis: Dict) -> Tuple[str, str, str]:
        """
        Recommend an imputation strategy based on analysis.
        Returns: (strategy, reasoning, code_snippet)
        """
        missing_pct = analysis['missing_percent']
        is_numeric = analysis['is_numeric']
        is_categorical = analysis['is_categorical']
        cardinality = analysis['cardinality']
        col_name = analysis['column_name']
        
        # Decision logic
        if missing_pct > 40:
            strategy = "DROP COLUMN or Use 'Unknown'"
            reasoning = (
                f"⚠️  CRITICAL: {missing_pct}% missing data is too high for reliable imputation.\n"
                f"   Recommendation: Consider dropping this column or keeping as 'Unknown'.\n"
                f"   Any imputation will create more noise than signal."
            )
            code = f"# Option 1: Drop the column\ndf.drop('{col_name}', axis=1, inplace=True)\n\n"
            code += f"# Option 2: Keep as 'Unknown'\ndf['{col_name}'].fillna('Unknown', inplace=True)"
        
        elif missing_pct > 20:
            strategy = "Use 'Unknown' or Advanced Methods"
            reasoning = (
                f"⚠️  WARNING: {missing_pct}% missing is substantial.\n"
                f"   Recommendation: Use 'Unknown' to preserve transparency.\n"
                f"   Consider advanced imputation methods for critical analyses."
            )
            if is_numeric:
                code = f"df['{col_name}'].fillna(-999, inplace=True)  # Use sentinel value\n"
                code += f"# Or create missing indicator: df['{col_name}_missing'] = df['{col_name}'].isna()"
            else:
                code = f"df['{col_name}'].fillna('Unknown', inplace=True)"
        
        elif missing_pct > 5:
            strategy = "'Unknown' (Preferred) or Mode/Median"
            reasoning = (
                f"⚡ CAUTION: {missing_pct}% missing is moderate.\n"
                f"   Recommendation: Use 'Unknown' for transparency, especially if missingness might be meaningful.\n"
                f"   Mode/median acceptable if missingness is truly random."
            )
            if is_numeric:
                code = f"# Preferred: Create missing indicator\ndf['{col_name}'].fillna(df['{col_name}'].median(), inplace=True)\n\n"
                code += f"# Alternative: Use sentinel\ndf['{col_name}'].fillna(-999, inplace=True)"
            else:
                code = f"# Preferred: Preserve missingness\ndf['{col_name}'].fillna('Unknown', inplace=True)\n\n"
                code += f"# Alternative: Use mode\ndf['{col_name}'].fillna(df['{col_name}'].mode()[0], inplace=True)"
        
        else:  # < 5%
            if is_numeric:
                strategy = "Median (Preferred) or Mean"
                reasoning = (
                    f"✅ SAFE: Only {missing_pct}% missing - imputation is low-risk.\n"
                    f"   Recommendation: Use median (robust to outliers) or mean.\n"
                    f"   Impact on analysis will be minimal."
                )
                code = f"# Preferred: Median (robust to outliers)\ndf['{col_name}'].fillna(df['{col_name}'].median(), inplace=True)\n\n"
                code += f"# Alternative: Mean\ndf['{col_name}'].fillna(df['{col_name}'].mean(), inplace=True)"
            
            elif is_categorical:
                if cardinality == "High":
                    strategy = "'Unknown' (safer for high cardinality)"
                    reasoning = (
                        f"✅ SAFE: Only {missing_pct}% missing, but {analysis['unique_values']} unique values.\n"
                        f"   Recommendation: Use 'Unknown' to avoid creating false patterns.\n"
                        f"   High cardinality makes mode less reliable."
                    )
                    code = f"df['{col_name}'].fillna('Unknown', inplace=True)"
                else:
                    strategy = "Mode (acceptable) or 'Unknown'"
                    mode_val = analysis.get('mode', 'N/A')
                    mode_pct = analysis.get('mode_percent', 0)
                    reasoning = (
                        f"✅ SAFE: Only {missing_pct}% missing with clear mode.\n"
                        f"   Mode '{mode_val}' appears in {mode_pct}% of non-missing values.\n"
                        f"   Recommendation: Mode is acceptable, but 'Unknown' is more transparent."
                    )
                    code = f"# Option 1: Mode (acceptable for low missingness)\ndf['{col_name}'].fillna(df['{col_name}'].mode()[0], inplace=True)\n\n"
                    code += f"# Option 2: 'Unknown' (more transparent)\ndf['{col_name}'].fillna('Unknown', inplace=True)"
            else:
                strategy = "Convert to appropriate type first"
                reasoning = "⚠️  Unable to determine data type clearly. Please check the column manually."
                code = f"# Check data type first\nprint(df['{col_name}'].dtype)\nprint(df['{col_name}'].head(10))"
        
        return strategy, reasoning, code
    
    def interactive_analysis(self, column_name: Optional[str] = None):
        """Run interactive analysis for a column."""
        if column_name is None:
            # Show columns with missing data
            missing_df = self.list_columns_with_missing()
            if missing_df.empty:
                return
            
            print("\nColumns with missing data:")
            print(missing_df.to_string(index=False))
            print("\nUse: helper.interactive_analysis('column_name') to analyze a specific column")
            return
        
        # Analyze the column
        analysis = self.analyze_column(column_name)
        
        # Print report
        self.print_analysis_report(analysis)
        
        # Get recommendation
        strategy, reasoning, code = self.recommend_strategy(analysis)
        
        print("\n" + "="*70)
        print("💡 RECOMMENDATION")
        print("="*70)
        print(f"\nStrategy: {strategy}")
        print(f"\n{reasoning}")
        
        print(f"\n📝 SUGGESTED CODE:")
        print("-" * 70)
        print(code)
        print("-" * 70)
        
        print("\n✨ NEXT STEPS:")
        print("   1. Review the analysis and recommendation above")
        print("   2. Copy and run the suggested code")
        print("   3. Verify the results with: df['{}'].isna().sum()".format(column_name))
        print("="*70 + "\n")


# ==============================================================================
# USAGE EXAMPLE
# ==============================================================================

if __name__ == "__main__":
    # Example usage
    print("="*70)
    print("MISSING DATA IMPUTATION HELPER - USAGE EXAMPLE")
    print("="*70)
    
    # Create sample data
    np.random.seed(42)
    sample_data = {
        'Temperature': [72, 68, np.nan, 75, 80, np.nan, 70, 73],
        'Weather': ['Sunny', 'Rainy', 'Sunny', np.nan, 'Sunny', 'Cloudy', np.nan, 'Sunny'],
        'Time_of_Day': ['Morning', np.nan, 'Evening', 'Afternoon', np.nan, 'Morning', 'Night', np.nan],
        'City': ['NYC', 'LA', np.nan, 'Chicago', 'NYC', np.nan, np.nan, np.nan]
    }
    df = pd.DataFrame(sample_data)
    
    print("\nSample DataFrame:")
    print(df)
    
    # Initialize helper
    helper = MissingDataHelper(df)
    
    print("\n" + "="*70)
    print("STEP 1: List all columns with missing data")
    print("="*70)
    helper.interactive_analysis()
    
    print("\n" + "="*70)
    print("STEP 2: Analyze a specific column")
    print("="*70)
    print("\nAnalyzing 'Weather' column...")
    helper.interactive_analysis('Weather')
    
    print("\n" + "="*70)
    print("HOW TO USE IN YOUR PROJECT")
    print("="*70)
    print("""
# 1. Import and initialize
from missing_data_helper import MissingDataHelper

helper = MissingDataHelper(your_dataframe)

# 2. See all columns with missing data
helper.interactive_analysis()

# 3. Analyze specific column
helper.interactive_analysis('your_column_name')

# 4. Copy and use the suggested code!
""")