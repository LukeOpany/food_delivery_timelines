"""
Feature Engineering Workflow Tool
A reusable script for intelligent feature engineering decisions

Usage:
    fe_tool = FeatureEngineeringWorkflow(df, target_column='target')
    fe_tool.run_interactive_workflow()
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineeringWorkflow:
    """
    Interactive workflow for feature engineering decisions.
    Analyzes columns and recommends binning/interaction strategies.
    """
    
    def __init__(self, df: pd.DataFrame, target_column: str, model_type: str = None):
        """
        Initialize the workflow.
        
        Args:
            df: Your dataset
            target_column: Name of target variable
            model_type: 'tree' or 'linear' (will prompt if not provided)
        """
        self.df = df.copy()
        self.target = target_column
        self.model_type = model_type
        self.recommendations = []
        
    def run_interactive_workflow(self):
        """Main workflow execution"""
        print("="*70)
        print("FEATURE ENGINEERING WORKFLOW")
        print("="*70)
        
        # Step 1: Determine model type
        if not self.model_type:
            self._get_model_type()
        
        # Step 2: Show dataset overview
        self._show_dataset_overview()
        
        # Step 3: Analyze all features
        print("\n" + "="*70)
        print("ANALYZING ALL FEATURES...")
        print("="*70)
        self._analyze_all_features()
        
        # Step 4: Interactive selection
        self._interactive_selection()
        
        # Step 5: Generate code
        self._generate_code()
        
    def _get_model_type(self):
        """Prompt user for model type"""
        print("\nWhat type of model will you use?")
        print("1. Tree-based (Random Forest, XGBoost, LightGBM, etc.)")
        print("2. Linear (Logistic Regression, Linear Regression, SVM, etc.)")
        
        while True:
            choice = input("\nEnter 1 or 2: ").strip()
            if choice == '1':
                self.model_type = 'tree'
                print("\n✓ Tree-based model selected")
                print("💡 Note: Tree models handle non-linearity automatically.")
                print("   Feature engineering is less critical but can still help.\n")
                break
            elif choice == '2':
                self.model_type = 'linear'
                print("\n✓ Linear model selected")
                print("💡 Note: Linear models benefit greatly from feature engineering!")
                print("   Binning and interactions are highly recommended.\n")
                break
            else:
                print("Invalid choice. Please enter 1 or 2.")
    
    def _show_dataset_overview(self):
        """Display dataset summary"""
        print("\n" + "-"*70)
        print("DATASET OVERVIEW")
        print("-"*70)
        print(f"Shape: {self.df.shape[0]} rows × {self.df.shape[1]} columns")
        print(f"Target: {self.target}")
        print(f"\nColumn types:")
        print(self.df.dtypes.value_counts())
        print(f"\nMissing values: {self.df.isnull().sum().sum()} total")
        
    def _analyze_all_features(self):
        """Analyze each feature and provide recommendations"""
        features = [col for col in self.df.columns if col != self.target]
        
        for col in features:
            analysis = self._analyze_column(col)
            self.recommendations.append(analysis)
            self._print_analysis(analysis)
    
    def _analyze_column(self, col: str) -> Dict:
        """Analyze a single column and recommend actions"""
        analysis = {
            'column': col,
            'dtype': str(self.df[col].dtype),
            'binning_eligible': False,
            'binning_method': None,
            'interaction_eligible': False,
            'reasons': [],
            'warnings': []
        }
        
        # Basic stats
        n_unique = self.df[col].nunique()
        missing_pct = (self.df[col].isnull().sum() / len(self.df)) * 100
        
        analysis['n_unique'] = n_unique
        analysis['missing_pct'] = missing_pct
        
        # Check if numeric
        is_numeric = pd.api.types.is_numeric_dtype(self.df[col])
        
        if missing_pct > 50:
            analysis['warnings'].append(f"⚠️  High missing rate ({missing_pct:.1f}%)")
        
        # BINNING ANALYSIS
        if is_numeric and n_unique > 10:
            analysis['binning_eligible'] = True
            
            # Determine best binning method
            skewness = self.df[col].skew()
            has_outliers = self._check_outliers(col)
            
            if abs(skewness) > 1 or has_outliers:
                analysis['binning_method'] = 'quantile'
                analysis['reasons'].append(
                    f"📊 Quantile binning recommended (skewness={skewness:.2f})"
                )
            else:
                analysis['binning_method'] = 'equal_width'
                analysis['reasons'].append(
                    "📊 Equal-width binning suitable (normal distribution)"
                )
            
            if has_outliers:
                analysis['reasons'].append("🎯 Binning will handle outliers")
            
            # Model-specific recommendation
            if self.model_type == 'linear':
                analysis['reasons'].append(
                    "✓ RECOMMENDED for linear models (captures non-linearity)"
                )
            else:
                analysis['reasons'].append(
                    "⚪ Optional for tree models (they handle this automatically)"
                )
        
        elif is_numeric and n_unique <= 10:
            analysis['warnings'].append(
                f"Already discrete ({n_unique} unique values) - no binning needed"
            )
        
        # INTERACTION ANALYSIS
        if is_numeric or (not is_numeric and n_unique < 10):
            analysis['interaction_eligible'] = True
            
            if self.model_type == 'linear':
                analysis['reasons'].append(
                    "🔗 Good candidate for interactions (linear model)"
                )
            else:
                analysis['reasons'].append(
                    "🔗 Interactions optional (tree models learn these)"
                )
        
        return analysis
    
    def _check_outliers(self, col: str) -> bool:
        """Check if column has outliers using IQR method"""
        try:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers = ((self.df[col] < lower) | (self.df[col] > upper)).sum()
            return outliers > 0
        except:
            return False
    
    def _print_analysis(self, analysis: Dict):
        """Pretty print analysis for a column"""
        print(f"\n{'─'*70}")
        print(f"Column: {analysis['column']}")
        print(f"Type: {analysis['dtype']} | Unique: {analysis['n_unique']} | Missing: {analysis['missing_pct']:.1f}%")
        
        if analysis['warnings']:
            for warning in analysis['warnings']:
                print(f"  {warning}")
        
        if analysis['binning_eligible']:
            print(f"\n  BINNING: ✓ Eligible ({analysis['binning_method']})")
        else:
            print(f"\n  BINNING: ✗ Not recommended")
        
        if analysis['interaction_eligible']:
            print(f"  INTERACTIONS: ✓ Eligible")
        else:
            print(f"  INTERACTIONS: ✗ Not recommended")
        
        if analysis['reasons']:
            print("\n  Reasoning:")
            for reason in analysis['reasons']:
                print(f"    • {reason}")
    
    def _interactive_selection(self):
        """Let user select features for engineering"""
        print("\n" + "="*70)
        print("INTERACTIVE SELECTION")
        print("="*70)
        
        # Filter eligible features
        binning_candidates = [r for r in self.recommendations if r['binning_eligible']]
        interaction_candidates = [r for r in self.recommendations if r['interaction_eligible']]
        
        self.selected_binning = []
        self.selected_interactions = []
        
        # Binning selection
        if binning_candidates:
            print("\n📊 BINNING CANDIDATES:")
            for i, rec in enumerate(binning_candidates, 1):
                priority = "HIGH" if self.model_type == 'linear' else "LOW"
                print(f"  {i}. {rec['column']} (Priority: {priority})")
            
            print("\nEnter column numbers to bin (comma-separated) or 'all' or 'none':")
            selection = input("Selection: ").strip().lower()
            
            if selection == 'all':
                self.selected_binning = [r['column'] for r in binning_candidates]
            elif selection != 'none':
                try:
                    indices = [int(x.strip()) - 1 for x in selection.split(',')]
                    self.selected_binning = [binning_candidates[i]['column'] for i in indices]
                except:
                    print("Invalid selection, skipping binning.")
        
        # Interaction selection
        if interaction_candidates and len(interaction_candidates) >= 2:
            print("\n🔗 INTERACTION CANDIDATES:")
            for i, rec in enumerate(interaction_candidates, 1):
                print(f"  {i}. {rec['column']}")
            
            print("\nCreate interactions? Enter pairs like '1,2' or '1,3 2,4' or 'none':")
            selection = input("Selection: ").strip().lower()
            
            if selection != 'none':
                try:
                    pairs = selection.split()
                    for pair in pairs:
                        idx1, idx2 = [int(x.strip()) - 1 for x in pair.split(',')]
                        col1 = interaction_candidates[idx1]['column']
                        col2 = interaction_candidates[idx2]['column']
                        self.selected_interactions.append((col1, col2))
                except:
                    print("Invalid selection, skipping interactions.")
        
        print(f"\n✓ Selected {len(self.selected_binning)} columns for binning")
        print(f"✓ Selected {len(self.selected_interactions)} interaction pairs")
    
    def _generate_code(self):
        """Generate executable code for selected features"""
        print("\n" + "="*70)
        print("GENERATED CODE")
        print("="*70)
        
        code_lines = [
            "# Feature Engineering Code",
            "# Copy and paste this into your pipeline\n",
            "import pandas as pd",
            "import numpy as np",
            "from sklearn.preprocessing import KBinsDiscretizer\n",
            "def engineer_features(df):",
            "    \"\"\"Apply feature engineering transformations\"\"\"",
            "    df = df.copy()\n"
        ]
        
        # Binning code
        if self.selected_binning:
            code_lines.append("    # BINNING")
            for col in self.selected_binning:
                rec = next(r for r in self.recommendations if r['column'] == col)
                method = rec['binning_method']
                
                if method == 'quantile':
                    code_lines.extend([
                        f"    # {col} - Quantile binning",
                        f"    df['{col}_binned'] = pd.qcut(df['{col}'], q=5, labels=False, duplicates='drop')"
                    ])
                else:
                    code_lines.extend([
                        f"    # {col} - Equal-width binning",
                        f"    df['{col}_binned'] = pd.cut(df['{col}'], bins=5, labels=False)"
                    ])
            code_lines.append("")
        
        # Interaction code
        if self.selected_interactions:
            code_lines.append("    # INTERACTIONS")
            for col1, col2 in self.selected_interactions:
                code_lines.extend([
                    f"    # Interaction: {col1} × {col2}",
                    f"    df['{col1}_x_{col2}'] = df['{col1}'] * df['{col2}']"
                ])
            code_lines.append("")
        
        code_lines.extend([
            "    return df\n",
            "# Apply to your data:",
            "# df_engineered = engineer_features(df)"
        ])
        
        final_code = "\n".join(code_lines)
        print(final_code)
        
        # Save to file
        with open('feature_engineering_code.py', 'w') as f:
            f.write(final_code)
        print("\n✓ Code saved to 'feature_engineering_code.py'")
        
        return final_code


# EXAMPLE USAGE
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    df = pd.DataFrame({
        'age': np.random.randint(18, 80, 1000),
        'income': np.random.exponential(50000, 1000),
        'credit_score': np.random.normal(700, 50, 1000),
        'num_accounts': np.random.poisson(3, 1000),
        'region': np.random.choice(['North', 'South', 'East', 'West'], 1000),
        'target': np.random.binomial(1, 0.3, 1000)
    })
    
    print("SAMPLE USAGE:")
    print("="*70)
    print("\n# Initialize the workflow")
    print("fe_tool = FeatureEngineeringWorkflow(df, target_column='target')")
    print("\n# Run interactive workflow")
    print("fe_tool.run_interactive_workflow()")
    print("\n" + "="*70)
    print("\nUncomment the lines below to run with sample data:\n")
    
    # Uncomment to run:
    # fe_tool = FeatureEngineeringWorkflow(df, target_column='target')
    # fe_tool.run_interactive_workflow()
