"""
Dimensionality reduction analysis module for DataAiPrep
Implements PCA and t-SNE for data visualization and pattern discovery
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import warnings


class DimensionalityAnalyzer:
    """Performs PCA and t-SNE analysis for dimensionality reduction and visualization"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.max_features_tsne = 50  # Limit features for t-SNE performance
        self.max_samples_tsne = 5000  # Limit samples for t-SNE performance
        
    def analyze(self, df: pd.DataFrame, target_column: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform comprehensive dimensionality reduction analysis
        
        Args:
            df: DataFrame to analyze
            target_column: Name of target column if available
            
        Returns:
            Dictionary containing PCA and t-SNE analysis results
        """
        results = {
            'total_features': len(df.columns) - (1 if target_column else 0),
            'total_samples': len(df),
            'target_column': target_column,
            'pca_analysis': {},
            'tsne_analysis': {},
            'data_preparation': {},
            'visualizations': {},
            'recommendations': [],
            'reduction_score': 0.0
        }
        
        try:
            # Prepare data for analysis
            X, y, feature_names = self._prepare_data(df, target_column)
            results['data_preparation'] = {
                'processed_features': len(feature_names),
                'processed_samples': len(X),
                'feature_names': feature_names,
                'has_target': y is not None,
                'target_classes': len(np.unique(y)) if y is not None else 0
            }
            
            if X.shape[1] < 2:
                results['error'] = "Need at least 2 features for dimensionality reduction"
                return results
            
            # Perform PCA analysis
            results['pca_analysis'] = self._perform_pca_analysis(X, feature_names, y)
            
            # Perform t-SNE analysis
            results['tsne_analysis'] = self._perform_tsne_analysis(X, y)
            
            # Generate visualizations data
            results['visualizations'] = self._prepare_visualizations(
                results['pca_analysis'], results['tsne_analysis'], y
            )
            
            # Generate recommendations
            results['recommendations'] = self._generate_recommendations(results)
            
            # Calculate reduction quality score
            results['reduction_score'] = self._calculate_reduction_score(results)
            
            self.logger.info(f"Dimensionality analysis completed. Score: {results['reduction_score']:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error in dimensionality analysis: {str(e)}")
            results['error'] = str(e)
            
        return results
        
    def _prepare_data(self, df: pd.DataFrame, target_column: Optional[str] = None) -> Tuple[np.ndarray, Optional[np.ndarray], List[str]]:
        """Prepare data for dimensionality reduction"""
        try:
            # Separate features and target
            if target_column and target_column in df.columns:
                feature_columns = [col for col in df.columns if col != target_column]
                y = df[target_column].copy()
            else:
                feature_columns = list(df.columns)
                y = None
                
            X = df[feature_columns].copy()
            
            # Handle missing values
            if X.isnull().any().any():
                self.logger.info("Handling missing values with median/mode imputation")
                numeric_cols = X.select_dtypes(include=[np.number]).columns
                categorical_cols = X.select_dtypes(exclude=[np.number]).columns
                
                if len(numeric_cols) > 0:
                    X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
                if len(categorical_cols) > 0:
                    X[categorical_cols] = X[categorical_cols].fillna(X[categorical_cols].mode().iloc[0])
            
            # Encode categorical features
            categorical_columns = X.select_dtypes(include=['object']).columns
            label_encoders = {}
            
            for col in categorical_columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                label_encoders[col] = le
                
            # Convert to numeric array
            X_numeric = X.select_dtypes(include=[np.number]).values
            feature_names = X.select_dtypes(include=[np.number]).columns.tolist()
            
            # Handle infinite values and extreme outliers
            # Replace infinity with NaN, then fill with column median
            X_numeric = np.where(np.isinf(X_numeric), np.nan, X_numeric)
            
            # Fill any remaining NaN values with column median
            imputer = SimpleImputer(strategy='median')
            X_numeric = imputer.fit_transform(X_numeric)
            
            # Handle extreme values (beyond 5 standard deviations)
            for col_idx in range(X_numeric.shape[1]):
                col_data = X_numeric[:, col_idx]
                mean_val = np.mean(col_data)
                std_val = np.std(col_data)
                
                if std_val > 0:  # Avoid division by zero
                    # Cap extreme values at 5 standard deviations
                    lower_bound = mean_val - 5 * std_val
                    upper_bound = mean_val + 5 * std_val
                    X_numeric[:, col_idx] = np.clip(col_data, lower_bound, upper_bound)
            
            # Handle target encoding if categorical
            if y is not None and not pd.api.types.is_numeric_dtype(y):
                le_target = LabelEncoder()
                y = le_target.fit_transform(y.astype(str))
            elif y is not None:
                y = y.values
                
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_numeric)
            
            # Final check for any remaining invalid values
            if not np.isfinite(X_scaled).all():
                self.logger.warning("Still have non-finite values after preprocessing, replacing with zeros")
                X_scaled = np.where(np.isfinite(X_scaled), X_scaled, 0)
            
            return X_scaled, y, feature_names
            
        except Exception as e:
            self.logger.error(f"Error preparing data: {str(e)}")
            raise
            
    def _perform_pca_analysis(self, X: np.ndarray, feature_names: List[str], y: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Perform Principal Component Analysis"""
        pca_results = {}
        
        try:
            # Determine number of components
            n_features = X.shape[1]
            n_samples = X.shape[0]
            max_components = min(n_features, n_samples, 50)  # Limit for performance
            
            # Fit PCA with all possible components
            pca_full = PCA(n_components=max_components)
            X_pca = pca_full.fit_transform(X)
            
            # Calculate explained variance
            explained_variance_ratio = pca_full.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance_ratio)
            
            # Find components for different variance thresholds
            components_80 = np.argmax(cumulative_variance >= 0.8) + 1
            components_90 = np.argmax(cumulative_variance >= 0.9) + 1
            components_95 = np.argmax(cumulative_variance >= 0.95) + 1
            
            pca_results = {
                'n_components': max_components,
                'explained_variance_ratio': explained_variance_ratio.tolist(),
                'cumulative_variance': cumulative_variance.tolist(),
                'components_for_80_variance': int(components_80),
                'components_for_90_variance': int(components_90),
                'components_for_95_variance': int(components_95),
                'total_variance_explained': float(cumulative_variance[-1]),
                'singular_values': pca_full.singular_values_.tolist(),
                'components': X_pca.tolist(),  # First few components for visualization
                'feature_importance': {}
            }
            
            # Analyze feature importance in first few components
            n_important_components = min(5, max_components)
            components_matrix = pca_full.components_[:n_important_components]
            
            for i, component in enumerate(components_matrix):
                component_name = f"PC{i+1}"
                feature_contributions = {}
                
                for j, feature_name in enumerate(feature_names):
                    if j < len(component):
                        feature_contributions[feature_name] = float(abs(component[j]))
                
                # Sort by contribution
                sorted_contributions = sorted(feature_contributions.items(), 
                                           key=lambda x: x[1], reverse=True)
                pca_results['feature_importance'][component_name] = sorted_contributions[:10]
            
            # Fit PCA for 2D visualization
            if max_components >= 2:
                pca_2d = PCA(n_components=2)
                X_pca_2d = pca_2d.fit_transform(X)
                pca_results['pca_2d'] = {
                    'coordinates': X_pca_2d.tolist(),
                    'explained_variance_ratio': pca_2d.explained_variance_ratio_.tolist(),
                    'total_variance_2d': float(sum(pca_2d.explained_variance_ratio_))
                }
            
            # Fit PCA for 3D visualization if possible
            if max_components >= 3:
                pca_3d = PCA(n_components=3)
                X_pca_3d = pca_3d.fit_transform(X)
                pca_results['pca_3d'] = {
                    'coordinates': X_pca_3d.tolist(),
                    'explained_variance_ratio': pca_3d.explained_variance_ratio_.tolist(),
                    'total_variance_3d': float(sum(pca_3d.explained_variance_ratio_))
                }
                
        except Exception as e:
            self.logger.error(f"Error in PCA analysis: {str(e)}")
            pca_results['error'] = str(e)
            
        return pca_results
        
    def _perform_tsne_analysis(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Perform t-SNE analysis"""
        tsne_results = {}
        
        try:
            # Prepare data for t-SNE (limit features and samples for performance)
            X_tsne = X.copy()
            y_tsne = y.copy() if y is not None else None
            
            # Limit features if too many
            if X_tsne.shape[1] > self.max_features_tsne:
                # Use PCA to reduce to manageable number of features first
                pca_reducer = PCA(n_components=self.max_features_tsne)
                X_tsne = pca_reducer.fit_transform(X_tsne)
                self.logger.info(f"Reduced features from {X.shape[1]} to {self.max_features_tsne} using PCA for t-SNE")
            
            # Limit samples if too many
            if X_tsne.shape[0] > self.max_samples_tsne:
                indices = np.random.choice(X_tsne.shape[0], self.max_samples_tsne, replace=False)
                X_tsne = X_tsne[indices]
                y_tsne = y_tsne[indices] if y_tsne is not None else None
                self.logger.info(f"Sampled {self.max_samples_tsne} points from {X.shape[0]} for t-SNE")
            
            # Perform t-SNE with different perplexity values
            perplexity_values = [5, 15, 30, 50]
            valid_perplexities = [p for p in perplexity_values if p < X_tsne.shape[0] / 3]
            
            if not valid_perplexities:
                valid_perplexities = [min(5, X_tsne.shape[0] // 4)]
            
            tsne_results = {
                'n_samples_used': X_tsne.shape[0],
                'n_features_used': X_tsne.shape[1],
                'perplexity_results': {},
                'best_perplexity': None,
                'preprocessing_applied': X.shape[1] > self.max_features_tsne
            }
            
            best_kl_divergence = float('inf')
            best_perplexity = valid_perplexities[0]
            
            for perplexity in valid_perplexities[:2]:  # Limit to 2 perplexity values for performance
                try:
                    # Fit t-SNE
                    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, 
                              max_iter=1000, learning_rate='auto', init='pca')
                    
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        X_tsne_2d = tsne.fit_transform(X_tsne)
                    
                    tsne_results['perplexity_results'][perplexity] = {
                        'coordinates': X_tsne_2d.tolist(),
                        'kl_divergence': float(tsne.kl_divergence_),
                        'n_iter_final': int(tsne.max_iter),
                        'learning_rate': float(tsne.learning_rate)
                    }
                    
                    # Track best perplexity (lowest KL divergence)
                    if tsne.kl_divergence_ < best_kl_divergence:
                        best_kl_divergence = tsne.kl_divergence_
                        best_perplexity = perplexity
                        
                except Exception as e:
                    self.logger.warning(f"t-SNE failed for perplexity {perplexity}: {str(e)}")
                    continue
            
            tsne_results['best_perplexity'] = best_perplexity
            
            # Try 3D t-SNE with best perplexity if we have enough data
            if X_tsne.shape[0] >= 50 and len(tsne_results['perplexity_results']) > 0:
                try:
                    tsne_3d = TSNE(n_components=3, perplexity=best_perplexity, random_state=42,
                                 max_iter=1000, learning_rate='auto', init='pca')
                    
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        X_tsne_3d = tsne_3d.fit_transform(X_tsne)
                    
                    tsne_results['tsne_3d'] = {
                        'coordinates': X_tsne_3d.tolist(),
                        'kl_divergence': float(tsne_3d.kl_divergence_),
                        'perplexity': best_perplexity
                    }
                    
                except Exception as e:
                    self.logger.warning(f"3D t-SNE failed: {str(e)}")
                    
        except Exception as e:
            self.logger.error(f"Error in t-SNE analysis: {str(e)}")
            tsne_results['error'] = str(e)
            
        return tsne_results
        
    def _prepare_visualizations(self, pca_results: Dict[str, Any], tsne_results: Dict[str, Any], 
                              y: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Prepare visualization data for the GUI"""
        visualizations = {
            'pca_scree_plot': {},
            'pca_2d_plot': {},
            'pca_3d_plot': {},
            'tsne_2d_plot': {},
            'tsne_3d_plot': {},
            'feature_importance_plot': {}
        }
        
        try:
            # PCA Scree Plot data
            if 'explained_variance_ratio' in pca_results:
                explained_var = pca_results['explained_variance_ratio']
                cumulative_var = pca_results['cumulative_variance']
                
                visualizations['pca_scree_plot'] = {
                    'components': list(range(1, len(explained_var) + 1)),
                    'explained_variance': explained_var,
                    'cumulative_variance': cumulative_var,
                    'components_80': pca_results.get('components_for_80_variance', 0),
                    'components_90': pca_results.get('components_for_90_variance', 0)
                }
            
            # PCA 2D Plot data
            if 'pca_2d' in pca_results:
                coords = pca_results['pca_2d']['coordinates']
                visualizations['pca_2d_plot'] = {
                    'x': [coord[0] for coord in coords],
                    'y': [coord[1] for coord in coords],
                    'labels': y.tolist() if y is not None else None,
                    'variance_explained': pca_results['pca_2d']['total_variance_2d']
                }
            
            # PCA 3D Plot data
            if 'pca_3d' in pca_results:
                coords = pca_results['pca_3d']['coordinates']
                visualizations['pca_3d_plot'] = {
                    'x': [coord[0] for coord in coords],
                    'y': [coord[1] for coord in coords],
                    'z': [coord[2] for coord in coords],
                    'labels': y.tolist() if y is not None else None,
                    'variance_explained': pca_results['pca_3d']['total_variance_3d']
                }
            
            # t-SNE 2D Plot data
            if tsne_results.get('best_perplexity') and tsne_results.get('perplexity_results'):
                best_perp = tsne_results['best_perplexity']
                coords = tsne_results['perplexity_results'][best_perp]['coordinates']
                
                visualizations['tsne_2d_plot'] = {
                    'x': [coord[0] for coord in coords],
                    'y': [coord[1] for coord in coords],
                    'labels': y.tolist() if y is not None else None,
                    'perplexity': best_perp,
                    'kl_divergence': tsne_results['perplexity_results'][best_perp]['kl_divergence']
                }
            
            # t-SNE 3D Plot data
            if 'tsne_3d' in tsne_results:
                coords = tsne_results['tsne_3d']['coordinates']
                visualizations['tsne_3d_plot'] = {
                    'x': [coord[0] for coord in coords],
                    'y': [coord[1] for coord in coords],
                    'z': [coord[2] for coord in coords],
                    'labels': y.tolist() if y is not None else None,
                    'perplexity': tsne_results['tsne_3d']['perplexity']
                }
            
            # Feature importance in PCA
            if 'feature_importance' in pca_results:
                visualizations['feature_importance_plot'] = pca_results['feature_importance']
                
        except Exception as e:
            self.logger.warning(f"Error preparing visualizations: {str(e)}")
            
        return visualizations
        
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on dimensionality analysis"""
        recommendations = []
        
        try:
            pca_results = results.get('pca_analysis', {})
            tsne_results = results.get('tsne_analysis', {})
            data_prep = results.get('data_preparation', {})
            
            # PCA recommendations
            if 'components_for_95_variance' in pca_results:
                total_features = data_prep.get('processed_features', 0)
                comp_95 = pca_results['components_for_95_variance']
                comp_90 = pca_results['components_for_90_variance']
                comp_80 = pca_results['components_for_80_variance']
                
                reduction_ratio_95 = (total_features - comp_95) / total_features * 100
                reduction_ratio_90 = (total_features - comp_90) / total_features * 100
                reduction_ratio_80 = (total_features - comp_80) / total_features * 100
                
                recommendations.append(f"📊 PCA Dimensionality Reduction Opportunities:")
                recommendations.append(f"  • Reduce to {comp_80} components: retain 80% variance ({reduction_ratio_80:.1f}% reduction)")
                recommendations.append(f"  • Reduce to {comp_90} components: retain 90% variance ({reduction_ratio_90:.1f}% reduction)")
                recommendations.append(f"  • Reduce to {comp_95} components: retain 95% variance ({reduction_ratio_95:.1f}% reduction)")
                
                if reduction_ratio_90 > 50:
                    recommendations.append("  💡 Significant dimensionality reduction possible!")
                    recommendations.append("  💡 Consider using PCA for preprocessing before ML training")
                elif reduction_ratio_90 > 20:
                    recommendations.append("  ✓ Moderate dimensionality reduction available")
                else:
                    recommendations.append("  ⚠ Limited dimensionality reduction benefits")
            
            # Feature importance recommendations
            if 'feature_importance' in pca_results:
                recommendations.append(f"\n🎯 Feature Importance Insights:")
                pc1_features = pca_results['feature_importance'].get('PC1', [])
                if pc1_features:
                    top_features = [f[0] for f in pc1_features[:3]]
                    recommendations.append(f"  • Most important features (PC1): {', '.join(top_features)}")
                    recommendations.append("  • These features explain the most variance in your data")
            
            # t-SNE recommendations
            if tsne_results.get('perplexity_results'):
                best_perp = tsne_results.get('best_perplexity')
                kl_div = tsne_results['perplexity_results'][best_perp]['kl_divergence']
                
                recommendations.append(f"\n🗺️ t-SNE Visualization Quality:")
                if kl_div < 1.0:
                    recommendations.append(f"  ✓ Excellent t-SNE convergence (KL divergence: {kl_div:.3f})")
                    recommendations.append("  ✓ Clear clusters and patterns should be visible")
                elif kl_div < 2.0:
                    recommendations.append(f"  ✓ Good t-SNE convergence (KL divergence: {kl_div:.3f})")
                    recommendations.append("  ✓ Reliable visualization for pattern discovery")
                else:
                    recommendations.append(f"  ⚠ Moderate t-SNE convergence (KL divergence: {kl_div:.3f})")
                    recommendations.append("  ⚠ Try different perplexity values or preprocessing")
                
                if tsne_results.get('preprocessing_applied'):
                    recommendations.append("  💡 Data was preprocessed with PCA for t-SNE performance")
            
            # Data-specific recommendations
            total_features = data_prep.get('processed_features', 0)
            total_samples = data_prep.get('processed_samples', 0)
            
            if total_features > 100:
                recommendations.append(f"\n🚀 High-Dimensional Data Recommendations:")
                recommendations.append("  • Consider PCA preprocessing for ML algorithms")
                recommendations.append("  • Use feature selection based on PCA importance")
                recommendations.append("  • Tree-based algorithms handle high dimensions well")
            
            if total_samples > 10000:
                recommendations.append(f"\n📈 Large Dataset Recommendations:")
                recommendations.append("  • Consider incremental PCA for memory efficiency")
                recommendations.append("  • Use sampling for t-SNE visualization")
                recommendations.append("  • Parallel processing will improve performance")
            
            # Clustering recommendations
            has_target = data_prep.get('has_target', False)
            target_classes = data_prep.get('target_classes', 0)
            
            if not has_target:
                recommendations.append(f"\n🔍 Unsupervised Learning Opportunities:")
                recommendations.append("  • Use t-SNE plots to discover natural clusters")
                recommendations.append("  • Consider k-means clustering on PCA components")
                recommendations.append("  • Look for patterns in the 2D/3D visualizations")
            elif target_classes > 1:
                recommendations.append(f"\n🎯 Supervised Learning Insights:")
                recommendations.append("  • Check if target classes separate in PCA/t-SNE plots")
                recommendations.append("  • Well-separated classes indicate good ML potential")
                recommendations.append("  • Overlapping classes may need feature engineering")
                
        except Exception as e:
            self.logger.warning(f"Error generating recommendations: {str(e)}")
            recommendations.append("⚠ Error generating specific recommendations")
            
        return recommendations
        
    def _calculate_reduction_score(self, results: Dict[str, Any]) -> float:
        """Calculate dimensionality reduction quality score (0-100)"""
        try:
            score = 0.0
            
            pca_results = results.get('pca_analysis', {})
            tsne_results = results.get('tsne_analysis', {})
            data_prep = results.get('data_preparation', {})
            
            # Base score
            score += 30
            
            # PCA quality scoring
            if 'components_for_90_variance' in pca_results:
                total_features = data_prep.get('processed_features', 1)
                comp_90 = pca_results['components_for_90_variance']
                reduction_ratio = (total_features - comp_90) / total_features
                
                if reduction_ratio > 0.7:  # >70% reduction possible
                    score += 25
                elif reduction_ratio > 0.5:  # >50% reduction
                    score += 20
                elif reduction_ratio > 0.3:  # >30% reduction  
                    score += 15
                else:
                    score += 10
            
            # t-SNE quality scoring
            if tsne_results.get('best_perplexity') and tsne_results.get('perplexity_results'):
                best_perp = tsne_results['best_perplexity']
                kl_div = tsne_results['perplexity_results'][best_perp]['kl_divergence']
                
                if kl_div < 1.0:
                    score += 25
                elif kl_div < 2.0:
                    score += 20
                elif kl_div < 3.0:
                    score += 15
                else:
                    score += 10
            
            # Data complexity bonus
            total_features = data_prep.get('processed_features', 0)
            if total_features > 50:
                score += 10
            elif total_features > 20:
                score += 5
                
            # Successful analysis bonus
            if 'pca_2d' in pca_results and tsne_results.get('perplexity_results'):
                score += 10
                
            return min(100.0, max(0.0, score))
            
        except Exception as e:
            self.logger.warning(f"Error calculating reduction score: {str(e)}")
            return 50.0