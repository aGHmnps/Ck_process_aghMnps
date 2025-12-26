import json
import pandas as pd
from typing import List, Dict, Optional, Any
from datetime import datetime


class PFMStructureBuilder:
    """
    Builder pattern for preparing PFM storage structure matching the schema of:
    - Table 1: ck_pfm_insights_main (metadata + modeling results)
    - Table 2: insight_values_Ndim (matrix values)
    
    Usage:
        builder = PFMStructureBuilder()
        builder.set_criteria("avg_duration_zev", "Average ZEV duration", "H0D", "7y_150k_km")
        builder.set_population(vins, "MHEV", "Segment_D", "Europe")
        builder.set_data_source("SMARTDATA", signals, "06/04/2023", "18/12/2023")
        builder.add_modeling_results(combined_df, "avg_duration_zev")
        result = builder.build()
    """
    
    def __init__(self):
        """Initialize builder with empty structure"""
        self._data = {
            # Required fields
            "Study_title": None,
            "Criteria_ref": "",  # Auto-generated or ignored
            "Criteria_label": None,
            "Criteria_description": None,
            "Norm_applied": None,
            "Energy_category": None,
            "Segment": None,
            "region_country": None,
            "Data_source": None,
            "Vin_list": None,
            "Period_date_start": None,
            "Period_date_end": None,
            "List_of_signal_names": None,
            
            # Modeling results (optional - default to None)
            "Final_stats_min_max_moyenne": "[]",
            "loi": None,
            "parameters": None,
            "class_index": None,
            "R2": None,
            "client1pct": None,
            "client50pct": None,
            "client99pct": None,
            "sample_client1pct": None,
            "sample_client50pct": None,
            "sample_client99pct": None,
            
            # Matrix (optional)
            "dfmatrix": None
        }
        
        self._dimension = None
        self._errors = []
    
    
    def set_study_title(self, title: str):
        """
        Set study title
        
        Parameters:
        -----------
        title : str
            Study title (will be used to generate Study_ref)
        """
        if not title or not isinstance(title, str):
            self._errors.append("Study title must be a non-empty string")
        else:
            self._data["Study_title"] = title.strip()
        return self
    
    
    def set_criteria(self, label: str, description: str, dimension: str, norm_applied: str):
        """
        Set criteria information
        
        Parameters:
        -----------
        label : str
            Criteria name (e.g., "avg_duree_fonctionnement_mode_zev_7y_150k_km")
        description : str
            Brief description of the criteria
        dimension : str
            One of: H0D, H1D, H2D, H3D
        norm_applied : str
            Norm applied (e.g., "7y_150k_km", "1y_60k_km")
        """
        # Validate dimension
        valid_dimensions = ["H0D", "H1D", "H2D", "H3D"]
        if dimension not in valid_dimensions:
            self._errors.append(f"Invalid dimension '{dimension}'. Must be one of {valid_dimensions}")
        
        self._data["Criteria_label"] = label
        self._data["Criteria_description"] = description
        self._data["Norm_applied"] = norm_applied
        self._dimension = dimension
        
        return self
    
    
    def set_population(self, vins: List[str], energy_category: str, segment: str, region_country: str):
        """
        Set population/sample information
        
        Parameters:
        -----------
        vins : list of str
            List of VINs in the sample
        energy_category : str
            Energy category (e.g., "BEV", "ICE", "MHEV", "PHEV")
        segment : str
            Vehicle segment (e.g., "Segment_A", "Segment_D")
        region_country : str
            Region or country (e.g., "Europe", "France")
        """
        if not vins or len(vins) == 0:
            self._errors.append("VIN list cannot be empty")
        else:
            # Convert list to comma-separated string
            self._data["Vin_list"] = ",".join(str(v) for v in vins)
        
        self._data["Energy_category"] = energy_category
        self._data["Segment"] = segment
        self._data["region_country"] = region_country
        
        return self
    
    
    def set_data_source(self, source: str, signals: List[str], start_date: str, end_date: str):
        """
        Set data source and time period information
        
        Parameters:
        -----------
        source : str
            Data source (e.g., "SMARTDATA", "IVDP", "CVDP")
        signals : list of str
            List of signal names used in the study
        start_date : str
            Period start date (format: "DD/MM/YYYY" or "YYYY-MM-DD")
        end_date : str
            Period end date (format: "DD/MM/YYYY" or "YYYY-MM-DD")
        """
        self._data["Data_source"] = source
        
        if signals:
            # Convert list to comma-separated string
            self._data["List_of_signal_names"] = ",".join(str(s) for s in signals)
        
        # Store dates as-is (strings) - Delta will handle conversion
        self._data["Period_date_start"] = start_date if start_date else None
        self._data["Period_date_end"] = end_date if end_date else None
        
        return self
    
    
    def add_modeling_results(self, combined_df, target_criteria: str):
        """
        Extract and add modeling results from combined_df
        
        Parameters:
        -----------
        combined_df : DataFrame or None
            Spark/Pandas DataFrame containing modeling results
        target_criteria : str
            Criteria name to filter for in the DataFrame
        
        Returns:
        --------
        self : for method chaining
        """
        if combined_df is None:
            print(f"⚠️  No modeling DataFrame provided - all modeling fields will be None")
            return self
        
        try:
            # Try Spark DataFrame first
            if hasattr(combined_df, 'filter'):
                row_data = combined_df.filter(combined_df.criteria == target_criteria).first()
            # Fallback to Pandas
            elif hasattr(combined_df, 'loc'):
                filtered = combined_df[combined_df['criteria'] == target_criteria]
                row_data = filtered.iloc[0] if len(filtered) > 0 else None
            else:
                print(f"⚠️  Unknown DataFrame type - skipping modeling results")
                return self
            
            if row_data:
                # Extract final stats
                try:
                    final_stats = [
                        float(row_data['data_min']) if row_data['data_min'] is not None else None,
                        float(row_data['data_max']) if row_data['data_max'] is not None else None,
                        float(row_data['data_mean']) if row_data['data_mean'] is not None else None
                    ]
                    self._data["Final_stats_min_max_moyenne"] = json.dumps(final_stats)
                except Exception as e:
                    print(f"⚠️  Error extracting final stats: {e}")
                    self._data["Final_stats_min_max_moyenne"] = "[]"
                
                # Extract modeling parameters
                self._data["loi"] = str(row_data['loi']) if row_data['loi'] is not None else None
                self._data["parameters"] = str(row_data['parameters']) if row_data['parameters'] is not None else None
                
                # Cast types properly
                try:
                    self._data["class_index"] = int(row_data['class_index']) if row_data['class_index'] is not None else None
                except (ValueError, TypeError):
                    self._data["class_index"] = None
                
                try:
                    self._data["R2"] = float(row_data['R2']) if row_data['R2'] is not None else None
                except (ValueError, TypeError):
                    self._data["R2"] = None
                
                # Extract percentiles with proper casting
                percentile_fields = [
                    'client1%', 'client50%', 'client99%',
                    'sample_client1%', 'sample_client50%', 'sample_client99%'
                ]
                percentile_mapping = {
                    'client1%': 'client1pct',
                    'client50%': 'client50pct',
                    'client99%': 'client99pct',
                    'sample_client1%': 'sample_client1pct',
                    'sample_client50%': 'sample_client50pct',
                    'sample_client99%': 'sample_client99pct'
                }
                
                for src_field, dest_field in percentile_mapping.items():
                    try:
                        value = row_data[src_field] if src_field in row_data else None
                        self._data[dest_field] = float(value) if value is not None else None
                    except (ValueError, TypeError, KeyError):
                        self._data[dest_field] = None
                
                print(f"✅ Successfully populated modeling results for '{target_criteria}'")
            else:
                print(f"⚠️  No data found in combined_df for criteria: '{target_criteria}'")
        
        except Exception as e:
            print(f"⚠️  Error extracting modeling results: {e}")
            self._errors.append(f"Modeling extraction failed: {e}")
        
        return self
    
    
    def add_matrix(self, nested_matrix: Dict, boundaries_dict: Dict):
        """
        Add matrix data with boundaries (for H1D, H2D, H3D)
        
        Parameters:
        -----------
        nested_matrix : dict
            3-level nested dictionary: {classA: {classB: {classC: value}}}
        boundaries_dict : dict
            Boundaries metadata with keys: x_axis, y_axis, x_bins, y_bins
            Example: {
                "x_axis": "regime_mel",
                "y_axis": "couple_mel",
                "x_bins": "min=-90;max=10410;step=50",
                "y_bins": "min=-4000;max=5000;step=5"
            }
        """
        if self._dimension == "H0D":
            print(f"⚠️  Dimension is H0D - matrix will be ignored")
            return self
        
        if not nested_matrix or not isinstance(nested_matrix, dict):
            self._errors.append("Matrix must be a non-empty dictionary")
            return self
        
        # Wrap matrix with boundaries metadata
        matrix_with_boundaries = {
            "x_axis": boundaries_dict.get("x_axis", "unknown"),
            "y_axis": boundaries_dict.get("y_axis", "unknown"),
            "x_bins": boundaries_dict.get("x_bins", ""),
            "y_bins": boundaries_dict.get("y_bins", ""),
            "matrix": nested_matrix
        }
        
        # Convert to JSON string
        self._data["dfmatrix"] = json.dumps(matrix_with_boundaries)
        
        print(f"✅ Matrix added with boundaries (dimension: {self._dimension})")
        
        return self
    
    
    def validate(self) -> bool:
        """
        Validate that all required fields are populated
        
        Returns:
        --------
        bool : True if valid, False otherwise
        """
        required_fields = [
            "Study_title",
            "Criteria_label",
            "Criteria_description",
            "Norm_applied",
            "Energy_category",
            "Segment",
            "region_country",
            "Data_source",
            "Vin_list",
            "List_of_signal_names"
        ]
        
        for field in required_fields:
            if self._data[field] is None or self._data[field] == "":
                self._errors.append(f"Required field '{field}' is missing")
        
        if self._errors:
            print(f"❌ Validation failed with {len(self._errors)} error(s):")
            for error in self._errors:
                print(f"   - {error}")
            return False
        
        print(f"✅ Validation passed")
        return True
    
    
    def build(self) -> Dict[str, Any]:
        """
        Build and return the final structure as a dictionary
        
        Returns:
        --------
        dict : Complete structure ready for store_insights_hnd(level="PFM")
        
        Raises:
        -------
        ValueError : If validation fails
        """
        if not self.validate():
            raise ValueError(f"Cannot build structure - validation failed with {len(self._errors)} error(s)")
        
        print(f"✅ Structure built successfully for criteria: '{self._data['Criteria_label']}'")
        
        # Return a copy to prevent external modifications
        return dict(self._data)
    
    
    def get_errors(self) -> List[str]:
        """Return list of validation errors"""
        return list(self._errors)
    
    
    def reset(self):
        """Reset builder to initial state"""
        self.__init__()
        return self


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def example_usage():
    """
    Example of how to use the PFMStructureBuilder
    """
    
    # Sample data
    criteria_label = "avg_duree_fonctionnement_mode_zev_7y_150k_km"
    criteria_description = "average duration in seconds of time that the vehicle spent in mode ZEV"
    dimension = "H2D"
    norm_applied = "7y_150k_km"
    
    vins = ['VF3MRHPYEPS013468', 'VF3MRHPYEPS013469', 'VF3MRHPYEPS013470']
    energy_category = "MHEV"
    segment = "Segment_D"
    region_country = "Europe"
    
    data_source = "SMARTDATA"
    signals = ['TIMESTAMP', 'LifeTimeCycle_ID', 'ETAT_PRINCIP_SEV:R', 'KILOMETRAGE:C']
    start_date = '06/04/2023'
    end_date = '18/12/2023'
    
    # Build structure
    builder = PFMStructureBuilder()
    
    result = (builder
        .set_study_title("[PWT] MHEV MEL IT116R Study")
        .set_criteria(criteria_label, criteria_description, dimension, norm_applied)
        .set_population(vins, energy_category, segment, region_country)
        .set_data_source(data_source, signals, start_date, end_date)
        .add_modeling_results(combined_df, criteria_label)  # combined_df from your code
        .add_matrix(nested_matrix, {
            "x_axis": "regime_mel",
            "y_axis": "couple_mel",
            "x_bins": "min=-90;max=10410;step=50",
            "y_bins": "min=-4000;max=5000;step=5"
        })
        .build()
    )
    
    # Now use with your storage function
    study_ref = store_insights_hnd(result, level="PFM", overwrite=False)
    
    print(f"✅ Stored with Study_ref: {study_ref}")


# ============================================================================
# HELPER: Quick validation without building
# ============================================================================

def preview_structure(builder: PFMStructureBuilder):
    """Preview the structure without validation"""
    import pprint
    print("\n📋 Current Structure Preview:")
    pprint.pprint(builder._data, width=120)
    print(f"\n⚠️  Errors: {len(builder._errors)}")
    for err in builder._errors:
        print(f"   - {err}")
