from datetime import datetime
from typing import Dict, List, Any, Optional
from pyspark.sql import functions as F

import json

class CK_PFM_InsightCollector:
    """
    Collects study metadata and insight values during notebook execution
    and inserts them into ck_pfm_insights_main and insight_values_Ndim tables.
    
    ✅ UPDATED to match actual table schemas in Databricks
    """
    
    def __init__(self):
        """Initialize the collector with empty containers"""
        # Main table data
        self.study_info = {}
        self.criteria_info = {}
        self.population_info = {}
        self.sample_info = {}
        
        # Insight values (will be a list of records)
        self.insight_values = []
        
    def set_study_info(self, study_ref: str, study_title: str, 
                       dafe_ck_team_pole: str, study_category: str):
        """Set study-level information"""
        self.study_info = {
            'Study_ref': study_ref,
            'Study_title': study_title,
            'Dafe_ck_team_pole': dafe_ck_team_pole,
            'Study_category': study_category
        }
        print(f"✓ Study info set: {study_ref}")
        
    def set_criteria_info(self, criteria_ref: str, criteria_label: str,
                         criteria_description: str, dimension: str, norm_applied: str):
        """Set criteria-level information"""
        self.criteria_info = {
            'Criteria_ref': criteria_ref,
            'Criteria_label': criteria_label,
            'Criteria_description': criteria_description,
            'Dimension': dimension,
            'Norm_applied': norm_applied
        }
        print(f"✓ Criteria info set: {criteria_ref}")
        
    def set_population_info(self, energy_category: str, segment: str, region_country: str):
        """Set population-level information"""
        self.population_info = {
            'Energy_category': energy_category,
            'Segment': segment,
            'region_country': region_country
        }
        print(f"✓ Population info set: {energy_category} - {segment}")
        
    def set_sample_info(self, data_source: str, vin_list: List[str],
                       period_start: str, period_end: str,
                       signal_names: List[str], final_stats: Dict[str, Any]):
        """Set sample-level information"""
        self.sample_info = {
            'Data_source': data_source,
            'Vin_list': json.dumps(vin_list),  # Convert to JSON string
            'Period_date_start': period_start,
            'Period_date_end': period_end,
            'List_of_signal_names': json.dumps(signal_names),  # Convert to JSON string
            'Final_stats_min_max_moyenne': json.dumps(final_stats)  # Convert to JSON string
        }
        print(f"✓ Sample info set: {len(vin_list)} VINs, {len(signal_names)} signals")
        
    def set_result_criteria_ref(self, result_criteria_ref: str):
        """Set result criteria reference"""
        self.result_criteria_ref = result_criteria_ref
        print(f"✓ Result criteria ref set: {result_criteria_ref}")
        
    def add_insight_value(self, ref_criteria: str, result_criteria_ref: str,
                         dimension: str, label_dim1: str, size_dim1: int, val_dim1: float,
                         criteria_result_value: float, criteria_result_unit: str,
                         label_dim2: Optional[str] = None, size_dim2: Optional[int] = None, 
                         val_dim2: Optional[float] = None,
                         label_dim3: Optional[str] = None, size_dim3: Optional[int] = None, 
                         val_dim3: Optional[float] = None):
        """Add a single insight value record"""
        insight = {
            # ✅ FIXED: Changed RefPFMstudy -> Study_ref
            'Study_ref': self.study_info.get('Study_ref'),
            'Ref_criteria': ref_criteria,
            'Result_criteria_ref': result_criteria_ref,
            'dimension': dimension,
            'Label_dim1': label_dim1,
            # ✅ FIXED: Convert to string to match table schema
            'Size_dim1': str(size_dim1),
            'Val_dim1': str(val_dim1),
            'Label_dim2': label_dim2,
            'Size_dim2': str(size_dim2) if size_dim2 is not None else None,
            'Val_dim2': str(val_dim2) if val_dim2 is not None else None,
            'Label_dim3': label_dim3,
            'Size_dim3': str(size_dim3) if size_dim3 is not None else None,
            'Val_dim3': str(val_dim3) if val_dim3 is not None else None,
            'Criteria_result_value': criteria_result_value,
            'Criteria_result_unit': criteria_result_unit
        }
        self.insight_values.append(insight)
        
    def add_insight_values_from_dataframe(self, df_insights: 'DataFrame', 
                                         ref_criteria: str, result_criteria_ref: str):
        """
        Bulk add insight values from a Spark DataFrame
        
        Expected DataFrame columns:
        - dimension, Label_dim1, Size_dim1, Val_dim1
        - Label_dim2, Size_dim2, Val_dim2 (optional)
        - Label_dim3, Size_dim3, Val_dim3 (optional)
        - Criteria_result_value, Criteria_result_unit
        """
        study_ref = self.study_info.get('Study_ref')
        
        # ✅ FIXED: Add Study_ref (not RefPFMstudy), and convert types to match schema
        df_with_refs = df_insights.withColumn("Study_ref", F.lit(study_ref)) \
                                  .withColumn("Ref_criteria", F.lit(ref_criteria)) \
                                  .withColumn("Result_criteria_ref", F.lit(result_criteria_ref)) \
                                  .withColumn("Size_dim1", F.col("Size_dim1").cast("string")) \
                                  .withColumn("Val_dim1", F.col("Val_dim1").cast("string")) \
                                  .withColumn("Size_dim2", F.col("Size_dim2").cast("string")) \
                                  .withColumn("Val_dim2", F.col("Val_dim2").cast("string")) \
                                  .withColumn("Size_dim3", F.col("Size_dim3").cast("string")) \
                                  .withColumn("Val_dim3", F.col("Val_dim3").cast("string"))
        
        # Convert to list of dictionaries
        insights_list = df_with_refs.toPandas().to_dict('records')
        self.insight_values.extend(insights_list)
        
        print(f"✓ Added {len(insights_list)} insight values from DataFrame")
        
    def insert_to_tables(self, catalog: str = "dafe_dev", schema: str = "sf96268"):
        """
        Insert all collected data into both tables
        
        Returns:
            Tuple[bool, str]: (success, message)
        """
        try:
            # Validate required fields
            if not self.study_info:
                return False, "❌ Study info not set. Call set_study_info() first."
            
            # ================================================================
            # 1. INSERT INTO ck_pfm_insights_main (Main Table)
            # ================================================================
            
            main_record = {
                **self.study_info,
                **self.criteria_info,
                **self.population_info,
                **self.sample_info,
                'Result_criteria_ref': getattr(self, 'result_criteria_ref', None),
                # ✅ Creation_date and Modification_date handled by DEFAULT in table
            }
            
            df_main = spark.createDataFrame([main_record])
            
            df_main.write.mode("append").saveAsTable(f"{catalog}.{schema}.ck_pfm_insights_main")
            
            print(f"✓ Inserted into {catalog}.{schema}.ck_pfm_insights_main")
            
            # ================================================================
            # 2. INSERT INTO insight_values_Ndim (Values Table)
            # ================================================================
            
            if self.insight_values:
                df_values = spark.createDataFrame(self.insight_values)
                
                # ✅ FIXED: Table name is insight_values_Ndim (lowercase 'insight')
                df_values.write.mode("append").saveAsTable(f"{catalog}.{schema}.insight_values_Ndim")
                
                print(f"✓ Inserted {len(self.insight_values)} rows into {catalog}.{schema}.insight_values_Ndim")
            else:
                print("⚠ No insight values to insert")
            
            # ================================================================
            # 3. RESET COLLECTOR FOR NEXT STUDY
            # ================================================================
            self._reset()
            
            return True, "✓ Data successfully inserted into both tables"
            
        except Exception as e:
            error_msg = f"❌ Error inserting data: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return False, error_msg
    
    def _reset(self):
        """Reset collector for next study"""
        self.study_info = {}
        self.criteria_info = {}
        self.population_info = {}
        self.sample_info = {}
        self.insight_values = []
        print("✓ Collector reset for next study")
        
    def get_summary(self):
        """Print summary of collected data"""
        print("\n" + "="*60)
        print("COLLECTED DATA SUMMARY")
        print("="*60)
        print(f"Study: {self.study_info.get('Study_ref', 'NOT SET')}")
        print(f"Criteria: {self.criteria_info.get('Criteria_ref', 'NOT SET')}")
        print(f"Population: {self.population_info.get('Energy_category', 'NOT SET')}")
        
        vin_list_str = self.sample_info.get('Vin_list', '[]')
        try:
            vin_count = len(json.loads(vin_list_str))
        except:
            vin_count = 0
        print(f"Sample: {vin_count} VINs")
        print(f"Insight values: {len(self.insight_values)} records")
        print("="*60 + "\n")

collector = CK_PFM_InsightCollector()
