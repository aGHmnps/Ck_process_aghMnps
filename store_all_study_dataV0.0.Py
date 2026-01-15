import json
import pandas as pd
from typing import List, Dict, Optional, Any


class PFMBatchStructureBuilder:
    """
    Batch builder for multiple criteria in one PFM study.
    
    ✅ Updated for new schema with Population_description and Sample_definition as JSON strings
    """
    
    def __init__(self):
        """Initialize batch builder"""
        self._study_metadata = {
            "Study_title": None,
            "Energy_category": None,
            "Segment": None,
            "region_country": None,
            "Data_source": None,
            "Vin_list": None,
            "Period_date_start": None,
            "Period_date_end": None,
            "List_of_signal_names": None
        }
        
        self._criteria_rows = []
        self._errors = []
    
    
    def set_study_title(self, title: str):
        """Set study title (shared across all criteria)"""
        if not title or not isinstance(title, str):
            self._errors.append("Study title must be a non-empty string")
        else:
            self._study_metadata["Study_title"] = title.strip()
        return self
    
    
    def set_population(self, vins: List[str], energy_category: str, segment: str, region_country: str):
        """Set population metadata (shared across all criteria)"""
        if not vins or len(vins) == 0:
            self._errors.append("VIN list cannot be empty")
        else:
            self._study_metadata["Vin_list"] = ",".join(str(v) for v in vins)
        
        self._study_metadata["Energy_category"] = energy_category
        self._study_metadata["Segment"] = segment
        self._study_metadata["region_country"] = region_country
        
        return self
    
    
    def set_data_source(self, source: str, signals: List[str], start_date: str, end_date: str):
        """Set data source metadata (shared across all criteria)"""
        self._study_metadata["Data_source"] = source
        
        if signals:
            self._study_metadata["List_of_signal_names"] = ",".join(str(s) for s in signals)
        else :
            self._study_metadata["List_of_signal_names"] = ""
        
        self._study_metadata["Period_date_start"] = start_date if start_date else ""
        self._study_metadata["Period_date_end"] = end_date if end_date else ""
        
        return self
    
    
    def add_criteria(self, 
                     label: str, 
                     description: str, 
                     dimension: str, 
                     norm_applied: str,
                     modeling_df=None,
                     matrix: Optional[Dict] = None,
                     boundaries: Optional[Dict] = None):
        """Add a criteria row to the batch"""
        valid_dimensions = ["H0D", "H1D", "H2D", "H3D"]
        if dimension not in valid_dimensions:
            self._errors.append(f"Invalid dimension '{dimension}' for criteria '{label}'")
            return self
        
        criteria_row = {
            "Criteria_label": label,
            "Criteria_description": description,
            "Norm_applied": norm_applied,
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
            "dfmatrix": None,
            "dimension": dimension,
            
        }
        
        if modeling_df is not None:
            self._extract_modeling_results(criteria_row, modeling_df, label)
        
        if dimension != "H0D" and matrix is not None:
            self._add_matrix_to_row(criteria_row, matrix, boundaries)
        
        self._criteria_rows.append(criteria_row)
        
        print(f"✅ Added criteria: {label} (dimension: {dimension})")
        
        return self
    
    
    def _extract_modeling_results(self, criteria_row: Dict, modeling_df, target_criteria: str):
        """Extract modeling results for a specific criteria (internal helper)"""
        try:
            if hasattr(modeling_df, 'filter'):
                row_data = modeling_df.filter(modeling_df.criteria == target_criteria).first()
            elif hasattr(modeling_df, 'loc'):
                filtered = modeling_df[modeling_df['criteria'] == target_criteria]
                row_data = filtered.iloc[0] if len(filtered) > 0 else None
            else:
                print(f"⚠️  Unknown DataFrame type for '{target_criteria}'")
                return
            
            if row_data:
                try:
                    final_stats = [
                        float(row_data['data_min']) if row_data['data_min'] is not None else None,
                        float(row_data['data_max']) if row_data['data_max'] is not None else None,
                        float(row_data['data_mean']) if row_data['data_mean'] is not None else None
                    ]
                    criteria_row["Final_stats_min_max_moyenne"] = json.dumps(final_stats)
                except Exception as e:
                    print(f"⚠️  Error extracting stats for '{target_criteria}': {e}")
                
                criteria_row["loi"] = str(row_data['loi']) if row_data['loi'] is not None else None
                criteria_row["parameters"] = str(row_data['parameters']) if row_data['parameters'] is not None else None
                
                try:
                    criteria_row["class_index"] = int(row_data['class_index']) if row_data['class_index'] is not None else None
                except (ValueError, TypeError):
                    criteria_row["class_index"] = None
                
                try:
                    criteria_row["R2"] = float(row_data['R2']) if row_data['R2'] is not None else None
                except (ValueError, TypeError):
                    criteria_row["R2"] = None
                
                percentile_mapping = {
                    'client1%': 'client1pct',
                    'client50%': 'client50pct',
                    'client99%': 'client99pct',
                    'sample_client1%': 'sample_client1pct',
                    'sample_client50%': 'sample_client50pct',
                    'sample_client99%': 'sample_client99pct'
                }
                
                for src, dest in percentile_mapping.items():
                    try:
                        value = row_data[src] if src in row_data else None
                        criteria_row[dest] = float(value) if value is not None else None
                    except (ValueError, TypeError, KeyError):
                        criteria_row[dest] = None
                
                print(f"  ✅ Modeling results extracted for '{target_criteria}'")
            else:
                print(f"  ⚠️  No modeling data found for '{target_criteria}'")
        
        except Exception as e:
            print(f"  ⚠️  Error extracting modeling for '{target_criteria}': {e}")
    
    
    def _add_matrix_to_row(self, criteria_row: Dict, matrix: Dict, boundaries: Optional[Dict]):
        """Add matrix with boundaries to criteria row (internal helper)"""
        if not matrix or not isinstance(matrix, dict):
            print(f"  ⚠️  Invalid matrix for '{criteria_row['Criteria_label']}'")
            return
        
        if boundaries:
            matrix_with_boundaries = {
                "x_axis": boundaries.get("x_axis", "unknown"),
                "y_axis": boundaries.get("y_axis", "unknown"),
                "x_bins": boundaries.get("x_bins", ""),
                "y_bins": boundaries.get("y_bins", ""),
                "matrix": matrix
            }
        else:
            matrix_with_boundaries = {"matrix": matrix}
        
        criteria_row["dfmatrix"] = json.dumps(matrix_with_boundaries)
        print(f"  ✅ Matrix added for '{criteria_row['Criteria_label']}'")
    
    
    def validate(self) -> bool:
        """Validate study metadata and criteria rows"""
        required_metadata = [
            "Study_title", "Energy_category", "Segment", "region_country",
            "Data_source", "Vin_list", "List_of_signal_names"
        ]
        
        for field in required_metadata:
            if self._study_metadata[field] is None : 
                self._errors.append(f"Required study metadata '{field}' is missing")
        
        if len(self._criteria_rows) == 0:
            self._errors.append("No criteria added - use add_criteria() to add at least one")
        
        if self._errors:
            print(f"❌ Validation failed with {len(self._errors)} error(s):")
            for error in self._errors:
                print(f"   - {error}")
            return False
        
        print(f"✅ Validation passed ({len(self._criteria_rows)} criteria)")
        return True
    
    
    def build(self) -> Dict[str, Any]:
        """Build and return the batch structure"""
        if not self.validate():
            raise ValueError(f"Cannot build - validation failed with {len(self._errors)} error(s)")
        
        result = {
            "study_metadata": dict(self._study_metadata),
            "criteria_rows": [dict(row) for row in self._criteria_rows]
        }
        
        print(f"✅ Batch structure built: {len(self._criteria_rows)} criteria for study '{self._study_metadata['Study_title']}'")
        
        return result
    
    
    def get_errors(self) -> List[str]:
        """Return list of validation errors"""
        return list(self._errors)
    
    
    def reset(self):
        """Reset builder to initial state"""
        self.__init__()
        return self

import hashlib, re, string, secrets
import pandas as pd
import json
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from typing import Dict

def generate_study_ref(study_title):
    """
    Generate unique Study_ref from study title
    Extracts prefix from [brackets] or uses 'PFM' default, adds random 6-char suffix
    """
    if not study_title:
        study_title = f"STUDY_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    
    match = re.search(r'\[([^\]]+)\]', study_title)
    
    if match:
        prefix = match.group(1).strip().upper()
    else:
        prefix = "PFM"
    
    suffix = ''.join(secrets.choice(string.ascii_uppercase + string.digits) for _ in range(6))
    
    return f"{prefix}_{suffix}"


def explode_matrix_to_dataframe(dfmatrix_json, study_ref, criteria_label):
    """
    Explode nested 3D JSON matrix into flat dataframe for Ndim table
    
    Expected structure (two layers):
    {
      "x_axis": "regime_mel",
      "y_axis": "couple_mel",
      "x_bins": "min=-90;max=10410;step=50",
      "y_bins": "min=-4000;max=5000;step=5",
      "matrix": {
        "classA": {"classB": {"classC": value}}
      }
    }
    
    Returns: pandas DataFrame with columns [Study_ref, Criteria_label, Dimension, 
                                           Dimensions_boudaries, ClassA, ClassB, ClassC, 
                                           Correspondant_value]
    """
    if not dfmatrix_json or dfmatrix_json in ["", "none", "None", None]:
        return None
    
    try:
        if isinstance(dfmatrix_json, str):
            wrapper_data = json.loads(dfmatrix_json)
        else:
            wrapper_data = dfmatrix_json
    except Exception as e:
        print(f"⚠️  Error parsing matrix JSON: {e}")
        return None
    
    if not isinstance(wrapper_data, dict):
        return None
    
    x_axis = wrapper_data.get("x_axis", "unknown")
    y_axis = wrapper_data.get("y_axis", "unknown")
    x_bins = wrapper_data.get("x_bins", "")
    y_bins = wrapper_data.get("y_bins", "")
    
    dimensions_boundaries = f"x_axis={x_axis}; y_axis={y_axis}; x_bins=[{x_bins}]; y_bins=[{y_bins}]"
    
    matrix_data = wrapper_data.get("matrix")
    
    if not matrix_data or not isinstance(matrix_data, dict) or len(matrix_data) == 0:
        print(f"⚠️  No valid 'matrix' key found in dfmatrix")
        return None
    
    records = []
    
    for classA, level2 in matrix_data.items():
        if not isinstance(level2, dict):
            continue
            
        for classB, level3 in level2.items():
            if not isinstance(level3, dict):
                continue
                
            for classC, value in level3.items():
                try:
                    records.append({
                        "Study_ref": study_ref,
                        "Criteria_label": criteria_label,
                        "Dimension": detect_dimension(classA, classB, classC),
                        "Dimensions_boudaries": dimensions_boundaries,
                        "ClassA": str(classA),
                        "ClassB": str(classB),
                        "ClassC": str(classC),
                        "Correspondant_value": float(value)
                    })
                except (ValueError, TypeError) as e:
                    print(f"⚠️  Skipping invalid value at [{classA}][{classB}][{classC}]: {e}")
                    continue
    
    if not records:
        return None
    
    return pd.DataFrame(records)


def detect_dimension_from_df(df):
    """
    Detect dimension from a DataFrame by checking classA, classB, classC columns
    """
    from pyspark.sql import DataFrame as SparkDataFrame
    import pandas as pd
    
    required_cols = ["classA", "classB", "classC"]
    
    if isinstance(df, SparkDataFrame):
        df_cols = df.columns
        if not all(col in df_cols for col in required_cols):
            raise ValueError(f"DataFrame must contain columns: {required_cols}. Found: {df_cols}")
        
        first_row = df.filter(
            (df.classA.isNotNull()) | 
            (df.classB.isNotNull()) | 
            (df.classC.isNotNull())
        ).first()
        
        if first_row is None:
            return "H0D"
        
        classA = first_row.classA if hasattr(first_row, 'classA') else None
        classB = first_row.classB if hasattr(first_row, 'classB') else None
        classC = first_row.classC if hasattr(first_row, 'classC') else None
        
    elif isinstance(df, pd.DataFrame):
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain columns: {required_cols}. Found: {list(df.columns)}")
        
        non_null_rows = df.dropna(subset=required_cols, how='all')
        
        if len(non_null_rows) == 0:
            return "H0D"
        
        first_row = non_null_rows.iloc[0]
        classA = first_row['classA']
        classB = first_row['classB']
        classC = first_row['classC']
    
    else:
        raise TypeError("DataFrame must be either Spark DataFrame or Pandas DataFrame")
    
    return detect_dimension(classA, classB, classC)


def detect_dimension(classA, classB, classC):
    """
    Detect dimension based on which classes are not 'none'
    """
    classA_filled = str(classA).lower() not in ["none", "", "null"] and classA is not None
    classB_filled = str(classB).lower() not in ["none", "", "null"] and classB is not None
    classC_filled = str(classC).lower() not in ["none", "", "null"] and classC is not None
    # print(f"class_A filled {classA_filled}, class_B filled {classB_filled}, class_C filled {classC_filled}")

    if classA_filled and classB_filled and classC_filled:
        return "H3D"
    elif classA_filled and classB_filled:
        return "H2D"
    elif classA_filled:
        return "H1D"
    else:
        return "H0D"


def auto_detect_dimensions(df, criteria_list):
    """
    Automatically detect dimension for each criteria in the list
    """
    from pyspark.sql import DataFrame as SparkDataFrame
    import pandas as pd
    
    dimensions_map = {}
    required_cols = ["classA", "classB", "classC"]
    
    if isinstance(df, SparkDataFrame):
        df_cols = df.columns
        
        if not all(col in df_cols for col in required_cols):
            raise ValueError(f"DataFrame must contain columns: {required_cols}. Found: {df_cols}")
        
        missing_criteria = [c for c in criteria_list if c not in df_cols]
        if missing_criteria:
            raise ValueError(f"Criteria not found in DataFrame: {missing_criteria}")
        
        print(f"🔍 Detecting dimensions for {len(criteria_list)} criteria...")
        
        for criteria in criteria_list:
            try:
                criteria_df = df.filter(df[criteria].isNotNull())
                
                non_none_rows = criteria_df.filter(
                    (F.lower(F.col("classA")) != "none") | 
                    (F.lower(F.col("classB")) != "none") | 
                    (F.lower(F.col("classC")) != "none")
                )
                
                if non_none_rows.count() == 0:
                    dimensions_map[criteria] = "H0D"
                    print(f"  ✅ {criteria}: H0D (all classes are 'none')")
                    continue
                
                # Get first row with actual class data
                first_row = non_none_rows.first()
                
                classA = first_row.classA if hasattr(first_row, 'classA') else None
                classB = first_row.classB if hasattr(first_row, 'classB') else None
                classC = first_row.classC if hasattr(first_row, 'classC') else None
                
                dimension = detect_dimension(classA, classB, classC)
                dimensions_map[criteria] = dimension
                
                # print(f"  ✅ {criteria}: {dimension} (classA={classA}, classB={classB}, classC={classC})")
                
            except Exception as e:
                print(f"  ⚠️  Error detecting dimension for '{criteria}': {e}")
                dimensions_map[criteria] = "H0D"
            
    elif isinstance(df, pd.DataFrame):
        df_cols = df.columns.tolist()
        
        if not all(col in df_cols for col in required_cols):
            raise ValueError(f"DataFrame must contain columns: {required_cols}. Found: {df_cols}")
        
        missing_criteria = [c for c in criteria_list if c not in df_cols]
        if missing_criteria:
            raise ValueError(f"Criteria not found in DataFrame: {missing_criteria}")
        
        print(f"🔍 Detecting dimensions for {len(criteria_list)} criteria...")
        
        for criteria in criteria_list:
            try:
                criteria_df = df[df[criteria].notna()]
                
                if len(criteria_df) == 0:
                    dimensions_map[criteria] = "H0D"
                    print(f"  ✅ {criteria}: H0D (no data)")
                    continue
                
                # ✅ FIXED: Filter out rows where ALL classes are "none"
                non_none_rows = criteria_df[
                    (criteria_df['classA'].str.lower() != 'none') | 
                    (criteria_df['classB'].str.lower() != 'none') | 
                    (criteria_df['classC'].str.lower() != 'none')
                ]
                
                if len(non_none_rows) == 0:
                    dimensions_map[criteria] = "H0D"
                    print(f"  ✅ {criteria}: H0D (all classes are 'none')")
                    continue
                
                first_row = non_none_rows.iloc[0]
                classA = first_row['classA']
                classB = first_row['classB']
                classC = first_row['classC']
                
                dimension = detect_dimension(classA, classB, classC)
                dimensions_map[criteria] = dimension
                
                # print(f"  ✅ {criteria}: {dimension} (classA={classA}, classB={classB}, classC={classC})")
                
            except Exception as e:
                print(f"  ⚠️  Error detecting dimension for '{criteria}': {e}")
                dimensions_map[criteria] = "H0D"
    
    else:
        raise TypeError("DataFrame must be either Spark DataFrame or Pandas DataFrame")
    
    print(f"✅ Dimension detection complete: {len(dimensions_map)} criteria mapped")
    
    return dimensions_map


def store_pfm_insights(df, table_name, overwrite=False, add_timestamps=True):
    """
    Simple function to store a dataframe to a Delta table
    """
    spark = SparkSession.builder.getOrCreate()
    spark.sparkContext.setCheckpointDir("/tmp/spark-checkpoints")
    
    if isinstance(df, pd.DataFrame):
        spark_df = spark.createDataFrame(df)
    else:
        spark_df = df
    
    if add_timestamps:
        df_with_dt = (
            spark_df
            .withColumn("Creation_date", F.current_timestamp())
            .withColumn("Modification_date", F.current_timestamp())
        )
    else:
        df_with_dt = spark_df
    
    if spark.catalog.tableExists(table_name):
        if overwrite:
            if "Study_ref" in df_with_dt.columns:
                study_refs = [row.Study_ref for row in df_with_dt.select("Study_ref").distinct().collect()]
                existing = spark.read.table(table_name)
                filtered = existing.filter(~F.col("Study_ref").isin(study_refs))
                df_final = filtered.union(df_with_dt)
                df_final.write.mode("overwrite").option("mergeSchema", "true").format("delta").saveAsTable(table_name)
            else:
                df_with_dt.write.mode("append").option("mergeSchema", "true").format("delta").saveAsTable(table_name)
        else:
            df_with_dt.write.mode("append").option("mergeSchema", "true").format("delta").saveAsTable(table_name)
    else:
        df_with_dt.write.mode("overwrite").option("overwriteSchema", "true").format("delta").saveAsTable(table_name)
    
    print(f"✅ Stored {df_with_dt.count()} rows to {table_name}")


def store_trip_insights(df_new, table_name, study_title=None, overwrite=False):
    """Store trip-level insights"""
    spark = SparkSession.builder.getOrCreate()
    spark.sparkContext.setCheckpointDir("/tmp/spark-checkpoints")

    key_cols = ["DRIVER_ID", "TRIP_ID", "criteria"]

    df_new = (
        df_new
        .withColumn("dt", F.current_date())
        .withColumn("study_title", F.lit(study_title))
    )

    if not spark.catalog.tableExists(table_name):
        df_new.write.format("delta") \
            .option("mergeSchema", "true") \
            .option("overwriteSchema", "true") \
            .partitionBy("DRIVER_ID", "TRIP_ID") \
            .mode("overwrite") \
            .saveAsTable(table_name)
        return

    df_existing = spark.read.table(table_name)

    if overwrite:
        cond = [df_existing[k] == df_new[k] for k in key_cols]
        df_filtered = df_existing.join(df_new, cond, "left_anti")
        df_final = df_filtered.unionByName(df_new)
    else:
        df_final = df_existing.unionByName(df_new)

    df_final.write.format("delta") \
        .option("mergeSchema", "true") \
        .option("overwriteSchema", "true") \
        .partitionBy("DRIVER_ID", "TRIP_ID") \
        .mode("overwrite") \
        .saveAsTable(table_name)


def store_drivers_insights(df_new, table_name, study_title=None, overwrite=False):
    """Store driver-level insights"""
    spark = SparkSession.builder.getOrCreate()
    spark.sparkContext.setCheckpointDir("/tmp/spark-checkpoints")

    key_cols = ["DRIVER_ID", "month", "year", "criteria"]

    df_new = (
        df_new
        .withColumn("dt", F.current_date())
        .withColumn("study_title", F.lit(study_title))
    )

    if not spark.catalog.tableExists(table_name):
        df_new.write.format("delta") \
            .option("mergeSchema", "true") \
            .option("overwriteSchema", "true") \
            .partitionBy("DRIVER_ID", "year", "month") \
            .mode("overwrite") \
            .saveAsTable(table_name)
        return

    df_existing = spark.read.table(table_name)

    if overwrite:
        cond = [df_existing[k] == df_new[k] for k in key_cols]
        df_filtered = df_existing.join(df_new, cond, "left_anti")
        df_final = df_filtered.unionByName(df_new)
    else:
        df_final = df_existing.unionByName(df_new)

    df_final.write.format("delta") \
        .option("mergeSchema", "true") \
        .option("overwriteSchema", "true") \
        .partitionBy("DRIVER_ID", "year", "month") \
        .mode("overwrite") \
        .saveAsTable(table_name)


def store_insights_hnd(batch_data: Dict, level: str = "PFM", table_name: str = None, overwrite: bool = False):
    """
    Store batch of criteria for one study
    
    NEW: Converts separate fields into JSON strings for Population_description and Sample_definition
    """
    if level != "PFM":
        raise ValueError("Batch storage only supported for PFM level")
    
    main_table = "dafe_dev.customer_knowledge_utils.ck_pfm_insights_main"
    ndim_table = "dafe_dev.customer_knowledge_utils.insight_values_Ndim"
    
    study_metadata = batch_data["study_metadata"]
    criteria_rows = batch_data["criteria_rows"]
    ###################"
    #check if ref study exists
    if spark.catalog.tableExists(main_table):
        existing_studies = spark.read.table(main_table) \
            .filter(F.col("Study_title") == study_metadata.get("Study_title")) \
            .select("Study_ref") \
            .distinct()
        
        if existing_studies.count() > 0:
            # Study exists - reuse Study_ref
            study_ref = existing_studies.first().Study_ref
            print(f"📚 Found existing study - reusing Study_ref: {study_ref}")
        else:
            # New study - generate new Study_ref
            study_ref = generate_study_ref(study_metadata.get("Study_title", "Untitled"))
            print(f"✨ New study - generated Study_ref: {study_ref}")
    else:
        # Table doesn't exist yet - generate new Study_ref
        study_ref = generate_study_ref(study_metadata.get("Study_title", "Untitled"))
        print(f"📊 Generated new Study_ref: {study_ref}")

    #################################
    # Build Population_description JSON string
    population_description = json.dumps({
        "Energy_category": study_metadata.get("Energy_category"),
        "Segment": study_metadata.get("Segment"),
        "Region_country": study_metadata.get("region_country")
    })
    
    # Build Sample_definition JSON string
    sample_definition = json.dumps({
        "Data_source": study_metadata.get("Data_source"),
        "Vin_list": study_metadata.get("Vin_list"),
        "Period_date_start": study_metadata.get("Period_date_start"),
        "Period_date_end": study_metadata.get("Period_date_end"),
        "List_of_signal_names": study_metadata.get("List_of_signal_names")
    })
    
    main_rows = []
    all_ndim_records = []
    
    for criteria_row in criteria_rows:
        # Detect if modeling results exist
        has_modeling = (
            criteria_row.get("loi") is not None
        )
        
        # Get dimension (from criteria_row or default to H0D)
        dimension = criteria_row.get("Dimension", "H0D")
        
        # Build composite key only when modeling absent
        if has_modeling:
            composite_key = None           
        else:
            composite_key = f"{criteria_row['Criteria_label']}_{study_ref}"
            print(f"📚 Generated Insights_values_Ndim_ref: {composite_key}")           
        
        # Build main row with new schema
        main_row = {
            "Study_ref": study_ref,
            "CriteriaValuesNdimRef": composite_key,
            "Study_title": study_metadata.get("Study_title"),
            "Criteria_label": criteria_row["Criteria_label"],
            "Norm_applied": criteria_row["Norm_applied"],
            "Population_description": population_description,  
            "Sample_definition": sample_definition,  
            "Final_stats_min_max_moyenne": criteria_row["Final_stats_min_max_moyenne"],
            "loi": criteria_row["loi"],
            "parameters": criteria_row["parameters"],
            "class_index": criteria_row["class_index"],
            "R2": criteria_row["R2"],
            "client1pct": criteria_row["client1pct"],
            "client50pct": criteria_row["client50pct"],
            "client99pct": criteria_row["client99pct"],
            "sample_client1pct": criteria_row["sample_client1pct"],
            "sample_client50pct": criteria_row["sample_client50pct"],
            "sample_client99pct": criteria_row["sample_client99pct"]
        }
        
        dfmatrix = criteria_row.get("dfmatrix")
        
        main_rows.append(main_row)
        
        if dfmatrix and dfmatrix not in ["", "none", "None", None]:
            df_ndim = explode_matrix_to_dataframe(
                dfmatrix_json=dfmatrix,
                study_ref=study_ref,
                criteria_label=criteria_row["Criteria_label"]
            )
            
            if df_ndim is not None and len(df_ndim) > 0:
                all_ndim_records.append(df_ndim)
    
    df_main = pd.DataFrame(main_rows)
    store_pfm_insights(df_main, table_name=main_table, overwrite=overwrite, add_timestamps=True)
    
    print(f"✅ Stored {len(main_rows)} criteria to main table")
    
    if len(all_ndim_records) > 0:
        df_ndim_combined = pd.concat(all_ndim_records, ignore_index=True)
        store_pfm_insights(df_ndim_combined, table_name=ndim_table, overwrite=overwrite, add_timestamps=False)
        
        dim_summary = df_ndim_combined.groupby(['Criteria_label', 'Dimension']).size().to_dict()
        print(f"✅ Stored matrix data:")
        for (criteria, dim), count in dim_summary.items():
            print(f"   - {criteria} ({dim}): {count} records")
    else:
        print(f"⚠️  No matrix data to store")
    
    return study_ref
