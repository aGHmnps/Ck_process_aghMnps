def get_optimal_partitions(df):
    """
    Calculate optimal shuffle partitions based on cluster config and DataFrame size.
    
    Args:
        df: The DataFrame being processed
        
    Returns:
        int: Optimal number of partitions
    """
    
    # Get cluster configuration
    sc = spark.sparkContext
    
    # Extract cluster specs
    num_executors = int(sc._conf.get("spark.executor.instances", 
                        sc._conf.get("spark.dynamicAllocation.maxExecutors", "2")))
    executor_cores = int(sc._conf.get("spark.executor.cores", "4"))
    executor_memory_str = sc._conf.get("spark.executor.memory", "8g")
    
    # Parse executor memory to GB
    executor_memory_gb = float(executor_memory_str.lower().replace('g', '').replace('m', '')) 
    if 'm' in executor_memory_str.lower():
        executor_memory_gb = executor_memory_gb / 1024
    
    # Calculate total cluster resources
    total_cores = num_executors * executor_cores
    total_memory_gb = num_executors * executor_memory_gb
    
    # Get DataFrame size
    try:
        # Try to get actual size from storage
        df.cache()
        df_size_bytes = df.rdd.map(lambda row: len(str(row))).sum()
        df.unpersist()
        df_size_gb = df_size_bytes / (1024**3)
    except:
        # Fallback: estimate from row count and schema
        row_count = df.count()
        avg_row_size_bytes = sum([
            8 if str(field.dataType) in ['IntegerType', 'LongType', 'DoubleType', 'FloatType'] 
            else 50 for field in df.schema.fields
        ])
        df_size_gb = (row_count * avg_row_size_bytes) / (1024**3)
    
    # Formula: Based on data size and cluster capacity
    target_partition_size_mb = 128  # Optimal partition size
    
    # Calculate partitions needed for data size
    data_based_partitions = int((df_size_gb * 1024) / target_partition_size_mb)
    
    # Calculate partitions based on parallelism (3x cores for good CPU utilization)
    parallelism_partitions = total_cores * 3
    
    # Calculate partitions based on memory (ensure partitions fit in memory)
    memory_safe_partitions = int((total_memory_gb * 1024 * 0.6) / target_partition_size_mb)
    
    # Take the maximum of data-based and parallelism, but cap at memory limit
    optimal_partitions = min(
        max(data_based_partitions, parallelism_partitions),
        memory_safe_partitions
    )
    
    # Apply reasonable bounds
    optimal_partitions = max(200, min(optimal_partitions, 10000))
    
    # Print summary
    print("="*60)
    print("CLUSTER CONFIGURATION:")
    print(f"  Executors: {num_executors}")
    print(f"  Cores per executor: {executor_cores}")
    print(f"  Total cores: {total_cores}")
    print(f"  Executor memory: {executor_memory_str}")
    print(f"  Total cluster memory: {total_memory_gb:.1f} GB")
    print("\nDATA ANALYSIS:")
    print(f"  DataFrame size: {df_size_gb:.2f} GB")
    print(f"  Current partitions: {df.rdd.getNumPartitions()}")
    print("\nCALCULATIONS:")
    print(f"  Data-based partitions: {data_based_partitions}")
    print(f"  Parallelism-based partitions: {parallelism_partitions}")
    print(f"  Memory-safe partitions: {memory_safe_partitions}")
    print(f"\n  OPTIMAL PARTITIONS: {optimal_partitions}")
    print("="*60)
    
    return optimal_partitions


# Usage:
# df = spark.read.table("your_delta_table")
# optimal = get_optimal_partitions(df)
# spark.conf.set("spark.sql.shuffle.partitions", optimal)


"""
usage example 

"""
