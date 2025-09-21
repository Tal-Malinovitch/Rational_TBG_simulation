#!/usr/bin/env python3
"""
Simple memory analysis focusing on actual TBG project structure.
"""

import sys
import gc
import constants
from TBG import tbg
from graph import periodic_graph

def memory_usage_check():
    """Check memory usage patterns in TBG project."""
    print("=== TBG MEMORY USAGE ANALYSIS ===\n")
    
    # Test different system sizes
    sizes = [(2, 2), (5, 5), (8, 8)]
    
    for maxsize_n, maxsize_m in sizes:
        print(f"Testing system size {maxsize_n}x{maxsize_m}")
        
        # Create system using context manager
        with tbg(maxsize_n=maxsize_n, maxsize_m=maxsize_m,
                a=5, b=1,
                interlayer_dist_threshold=1.5,
                intralayer_dist_threshold=1.5,
                unit_cell_radius_factor=1) as tbg_system:
            
            # Basic stats
            num_nodes = len(tbg_system.full_graph.nodes)
            print(f"  Nodes: {num_nodes}")
            
            # Check if we can create periodic version
            try:
                with tbg_system.full_graph.create_periodic_copy(
                        tbg_system.lattice_vectors, (0.1, 0.1)) as periodic_version:
                    
                    print(f"  Periodic copy created: {len(periodic_version.nodes)} nodes")
                    
                    # Check matrix creation if possible
                    if hasattr(periodic_version, 'matrix_handler'):
                        periodic_version.matrix_handler.build_adj_matrix()
                        matrix = periodic_version.matrix_handler.adj_matrix
                        if matrix is not None:
                            matrix_size = matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
                            print(f"  Matrix memory: {matrix_size / 1024:.1f} KB")
                            print(f"  Matrix density: {matrix.nnz / (matrix.shape[0] * matrix.shape[1]) * 100:.1f}%")
                
            except Exception as e:
                print(f"  Periodic copy failed: {e}")
        gc.collect()
        print()

def check_cleanup_effectiveness():
    """Check if cleanup methods are working properly."""
    print("=== CLEANUP EFFECTIVENESS TEST ===\n")
    
    initial_objects = len(gc.get_objects())
    print(f"Initial objects: {initial_objects}")
    
    # Create and destroy multiple systems using context managers
    for i in range(3):
        with tbg(maxsize_n=3, maxsize_m=3, a=5, b=1) as tbg_system:
            
            # Create periodic copy to test more complex cleanup
            with tbg_system.full_graph.create_periodic_copy(
                    tbg_system.lattice_vectors, (0.0, 0.0)) as periodic_copy:
                pass  # Just test creation and cleanup
        
        # Force garbage collection
        gc.collect()
    
    final_objects = len(gc.get_objects())
    print(f"Final objects: {final_objects}")
    print(f"Object difference: {final_objects - initial_objects}")
    
    if final_objects - initial_objects > 100:
        print("[WARNING] Significant object count increase detected")
    else:
        print("[OK] Object count appears stable")

def analyze_large_objects():
    """Find largest objects in memory."""
    print("=== LARGE OBJECTS ANALYSIS ===\n")
    
    # Create a reasonably sized system
    with tbg(maxsize_n=6, maxsize_m=6, a=5, b=1) as tbg_system:
        
        # Get all objects and find the largest ones
        objects = gc.get_objects()
        large_objects = []
        
        for obj in objects:
            try:
                size = sys.getsizeof(obj)
                if size > 10000:  # Objects > 10KB
                    obj_type = type(obj).__name__
                    large_objects.append((size, obj_type, str(obj)[:100]))
            except:
                pass
        
        # Sort by size
        large_objects.sort(reverse=True)
        
        print("Top 10 largest objects:")
        for i, (size, obj_type, description) in enumerate(large_objects[:10]):
            print(f"{i+1:2d}. {size:8d} bytes - {obj_type:15s} - {description}")
        
        # Check for TBG-related objects
        tbg_objects = [obj for obj in large_objects if any(
            keyword in obj[1].lower() or keyword in obj[2].lower() 
            for keyword in ['tbg', 'graph', 'matrix', 'node']
        )]
        
        if tbg_objects:
            print(f"\nTBG-related large objects: {len(tbg_objects)}")
            for size, obj_type, description in tbg_objects[:5]:
                print(f"  {size:8d} bytes - {obj_type:15s}")

def main():
    """Run memory analysis."""
    print("SIMPLE MEMORY ANALYSIS FOR TBG PROJECT")
    print("=" * 50)
    
    try:
        memory_usage_check()
        check_cleanup_effectiveness()
        analyze_large_objects()
        
        print("\n" + "=" * 50)
        print("MEMORY ANALYSIS COMPLETE")
        
    except Exception as e:
        print(f"[ERROR] Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()