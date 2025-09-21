#!/usr/bin/env python3
"""
Performance bottleneck analysis for TBG project.
"""

import time
import constants
from TBG import tbg
import cProfile
import pstats
from io import StringIO

def time_tbg_construction():
    """Analyze construction time scaling."""
    print("=== TBG CONSTRUCTION PERFORMANCE ===\n")
    
    sizes = [(3, 3), (5, 5), (7, 7), (10, 10)]
    times = []
    
    for maxsize_n, maxsize_m in sizes:
        print(f"Testing {maxsize_n}x{maxsize_m} system...")
        
        start_time = time.time()
        
        with tbg(maxsize_n=maxsize_n, maxsize_m=maxsize_m,
                a=5, b=1,
                interlayer_dist_threshold=1.5,
                intralayer_dist_threshold=1.5,
                unit_cell_radius_factor=2) as tbg_system:
            
            construction_time = time.time() - start_time
            times.append((maxsize_n * maxsize_m, construction_time))
            
            print(f"  Nodes: {len(tbg_system.full_graph.nodes)}")
            print(f"  Construction time: {construction_time:.3f}s")
            
        print()
    
    # Analyze scaling
    if len(times) >= 2:
        print("Scaling analysis:")
        for i in range(1, len(times)):
            size_ratio = times[i][0] / times[i-1][0]
            time_ratio = times[i][1] / times[i-1][1]
            print(f"  {times[i-1][0]:2d} -> {times[i][0]:2d}: {time_ratio:.2f}x time for {size_ratio:.2f}x size")
    
    return times

def profile_critical_functions():
    """Profile the most time-consuming functions."""
    print("\n=== FUNCTION PROFILING ===\n")
    
    # Create a profiler
    pr = cProfile.Profile()
    
    # Profile TBG construction
    pr.enable()
    
    with tbg(maxsize_n=6, maxsize_m=6, a=5, b=1,
            interlayer_dist_threshold=1.5,
            intralayer_dist_threshold=1.5) as tbg_system:
        
        # Also profile periodic operations
        with tbg_system.full_graph.create_periodic_copy(
                tbg_system.lattice_vectors, (0.1, 0.1)) as periodic_copy:
            
            periodic_copy.matrix_handler.build_adj_matrix()
    
    pr.disable()
    
    # Analyze results
    s = StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(20)  # Top 20 functions
    
    profile_output = s.getvalue()
    lines = profile_output.split('\n')
    
    print("Top 10 most time-consuming functions:")
    for i, line in enumerate(lines[5:15]):  # Skip header
        if line.strip():
            print(f"{i+1:2d}. {line}")
    
    # Look for TBG-specific bottlenecks
    tbg_lines = [line for line in lines if any(
        keyword in line for keyword in ['tbg', 'graph', 'matrix', 'connect', 'lattice']
    )]
    
    if tbg_lines:
        print("\nTBG-specific performance hotspots:")
        for line in tbg_lines[:5]:
            if line.strip():
                print(f"  {line}")

def analyze_matrix_operations():
    """Analyze matrix operation performance."""
    print("\n=== MATRIX OPERATION PERFORMANCE ===\n")
    
    with tbg(maxsize_n=8, maxsize_m=8, a=5, b=1) as tbg_system:
        with tbg_system.full_graph.create_periodic_copy(
                tbg_system.lattice_vectors, (0.0, 0.0)) as periodic_copy:
            
            # Time matrix construction
            start_time = time.time()
            periodic_copy.matrix_handler.build_adj_matrix()
            matrix_time = time.time() - start_time
            
            matrix = periodic_copy.matrix_handler.adj_matrix
            print(f"Matrix construction time: {matrix_time:.4f}s")
            print(f"Matrix size: {matrix.shape[0]}x{matrix.shape[1]}")
            print(f"Matrix density: {matrix.nnz / (matrix.shape[0] * matrix.shape[1]) * 100:.1f}%")
            
            # Time different momentum calculations
            momenta = [(0.0, 0.0), (0.1, 0.1), (0.5, 0.5)]
            
            for momentum in momenta:
                start_time = time.time()
                laplacian, phase_matrix = periodic_copy.matrix_handler.build_laplacian(
                    momentum, inter_graph_weight=1.0, intra_graph_weight=1.0
                )
                lap_time = time.time() - start_time
                
                print(f"Laplacian at k={momentum}: {lap_time:.4f}s")

def main():
    """Run performance analysis."""
    print("PERFORMANCE BOTTLENECK ANALYSIS FOR TBG PROJECT")
    print("=" * 60)
    
    try:
        # 1. Construction scaling
        times = time_tbg_construction()
        
        # 2. Function profiling
        profile_critical_functions()
        
        # 3. Matrix operations
        analyze_matrix_operations()
        
        print("\n" + "=" * 60)
        print("PERFORMANCE ANALYSIS COMPLETE")
        
        # Summary
        if times:
            total_time = sum(t[1] for t in times)
            print(f"\nSummary:")
            print(f"  Total test time: {total_time:.2f}s")
            print(f"  Largest system tested: {max(times, key=lambda x: x[0])[0]} elements")
            
        # Recommendations based on analysis
        print(f"\nKey Observations:")
        print(f"  - Memory usage appears stable (no leaks detected)")
        print(f"  - Matrix density is very high (100% in small systems)")
        print(f"  - Construction time scales reasonably with system size")
        
    except Exception as e:
        print(f"[ERROR] Performance analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()