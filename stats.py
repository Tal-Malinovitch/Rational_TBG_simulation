"""
Statistics and Performance Analysis Module for TBG Training Data Generation.

This module provides comprehensive timing and performance analysis capabilities for
twisted bilayer graphene (TBG) computations. It tracks computational bottlenecks,
system scaling behavior, and provides metrics to evaluate neural network speedup potential.

Key Features:
    - Real-time performance monitoring during training data generation
    - System size and parameter-based timing analysis
    - Success rate tracking for Dirac point finding algorithms
    - Detailed statistics export for further analysis
    - Memory usage and computational complexity insights

Main Components:
    - statistics: Core class for collecting and analyzing performance metrics
    - Timing categorization by system parameters (N_scale, system size)
    - Success/failure rate tracking with detailed breakdown
    - Export functionality for statistical analysis

Usage:
    Basic usage:
        stats = statistics()
        stats.log_combination(duration=2.5, system_size=500, success=True)
        stats.print_summary()
        stats.save_results_to_file("performance_report.txt")

    Integration with training data generation:
        Used automatically by Generate_training_data.py to track
        computational performance across parameter sweeps.

Statistical Metrics:
    - Total computation time and average per system
    - Success rates for different parameter ranges
    - Scaling behavior analysis (time vs system size)
    - Bottleneck identification for optimization

Author: Tal Malinovitch
License: MIT (Academic Use)
"""

import constants
from constants import logging, time, csv, os, defaultdict
from constants import List, Optional, Dict, DefaultDict, Union, Any
class statistics():
    """
    Class for tracking and analyzing timing statistics during training data generation.
    
    This class collects performance metrics for TBG computations, categorizes them by
    system parameters, and provides analysis tools to understand computational bottlenecks
    and estimate neural network speedup potential.
    
    Attributes:
        combination_times (List[float]): Duration of each parameter combination computation.
        intersection_times (List[float]): Reserved for intersection finding timing.
        dirac_times (List[float]): Reserved for Dirac point optimization timing.
        system_size_times (defaultdict): Timing data grouped by number of nodes.
        n_scale_times (defaultdict): Timing data grouped by N_scale values.
        start_time (Optional[float]): Timestamp when statistics collection started.
        total_combinations (int): Total number of parameter combinations attempted.
        successful_combinations (int): Number of combinations that found Dirac points.
        successful_combinations_num_of_Dirac (int): Total Dirac points found across all combinations.
        failed_no_intersections (int): Number of failures due to no intersection points.
        failed_no_Dirac (int): Number of failures due to no Dirac points from intersections.
    """
    def __init__(self) -> None:
        """Initialize empty statistics collection."""
        self.combination_times: List[float] = []
        self.intersection_times: List[float] = []        
        self.dirac_times: List[float] = []
        self.system_size_times: DefaultDict[int, List[float]] = defaultdict(list)
        self.n_scale_times: DefaultDict[float, List[float]] = defaultdict(list)
        self.start_time: Optional[float] = None
        self.total_combinations: int = 0
        self.successful_combinations: int = 0
        self.successful_combinations_num_of_Dirac: int = 0
        self.failed_no_intersections: int = 0
        self.failed_no_Dirac: int = 0

    def log_combination(self, duration: float = 0.0, system_size: int = 0, n_scale: float = 0.0,
                           success: bool = True, no_intersection: bool = True, num_of_Dirac: int = 0) -> None:       
        """
        Record timing and outcome data for a single parameter combination.
        
        Args:
            duration (float): Time taken for this combination in seconds
            system_size (int): Number of nodes in the TBG system
            n_scale (float): N_scale parameter (physical scaling factor)
            success (bool): Whether Dirac points were successfully found
            no_intersection (bool): If failed, whether failure was due to no intersections
            num_of_Dirac (int): Number of Dirac points found (0 if failed)
        """
        self.combination_times.append(duration)     
        self.system_size_times[system_size].append(duration)
        self.n_scale_times[round(n_scale, 1)].append(duration)
        self.total_combinations += 1  # Always increment total
        if success:
            self.successful_combinations += 1
            self.successful_combinations_num_of_Dirac+=num_of_Dirac
        else:
            if no_intersection:
                self.failed_no_intersections+=1
            else:
                self.failed_no_Dirac+=1

    def log_statistics(self) -> None:
        """
        Log comprehensive timing statistics to console.
        
        Prints summary statistics including total runtime, success rates,
        and detailed breakdowns by system size and N_scale parameters.
        """
        if not self.combination_times:      
            return
        total_time = time.time() -self.start_time if self.start_time else 0
        avg_time =sum(self.combination_times) /len(self.combination_times)

        logging.info(f"\n=== TIMING STATISTICS ===")
        logging.info(f"Total runtime:{total_time/3600:.1f} hours")
        logging.info(f"Average time percombination: {avg_time:.2f} seconds")       
        logging.info(f"Successful combinations: {self.successful_combinations}/{self.total_combinations}")
        logging.info(f"Time per successful combination: {total_time/max(self.successful_combinations, 1):.2f} seconds")      

        if self.successful_combinations!= self.total_combinations:
            logging.info(f"  Out of Failed cases, the percent of no intersection found is: {self.failed_no_intersections/(self.total_combinations-self.successful_combinations)*100:.3f}%")
        else:
            logging.info(f"  No failed cases - all combinations successful!")
        
        logging.info(f"  In total found {self.successful_combinations_num_of_Dirac} Dirac points")
        # System size analysis
        logging.info(f"\nTime by systemsize:")
        for size, times in sorted(self.system_size_times.items()):     
            logging.info(f"  {size:3d} nodes:{sum(times)/len(times):6.2f}s avg({len(times):3d} samples)")

        # N_scale analysis
        logging.info(f"\nTime by N_scale:")        
        for n_scale, times in sorted(self.n_scale_times.items()):
            logging.info(f"  N_scale{n_scale:4.1f}:{sum(times)/len(times):6.2f}s avg({len(times):3d} samples)")
    def save_statistics(self, filename: Optional[str] = None) -> None:
        """
        Save detailed timing statistics to CSV and summary files.
        
        Creates two output files:
        1. CSV file with detailed timing data categorized by system parameters
        2. Summary text file with human-readable statistics
        
        Args:
            filename (str, optional): Base filename for output. Defaults to timing_statistics.csv
        """
        if filename is None:
            filename = constants.PATH + "/timing_statistics.csv"
        
        if not self.combination_times:
            logging.warning("No timing data to save")
            return
        

        
        # Prepare summary statistics
        total_time = time.time() - self.start_time if self.start_time else 0
        avg_time = sum(self.combination_times) / len(self.combination_times)
        
        # Create detailed data for CSV
        detailed_data: List[Dict[str, Union[str, int, float]]] = []
        
        # Add system size statistics
        for system_size, times in sorted(self.system_size_times.items()):
            detailed_data.append({
                'category': 'system_size',
                'value': system_size,
                'avg_time_seconds': sum(times) / len(times),
                'sample_count': len(times),
                'total_time_seconds': sum(times),
                'min_time': min(times),
                'max_time': max(times)
            })
        
        # Add N_scale statistics
        for n_scale, times in sorted(self.n_scale_times.items()):
            detailed_data.append({
                'category': 'n_scale',
                'value': n_scale,
                'avg_time_seconds': sum(times) / len(times),
                'sample_count': len(times),
                'total_time_seconds': sum(times),
                'min_time': min(times),
                'max_time': max(times)
            })
        
        # Ensure directory exists before creating file
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Save detailed statistics to CSV
        with open(filename, 'w', newline='') as csvfile:
            if detailed_data:
                fieldnames = detailed_data[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(detailed_data)
        
        # Save summary statistics to a separate file
        summary_filename = filename.replace('.csv', '_summary.txt')
        # Ensure directory exists before creating summary file
        os.makedirs(os.path.dirname(summary_filename), exist_ok=True)
        
        with open(summary_filename, 'w') as f:
            f.write("=== TIMING STATISTICS SUMMARY ===\n")
            f.write(f"Total runtime: {total_time/3600:.1f} hours\n")
            f.write(f"Average time per combination: {avg_time:.2f} seconds\n")
            f.write(f"Successful combinations: {self.successful_combinations}/{self.total_combinations}\n")
            f.write(f"Success rate: {self.successful_combinations/max(self.total_combinations,1)*100:.2f}%\n")
            f.write(f"Time per successful combination: {total_time/max(self.successful_combinations, 1):.2f} seconds\n")
            f.write(f"Total Dirac points found: {self.successful_combinations_num_of_Dirac}\n")
            f.write(f"Failed (no intersections): {self.failed_no_intersections}\n")
            f.write(f"Failed (no Dirac points): {self.failed_no_Dirac}\n")
        
        logging.info(f"Statistics saved to {filename} and {summary_filename}")

    def cleanup(self) -> None:
        """Clean up statistics by clearing all collected data."""
        self.combination_times.clear()
        self.intersection_times.clear()
        self.dirac_times.clear()
        self.system_size_times.clear()
        self.n_scale_times.clear()
        self.start_time = None
        self.total_combinations = 0
        self.successful_combinations = 0
        self.successful_combinations_num_of_Dirac = 0
        self.failed_no_intersections = 0
        self.failed_no_Dirac = 0

    def __enter__(self) -> 'statistics':
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Optional[type], exc_value: Optional[Exception], traceback: Optional[Any]) -> bool:
        """Context manager exit with cleanup."""
        self.cleanup()
        return False
