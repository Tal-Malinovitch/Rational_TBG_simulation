#!/usr/bin/env python3
"""
Comprehensive test suite runner for TBG project.

Runs all unit tests, integration tests, and performance tests.
Generates coverage reports and test summaries.
"""

import unittest
import sys
import os
import time
import logging
from io import StringIO

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TestResult:
    """Container for test results."""
    
    def __init__(self):
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        self.error_tests = 0
        self.skipped_tests = 0
        self.execution_time = 0.0
        self.failures = []
        self.errors = []


class TBGTestRunner:
    """Main test runner for TBG project."""
    
    def __init__(self, verbosity=2):
        self.verbosity = verbosity
        self.results = {}
    
    def discover_tests(self, test_dir):
        """Discover all test modules in directory."""
        loader = unittest.TestLoader()
        start_dir = test_dir
        pattern = 'test*.py'
        
        try:
            test_suite = loader.discover(start_dir, pattern=pattern)
            return test_suite
        except Exception as e:
            logger.error(f"Failed to discover tests in {test_dir}: {e}")
            return unittest.TestSuite()
    
    def run_test_suite(self, test_suite, suite_name):
        """Run a test suite and collect results."""
        logger.info(f"Running {suite_name} tests...")
        
        # Capture test output
        stream = StringIO()
        runner = unittest.TextTestRunner(
            stream=stream,
            verbosity=self.verbosity,
            failfast=False
        )
        
        start_time = time.time()
        result = runner.run(test_suite)
        end_time = time.time()
        
        # Create result summary
        test_result = TestResult()
        test_result.total_tests = result.testsRun
        test_result.passed_tests = result.testsRun - len(result.failures) - len(result.errors)
        test_result.failed_tests = len(result.failures)
        test_result.error_tests = len(result.errors)
        test_result.skipped_tests = len(getattr(result, 'skipped', []))
        test_result.execution_time = end_time - start_time
        test_result.failures = result.failures
        test_result.errors = result.errors
        
        # Store results
        self.results[suite_name] = test_result
        
        # Print summary
        self.print_suite_summary(suite_name, test_result)
        
        return test_result
    
    def print_suite_summary(self, suite_name, result):
        """Print summary for a test suite."""
        print(f"\n{suite_name} Test Results:")
        print("=" * 50)
        print(f"Total tests: {result.total_tests}")
        print(f"Passed: {result.passed_tests}")
        print(f"Failed: {result.failed_tests}")
        print(f"Errors: {result.error_tests}")
        print(f"Skipped: {result.skipped_tests}")
        print(f"Execution time: {result.execution_time:.2f} seconds")
        
        if result.failures:
            print(f"\nFailures ({len(result.failures)}):")
            for test, traceback in result.failures:
                print(f"  - {test}")
        
        if result.errors:
            print(f"\nErrors ({len(result.errors)}):")
            for test, traceback in result.errors:
                print(f"  - {test}")
        
        success_rate = (result.passed_tests / result.total_tests * 100) if result.total_tests > 0 else 0
        print(f"Success rate: {success_rate:.1f}%")
    
    def print_overall_summary(self):
        """Print overall test summary."""
        print("\n" + "=" * 60)
        print("OVERALL TEST SUMMARY")
        print("=" * 60)
        
        total_tests = sum(r.total_tests for r in self.results.values())
        total_passed = sum(r.passed_tests for r in self.results.values())
        total_failed = sum(r.failed_tests for r in self.results.values())
        total_errors = sum(r.error_tests for r in self.results.values())
        total_time = sum(r.execution_time for r in self.results.values())
        
        print(f"Test suites run: {len(self.results)}")
        print(f"Total tests: {total_tests}")
        print(f"Total passed: {total_passed}")
        print(f"Total failed: {total_failed}")
        print(f"Total errors: {total_errors}")
        print(f"Total execution time: {total_time:.2f} seconds")
        
        if total_tests > 0:
            success_rate = (total_passed / total_tests * 100)
            print(f"Overall success rate: {success_rate:.1f}%")
        
        # Print suite breakdown
        print("\nSuite Breakdown:")
        for suite_name, result in self.results.items():
            suite_success = (result.passed_tests / result.total_tests * 100) if result.total_tests > 0 else 0
            print(f"  {suite_name:20} {result.passed_tests:3d}/{result.total_tests:3d} ({suite_success:5.1f}%)")
        
        return total_failed + total_errors == 0


def run_unit_tests():
    """Run all unit tests."""
    runner = TBGTestRunner(verbosity=2)
    
    # Discover and run unit tests
    test_dir = os.path.join(os.path.dirname(__file__))
    test_suite = runner.discover_tests(test_dir)
    
    if test_suite.countTestCases() == 0:
        logger.warning("No unit tests found!")
        return True
    
    result = runner.run_test_suite(test_suite, "Unit Tests")
    runner.print_overall_summary()
    
    return result.failed_tests + result.error_tests == 0


def run_specific_test_modules():
    """Run specific test modules individually."""
    runner = TBGTestRunner(verbosity=1)
    
    # Define test modules to run
    test_modules = [
        'test_utils',
        'test_graph', 
        'test_band_computation',
        'test_integration',
        'tests_lattice',  # Existing test file
        'test_error_handling',  # Existing test file
    ]
    
    all_passed = True
    
    for module_name in test_modules:
        try:
            # Import the test module
            test_module = __import__(module_name)
            
            # Load tests from module
            loader = unittest.TestLoader()
            suite = loader.loadTestsFromModule(test_module)
            
            if suite.countTestCases() > 0:
                result = runner.run_test_suite(suite, module_name)
                if result.failed_tests + result.error_tests > 0:
                    all_passed = False
            else:
                logger.warning(f"No tests found in module {module_name}")
                
        except ImportError as e:
            logger.error(f"Could not import test module {module_name}: {e}")
            all_passed = False
        except Exception as e:
            logger.error(f"Error running tests for {module_name}: {e}")
            all_passed = False
    
    runner.print_overall_summary()
    return all_passed


def run_performance_tests():
    """Run performance tests separately."""
    logger.info("Running performance tests...")
    
    try:
        # Import performance test module
        from test_preformence import test_performance
        
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromTestCase(test_performance)
        
        runner = TBGTestRunner(verbosity=2)
        result = runner.run_test_suite(suite, "Performance Tests")
        
        return result.failed_tests + result.error_tests == 0
        
    except ImportError as e:
        logger.error(f"Could not import performance tests: {e}")
        return False


def check_test_coverage():
    """Check test coverage of main modules."""
    print("\n" + "=" * 50)
    print("TEST COVERAGE ANALYSIS")
    print("=" * 50)
    
    # Define main modules and their test counterparts
    coverage_map = {
        'utils.py': 'test_utils.py',
        'graph.py': 'test_graph.py',
        'band_comp_and_plot.py': 'test_band_computation.py',
        'TBG.py': 'test_integration.py',
        'constants.py': 'test_error_handling.py',
        'Generate_training_data.py': 'test_integration.py',
        'stats.py': 'test_integration.py',
    }
    
    project_dir = os.path.dirname(os.path.dirname(__file__))
    test_dir = os.path.dirname(__file__)
    
    for module, test_file in coverage_map.items():
        module_path = os.path.join(project_dir, module)
        test_path = os.path.join(test_dir, test_file)
        
        module_exists = os.path.exists(module_path)
        test_exists = os.path.exists(test_path)
        
        status = "✓" if (module_exists and test_exists) else "✗"
        print(f"{status} {module:25} -> {test_file}")
        
        if module_exists and not test_exists:
            print(f"  WARNING: No test file found for {module}")
    
    print("\nTest files created:")
    test_files = [f for f in os.listdir(test_dir) if f.startswith('test_') and f.endswith('.py')]
    for test_file in sorted(test_files):
        print(f"  ✓ {test_file}")


def main():
    """Main test runner entry point."""
    print("TBG PROJECT COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    
    # Check Python version
    print(f"Python version: {sys.version}")
    print(f"Project root: {project_root}")
    print(f"Test directory: {os.path.dirname(__file__)}")
    
    # Check test coverage
    check_test_coverage()
    
    overall_success = True
    
    try:
        # Run unit tests by module
        print("\n🧪 RUNNING UNIT TESTS BY MODULE")
        unit_success = run_specific_test_modules()
        overall_success = overall_success and unit_success
        
        # Run performance tests
        print("\n⚡ RUNNING PERFORMANCE TESTS")
        perf_success = run_performance_tests()
        overall_success = overall_success and perf_success
        
        # Final summary
        print("\n" + "=" * 60)
        if overall_success:
            print("🎉 ALL TESTS PASSED! The TBG project is ready for production.")
        else:
            print("❌ Some tests failed. Please review the failures above.")
        print("=" * 60)
        
        return overall_success
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Test run interrupted by user.")
        return False
    except Exception as e:
        logger.error(f"Test runner encountered an unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)