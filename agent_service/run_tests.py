#!/usr/bin/env python3
"""
Test runner script for the Chatwoot Agent Service test suite.

This script provides convenient commands to run different types of tests
with appropriate configurations and reporting.
"""

import sys
import subprocess
import argparse
import os
from pathlib import Path
from typing import List, Optional


def run_command(cmd: List[str], description: str) -> int:
    """
    Run a command and return the exit code.
    
    Args:
        cmd: Command and arguments to run
        description: Description of what the command does
        
    Returns:
        Exit code from the command
    """
    print(f"\n🔄 {description}")
    print(f"Running: {' '.join(cmd)}")
    print("-" * 60)
    
    try:
        result = subprocess.run(cmd, check=False)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
        else:
            print(f"❌ {description} failed with exit code {result.returncode}")
        return result.returncode
    except KeyboardInterrupt:
        print(f"\n⚠️  {description} interrupted by user")
        return 130
    except Exception as e:
        print(f"❌ Error running {description}: {e}")
        return 1


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(
        description="Run Chatwoot Agent Service tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py --unit                    # Run only unit tests
  python run_tests.py --integration            # Run only integration tests
  python run_tests.py --e2e                   # Run only end-to-end tests
  python run_tests.py --fast                  # Run fast tests only
  python run_tests.py --coverage              # Run with coverage report
  python run_tests.py --performance           # Run performance tests
  python run_tests.py --security              # Run security tests
  python run_tests.py --all                   # Run all tests
  python run_tests.py --verbose               # Run with verbose output
  python run_tests.py --parallel              # Run tests in parallel
        """
    )
    
    # Test selection options
    parser.add_argument(
        "--unit", 
        action="store_true", 
        help="Run unit tests only"
    )
    parser.add_argument(
        "--integration", 
        action="store_true", 
        help="Run integration tests only"
    )
    parser.add_argument(
        "--e2e", 
        action="store_true", 
        help="Run end-to-end tests only"
    )
    parser.add_argument(
        "--performance", 
        action="store_true", 
        help="Run performance tests only"
    )
    parser.add_argument(
        "--security", 
        action="store_true", 
        help="Run security tests only"
    )
    parser.add_argument(
        "--governance", 
        action="store_true", 
        help="Run governance tests only"
    )
    parser.add_argument(
        "--multimodal", 
        action="store_true", 
        help="Run multimodal tests only"
    )
    parser.add_argument(
        "--fast", 
        action="store_true", 
        help="Run fast tests only (exclude slow tests)"
    )
    parser.add_argument(
        "--slow", 
        action="store_true", 
        help="Run slow tests only"
    )
    parser.add_argument(
        "--all", 
        action="store_true", 
        help="Run all tests including slow ones"
    )
    
    # Output and reporting options
    parser.add_argument(
        "--coverage", 
        action="store_true", 
        help="Generate coverage report"
    )
    parser.add_argument(
        "--html-coverage", 
        action="store_true", 
        help="Generate HTML coverage report"
    )
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true", 
        help="Verbose output"
    )
    parser.add_argument(
        "--quiet", "-q", 
        action="store_true", 
        help="Quiet output"
    )
    parser.add_argument(
        "--parallel", "-n", 
        type=int, 
        metavar="N", 
        help="Run tests in parallel using N workers"
    )
    
    # Test configuration options
    parser.add_argument(
        "--fail-fast", "-x", 
        action="store_true", 
        help="Stop on first failure"
    )
    parser.add_argument(
        "--pdb", 
        action="store_true", 
        help="Drop into debugger on failures"
    )
    parser.add_argument(
        "--lf", 
        action="store_true", 
        help="Run last failed tests only"
    )
    parser.add_argument(
        "--tb", 
        choices=["short", "long", "auto", "line", "native", "no"],
        default="short",
        help="Traceback print mode"
    )
    
    # Specific test selection
    parser.add_argument(
        "--file", 
        type=str, 
        help="Run specific test file"
    )
    parser.add_argument(
        "--test", 
        type=str, 
        help="Run specific test function"
    )
    parser.add_argument(
        "--keyword", "-k", 
        type=str, 
        help="Run tests matching keyword expression"
    )
    
    args = parser.parse_args()
    
    # Set up environment
    os.environ["PYTHONPATH"] = str(Path(__file__).parent)
    
    # Build pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add basic options
    cmd.extend(["--tb", args.tb])
    
    if args.verbose:
        cmd.append("-v")
    elif args.quiet:
        cmd.append("-q")
    
    if args.fail_fast:
        cmd.append("-x")
    
    if args.pdb:
        cmd.append("--pdb")
    
    if args.lf:
        cmd.append("--lf")
    
    # Add parallel execution
    if args.parallel:
        cmd.extend(["-n", str(args.parallel)])
    
    # Add coverage options
    if args.coverage or args.html_coverage:
        cmd.extend([
            "--cov=src",
            "--cov-report=term-missing",
            "--cov-report=json:coverage.json",
            "--cov-report=xml:coverage.xml",
        ])
        
        if args.html_coverage:
            cmd.extend(["--cov-report=html:htmlcov"])
    
    # Add test selection markers
    markers = []
    
    if args.unit:
        markers.append("unit")
    if args.integration:
        markers.append("integration")
    if args.e2e:
        markers.append("e2e")
    if args.performance:
        markers.append("performance")
    if args.security:
        markers.append("security")
    if args.governance:
        markers.append("governance")
    if args.multimodal:
        markers.append("multimodal")
    
    if args.fast:
        markers.append("not slow")
    elif args.slow:
        markers.append("slow")
    
    if markers:
        cmd.extend(["-m", " or ".join(markers)])
    
    # Add keyword filter
    if args.keyword:
        cmd.extend(["-k", args.keyword])
    
    # Add specific file or test
    if args.file:
        cmd.append(args.file)
        if args.test:
            cmd[-1] += f"::{args.test}"
    elif args.test:
        cmd.extend(["-k", args.test])
    
    # Default to running fast tests if no specific selection
    if not any([
        args.unit, args.integration, args.e2e, args.performance,
        args.security, args.governance, args.multimodal,
        args.fast, args.slow, args.all, args.file, args.test, args.keyword
    ]):
        cmd.extend(["-m", "not slow"])
    
    # Run the tests
    print("🧪 Chatwoot Agent Service Test Suite")
    print("=" * 60)
    
    exit_code = run_command(cmd, "Running tests")
    
    if exit_code == 0:
        print("\n🎉 All tests passed!")
        
        if args.coverage or args.html_coverage:
            print("\n📊 Coverage Report Generated:")
            if args.html_coverage:
                print("  HTML: htmlcov/index.html")
            print("  JSON: coverage.json")
            print("  XML: coverage.xml")
    else:
        print(f"\n💥 Tests failed with exit code {exit_code}")
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())