"""
CLI arguments to extract features
"""

import sys
import argparse
from pathlib import Path
import json

from config import config
from extractor import FeatureExtractor
from utils.logging_config import logger, log_extraction_start, log_extraction_complete
from utils.pdf_utils import list_pdf_files


def setup_argparse() -> argparse.ArgumentParser:
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Extract features from research papers using multiple AI models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
                Examples:
                # single pdf
                python main.py paper.pdf
                
                # folder of pdfs
                python main.py --batch papers_folder/

                # test pipeline: essential extraction + similarity vs ground_truth/
                python main.py --test-similarity --training-pdf pdf_1.pdf
        """
    )
    
    # Input
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "pdf_path",
        nargs="?", # optional argument in case of batch
        type=str,
        help="Path to a single PDF file"
    )
    input_group.add_argument(
        "--batch",
        type=str,
        help="Process PDFs in specified folder"
    )
    input_group.add_argument(
        "--test-similarity",
        action="store_true",
        help=(
            "Essential extraction only (TestPaperExtraction), then cosine similarity "
            "vs ground_truth JSON for the same PDF stem (light; needs OPENAI_API_KEY only)"
        ),
    )
    
    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output", #default output dir est. in config
        help="Output directory for batch processing (default: output/)"
    )
    
    # Model disabling
    parser.add_argument(
        "--no-openai",
        action="store_true", #boolean flag 
        help="Disable direct OpenAI model"
    )
    parser.add_argument(
        "--openrouter",
        action="store_true",
        help=(
            "Use OpenRouter for the GPT slot instead of direct OpenAI "
            "(set OPENROUTER_API_KEY; optional OPENROUTER_MODEL, e.g. openai/gpt-4o-mini)"
        ),
    )
    parser.add_argument(
        "--no-claude",
        action="store_true", #boolean flag
        help="Disable Claude model"
    )
    parser.add_argument(
        "--no-gemini",
        action="store_true", #boolean flag
        help="Disable Gemini model"
    )
    parser.add_argument(
        "--no-expert",
        action="store_true", #boolean flag
        help="Disable expert evaluation"
    )
    parser.add_argument(
        "--no-similarity",
        action="store_true",
        help="Disable cosine similarity scoring against ground truth"
    )
    
    # Processing qualifier
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit batch processing to first N files"
    )
    parser.add_argument(
        "--training-pdf",
        type=str,
        default=None,
        help="PDF in training_papers dir for --test-similarity (default: pdf_1.pdf)",
    )

    return parser


def process_single_pdf(pdf_path: str, extractor: FeatureExtractor, output_path: str = config.OUTPUT_DIR) -> bool:
    """
    Process a single PDF with AI extractor specified.
    
    Args:
        pdf_path: Path to PDF file
        extractor: FeatureExtractor instance
        output_path: Optional output path for results
    
    Returns:
        True if successful, False otherwise
    """
    try:
        pdf_path = Path(pdf_path) # type conversion to Path object
        log_extraction_start(logger, str(pdf_path)) # logger start 
        
        # Extract features
        result = extractor.extract_from_pdf(pdf_path) 
        
        saved_path = extractor.save_result(result, output_path)
        
        log_extraction_complete(
            logger,
            str(pdf_path),
            result.total_processing_time,
            success=True
        )
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"  Extraction completed successfully")
        print(f"  Paper: {pdf_path.name}")
        print(f"  Output: {saved_path}")
        print(f"  Time: {result.total_processing_time:.2f}s")
        print(f"{'='*80}\n")
        
        return True
    
    except Exception as e:
        logger.error(f"Failed to process {pdf_path}: {str(e)}")
        log_extraction_complete(
            logger, 
            str(pdf_path), 
            0, 
            success=False)
        print(f"\n✗ Extraction failed: {str(e)}\n")
        return False


def process_batch(folder_path: str, extractor: FeatureExtractor, output_dir: str = "output", limit: int = None) -> dict:
    """
    Process all PDFs in a folder.
    
    Args:
        folder_path: Path to folder containing PDFs
        extractor: FeatureExtractor instance
        output_dir: Directory for output files
        limit: Maximum number of files to process
    
    Returns:
        Dictionary with processing statistics
    """
    folder_path = Path(folder_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get list of PDFs
    pdf_files = list_pdf_files(folder_path)
    
    if limit:
        pdf_files = pdf_files[:limit]
    
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    
    stats = {
        "total": len(pdf_files),
        "successful": 0,
        "failed": 0,
        "errors": []
    }
    
    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"\n[{i}/{len(pdf_files)}] Processing: {pdf_file.name}")
        
        # Generate output path
        output_path = output_dir / f"{pdf_file.stem}_extraction.json"
        
        success = process_single_pdf(str(pdf_file), extractor, str(output_path))
        
        if success:
            stats["successful"] += 1
        else:
            stats["failed"] += 1
            stats["errors"].append({
                "file": str(pdf_file),
                "index": i
            })
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"  Total files: {stats['total']}")
    print(f"  Successful: {stats['successful']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Output directory: {output_dir}")
    print(f"{'='*80}\n")
    
    # Save summary
    summary_path = output_dir / "batch_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Batch summary saved to: {summary_path}")
    
    return stats


def validate_test_similarity() -> bool:
    """Only OpenAI is required for essential-feature extraction."""
    if not config.OPENAI_API_KEY:
        print("Configuration error: OPENAI_API_KEY must be set for --test-similarity")
        return False
    return True


def run_test_similarity_pipeline(pdf_path: Path, output_dir: Path) -> Path:
    """
    Extract TestPaperExtraction via OpenAI, load ground truth for the same stem,
    compute sentence-transformer cosine similarity on overlapping fields.
    """
    from models.openai_extractor import OpenAIExtractor
    from models.gemini_extractor import GeminiExtractor
    from models.claude_extractor import ClaudeExtractor
    from models.similarity_scorer import SimilarityScorer, load_ground_truth

    pdf_path = Path(pdf_path).resolve()
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # //TODO Update with extractors to test expert evaluator here.
    extractor = GeminiExtractor()
    essential = extractor.extract_essential_features(pdf_path)

    ground_truth = load_ground_truth(pdf_path)
    if ground_truth is None:
        raise RuntimeError(
            f"Could not load ground truth for stem {pdf_path.stem!r}. "
            f"Expected {config.GROUND_TRUTH_DIR / (pdf_path.stem + '.json')} "
            f"(or mapping.json → paper_N.json). "
            f"If the file exists, check logs: invalid JSON (e.g. trailing comma) or schema mismatch."
        )

    scorer = SimilarityScorer()
    similarity = scorer.score_essential(essential, ground_truth, paper_path=str(pdf_path))

    payload = {
        "paper_path": str(pdf_path),
        "essential_extraction": essential.model_dump(mode="json"),
        "similarity": similarity.model_dump(mode="json"),
    }
    out_path = output_dir / f"{pdf_path.stem}_test_similarity.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    logger.info(f"Test similarity pipeline wrote {out_path}")
    print(f"\n{'='*80}")
    print("  Test similarity pipeline complete")
    print(f"  PDF: {pdf_path.name}")
    print(f"  Mean similarity (essential fields): {similarity.mean_overall_similarity}")
    print(f"  Fields passing: {similarity.fields_passing}/{similarity.fields_total}")
    print(f"  Output: {out_path}")
    print(f"{'='*80}\n")

    return out_path


def main():
    """Main entry point."""
    parser = setup_argparse()
    args = parser.parse_args()
    
    if args.test_similarity:
        if not validate_test_similarity():
            sys.exit(1)
        config.ensure_directories()
        default_pdf = config.TRAINING_DIR / "pdf_1.pdf"
        pdf_path = Path(f"{config.TRAINING_DIR}/{args.training_pdf}") if args.training_pdf else default_pdf
        try:
            run_test_similarity_pipeline(pdf_path, Path(args.output_dir))
        except Exception as e:
            logger.error(str(e))
            print(f"\n✗ Test similarity pipeline failed: {e}\n")
            sys.exit(1)
        sys.exit(0)

    # Validate config (full pipeline: all API keys)
    if not config.validate():
        print("\nConfiguration validation failed!")
        sys.exit(1)
    if args.openrouter and not config.OPENROUTER_API_KEY:
        print("\nConfiguration error: OPENROUTER_API_KEY must be set when using --openrouter")
        sys.exit(1)
    config.ensure_directories()
    
    # Init extractor
    try:
        extractor = FeatureExtractor(
            use_openai=not args.no_openai,
            use_openrouter=args.openrouter,
            use_claude=not args.no_claude,
            use_gemini=not args.no_gemini,
            use_expert_evaluation=not args.no_expert,
            use_similarity_scoring=not args.no_similarity,
        )
    except Exception as e:
        logger.error(f"Failed to initialize extractor: {str(e)}")
        print(f"\nInitialization failed: {str(e)}\n")
        sys.exit(1)
    
    # Process file (batch/single)
    if args.batch:
        # Batch processing
        stats = process_batch(
            args.batch,
            extractor,
            args.output_dir,
            args.limit
        )
        
        # Exit with error code if any failures
        sys.exit(0 if stats["failed"] == 0 else 1)
    
    else:
        # Single file processing
        success = process_single_pdf(
            args.pdf_path,
            extractor,
            config.OUTPUT_DIR
        )
        
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
