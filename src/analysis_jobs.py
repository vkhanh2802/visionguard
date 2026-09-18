import logging
from pathlib import Path

from src.analysis_service import create_analysis_job, execute_analysis_job
from src.config import AppConfig
from src.database import SQLiteRepository
from src.logging_config import configure_logging
from src.pipeline import PipelineResult


def execute_queued_analysis(
    run_id: str,
    config_data: dict,
    source_path: str,
    output_path: str,
    database_path: str,
) -> PipelineResult:
    repository = SQLiteRepository(database_path)
    repository.start_run(run_id)

    try:
        config = AppConfig.model_validate(config_data)
        logger = configure_logging(config.logging.level)
        job = create_analysis_job(
            config=config,
            source_path=Path(source_path),
            output_path=Path(output_path),
            repository=repository,
            logger=logger,
            run_id=run_id,
            create_run=False,
        )
        return execute_analysis_job(job)
    except Exception as error:
        run = repository.get_run(run_id)
        if run is not None and run["status"] != "failed":
            repository.fail_run(run_id, str(error))
        raise
