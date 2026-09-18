import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ApiSettings:
    source_root: Path = Path("C:/VisionGuard/videos")
    output_root: Path = Path("C:/VisionGuard/outputs")
    config_root: Path = Path("C:/VisionGuard/configs")
    allowed_origins: tuple[str, ...] = (
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    )
    redis_url: str = "redis://127.0.0.1:6379/0"
    queue_name: str = "visionguard"

    @classmethod
    def from_environment(cls) -> "ApiSettings":
        origins = os.environ.get(
            "VISIONGUARD_ALLOWED_ORIGINS",
            "http://127.0.0.1:5173,http://localhost:5173",
        )
        return cls(
            source_root=Path(
                os.environ.get("VISIONGUARD_SOURCE_ROOT", "C:/VisionGuard/videos")
            ),
            output_root=Path(
                os.environ.get("VISIONGUARD_OUTPUT_ROOT", "C:/VisionGuard/outputs")
            ),
            config_root=Path(
                os.environ.get("VISIONGUARD_CONFIG_ROOT", "C:/VisionGuard/configs")
            ),
            allowed_origins=tuple(
                origin.strip()
                for origin in origins.split(",")
                if origin.strip()
            ),
            redis_url=os.environ.get(
                "VISIONGUARD_REDIS_URL",
                "redis://127.0.0.1:6379/0",
            ),
            queue_name=os.environ.get("VISIONGUARD_QUEUE_NAME", "visionguard"),
        )

    def validate_analysis_paths(
        self,
        source_path: str,
        output_path: str,
        config_path: str,
    ) -> tuple[Path, Path, Path]:
        source = self._resolve_within_root(
            source_path,
            self.source_root,
            "Source video",
        )
        if not source.is_file():
            raise FileNotFoundError(f"Source video does not exist: {source}")

        output = self.output_path(output_path)
        if source == output:
            raise ValueError("Input and output video paths must be different.")

        config = self._resolve_within_root(
            config_path,
            self.config_root,
            "Configuration",
        )
        if not config.is_file():
            raise FileNotFoundError(f"Configuration file does not exist: {config}")

        return source, output, config

    def output_path(self, output_path: str | Path) -> Path:
        return self._resolve_within_root(
            output_path,
            self.output_root,
            "Output video",
        )

    @staticmethod
    def _resolve_within_root(
        candidate_path: str | Path,
        root_path: Path,
        label: str,
    ) -> Path:
        root = Path(root_path).resolve()
        candidate = Path(candidate_path).resolve()

        try:
            candidate.relative_to(root)
        except ValueError as error:
            raise ValueError(f"{label} path must be inside: {root}") from error

        return candidate
