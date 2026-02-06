from pipeline.config import PipelineConfig


def run_pipeline(*args, **kwargs):
    from pipeline.main import run_pipeline as _run_pipeline

    return _run_pipeline(*args, **kwargs)


__all__ = ["PipelineConfig", "run_pipeline"]
