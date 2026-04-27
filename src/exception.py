import sys
import traceback


class PipelineException(Exception):
    """Custom exception for the EGFR Drug Discovery Pipeline."""

    def __init__(self, error_message: str, error_detail: sys = sys):
        super().__init__(error_message)
        self.error_message = self._build_error_message(error_message, error_detail)

    @staticmethod
    def _build_error_message(error_message: str, error_detail: sys) -> str:
        _, _, exc_tb = error_detail.exc_info()
        if exc_tb is not None:
            file_name = exc_tb.tb_frame.f_code.co_filename
            line_number = exc_tb.tb_lineno
        else:
            file_name = "unknown"
            line_number = 0
        return (
            f"Error in [{file_name}] at line [{line_number}]: {error_message}"
        )

    def __str__(self):
        return self.error_message

    def __repr__(self):
        return f"PipelineException({self.error_message!r})"
