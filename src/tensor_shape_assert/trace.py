import inspect

from typing import NamedTuple
from .types import VariablesType


class TracedVariableAssignment(NamedTuple):
    name: str | None
    annotation: str | None
    shape: tuple[int, ...]
    assignments: VariablesType

    def __str__(self) -> str:
        return (
            f"{self.name} : ({self.annotation}) -> shape {self.shape} => {self.assignments}"
        )

class TracedFunctionCall(NamedTuple):
    function_name: str | None
    file: str | None
    line: int
    stack_index: int
    call_index: int

    def __str__(self) -> str:
        return (
            f"{self.function_name} (defined at {self.file}:{self.line}), "
            f"stack index: {self.stack_index}, call index: {self.call_index}"
        )

class TraceRecord(NamedTuple):
    # function metadata
    function: TracedFunctionCall
    assignment: TracedVariableAssignment


_trace_stack: list[TracedFunctionCall] = []
_trace_records: list[TraceRecord] = []
_trace_enabled: bool = False

def add_function_trace(fn):
    global _trace_enabled
    if not _trace_enabled:
        return

    try:
        source_file = inspect.getsourcefile(fn)
        source_line = inspect.getsourcelines(fn)[1]
    except OSError:
        source_file = f"{fn.__module__}.{fn.__qualname__}"
        source_line = -1

    trace_record = TracedFunctionCall(
        function_name=fn.__name__,
        file=source_file,
        line=source_line,
        stack_index=len(_trace_stack),
        call_index=len(_trace_records)
    )
    _trace_stack.append(trace_record)

def add_assignment_trace(
        name: str | None,
        annotation: str | None,
        shape: tuple[int, ...],
        assignments: VariablesType
    ):
    global _trace_enabled
    if not _trace_enabled:
        return
    
    if len(_trace_stack) == 0:
        raise RuntimeError(
            "Internal error: Tried to add assignment trace without an active "
            "function trace."
        )
    
    function_trace = _trace_stack[-1]
    _trace_records.append(
        TraceRecord(
            function=function_trace,
            assignment=TracedVariableAssignment(
                name=name,
                annotation=annotation,
                shape=shape,
                assignments=assignments.copy()
            )
        )
    )

def finalize_function_trace():
    global _trace_enabled
    if not _trace_enabled:
        return
    
    if len(_trace_stack) == 0:
        raise RuntimeError(
            "Internal error: Tried to finalize function trace without an active "
            "function trace."
        )
    
    _trace_stack.pop()


def start_trace_recording():
    global _trace_enabled
    _trace_enabled = True
    _trace_records.clear()
    _trace_stack.clear()

def stop_trace_recording() -> list[TraceRecord]:
    global _trace_enabled
    _trace_enabled = False
    records = _trace_records.copy()
    _trace_records.clear()
    _trace_stack.clear()
    return records

def trace_records_to_string(records: list[TraceRecord]) -> str:
    lines = []
    mentioned_calls = set()

    for record in records:
        indentation = "|   " * record.function.stack_index

        if record.function not in mentioned_calls:
            lines.append(f"{indentation}\n{indentation}{record.function}")
            mentioned_calls.add(record.function)
                
        lines.append(f"{indentation}|   {record.assignment}")
    return "\n".join(lines)
