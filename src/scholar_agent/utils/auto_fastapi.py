# auto_fastapi.py
from __future__ import annotations

# Standard
from typing import Any, Dict, Type
import inspect
import typing

from fastapi import APIRouter, Depends
from pydantic import BaseModel, create_model


# --------------------------------------------------------------------------- #
# 1. Helper: Turn an arbitrary type into a Pydantic model if it isn’t one
# --------------------------------------------------------------------------- #
def _to_pydantic_type(tp: Any) -> Any:
    """
    Convert a Python type hint into a Pydantic type.
    - If it's already a Pydantic BaseModel subclass, keep it.
    - If it's a simple builtin, keep it.
    - If it's a complex type (List[int] etc.) it is accepted by pydantic directly.
    """
    if isinstance(tp, type) and issubclass(tp, BaseModel):
        return tp
    # FastAPI/pydantic understand typing constructs as is
    return tp


# --------------------------------------------------------------------------- #
# 2. Core decorator
# --------------------------------------------------------------------------- #
def auto_fastapi(cls: Type[Any]) -> Type[Any]:
    """
    Decorator that turns a plain Python class into a FastAPI router.
    The router is stored on the class as `__router__` and can be added to
    any FastAPI instance with `app.include_router(MyClass.__router__)`.

    Public methods (no leading underscore) become POST endpoints.
    """

    router = APIRouter()

    for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
        if name.startswith("_"):
            continue  # skip private helpers

        # ------------------------------------------------------------------- #
        # Build request model from method parameters
        # ------------------------------------------------------------------- #
        sig = inspect.signature(method)
        request_fields: Dict[str, Any] = {}

        for pname, p in sig.parameters.items():
            if pname == "self":
                continue

            # Type hint or default to `Any`
            p_type = _to_pydantic_type(
                p.annotation if p.annotation is not p.empty else Any
            )
            required = p.default is p.empty
            request_fields[pname] = (p_type, ... if required else p.default)

        RequestModel = create_model(
            f"{cls.__name__}{name.capitalize()}Request", **request_fields
        )
        RequestModel.model_rebuild()

        # ------------------------------------------------------------------- #
        # Build response model from return annotation
        # ------------------------------------------------------------------- #
        return_type = _to_pydantic_type(
            sig.return_annotation if sig.return_annotation is not sig.empty else Any
        )

        # If the return type is already a BaseModel we can use it as‑is
        if isinstance(return_type, type) and issubclass(return_type, BaseModel):
            ResponseModel = return_type
        else:
            # Wrap primitive return values in a generic `result` field
            ResponseModel = create_model(
                f"{cls.__name__}{name.capitalize()}Response", result=(return_type, ...)
            )
            ResponseModel.model_rebuild()

        # ------------------------------------------------------------------- #
        # Create the actual endpoint
        # ------------------------------------------------------------------- #
        async def endpoint(
            req: RequestModel,  # type: ignore[assignment]
            instance=Depends(lambda: cls()),
        ):
            """
            Dependency injection ensures that each HTTP request receives
            a fresh instance of the underlying class (you can change this
            to a shared instance if you want a singleton).
            """
            # Call the original method
            args = req.dict()
            result = await _maybe_await(method(instance, **args))

            # If the method already returned a BaseModel, use it directly
            if isinstance(result, BaseModel):
                return result

            # Otherwise wrap it
            return ResponseModel(result=result)

        # Register endpoint
        router.add_api_route(
            path=f"/{name}",  # e.g. /add
            endpoint=endpoint,
            methods=["POST"],
            summary=f"Proxy for {cls.__name__}.{name}",
            response_model=ResponseModel,
        )

    # Attach router to the class for later inclusion
    setattr(cls, "__router__", router)
    return cls


# --------------------------------------------------------------------------- #
# 3. Helper: Await if the underlying method is async
# --------------------------------------------------------------------------- #
async def _maybe_await(value):
    """
    Utility that awaits the value if it is a coroutine, otherwise returns it.
    """
    if typing.iscoroutine(value):
        return await value
    return value
