import abc
from asyncio import Lock
from dataclasses import dataclass
import dataclasses
from enum import IntFlag, auto
import functools
import types
import typing as t
import warnings
from inspect import Signature, Parameter, isclass, iscoroutinefunction
from contextlib import asynccontextmanager


def is_lambda_function(obj):
    return isinstance(obj, types.LambdaType) and obj.__name__ == "<lambda>"


def is_context_manager(obj):
    return hasattr(obj, "__aenter__") and hasattr(obj, "__aexit__")


class ServiceLifetime(IntFlag):
    TRANSIENT = auto()
    SCOPED = auto()
    SINGLETON = auto()
    ONCE = SCOPED | SINGLETON


T = t.TypeVar("T")
U = t.TypeVar("U")


class DIException(Exception):
    pass


class CircularDependencyException(DIException):
    pass


class InstantiationException(DIException):
    pass


class DependencyResolutionContext:
    def __init__(self, container: "Container"):
        self._container = container
        self._chain: t.Set[t.Type] = set()
        self._stack: t.List[t.Tuple[t.Type, ServiceLifetime]] = []

    def push(self, typ: t.Type[T], lifetime: ServiceLifetime):
        if typ in self._chain:
            raise CircularDependencyException()

        self._chain.add(typ)
        self._stack.append((typ, lifetime))

    def remove(self, typ: t.Type[T]):
        self._chain.remove(typ)

    def check_lifetime_compatibility(self, dependency_type: t.Type, dependency_lifetime: ServiceLifetime):
        """Check if a Singleton service is depending on a Scoped service."""
        if not self._stack:
            return
        
        # Get the current service being resolved (parent)
        parent_type, parent_lifetime = self._stack[-1]
        
        # Check if Singleton is depending on Scoped
        if (parent_lifetime == ServiceLifetime.SINGLETON and 
            dependency_lifetime == ServiceLifetime.SCOPED):
            parent_name = parent_type.__name__ if hasattr(parent_type, '__name__') else str(parent_type)
            dependency_name = dependency_type.__name__ if hasattr(dependency_type, '__name__') else str(dependency_type)
            warnings.warn(
                f"Singleton service '{parent_name}' is depending on Scoped service '{dependency_name}'. "
                f"This may lead to unexpected behavior as the Scoped service will be "
                f"captured and live for the entire application lifetime.",
                UserWarning,
                stacklevel=4
            )

    def __call__(self, typ: t.Type, lifetime: ServiceLifetime):
        self.push(typ, lifetime)
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        elem = self._stack.pop()
        self._chain.remove(elem[0])


@dataclass
class ServiceProvider:
    base_type: t.Type
    concrete_type_or_factory: t.Type | t.Callable[..., t.Any]
    lifetime: ServiceLifetime
    instance: t.Any
    lock: Lock


class Container:
    _UNRESOLVED = object()
    PRIMITIVE_TYPES = (
        str,
        int,
    )

    def __init__(self, container: "Container" = None): #type: ignore
        self._registered_services: t.Dict[t.Any, list[ServiceProvider]] = {}
        self._pending_ctx_managers = []

        if container:
            for base_type, providers in container.registered_services.items():
                for provider in providers:
                    if provider.lifetime is ServiceLifetime.SINGLETON:
                        self._add_provider(base_type, provider)
                    else:
                        self._add_provider(
                            base_type, dataclasses.replace(provider, instance=None)
                        )

    def _add_provider(self, key: str | t.Type, provider: ServiceProvider):
        if providers := self._registered_services.get(key, None):
            providers.append(provider)
        else:
            self._registered_services[key] = [provider]

    def register(
        self,
        base_type: t.Type[T],
        concrete_type_or_factory: t.Type[U] | t.Callable[..., U] | t.Any,
        lifetime: ServiceLifetime,
    ):

        if lifetime not in ServiceLifetime.ONCE:
            if not callable(concrete_type_or_factory) and not isclass(
                concrete_type_or_factory
            ):
                raise DIException(
                    "Transient lifetime requires a callable or class type, not a instance."
                )

        service_provider = ServiceProvider(
            base_type, concrete_type_or_factory, lifetime, None, Lock()
        )

        self._add_provider(base_type, service_provider)
        self._add_provider(base_type.__name__, service_provider)

    @property
    def registered_services(self):
        return self._registered_services

    def _get_service_definition(
        self, name_or_type: str | t.Type
    ) -> list[ServiceProvider]:
        return self._registered_services.get(name_or_type, [])

    def add_transient(self, typ, concrete_typ=None):
        return self.register(typ, concrete_typ or typ, ServiceLifetime.TRANSIENT)

    def add_scoped(self, typ, concrete_typ=None):
        return self.register(typ, concrete_typ or typ, ServiceLifetime.SCOPED)

    def add_singleton(self, typ, concrete_typ=None):
        return self.register(typ, concrete_typ or typ, ServiceLifetime.SINGLETON)

    def create_scope(self):
        scoped_container = Container(self)
        return scoped_container

    @asynccontextmanager
    async def scoped(self):
        scope = self.create_scope()
        try:
            yield scope
        finally:
            await scope.close()

    @staticmethod
    def _is_collection_type(typ) -> tuple[bool, t.Type | None, type | None]:
        """Check if typ is list[X] or tuple[X]. Returns (is_collection, inner_type, collection_type)."""
        origin = t.get_origin(typ)
        if origin is list or origin is tuple:
            args = t.get_args(typ)
            if args:
                return True, args[0], origin
        return False, None, None

    @staticmethod
    def _is_union_type(typ) -> bool:
        origin = t.get_origin(typ)
        return origin in (t.Union, types.UnionType) or isinstance(typ, types.UnionType)

    @staticmethod
    def _format_type_name(typ: t.Any) -> str:
        if isinstance(typ, str):
            return typ
        return getattr(typ, "__name__", str(typ))

    async def _resolve_annotation(
        self,
        annotation: t.Any,
        context: DependencyResolutionContext,
        strict: bool = True,
        check_lifetime_compatibility: bool = False,
    ) -> t.Any:
        is_collection, inner_type, collection_type = self._is_collection_type(annotation)
        if is_collection:
            resolved = await self._resolve_all(inner_type, context)
            if collection_type is tuple:
                return tuple(resolved)
            return resolved

        if self._is_union_type(annotation):
            return await self._resolve_union(
                annotation,
                context,
                strict=strict,
                check_lifetime_compatibility=check_lifetime_compatibility,
            )

        return await self._resolve_registered_type(
            annotation,
            context,
            strict=strict,
            check_lifetime_compatibility=check_lifetime_compatibility,
        )

    async def _resolve_union(
        self,
        desired_type: t.Any,
        context: DependencyResolutionContext,
        strict: bool = True,
        check_lifetime_compatibility: bool = False,
    ) -> t.Any:
        for branch_type in t.get_args(desired_type):
            if branch_type is type(None):
                return None

            resolved = await self._resolve_annotation(
                branch_type,
                context,
                strict=False,
                check_lifetime_compatibility=check_lifetime_compatibility,
            )
            if resolved is not self._UNRESOLVED:
                return resolved

        if strict:
            raise DIException(
                f"Type {self._format_type_name(desired_type)} could not be resolved."
            )

        return self._UNRESOLVED

    async def _resolve_registered_type(
        self,
        desired_type_or_callable: t.Type[T] | t.Callable[..., T],
        context: DependencyResolutionContext,
        strict: bool = True,
        check_lifetime_compatibility: bool = False,
    ) -> T | object:

        if desired_type_or_callable is self.__class__:
            return self

        desired_type = desired_type_or_callable
        providers = self._get_service_definition(desired_type)

        if not providers:
            if strict:
                raise DIException(
                    f"Type {self._format_type_name(desired_type)} could not be resolved."
                )
            return self._UNRESOLVED

        provider = providers[-1]
        resolved = await self._resolve_provider(provider, desired_type, context)

        if check_lifetime_compatibility:
            context.check_lifetime_compatibility(provider.base_type, provider.lifetime)

        return resolved

    def extract_depedencies(self, callable_: t.Callable[..., t.Any]):
        signature = Signature.from_callable(callable_)

        dependencies = {}
        for param_name, param in signature.parameters.items():
            if param.annotation is Parameter.empty:
                continue
            dependencies[param_name] = param.annotation
        return dependencies

    async def _instantiate_provider(
        self,
        factory: t.Any,
        context: DependencyResolutionContext,
    ) -> t.Any:
        kwargs = {}

        if isclass(factory):
            dependencies = self.extract_depedencies(factory.__init__)
        elif callable(factory):
            dependencies = self.extract_depedencies(factory)
        else:
            dependencies = {}

        for key, typ in dependencies.items():
            if typ in self.PRIMITIVE_TYPES:
                continue

            resolved = await self._resolve_annotation(
                typ,
                context,
                check_lifetime_compatibility=True,
            )
            if resolved is self._UNRESOLVED:
                continue
            kwargs[key] = resolved

        if iscoroutinefunction(factory):
            instance = await factory(**kwargs)
        elif callable(factory):
            instance = factory(**kwargs)
            if is_context_manager(instance):
                self._pending_ctx_managers.append(instance)
                instance = await instance.__aenter__()
        else:
            instance = factory

        return instance

    async def _resolve_provider(
        self,
        provider: ServiceProvider,
        desired_type: t.Type,
        context: DependencyResolutionContext,
    ) -> t.Any:
        concrete_type_or_callable = provider.concrete_type_or_factory
        lifetime = provider.lifetime

        factory = concrete_type = concrete_type_or_callable

        if not isclass(concrete_type_or_callable) and callable(
            concrete_type_or_callable
        ):
            if is_lambda_function(concrete_type_or_callable):
                concrete_type = desired_type
            else:
                sig = Signature.from_callable(concrete_type_or_callable)
                if (
                    not sig.return_annotation
                    or sig.return_annotation is Signature.empty
                ):
                    raise DIException(
                        "Callable is not a lambda function AND has no return type."
                    )
                concrete_type = sig.return_annotation

            factory = concrete_type_or_callable

        # Push the concrete type to the context chain, prevent circular dependency loop.
        with context(concrete_type, lifetime):
            if lifetime in ServiceLifetime.ONCE:
                async with provider.lock:
                    if provider.instance is not None:
                        return provider.instance

                    instance = await self._instantiate_provider(factory, context)
                    provider.instance = instance
                    return instance

            return await self._instantiate_provider(factory, context)

    async def _resolve(
        self,
        desired_type_or_callable: t.Type[T] | t.Callable[..., T],
        context: DependencyResolutionContext,
        strict: bool = True,
    ) -> T | None:
        resolved = await self._resolve_registered_type(
            desired_type_or_callable,
            context,
            strict=strict,
        )
        if resolved is self._UNRESOLVED:
            return None
        return resolved  # type: ignore

    async def _resolve_all(
        self,
        desired_type: t.Type[T],
        context: DependencyResolutionContext,
    ) -> list[T]:
        providers = self._get_service_definition(desired_type)
        if not providers:
            return []

        return [
            await self._resolve_provider(provider, desired_type, context)
            for provider in providers
        ]

    async def get(self, desired_type: t.Type[T]) -> T| t.Sequence[T]:
        return await self.resolve(desired_type, True)  # type: ignore

    async def try_get(self, desired_type: t.Type[T]) -> T | t.Sequence[T] | None:
        return await self.resolve(desired_type, False)

    async def resolve(self, desired_type: t.Type[T], strict: bool = True) -> T | t.Sequence[T] | None:
        context = DependencyResolutionContext(self)
        resolved = await self._resolve_annotation(desired_type, context, strict)
        if resolved is self._UNRESOLVED:
            return None
        return resolved

    async def get_executor(
        self, callable_: t.Callable[..., T], strict: bool = True
    ) -> t.Callable[..., T]:
        dependencies = self.extract_depedencies(callable_)
        context = DependencyResolutionContext(self)
        kwargs = {}
        for key, typ in dependencies.items():
            if typ in self.PRIMITIVE_TYPES:
                continue

            resolved = await self._resolve_annotation(typ, context, strict)
            if resolved is self._UNRESOLVED:
                continue

            kwargs[key] = resolved

        return functools.partial(callable_, **kwargs)

    async def close(self):
        for ctx_manager in self._pending_ctx_managers:
            await ctx_manager.__aexit__(None, None, None)
        self._pending_ctx_managers.clear()
