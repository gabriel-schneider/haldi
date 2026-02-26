import abc
import warnings

from haldi import (
    Container,
    ServiceLifetime,
    DIException,
    CircularDependencyException,
)
import pytest


class Interface:
    ...


class ImplA(Interface):
    ...


class ImplB(Interface):
    ...


# ──────────────────────────────────────────────
# Basic resolution
# ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_single_resolve():
    container = Container()
    container.add_transient(Interface, ImplA)

    instance = await container.resolve(Interface)
    assert isinstance(instance, ImplA)


@pytest.mark.asyncio
async def test_multiple_providers_returns_latest():
    container = Container()
    container.add_transient(Interface, ImplA)
    container.add_transient(Interface, ImplB)

    instance = await container.resolve(Interface)
    assert isinstance(instance, ImplB)


@pytest.mark.asyncio
async def test_resolve_unregistered_type_raises():
    container = Container()

    with pytest.raises(DIException, match="could not be resolved"):
        await container.resolve(Interface)


@pytest.mark.asyncio
async def test_try_get_returns_none_for_unregistered():
    container = Container()

    result = await container.try_get(Interface)
    assert result is None


@pytest.mark.asyncio
async def test_get_raises_for_unregistered():
    container = Container()

    with pytest.raises(DIException):
        await container.get(Interface)


@pytest.mark.asyncio
async def test_resolve_non_strict_returns_none():
    container = Container()

    result = await container.resolve(Interface, strict=False)
    assert result is None


# ──────────────────────────────────────────────
# Lifetime: Transient
# ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_transient_is_not_same_instance_simple():
    container = Container()
    container.add_transient(Interface, ImplA)

    assert await container.resolve(Interface) is not await container.resolve(Interface)


@pytest.mark.asyncio
async def test_transient_rejects_instance_registration():
    container = Container()

    with pytest.raises(DIException, match="callable or class type"):
        container.register(Interface, ImplA(), ServiceLifetime.TRANSIENT)


# ──────────────────────────────────────────────
# Lifetime: Singleton
# ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_singleton_simple():
    container = Container()
    container.add_singleton(Interface, ImplA)

    assert await container.resolve(Interface) is await container.resolve(Interface)


@pytest.mark.asyncio
async def test_singleton_accepts_instance():
    container = Container()
    instance = ImplA()
    container.register(Interface, instance, ServiceLifetime.SINGLETON)

    resolved = await container.resolve(Interface)
    assert resolved is instance


@pytest.mark.asyncio
async def test_singleton_persists_across_scopes():
    container = Container()
    container.add_singleton(Interface, ImplA)

    root_instance = await container.resolve(Interface)

    scope = container.create_scope()
    scoped_instance = await scope.resolve(Interface)

    assert root_instance is scoped_instance


# ──────────────────────────────────────────────
# Lifetime: Scoped
# ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_scoped_context():
    container = Container()
    container.add_scoped(Interface, ImplA)

    instance_a = await container.resolve(Interface)

    scope_b = container.create_scope()
    instance_b = await scope_b.resolve(Interface)
    assert instance_b is await scope_b.resolve(Interface)

    scope_c = container.create_scope()
    instance_c = await scope_c.resolve(Interface)
    assert instance_c is await scope_c.resolve(Interface)

    assert instance_a is not instance_b
    assert instance_b is not instance_c


@pytest.mark.asyncio
async def test_scoped_async_context_manager():
    container = Container()
    container.add_scoped(Interface, ImplA)

    async with container.scoped() as scope:
        instance = await scope.resolve(Interface)
        assert isinstance(instance, ImplA)
        assert instance is await scope.resolve(Interface)


@pytest.mark.asyncio
async def test_scoped_accepts_instance():
    container = Container()
    instance = ImplA()
    container.register(Interface, instance, ServiceLifetime.SCOPED)

    resolved = await container.resolve(Interface)
    assert resolved is instance


# ──────────────────────────────────────────────
# Collection injection (list / tuple)
# ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_dependency_resolves_all_providers():
    container = Container()
    container.add_transient(Interface, ImplA)
    container.add_transient(Interface, ImplB)

    class Consumer:
        def __init__(self, impls: list[Interface]):
            self.impls = impls

    container.add_transient(Consumer)

    consumer = await container.resolve(Consumer)
    assert isinstance(consumer.impls, list)
    assert len(consumer.impls) == 2
    assert isinstance(consumer.impls[0], ImplA)
    assert isinstance(consumer.impls[1], ImplB)


@pytest.mark.asyncio
async def test_tuple_dependency_resolves_all_providers():
    container = Container()
    container.add_transient(Interface, ImplA)
    container.add_transient(Interface, ImplB)

    class Consumer:
        def __init__(self, impls: tuple[Interface]):
            self.impls = impls

    container.add_transient(Consumer)

    consumer = await container.resolve(Consumer)
    assert isinstance(consumer.impls, tuple)
    assert len(consumer.impls) == 2
    assert isinstance(consumer.impls[0], ImplA)
    assert isinstance(consumer.impls[1], ImplB)


@pytest.mark.asyncio
async def test_list_dependency_empty_when_no_providers():
    container = Container()

    class Unregistered:
        ...

    class Consumer:
        def __init__(self, impls: list[Unregistered]):
            self.impls = impls

    container.add_transient(Consumer)

    consumer = await container.resolve(Consumer)
    assert consumer.impls == []


@pytest.mark.asyncio
async def test_list_dependency_single_provider():
    container = Container()
    container.add_transient(Interface, ImplA)

    class Consumer:
        def __init__(self, impls: list[Interface]):
            self.impls = impls

    container.add_transient(Consumer)

    consumer = await container.resolve(Consumer)
    assert len(consumer.impls) == 1
    assert isinstance(consumer.impls[0], ImplA)


# ──────────────────────────────────────────────
# Nested / transitive dependencies
# ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_nested_dependency_resolution():
    class Repository:
        pass

    class Service:
        def __init__(self, repo: Repository):
            self.repo = repo

    class Controller:
        def __init__(self, service: Service):
            self.service = service

    container = Container()
    container.add_transient(Repository)
    container.add_transient(Service)
    container.add_transient(Controller)

    controller = await container.resolve(Controller)
    assert isinstance(controller, Controller)
    assert isinstance(controller.service, Service)
    assert isinstance(controller.service.repo, Repository)


@pytest.mark.asyncio
async def test_deep_dependency_chain():
    class A:
        pass

    class B:
        def __init__(self, a: A):
            self.a = a

    class C:
        def __init__(self, b: B):
            self.b = b

    class D:
        def __init__(self, c: C):
            self.c = c

    container = Container()
    container.add_transient(A)
    container.add_transient(B)
    container.add_transient(C)
    container.add_transient(D)

    d = await container.resolve(D)
    assert isinstance(d.c.b.a, A)


# ──────────────────────────────────────────────
# Circular dependency detection
# ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_circular_dependency_raises():
    class ServiceA:
        def __init__(self, b: "ServiceB"):
            self.b = b

    class ServiceB:
        def __init__(self, a: ServiceA):
            self.a = a

    container = Container()
    container.add_transient(ServiceA)
    container.add_transient(ServiceB)

    with pytest.raises(CircularDependencyException):
        await container.resolve(ServiceA)


@pytest.mark.asyncio
async def test_self_referencing_circular_dependency():
    class SelfRef:
        def __init__(self, me: "SelfRef"):
            self.me = me

    container = Container()
    container.add_transient(SelfRef)

    with pytest.raises(CircularDependencyException):
        await container.resolve(SelfRef)


# ──────────────────────────────────────────────
# Factory functions
# ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_lambda_factory():
    container = Container()
    container.register(Interface, lambda: ImplA(), ServiceLifetime.TRANSIENT)

    instance = await container.resolve(Interface)
    assert isinstance(instance, ImplA)


@pytest.mark.asyncio
async def test_named_factory_with_return_annotation():
    def create_impl() -> Interface:
        return ImplA()

    container = Container()
    container.register(Interface, create_impl, ServiceLifetime.TRANSIENT)

    instance = await container.resolve(Interface)
    assert isinstance(instance, ImplA)


@pytest.mark.asyncio
async def test_factory_without_return_annotation_raises():
    def bad_factory():
        return ImplA()

    container = Container()
    container.register(Interface, bad_factory, ServiceLifetime.TRANSIENT)

    with pytest.raises(DIException, match="return type"):
        await container.resolve(Interface)


@pytest.mark.asyncio
async def test_async_factory():
    async def create_impl() -> Interface:
        return ImplA()

    container = Container()
    container.register(Interface, create_impl, ServiceLifetime.TRANSIENT)

    instance = await container.resolve(Interface)
    assert isinstance(instance, ImplA)


@pytest.mark.asyncio
async def test_factory_with_injected_dependencies():
    class Config:
        pass

    def create_impl(config: Config) -> Interface:
        return ImplA()

    container = Container()
    container.add_transient(Config)
    container.register(Interface, create_impl, ServiceLifetime.TRANSIENT)

    instance = await container.resolve(Interface)
    assert isinstance(instance, ImplA)


@pytest.mark.asyncio
async def test_singleton_lambda_factory_caches():
    call_count = 0

    def make():
        nonlocal call_count
        call_count += 1
        return ImplA()

    container = Container()
    container.register(Interface, lambda: make(), ServiceLifetime.SINGLETON)

    a = await container.resolve(Interface)
    b = await container.resolve(Interface)
    assert a is b
    assert call_count == 1


# ──────────────────────────────────────────────
# Container self-injection
# ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_container_self_injection():
    class NeedsContainer:
        def __init__(self, container: Container):
            self.container = container

    container = Container()
    container.add_transient(NeedsContainer)

    instance = await container.resolve(NeedsContainer)
    assert instance.container is container


# ──────────────────────────────────────────────
# Async context manager support
# ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_async_context_manager_service():
    entered = False
    exited = False

    class ManagedResource:
        async def __aenter__(self):
            nonlocal entered
            entered = True
            return self

        async def __aexit__(self, *args):
            nonlocal exited
            exited = True

    container = Container()
    container.add_transient(ManagedResource)

    instance = await container.resolve(ManagedResource)
    assert entered
    assert isinstance(instance, ManagedResource)

    await container.close()
    assert exited


@pytest.mark.asyncio
async def test_scoped_close_cleans_context_managers():
    closed = False

    class Resource:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            nonlocal closed
            closed = True

    container = Container()
    container.add_scoped(Resource)

    async with container.scoped() as scope:
        await scope.resolve(Resource)

    assert closed


# ──────────────────────────────────────────────
# get_executor
# ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_executor_injects_dependencies():
    class Dep:
        pass

    def my_func(dep: Dep) -> str:
        return "ok"

    container = Container()
    container.add_transient(Dep)

    executor = await container.get_executor(my_func)
    result = executor()
    assert result == "ok"


@pytest.mark.asyncio
async def test_get_executor_with_collection():
    class Plugin:
        pass

    class PluginA(Plugin):
        pass

    class PluginB(Plugin):
        pass

    def handler(plugins: list[Plugin]) -> int:
        return len(plugins)

    container = Container()
    container.add_transient(Plugin, PluginA)
    container.add_transient(Plugin, PluginB)

    executor = await container.get_executor(handler)
    assert executor() == 2


@pytest.mark.asyncio
async def test_get_executor_non_strict_skips_missing():
    """When strict=False, unresolved deps are not injected — caller must supply them or the function needs defaults."""

    def my_func(dep: Interface = None) -> str:
        return "fallback" if dep is None else "resolved"

    container = Container()

    executor = await container.get_executor(my_func, strict=False)
    result = executor()
    assert result == "fallback"


@pytest.mark.asyncio
async def test_get_executor_non_strict_no_default_raises_at_call_time():
    """When strict=False and param has no default, the partial won't fill it — calling raises TypeError."""

    def my_func(dep: Interface) -> str:
        return "nope"

    container = Container()

    executor = await container.get_executor(my_func, strict=False)
    with pytest.raises(TypeError):
        executor()


@pytest.mark.asyncio
async def test_get_executor_strict_raises_for_missing():
    def my_func(dep: Interface) -> str:
        return "nope"

    container = Container()

    with pytest.raises(DIException):
        await container.get_executor(my_func, strict=True)


# ──────────────────────────────────────────────
# Primitive type skipping
# ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_primitive_params_are_skipped():
    class MyService:
        def __init__(self, name: str = "default", count: int = 0):
            self.name = name
            self.count = count

    container = Container()
    container.add_transient(MyService)

    instance = await container.resolve(MyService)
    assert instance.name == "default"
    assert instance.count == 0


# ──────────────────────────────────────────────
# Lifetime compatibility warning
# ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_singleton_depending_on_scoped_warns():
    class ScopedService:
        pass

    class SingletonService:
        def __init__(self, scoped: ScopedService):
            self.scoped = scoped

    container = Container()
    container.add_scoped(ScopedService)
    container.add_singleton(SingletonService)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        await container.resolve(SingletonService)

    assert len(w) == 1
    assert "Singleton service" in str(w[0].message)
    assert "Scoped service" in str(w[0].message)


@pytest.mark.asyncio
async def test_transient_depending_on_scoped_no_warning():
    class ScopedService:
        pass

    class TransientService:
        def __init__(self, scoped: ScopedService):
            self.scoped = scoped

    container = Container()
    container.add_scoped(ScopedService)
    container.add_transient(TransientService)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        await container.resolve(TransientService)

    assert len(w) == 0


# ──────────────────────────────────────────────
# Scope isolation
# ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_scoped_services_are_independent_per_scope():
    class ScopedService:
        pass

    container = Container()
    container.add_scoped(ScopedService)

    scope_a = container.create_scope()
    scope_b = container.create_scope()

    a = await scope_a.resolve(ScopedService)
    b = await scope_b.resolve(ScopedService)

    assert a is not b
    assert a is await scope_a.resolve(ScopedService)
    assert b is await scope_b.resolve(ScopedService)


@pytest.mark.asyncio
async def test_transient_in_scope_creates_new_each_time():
    container = Container()
    container.add_transient(Interface, ImplA)

    scope = container.create_scope()
    a = await scope.resolve(Interface)
    b = await scope.resolve(Interface)
    assert a is not b


@pytest.mark.asyncio
async def test_scope_does_not_share_scoped_instance_with_parent():
    class ScopedService:
        pass

    container = Container()
    container.add_scoped(ScopedService)

    parent_instance = await container.resolve(ScopedService)

    scope = container.create_scope()
    child_instance = await scope.resolve(ScopedService)

    assert parent_instance is not child_instance


# ──────────────────────────────────────────────
# Mixed lifetimes
# ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_mixed_lifetimes_in_dependency_graph():
    class SingletonDep:
        pass

    class TransientDep:
        pass

    class Root:
        def __init__(self, s: SingletonDep, t: TransientDep):
            self.s = s
            self.t = t

    container = Container()
    container.add_singleton(SingletonDep)
    container.add_transient(TransientDep)
    container.add_transient(Root)

    root1 = await container.resolve(Root)
    root2 = await container.resolve(Root)

    assert root1 is not root2
    assert root1.s is root2.s  # singleton shared
    assert root1.t is not root2.t  # transient not shared


# ──────────────────────────────────────────────
# Abstract base class as interface
# ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_abc_as_interface():
    class AbstractRepo(abc.ABC):
        @abc.abstractmethod
        def get(self):
            ...

    class ConcreteRepo(AbstractRepo):
        def get(self):
            return 42

    container = Container()
    container.add_transient(AbstractRepo, ConcreteRepo)

    repo = await container.resolve(AbstractRepo)
    assert isinstance(repo, ConcreteRepo)
    assert repo.get() == 42


# ──────────────────────────────────────────────
# Parameters without annotation are skipped
# ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_unannotated_params_are_skipped():
    class MyService:
        def __init__(self, unknown="hello"):
            self.unknown = unknown

    container = Container()
    container.add_transient(MyService)

    instance = await container.resolve(MyService)
    assert instance.unknown == "hello"


# ──────────────────────────────────────────────
# Register same type with same concrete
# ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_registering_same_type_twice_appends():
    container = Container()
    container.add_transient(Interface, ImplA)
    container.add_transient(Interface, ImplA)

    class Consumer:
        def __init__(self, impls: list[Interface]):
            self.impls = impls

    container.add_transient(Consumer)

    consumer = await container.resolve(Consumer)
    assert len(consumer.impls) == 2


# ──────────────────────────────────────────────
# Close on empty container
# ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_close_on_empty_container():
    container = Container()
    await container.close()  # should not raise
