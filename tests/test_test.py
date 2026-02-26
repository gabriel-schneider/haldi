from haldi import Container
import pytest


class Interface:
    ...


class ImplA(Interface):
    ...


class ImplB(Interface):
    ...


@pytest.mark.asyncio
async def test_multiple_resolves():
    container = Container()

    container.add_transient(Interface, ImplA)
    container.add_transient(Interface, ImplB)

    instances = await container.resolve(Interface)
    assert isinstance(instances, list)

    impla, implb = instances
    assert isinstance(impla, ImplA)
    assert isinstance(implb, ImplB)


@pytest.mark.asyncio
async def test_single_resolve():
    container = Container()

    container.add_transient(Interface, ImplA)    

    instance = await container.resolve(Interface)
    assert isinstance(instance, ImplA)

@pytest.mark.asyncio
async def test_singleton_simple():
    container = Container()

    container.add_singleton(Interface, ImplA)    

    assert await container.resolve(Interface) is await container.resolve(Interface)


@pytest.mark.asyncio
async def test_transient_is_not_same_instance_simple():
    container = Container()

    container.add_transient(Interface, ImplA)    

    assert await container.resolve(Interface) is not await container.resolve(Interface)
    
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